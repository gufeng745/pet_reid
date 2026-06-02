import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image
import numpy as np


def extract_pet_color_histogram(img_tensor, trimap_tensor, bins=16):
    """在 trimap 前景区域上提取 HSV 颜色直方图

    Args:
        img_tensor: (3, H, W) RGB 图像，值域 [0, 1]
        trimap_tensor: (1, H, W) 分割图，1=前景, 2=背景, 3=边界
        bins: 每通道的直方图 bin 数
    Returns:
        color_feat: (bins*3,) 归一化颜色直方图
    """
    # 只取前景像素 (trimap == 1)
    mask = (trimap_tensor.squeeze(0) == 1)  # (H, W)
    if mask.sum() < 50:
        # 前景像素太少，退回全图
        mask = torch.ones_like(mask, dtype=torch.bool)

    # RGB -> HSV (手动实现，兼容纯 torch)
    r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
    max_c, _ = torch.stack([r, g, b]).max(dim=0)
    min_c, _ = torch.stack([r, g, b]).min(dim=0)
    diff = max_c - min_c + 1e-8

    # H channel
    h = torch.zeros_like(r)
    mask_r = (max_c == r) & (diff > 1e-8)
    mask_g = (max_c == g) & (diff > 1e-8)
    mask_b = (max_c == b) & (diff > 1e-8)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    s = torch.where(max_c > 1e-8, diff / (max_c + 1e-8), torch.zeros_like(max_c))
    v = max_c

    # 只在前景像素上计算直方图
    h_fg = h[mask]
    s_fg = s[mask]
    v_fg = v[mask]

    hist_h = torch.histc(h_fg, bins=bins, min=0, max=360)
    hist_s = torch.histc(s_fg, bins=bins, min=0, max=1)
    hist_v = torch.histc(v_fg, bins=bins, min=0, max=1)

    color_feat = torch.cat([hist_h, hist_s, hist_v]).float()
    color_feat = color_feat / (color_feat.sum() + 1e-8)  # 归一化为概率分布
    return color_feat


def get_dino_augmentation(crop_scale=(0.4, 1.0)):
    """DINO 风格数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    """评估用的标准变换"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class PetDistillationDataset(Dataset):
    """蒸馏用数据集：对同一张图生成两个不同增强视角

    当 use_trimap=True 时，额外返回原始图的颜色直方图（基于 trimap 前景区域），
    用于颜色感知训练。
    """

    def __init__(self, root, split='trainval', transform1=None, transform2=None, use_trimap=False):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            download=True,
            target_types='segmentation' if use_trimap else 'category',
        )
        self.transform1 = transform1 or get_dino_augmentation(crop_scale=(0.4, 1.0))
        self.transform2 = transform2 or get_dino_augmentation(crop_scale=(0.4, 1.0))
        self.use_trimap = use_trimap
        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_trimap:
            img, trimap = self.dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            if not isinstance(trimap, Image.Image):
                trimap = Image.fromarray(np.array(trimap).astype(np.uint8))

            # 原始图用于提取颜色（不能做颜色增强，否则颜色信息被破坏）
            img_tensor = self.base_transform(img)  # (3, 224, 224), [0,1]
            # trimap 需要同步 resize + center crop
            trimap_resized = transforms.functional.resize(
                trimap, 256, interpolation=transforms.InterpolationMode.NEAREST
            )
            trimap_cropped = transforms.functional.center_crop(trimap_resized, 224)
            trimap_tensor = torch.from_numpy(np.array(trimap_cropped)).long()  # (224, 224)
            trimap_tensor = trimap_tensor.unsqueeze(0)  # (1, 224, 224)
            color_feat = extract_pet_color_histogram(img_tensor, trimap_tensor)

            # 增强视图用于蒸馏
            view1 = self.transform1(img)
            view2 = self.transform2(img)
            return view1, view2, color_feat
        else:
            img, label = self.dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            view1 = self.transform1(img)
            view2 = self.transform2(img)
            return view1, view2, label


class PetEvalDataset(Dataset):
    """评估用数据集：单视角 + 标签"""

    def __init__(self, root, split='test', transform=None):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            download=True,
        )
        self.transform = transform or get_eval_transform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return self.transform(img), label


def create_dataloaders(dataset_root, batch_size=64, num_workers=0, use_trimap=False):
    """创建训练和评估 DataLoader

    Args:
        use_trimap: 是否加载 trimap 标注用于颜色感知训练
    """
    train_dataset = PetDistillationDataset(root=dataset_root, split='trainval', use_trimap=use_trimap)
    eval_dataset = PetEvalDataset(root=dataset_root, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, eval_loader


if __name__ == '__main__':
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_loader, eval_loader = create_dataloaders(dataset_root, batch_size=8)
    v1, v2, labels = next(iter(train_loader))
    print(f"view1: {v1.shape}, view2: {v2.shape}, labels: {labels.shape}")
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
