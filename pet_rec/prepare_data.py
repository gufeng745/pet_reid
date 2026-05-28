import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image


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
    """蒸馏用数据集：对同一张图生成两个不同增强视角"""

    def __init__(self, root, split='trainval', transform1=None, transform2=None):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            download=True,
        )
        self.transform1 = transform1 or get_dino_augmentation(crop_scale=(0.4, 1.0))
        self.transform2 = transform2 or get_dino_augmentation(crop_scale=(0.4, 1.0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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


def create_dataloaders(dataset_root, batch_size=64, num_workers=0):
    """创建训练和评估 DataLoader"""
    train_dataset = PetDistillationDataset(root=dataset_root, split='trainval')
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
