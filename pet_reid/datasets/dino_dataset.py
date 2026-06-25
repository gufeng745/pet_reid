"""
DINOv3 数据集

实现DINOv3的多视图数据增强：
- 2个全局视图 (224×224, 覆盖40%-100%)
- 6个局部视图 (96×96, 覆盖5%-40%)
"""

import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class MultiCropTransform:
    """DINOv3多视图数据增强

    对同一张图片生成多个不同尺度的crop：
    - global_crops: 2个大尺度crop (覆盖40%-100%)
    - local_crops: 多个小尺度crop (覆盖5%-40%)
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        global_size: int = 224,
        local_size: int = 96,
        num_local_crops: int = 6
    ):
        self.global_size = global_size
        self.local_size = local_size
        self.num_local_crops = num_local_crops

        # 全局视图增强
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # 局部视图增强
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """
        Args:
            image: PIL Image

        Returns:
            crops: list of tensors
                - crops[0:2]: 2个全局视图 (3, 224, 224)
                - crops[2:]: 6个局部视图 (3, 96, 96)
        """
        crops = []

        # 生成2个全局视图
        for _ in range(2):
            crops.append(self.global_transform(image))

        # 生成6个局部视图
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops


class DINODataset(Dataset):
    """DINOv3 数据集

    不需要标签，只需要图片
    返回同一图片的多个增强视图

    支持两种目录结构：
    1. Re-ID格式（按ID组织）：
        root/
        ├── cat/
        │   ├── 1/
        │   │   ├── img1.png
        │   │   └── img2.png
        │   └── 2/
        │       └── ...
        └── dog/
            └── ...

    2. 扁平格式（所有图片在一个文件夹）：
        root/
        ├── cat.0.jpg
        ├── cat.1.jpg
        ├── dog.0.jpg
        └── ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[MultiCropTransform] = None,
        species: Optional[str] = None
    ):
        """
        Args:
            root: 数据集根目录
            transform: 多视图数据增强
            species: 'cat', 'dog', 或 None (两者都包含)
        """
        self.root = root
        self.transform = transform or MultiCropTransform()

        # 收集所有图片路径
        self.image_paths = []

        # 检测数据集格式
        if self._is_flat_format(root):
            print(f"[DINODataset] 检测到扁平格式")
            self._collect_flat_images(root, species)
        else:
            print(f"[DINODataset] 检测到Re-ID格式")
            if species is None or species == 'cat':
                cat_dir = os.path.join(root, 'cat')
                if os.path.isdir(cat_dir):
                    self._collect_images(cat_dir)

            if species is None or species == 'dog':
                dog_dir = os.path.join(root, 'dog')
                if os.path.isdir(dog_dir):
                    self._collect_images(dog_dir)

        print(f"[DINODataset] 加载了 {len(self.image_paths)} 张图片")

    def _is_flat_format(self, root: str) -> bool:
        """检测是否是扁平格式（图片直接在根目录下）"""
        # 检查是否有train子目录（Kaggle格式）
        train_dir = os.path.join(root, 'train')
        if os.path.isdir(train_dir):
            # 检查train目录下是否有图片
            for file in os.listdir(train_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    return True
        return False

    def _collect_flat_images(self, root: str, species: Optional[str] = None):
        """收集扁平格式的图片"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # Kaggle格式：图片在train子目录下
        train_dir = os.path.join(root, 'train')
        if os.path.isdir(train_dir):
            target_dir = train_dir
        else:
            target_dir = root

        for file in os.listdir(target_dir):
            if not os.path.isfile(os.path.join(target_dir, file)):
                continue

            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                continue

            # 根据species过滤
            if species == 'cat' and not file.startswith('cat.'):
                continue
            if species == 'dog' and not file.startswith('dog.'):
                continue

            self.image_paths.append(os.path.join(target_dir, file))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """
        Args:
            idx: 索引

        Returns:
            crops: list of 8 tensors
                - 2个全局视图 (3, 224, 224)
                - 6个局部视图 (3, 96, 96)
        """
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[DINODataset] 警告: 无法加载图片 {img_path}: {e}")
            # 返回黑色图片
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        crops = self.transform(img)
        return crops


class DINODataLoader:
    """DINOv3 数据加载器

    自定义collate_fn将多个crop堆叠
    """

    @staticmethod
    def collate_fn(batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: list of (list of 8 tensors)

        Returns:
            global_views: (B, 2, 3, 224, 224)
            local_views: (B, 6, 3, 96, 96)
        """
        # batch[i] 是一个列表，包含8个tensor
        # batch[i][0:2] 是全局视图
        # batch[i][2:8] 是局部视图

        global_views = []
        local_views = []

        for crops in batch:
            global_views.append(torch.stack(crops[:2]))
            local_views.append(torch.stack(crops[2:]))

        global_views = torch.stack(global_views)  # (B, 2, 3, 224, 224)
        local_views = torch.stack(local_views)  # (B, 6, 3, 96, 96)

        return global_views, local_views
