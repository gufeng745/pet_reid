"""
Re-ID 数据集

支持：
- 按ID划分train/val
- PK Sampler (身份感知采样)
- 数据增强
"""

import os
import random
from collections import defaultdict
from typing import Optional, Set, Tuple, List

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import numpy as np


class ReIDDataset(Dataset):
    """Re-ID 身份识别数据集

    目录结构：
        root/
        ├── cat/
        │   ├── {id}/
        │   │   ├── img1.png
        │   │   └── img2.png
        │   └── ...
        └── dog/
            └── {id}/
                └── ...

    每个子文件夹是一个身份ID
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        species: Optional[str] = None,
        id_list: Optional[Set[str]] = None
    ):
        """
        Args:
            root: 数据集根目录
            transform: 数据增强
            species: 'cat', 'dog', 或 None (两者都包含)
            id_list: 指定使用的ID集合（用于train/val划分）
        """
        self.root = root
        self.transform = transform or self._default_transform()
        self.samples = []  # [(pet_id, image_path), ...]
        self.id_to_label = {}  # pet_id -> label (连续整数)
        self.label_to_id = {}  # label -> pet_id

        # 收集所有样本
        pet_id_counter = 0

        if species is None or species == 'cat':
            cat_dir = os.path.join(root, 'cat')
            if os.path.isdir(cat_dir):
                for pet_id in sorted(os.listdir(cat_dir),
                                    key=lambda x: int(x) if x.isdigit() else x):
                    if id_list is not None and f"cat_{pet_id}" not in id_list:
                        continue
                    pet_dir = os.path.join(cat_dir, pet_id)
                    if not os.path.isdir(pet_dir):
                        continue

                    full_id = f"cat_{pet_id}"
                    self.id_to_label[full_id] = pet_id_counter
                    self.label_to_id[pet_id_counter] = full_id

                    for img_name in os.listdir(pet_dir):
                        img_path = os.path.join(pet_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((full_id, img_path))

                    pet_id_counter += 1

        if species is None or species == 'dog':
            dog_dir = os.path.join(root, 'dog')
            if os.path.isdir(dog_dir):
                for pet_id in sorted(os.listdir(dog_dir),
                                    key=lambda x: int(x) if x.isdigit() else x):
                    if id_list is not None and f"dog_{pet_id}" not in id_list:
                        continue
                    pet_dir = os.path.join(dog_dir, pet_id)
                    if not os.path.isdir(pet_dir):
                        continue

                    full_id = f"dog_{pet_id}"
                    self.id_to_label[full_id] = pet_id_counter
                    self.label_to_id[pet_id_counter] = full_id

                    for img_name in os.listdir(pet_dir):
                        img_path = os.path.join(pet_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((full_id, img_path))

                    pet_id_counter += 1

        self.num_classes = pet_id_counter
        print(f"[ReIDDataset] {len(self.samples)} 张图片, {self.num_classes} 个身份")

    def _default_transform(self) -> transforms.Compose:
        """默认训练数据增强"""
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.4, 1.0),
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: 索引

        Returns:
            image: (3, 224, 224)
            label: 身份标签
        """
        pet_id, img_path = self.samples[idx]
        label = self.id_to_label[pet_id]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ReIDDataset] 警告: 无法加载图片 {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


class PKSampler(Sampler):
    """PK Sampler (身份感知采样器)

    每个Batch包含P个ID，每个ID抽K张图片
    保证每个Batch内都有正负样本，提升度量学习效果

    Args:
        dataset: ReIDDataset实例
        P: 每个batch的ID数
        K: 每个ID的样本数
        drop_last: 是否丢弃最后一个不完整的batch
    """

    def __init__(self, dataset: ReIDDataset, P: int = 16, K: int = 4, drop_last: bool = True):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.drop_last = drop_last

        # 按ID分组索引
        self.id_to_indices = defaultdict(list)
        for idx, (pet_id, _) in enumerate(dataset.samples):
            self.id_to_indices[pet_id].append(idx)

        # 过滤掉样本数不足K的ID
        self.valid_ids = [pid for pid, idxs in self.id_to_indices.items()
                         if len(idxs) >= K]

        if len(self.valid_ids) < P:
            print(f"[PKSampler] 警告: 有效ID数 ({len(self.valid_ids)}) 小于 P ({P})")
            self.P = len(self.valid_ids)

        print(f"[PKSampler] {len(self.valid_ids)} 个有效ID，每个batch {self.P} 个ID × {self.K} 张图")

    def __iter__(self):
        num_batches = len(self.valid_ids) // self.P

        for _ in range(num_batches):
            selected_ids = random.sample(self.valid_ids, self.P)
            batch_indices = []

            for pid in selected_ids:
                indices = random.sample(self.id_to_indices[pid], self.K)
                batch_indices.extend(indices)

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.valid_ids) // self.P
        else:
            return (len(self.valid_ids) + self.P - 1) // self.P


def split_dataset_by_id(
    data_root: str,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """按ID划分训练集和验证集

    Args:
        data_root: 数据集根目录
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        train_ids: 训练ID集合
        val_ids: 验证ID集合
    """
    # 收集所有ID
    all_ids = []

    cat_dir = os.path.join(data_root, 'cat')
    if os.path.isdir(cat_dir):
        for pet_id in os.listdir(cat_dir):
            if os.path.isdir(os.path.join(cat_dir, pet_id)):
                all_ids.append(f"cat_{pet_id}")

    dog_dir = os.path.join(data_root, 'dog')
    if os.path.isdir(dog_dir):
        for pet_id in os.listdir(dog_dir):
            if os.path.isdir(os.path.join(dog_dir, pet_id)):
                all_ids.append(f"dog_{pet_id}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(all_ids)

    # 划分
    val_size = max(1, int(len(all_ids) * val_ratio))
    val_ids = set(all_ids[:val_size])
    train_ids = set(all_ids[val_size:])

    print(f"[split_dataset_by_id] 总身份数: {len(all_ids)}")
    print(f"  训练集: {len(train_ids)} 个身份")
    print(f"  验证集: {len(val_ids)} 个身份")

    return train_ids, val_ids


def get_train_transform() -> transforms.Compose:
    """训练数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.4, 1.0),
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


def get_val_transform() -> transforms.Compose:
    """验证数据增强"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
