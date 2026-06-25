"""
数据增强模块

提供DINOv3和Re-ID的数据增强
"""

from torchvision import transforms
from typing import Tuple, List
from datasets.dino_dataset import MultiCropTransform


def get_dino_transforms(
    global_size: int = 224,
    local_size: int = 96,
    num_local: int = 6,
    global_scale: Tuple[float, float] = (0.4, 1.0),
    local_scale: Tuple[float, float] = (0.05, 0.4)
) -> MultiCropTransform:
    """获取DINOv3多视图数据增强

    Args:
        global_size: 全局视图尺寸
        local_size: 局部视图尺寸
        num_local: 局部视图数量
        global_scale: 全局视图缩放范围
        local_scale: 局部视图缩放范围

    Returns:
        transform: MultiCropTransform实例
    """
    return MultiCropTransform(
        global_crops_scale=global_scale,
        local_crops_scale=local_scale,
        global_size=global_size,
        local_size=local_size,
        num_local_crops=num_local
    )


def get_train_transform(
    crop_scale: Tuple[float, float] = (0.4, 1.0),
    image_size: int = 224
) -> transforms.Compose:
    """获取Re-ID训练数据增强

    Args:
        crop_scale: 随机裁剪缩放范围
        image_size: 图像尺寸

    Returns:
        transform: Compose实例
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=crop_scale,
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


def get_val_transform(
    resize_size: int = 256,
    crop_size: int = 224
) -> transforms.Compose:
    """获取验证/推理数据增强

    Args:
        resize_size: 缩放尺寸
        crop_size: 裁剪尺寸

    Returns:
        transform: Compose实例
    """
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_test_time_augmentation(
    num_crops: int = 5,
    image_size: int = 224
) -> List[transforms.Compose]:
    """获取测试时数据增强（TTA）

    生成多个增强版本，用于推理时的特征融合

    Args:
        num_crops: 裁剪数量
        image_size: 图像尺寸

    Returns:
        transforms: 列表，包含多个数据增强
    """
    transforms_list = []

    # 原始图像
    transforms_list.append(get_val_transform())

    # 水平翻转
    transforms_list.append(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    # 多尺度裁剪
    for scale in [0.8, 0.9, 1.0]:
        transforms_list.append(transforms.Compose([
            transforms.Resize(int(256 * scale)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    return transforms_list[:num_crops]
