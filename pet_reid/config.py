"""
Pet Re-ID 配置文件

基于DINOv3自监督学习的宠物Re-ID系统配置
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class DINOConfig:
    """DINOv3自监督预训练配置"""

    # ==================== 模型配置 ====================
    backbone: str = "mobilenetv3_large_100"
    proj_dim: int = 512  # 改为512维，与Re-ID对齐
    hidden_dim: int = 2048
    predictor_hidden_dim: int = 1024

    # ==================== 数据增强配置 ====================
    global_crop_size: int = 224
    local_crop_size: int = 96
    num_global_views: int = 2
    num_local_views: int = 6
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)

    # ==================== 训练配置 ====================
    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True

    # 优化器
    lr: float = 5e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # ==================== DINOv3特有配置 ====================
    # Teacher EMA
    teacher_momentum_start: float = 0.996
    teacher_momentum_end: float = 1.0

    # 温度系数
    teacher_temp: float = 0.04
    student_temp: float = 0.1

    # Centering
    center_momentum: float = 0.9

    # ==================== 其他配置 ====================
    seed: int = 42
    use_amp: bool = True
    max_grad_norm: float = 3.0

    # 保存
    save_interval: int = 20
    log_interval: int = 10

    # 输出目录
    output_dir: str = "outputs/dino"
    checkpoint_dir: str = "checkpoints/dino"
    log_dir: str = "logs/dino"

    # 数据路径（默认使用datasets文件夹）
    data_root: str = "../pet_rec/datasets"


@dataclass
class ReIDConfig:
    """Re-ID监督微调配置"""

    # ==================== 模型配置 ====================
    backbone: str = "mobilenetv3_large_100"
    proj_dim: int = 512
    num_classes: int = 82  # 将根据数据集自动设置

    # 模型结构
    use_gem_pool: bool = True
    use_se: bool = True
    use_bnneck: bool = True
    se_reduction: int = 16

    # ==================== 数据配置 ====================
    image_size: int = 224
    val_ratio: float = 0.1
    min_images_per_id: int = 3

    # PK Sampler
    P: int = 16  # 每个batch的ID数
    K: int = 4   # 每个ID的样本数

    # ==================== 训练配置 ====================
    epochs: int = 80
    batch_size: int = 64
    num_workers: int = 4

    # 学习率（双学习率）
    lr_backbone: float = 5e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.04
    warmup_epochs: int = 10

    # ==================== 损失函数权重 ====================
    lambda_id: float = 0.5
    lambda_triplet: float = 0.3
    lambda_contrastive: float = 0.2
    lambda_ortho: float = 0.05

    # 损失函数参数
    id_label_smoothing: float = 0.1
    triplet_margin: float = 0.3
    contrastive_temp: float = 0.1

    # ==================== 训练技巧 ====================
    use_amp: bool = True
    max_grad_norm: float = 1.0
    use_early_stopping: bool = True
    patience: int = 25

    # 预训练模型
    pretrained_dino: Optional[str] = None  # DINOv3预训练权重路径

    # ==================== 其他配置 ====================
    seed: int = 42
    save_interval: int = 10
    log_interval: int = 5

    # 输出目录
    output_dir: str = "outputs/reid"
    checkpoint_dir: str = "checkpoints/reid"


@dataclass
class EvalConfig:
    """评估配置"""

    # 模型路径
    model_path: str = "checkpoints/reid/best_reid.pth"

    # 数据集
    data_root: str = "../pet_rec/reid_dataset"
    species: Optional[str] = None  # 'cat', 'dog', None

    # 评估参数
    num_trials: int = 10
    min_images: int = 3
    batch_size: int = 32

    # 输出
    output_dir: str = "outputs/eval"


@dataclass
class ExportConfig:
    """ONNX导出配置"""

    # 模型路径
    model_path: str = "checkpoints/reid/best_reid.pth"

    # 导出参数
    opset_version: int = 11
    simplify: bool = True
    int8_quantize: bool = False

    # 输入尺寸
    input_height: int = 224
    input_width: int = 224

    # 输出
    output_dir: str = "outputs/onnx"


def get_dino_config(**kwargs) -> DINOConfig:
    """获取DINOv3配置，支持覆盖参数"""
    config = DINOConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_reid_config(**kwargs) -> ReIDConfig:
    """获取Re-ID配置，支持覆盖参数"""
    config = ReIDConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_eval_config(**kwargs) -> EvalConfig:
    """获取评估配置，支持覆盖参数"""
    config = EvalConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_export_config(**kwargs) -> ExportConfig:
    """获取导出配置，支持覆盖参数"""
    config = ExportConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
