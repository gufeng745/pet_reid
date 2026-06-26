"""
预训练模型管理模块

支持从本地加载预训练模型，包括：
- DINOv3预训练模型
- ImageNet预训练模型
- 自定义预训练模型

预训练模型目录结构：
pretrained_models/
├── dino/
│   ├── best_dino.pth          # DINOv3预训练模型
│   └── best_dino.onnx         # ONNX格式
├── imagenet/
│   ├── mobilenetv3_large.pth  # ImageNet预训练
│   └── ...
└── custom/
    └── ...
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path


# 预训练模型注册表
PRETRAINED_MODELS = {
    # DINOv3预训练模型（在ImageNet基础上训练）
    'dino_mobilenetv3_large': {
        'file': 'best_dino.pth',
        'dir': 'dino',
        'description': 'DINOv3预训练的MobileNetV3-Large (512维宠物特征)',
        'backbone': 'mobilenetv3_large_100',
        'proj_dim': 512,
        'note': '在ImageNet预训练基础上，使用宠物数据自监督训练',
    },
    'dino_mobilenetv3_small': {
        'file': 'best_dino.pth',
        'dir': 'dino_small',
        'description': 'DINOv3预训练的MobileNetV3-Small',
        'backbone': 'mobilenetv3_small_100',
        'proj_dim': 512,
        'note': '在ImageNet预训练基础上，使用宠物数据自监督训练',
    },

    # ImageNet预训练模型（原始预训练权重）
    'imagenet_mobilenetv3_large': {
        'source': 'timm',
        'model_name': 'mobilenetv3_large_100',
        'description': 'ImageNet预训练的MobileNetV3-Large (1280维通用特征)',
        'note': '原始ImageNet预训练权重，通用图像特征',
    },
    'imagenet_mobilenetv3_small': {
        'source': 'timm',
        'model_name': 'mobilenetv3_small_100',
        'description': 'ImageNet预训练的MobileNetV3-Small',
        'note': '原始ImageNet预训练权重，通用图像特征',
    },
    'imagenet_efficientnet_b0': {
        'source': 'timm',
        'model_name': 'efficientnet_b0',
        'description': 'ImageNet预训练的EfficientNet-B0',
        'note': '原始ImageNet预训练权重，通用图像特征',
    },

    # 本地ImageNet预训练模型
    'local_mobilenetv3_large': {
        'file': 'mobilenetv3_large_100_ra-f55367b5.pth',
        'dir': 'imagenet',
        'description': '本地MobileNetV3-Large ImageNet预训练权重',
        'backbone': 'mobilenetv3_large_100',
        'note': '从timm缓存或手动下载的ImageNet权重',
    },
}


class PretrainedModelManager:
    """预训练模型管理器

    管理和加载预训练模型
    """

    def __init__(self, base_dir: str = 'pretrained_models'):
        """
        Args:
            base_dir: 预训练模型基础目录
        """
        self.base_dir = base_dir
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保目录结构存在"""
        dirs = [
            self.base_dir,
            os.path.join(self.base_dir, 'dino'),
            os.path.join(self.base_dir, 'imagenet'),
            os.path.join(self.base_dir, 'custom'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取预训练模型路径

        Args:
            model_name: 模型名称（在PRETRAINED_MODELS中注册的名称）

        Returns:
            model_path: 模型文件路径，如果不存在返回None
        """
        if model_name not in PRETRAINED_MODELS:
            print(f"警告: 未知的预训练模型: {model_name}")
            print(f"可用的模型: {list(PRETRAINED_MODELS.keys())}")
            return None

        model_info = PRETRAINED_MODELS[model_name]

        # 如果是timm模型，返回None（由timm自动处理）
        if model_info.get('source') == 'timm':
            return None

        # 构建路径
        model_dir = os.path.join(self.base_dir, model_info['dir'])
        model_path = os.path.join(model_dir, model_info['file'])

        if os.path.exists(model_path):
            return model_path
        else:
            print(f"警告: 预训练模型文件不存在: {model_path}")
            print(f"请将模型文件放置到: {model_dir}/")
            return None

    def list_models(self):
        """列出所有可用的预训练模型"""
        print("\n可用的预训练模型:")
        print("=" * 60)

        for name, info in PRETRAINED_MODELS.items():
            print(f"\n{name}:")
            print(f"  描述: {info['description']}")

            if info.get('source') == 'timm':
                print(f"  来源: timm (自动下载)")
                print(f"  模型: {info['model_name']}")
            else:
                model_path = self.get_model_path(name)
                if model_path:
                    size_mb = os.path.getsize(model_path) / (1024*1024)
                    print(f"  路径: {model_path}")
                    print(f"  大小: {size_mb:.1f} MB")
                    print(f"  状态: ✓ 可用")
                else:
                    model_dir = os.path.join(self.base_dir, info['dir'])
                    print(f"  期望路径: {model_dir}/{info['file']}")
                    print(f"  状态: ✗ 未找到")

    def check_models(self) -> Dict[str, bool]:
        """检查所有预训练模型的可用性

        Returns:
            status: {model_name: is_available}
        """
        status = {}
        for name in PRETRAINED_MODELS:
            model_info = PRETRAINED_MODELS[name]
            if model_info.get('source') == 'timm':
                status[name] = True  # timm模型总是可用
            else:
                status[name] = self.get_model_path(name) is not None
        return status


# 全局管理器实例
_manager = None


def get_manager(base_dir: str = 'pretrained_models') -> PretrainedModelManager:
    """获取全局预训练模型管理器"""
    global _manager
    if _manager is None:
        _manager = PretrainedModelManager(base_dir)
    return _manager


def load_pretrained_backbone(
    model_name: str = 'dino_mobilenetv3_large',
    base_dir: str = 'pretrained_models',
    **kwargs
):
    """加载预训练的backbone

    Args:
        model_name: 预训练模型名称
        base_dir: 预训练模型目录
        **kwargs: 其他参数

    Returns:
        model: 加载了预训练权重的模型
    """
    from .backbone import CNNBackbone

    manager = get_manager(base_dir)

    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model: {model_name}")

    model_info = PRETRAINED_MODELS[model_name]

    # timm模型
    if model_info.get('source') == 'timm':
        print(f"[Pretrained] Loading timm model: {model_info['model_name']}")
        backbone = CNNBackbone(
            model_name=model_info['model_name'],
            pretrained=True,
            **kwargs
        )
        return backbone

    # 本地模型
    model_path = manager.get_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"Pretrained model not found: {model_name}\n"
            f"Please download the model and place it in: {base_dir}/{model_info['dir']}/"
        )

    print(f"[Pretrained] Loading local model: {model_path}")

    # 加载checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # 创建backbone
    backbone = CNNBackbone(
        model_name=model_info['backbone'],
        pretrained=False,
        **kwargs
    )

    # 加载权重
    if 'student_backbone' in ckpt:
        backbone.load_state_dict(ckpt['student_backbone'], strict=False)
        print("[Pretrained] Loaded student_backbone weights")
    elif 'backbone' in ckpt:
        backbone.load_state_dict(ckpt['backbone'], strict=False)
        print("[Pretrained] Loaded backbone weights")
    else:
        # 尝试直接加载
        try:
            backbone.load_state_dict(ckpt, strict=False)
            print("[Pretrained] Loaded weights directly")
        except Exception as e:
            print(f"[Pretrained] Warning: Could not load weights: {e}")

    return backbone


def load_pretrained_dino(
    model_name: str = 'dino_mobilenetv3_large',
    base_dir: str = 'pretrained_models',
    proj_dim: int = 512
):
    """加载完整的DINOv3预训练模型（backbone + projector）

    Args:
        model_name: 预训练模型名称
        base_dir: 预训练模型目录
        proj_dim: 投影维度

    Returns:
        model: DINOFeatureExtractor模型
    """
    from .backbone import CNNBackbone
    import torch.nn as nn

    manager = get_manager(base_dir)

    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model: {model_name}")

    model_info = PRETRAINED_MODELS[model_name]
    model_path = manager.get_model_path(model_name)

    if model_path is None:
        raise FileNotFoundError(
            f"Pretrained model not found: {model_name}\n"
            f"Please download the model and place it in: {base_dir}/{model_info['dir']}/"
        )

    print(f"[Pretrained] Loading DINOv3 model: {model_path}")

    # 加载checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # 创建backbone
    backbone = CNNBackbone(
        model_name=model_info['backbone'],
        pretrained=False
    )

    # 创建projector
    feat_dim = backbone.feature_dim
    projector = nn.Sequential(
        nn.Linear(feat_dim, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, proj_dim)
    )

    # 加载backbone权重
    if 'student_backbone' in ckpt:
        backbone.load_state_dict(ckpt['student_backbone'], strict=False)
        print("[Pretrained] Loaded student_backbone weights")

    # 加载projector权重
    if 'student_projector' in ckpt:
        projector.load_state_dict(ckpt['student_projector'], strict=False)
        print("[Pretrained] Loaded student_projector weights")

    # 创建完整的特征提取器
    class DINOFeatureExtractor(nn.Module):
        def __init__(self, backbone, projector):
            super().__init__()
            self.backbone = backbone
            self.projector = projector

        def forward(self, x):
            feat = self.backbone(x)
            proj = self.projector(feat)
            return nn.functional.normalize(proj, p=2, dim=1)

    model = DINOFeatureExtractor(backbone, projector)
    print(f"[Pretrained] DINOv3 model loaded successfully")

    return model


def setup_pretrained_models(source_dir: str, target_dir: str = 'pretrained_models'):
    """设置预训练模型目录

    从源目录复制模型到预训练模型目录

    Args:
        source_dir: 源目录（包含模型文件）
        target_dir: 目标目录（预训练模型目录）
    """
    import shutil

    manager = get_manager(target_dir)

    print(f"\n设置预训练模型目录")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")

    # 检查源目录
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return

    # 复制模型文件
    for model_name, model_info in PRETRAINED_MODELS.items():
        if model_info.get('source') == 'timm':
            continue

        source_file = os.path.join(source_dir, model_info['file'])
        target_file = os.path.join(target_dir, model_info['dir'], model_info['file'])

        if os.path.exists(source_file):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)
            size_mb = os.path.getsize(target_file) / (1024*1024)
            print(f"✓ {model_name}: {size_mb:.1f} MB")
        else:
            print(f"✗ {model_name}: 文件不存在 {source_file}")

    print("\n设置完成!")
    manager.list_models()
