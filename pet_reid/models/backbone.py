"""
CNN Backbone 模块

支持多种CNN架构：MobileNetV2, MobileNetV3, EfficientNet等
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple
import os


class CNNBackbone(nn.Module):
    """CNN Backbone 封装

    支持的模型：
    - mobilenetv2_100: MobileNetV2 (2.22M params, 1280-dim)
    - mobilenetv3_large_100: MobileNetV3-Large (4.20M params, 1280-dim)
    - mobilenetv3_small_100: MobileNetV3-Small (1.52M params, 1024-dim)
    - efficientnet_b0: EfficientNet-B0 (4.01M params, 1280-dim)
    - efficientnet_b1: EfficientNet-B1 (6.51M params, 1280-dim)
    """

    # 模型输出维度映射 (forward输出维度)
    FEATURE_DIMS = {
        'mobilenetv2_100': 1280,
        'mobilenetv3_large_100': 1280,
        'mobilenetv3_small_100': 1024,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
    }

    # forward_features输出维度（特征图维度）
    FEATURE_MAP_DIMS = {
        'mobilenetv2_100': 1280,
        'mobilenetv3_large_100': 960,
        'mobilenetv3_small_100': 576,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
    }

    def __init__(
        self,
        model_name: str = 'mobilenetv3_large_100',
        pretrained: bool = True,
        num_classes: int = 0,
        local_weight_path: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            pretrained: 是否使用ImageNet预训练权重
            num_classes: 分类数（0表示只提取特征）
            local_weight_path: 本地权重路径（优先于pretrained）
        """
        super().__init__()

        self.model_name = model_name
        self.feature_dim = self.FEATURE_DIMS.get(model_name, 1280)
        self.feature_map_dim = self.FEATURE_MAP_DIMS.get(model_name, 1280)

        # 创建模型
        if local_weight_path and os.path.exists(local_weight_path):
            print(f"[Backbone] 从本地加载权重: {local_weight_path}")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            state_dict = torch.load(local_weight_path, map_location='cpu')
            # 移除可能的前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
        elif pretrained:
            print(f"[Backbone] 使用ImageNet预训练权重: {model_name}")
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            print(f"[Backbone] 随机初始化: {model_name}")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            features: 特征向量 (B, feature_dim)
        """
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征图（用于GeM Pooling等）

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            feature_map: 特征图 (B, C, H, W) 或 (B, C)
        """
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        else:
            return self.backbone(x)

    @classmethod
    def get_feature_dim(cls, model_name: str) -> int:
        """获取模型输出特征维度"""
        return cls.FEATURE_DIMS.get(model_name, 1280)

    @classmethod
    def list_models(cls) -> list:
        """列出支持的模型"""
        return list(cls.FEATURE_DIMS.keys())


class GeMPooling(nn.Module):
    """Generalized Mean Pooling

    通过可学习参数 p 控制池化行为：
    - p=1 时退化为平均池化
    - p→∞ 时退化为最大池化
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 特征图

        Returns:
            pooled: (B, C) 池化后的特征
        """
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=x.size()[2:]
        ).pow(1. / self.p)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块

    通过学习通道间的依赖关系，增强重要特征通道
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C) 特征向量

        Returns:
            out: (B, C) 注意力加权后的特征
        """
        b, c = x.shape
        y = self.squeeze(x.unsqueeze(-1)).view(b, c)
        y = self.excitation(y)
        return x * y


class BNNeck(nn.Module):
    """BatchNorm Neck for Metric Learning

    在特征向量和分类器之间插入BN层
    训练时分类器使用BN后的特征，推理比对时使用BN前的特征
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
