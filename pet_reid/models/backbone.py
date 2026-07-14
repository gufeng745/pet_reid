"""
CNN Backbone 模块

支持多种 CNN 架构：MobileNetV2, MobileNetV3, EfficientNet 等
支持 .pth 和 .safetensors 格式的权重加载
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            pretrained: 是否使用 ImageNet 预训练权重
            num_classes: 分类数（0 表示只提取特征）
            local_weight_path: 本地权重路径（支持 .pth 和 .safetensors 格式）
        """
        super().__init__()

        self.model_name = model_name
        self.feature_dim = self.FEATURE_DIMS.get(model_name, 1280)
        self.feature_map_dim = self.FEATURE_MAP_DIMS.get(model_name, 1280)

        # 先创建模型（不加载权重）
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        # 加载权重
        if local_weight_path and os.path.exists(local_weight_path):
            print(f"[Backbone] 从本地加载权重：{local_weight_path}")
            state_dict = self._load_weights(local_weight_path)
            self.backbone.load_state_dict(state_dict, strict=False)
        elif pretrained:
            print(f"[Backbone] 使用 ImageNet 预训练权重：{model_name}")
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            print(f"[Backbone] 随机初始化：{model_name}")

    def _load_weights(self, path: str) -> dict:
        """
        加载权重文件（支持 .pth 和 .safetensors 格式）
        
        Args:
            path: 权重文件路径
            
        Returns:
            state_dict: 模型状态字典
        """
        if path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(path, device='cpu')
                print(f"[Backbone] 使用 safetensors 格式加载权重")
                return state_dict
            except ImportError:
                print("[Backbone] 错误：safetensors 未安装，请运行：pip install safetensors")
                raise
        else:
            # 传统的 .pth 格式
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            # 移除可能的前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            return state_dict

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
        # 使用 softplus 确保 p 始终为正数，避免数值不稳定
        self.p = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(p - 1.0)))
        self.eps = eps

    def _get_p(self) -> torch.Tensor:
        """获取 p 值，确保 1 <= p <= 10（防止溢出）"""
        return F.softplus(self.p).clamp(max=9.0) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 特征图

        Returns:
            pooled: (B, C) 池化后的特征
        """
        p = self._get_p()
        # 确保输入为非负数（防止负数的非整数次幂产生NaN）
        x_clamped = x.clamp(min=self.eps)
        # 计算 GeM 池化
        pooled = F.avg_pool2d(
            x_clamped.pow(p),
            kernel_size=x.size()[2:]
        )
        # 避免除以 0，p 也 clamp 到合理范围
        return pooled.pow(1.0 / p.clamp(min=1.0, max=10.0))


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
