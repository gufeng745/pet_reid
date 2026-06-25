"""
Re-ID 模型

基于预训练CNN backbone的宠物Re-ID模型
包含：backbone + GeM Pooling + SE + 投影头 + BNNeck + ID分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .backbone import CNNBackbone, GeMPooling, SEBlock, BNNeck


class ReIDModel(nn.Module):
    """Re-ID 模型

    结构：
    - Backbone: CNN (MobileNetV3/EfficientNet)
    - GeM Pooling: 可学习的池化
    - SE Block: 通道注意力
    - Projector: 特征投影
    - BNNeck: BatchNorm Neck
    - ID Head: 身份分类头

    训练时：返回特征 + ID预测
    推理时：只返回特征向量
    """

    def __init__(
        self,
        backbone_name: str = 'mobilenetv3_large_100',
        proj_dim: int = 512,
        num_classes: int = 82,
        pretrained_backbone: bool = True,
        use_gem_pool: bool = True,
        use_se: bool = True,
        use_bnneck: bool = True,
        se_reduction: int = 16,
        pretrained_dino_path: Optional[str] = None
    ):
        """
        Args:
            backbone_name: backbone模型名称
            proj_dim: 投影维度
            num_classes: 身份类别数
            pretrained_backbone: 是否使用ImageNet预训练
            use_gem_pool: 是否使用GeM Pooling
            use_se: 是否使用SE注意力
            use_bnneck: 是否使用BNNeck
            se_reduction: SE注意力降维比例
            pretrained_dino_path: DINOv3预训练权重路径
        """
        super().__init__()

        self.use_gem_pool = use_gem_pool
        self.use_se = use_se
        self.use_bnneck = use_bnneck
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # ========== Backbone ==========
        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone
        )
        # 使用forward_features的输出维度（特征图维度）
        feat_dim = self.backbone.feature_map_dim

        # 加载DINOv3预训练权重（如果有）
        if pretrained_dino_path:
            self._load_dino_pretrained(pretrained_dino_path)

        # ========== Pooling ==========
        if use_gem_pool:
            self.gem_pool = GeMPooling(p=3.0)
            print("[ReIDModel] 使用 GeM Pooling")
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        # ========== SE注意力 ==========
        if use_se:
            self.se_block = SEBlock(feat_dim, reduction=se_reduction)
            print(f"[ReIDModel] 使用 SE 注意力 (reduction={se_reduction})")

        # ========== 投影头 ==========
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # ========== BNNeck ==========
        if use_bnneck:
            self.bnneck = BNNeck(proj_dim)
            print("[ReIDModel] 使用 BNNeck")

        # ========== ID分类头 ==========
        self.id_head = nn.Linear(proj_dim, num_classes, bias=False)
        print(f"[ReIDModel] ID Head: Linear({proj_dim}, {num_classes})")

        # 统计
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[ReIDModel] 总参数量: {total_params/1e6:.2f}M")
        print(f"[ReIDModel] 特征维度: {proj_dim}, 类别数: {num_classes}")

    def _load_dino_pretrained(self, path: str):
        """加载DINOv3预训练的backbone权重"""
        ckpt = torch.load(path, map_location='cpu')

        if 'student_backbone' in ckpt:
            backbone_state = ckpt['student_backbone']
        elif 'backbone' in ckpt:
            backbone_state = ckpt['backbone']
        else:
            backbone_state = ckpt

        # 尝试加载
        try:
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"[ReIDModel] 加载DINOv3预训练权重: {path}")
        except Exception as e:
            print(f"[ReIDModel] 警告: 加载DINOv3权重失败: {e}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征图

        Args:
            x: (B, 3, H, W)

        Returns:
            feat: (B, feat_dim)
        """
        # Backbone
        feat_map = self.backbone.forward_features(x)  # (B, C, H, W)

        # Pooling
        if self.use_gem_pool:
            feat = self.gem_pool(feat_map).flatten(1)  # (B, C)
        else:
            feat = self.gap(feat_map).flatten(1)  # (B, C)

        # SE注意力
        if self.use_se:
            feat = self.se_block(feat)

        return feat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练用：返回特征 + ID预测

        Args:
            x: (B, 3, H, W)

        Returns:
            emb: (B, proj_dim) L2归一化的特征（BN前，用于Metric Loss）
            id_logits: (B, num_classes) 身份预测（BN后，用于ID Loss）
        """
        # 提取特征
        feat = self.forward_features(x)

        # 投影
        emb = self.projector(feat)  # (B, proj_dim)

        # BN前的特征（用于Metric Loss和推理）
        emb_norm = F.normalize(emb, dim=-1)

        # BN后的特征（用于ID Loss）
        if self.use_bnneck:
            feat_bn = self.bnneck(emb)
        else:
            feat_bn = emb

        # ID分类
        id_logits = self.id_head(feat_bn)

        return emb_norm, id_logits

    def forward_emb(self, x: torch.Tensor) -> torch.Tensor:
        """推理用：只返回特征向量

        Args:
            x: (B, 3, H, W)

        Returns:
            emb: (B, proj_dim) L2归一化的特征
        """
        feat = self.forward_features(x)
        emb = self.projector(feat)
        return F.normalize(emb, dim=-1)

    def forward_feat_for_metric(self, x: torch.Tensor) -> torch.Tensor:
        """度量学习用：返回BN前的原始特征

        Args:
            x: (B, 3, H, W)

        Returns:
            feat: (B, feat_dim)
        """
        return self.forward_features(x)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        backbone_name: str = 'mobilenetv3_large_100',
        **kwargs
    ) -> 'ReIDModel':
        """从预训练checkpoint加载模型

        Args:
            path: checkpoint路径
            backbone_name: backbone名称
            **kwargs: 其他模型参数

        Returns:
            model: ReIDModel实例
        """
        ckpt = torch.load(path, map_location='cpu')

        # 从checkpoint获取参数
        args = ckpt.get('args', {})
        num_classes = ckpt.get('num_classes', args.get('num_classes', 82))
        proj_dim = args.get('proj_dim', 512)
        use_se = args.get('use_se', True)
        use_bnneck = args.get('use_bnneck', True)

        # 创建模型
        model = cls(
            backbone_name=backbone_name,
            proj_dim=proj_dim,
            num_classes=num_classes,
            pretrained_backbone=False,
            use_se=use_se,
            use_bnneck=use_bnneck,
            **kwargs
        )

        # 加载权重
        if 'student' in ckpt:
            model.load_state_dict(ckpt['student'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

        print(f"[ReIDModel] 从checkpoint加载: {path}")
        return model
