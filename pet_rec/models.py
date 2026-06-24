import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from typing import Optional


# ==================== 注意力模块 ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块

    通过学习通道间的依赖关系，增强重要特征通道，抑制不重要通道。
    弥补 Depthwise 卷积缺乏通道交互的缺陷。
    """

    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例，默认 16
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C) 特征向量
        Returns:
            (B, C) 注意力加权后的特征
        """
        b, c = x.shape
        y = self.squeeze(x.unsqueeze(-1)).view(b, c)
        y = self.excitation(y)
        return x * y


class ECABlock(nn.Module):
    """Efficient Channel Attention 模块

    比 SE 更轻量，使用 1D 卷积代替全连接层。
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C) 特征向量
        Returns:
            (B, C) 注意力加权后的特征
        """
        b, c = x.shape
        y = self.avg_pool(x.unsqueeze(-1)).transpose(-1, -2)  # (B, 1, 1)
        y = self.conv(y).transpose(-1, -2)  # (B, 1, 1)
        y = self.sigmoid(y).view(b, c)
        return x * y


# ==================== 池化模块 ====================

class GeMPooling(nn.Module):
    """Generalized Mean Pooling

    通过可学习参数 p 控制池化行为：
    - p=1 时退化为平均池化
    - p→∞ 时退化为最大池化
    - 可学习的 p 能够放大特征图中的显著响应（如独特的斑点）
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (B, C) 或 (B, C, H, W) 特征
        Returns:
            (B, C) 池化后的特征
        """
        if x.dim() == 2:
            # 已经是 (B, C)，直接返回
            return x
        # (B, C, H, W) -> GeM Pooling
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=x.size()[2:]
        ).pow(1. / self.p)


class BNNeck(nn.Module):
    """BatchNorm Neck for Metric Learning

    在特征向量和分类器之间插入 BatchNorm 层。
    训练时分类器使用 BN 后的特征，推理比对时使用 BN 前的特征，
    显著提升度量准确性。

    原理：BN 将特征分布标准化到均值 0 方差 1，
    减少了类内方差，使特征更适合度量学习。
    """

    def __init__(self, feat_dim):
        """
        Args:
            feat_dim: 特征维度
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, D) 特征向量
        Returns:
            (B, D) BN 后的特征向量
        """
        return self.bn(x)


# 本地预训练权重目录
LOCAL_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre_weights')

# 模型名称到 HuggingFace 缓存目录名的映射
HF_DIR_MAP = {
    'vit_small_patch16_dinov3': 'models--timm--vit_small_patch16_dinov3.lvd1689m',
    'vit_small_patch14_reg4_dinov2': 'models--timm--vit_small_patch14_reg4_dinov2.lvd142m',
    'mobilenetv2_100': 'models--timm--mobilenetv2_100.ra_in1k',
}


def get_hf_cache_weight_path(model_name: str) -> Optional[str]:
    """从 pre_weights 中的 HuggingFace 缓存目录查找权重文件
    
    期望目录结构：
    pre_weights/
    ├── models--timm--mobilenetv2_100.ra_in1k/
    │   └── snapshots/
    │       └── <commit_hash>/
    │           └── model.safetensors
    ├── models--timm--vit_small_patch16_dinov3.lvd1689m/
    │   └── snapshots/
    │       └── <commit_hash>/
    │           └── model.safetensors
    """
    hf_dir_name = HF_DIR_MAP.get(model_name)
    if not hf_dir_name:
        return None
    
    hf_model_dir = os.path.join(LOCAL_WEIGHTS_DIR, hf_dir_name)
    if not os.path.exists(hf_model_dir):
        return None
    
    # 查找 snapshots 目录下的权重文件
    snapshots_dir = os.path.join(hf_model_dir, 'snapshots')
    if not os.path.exists(snapshots_dir):
        return None
    
    # 遍历 snapshots 下的所有 commit 目录
    for commit_hash in os.listdir(snapshots_dir):
        commit_dir = os.path.join(snapshots_dir, commit_hash)
        if os.path.isdir(commit_dir):
            weight_file = os.path.join(commit_dir, 'model.safetensors')
            if os.path.exists(weight_file):
                return weight_file
    
    return None


def get_local_weight_path(model_name: str) -> Optional[str]:
    """获取本地权重文件路径，如果存在则返回路径，否则返回 None"""
    return get_hf_cache_weight_path(model_name)


def load_safetensors_weight(weight_path: str) -> dict:
    """加载 safetensors 格式的权重文件"""
    try:
        from safetensors.torch import load_file
        return load_file(weight_path)
    except ImportError:
        raise ImportError("需要安装 safetensors: pip install safetensors")


class DINOv3Teacher(nn.Module):
    """冻结的 DINOv3 ViT-S teacher，输出 384 维 L2 归一化特征"""

    def __init__(self, model_name='vit_small_patch16_dinov3'):
        super().__init__()
        # 优先尝试从本地加载权重
        local_weight_path = get_local_weight_path(model_name)
        if local_weight_path:
            print(f"[DINOv3] 从本地加载权重：{local_weight_path}")
            self.model = timm.create_model(model_name, pretrained=False)
            # 加载 safetensors 格式的权重
            state_dict = load_safetensors_weight(local_weight_path)
            # 移除 'module.' 前缀（如果有）
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[DINOv3] 本地权重不存在，尝试在线下载...")
            self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.feature_dim = 384

    def forward(self, x):
        with torch.no_grad():
            feat = self.model.forward_features(x)
            global_feat = self.model.forward_head(feat, pre_logits=True)
            return F.normalize(global_feat, dim=-1)


class DINOv2Teacher(nn.Module):
    """DINOv2 ViT-S 备用 teacher（如果 DINOv3 下载失败）"""

    def __init__(self, model_name='vit_small_patch14_reg4_dinov2'):
        super().__init__()
        # 优先尝试从本地加载权重
        local_weight_path = get_local_weight_path(model_name)
        if local_weight_path:
            print(f"[DINOv2] 从本地加载权重：{local_weight_path}")
            self.model = timm.create_model(model_name, pretrained=False)
            # 加载 safetensors 格式的权重
            state_dict = load_safetensors_weight(local_weight_path)
            # 移除 'module.' 前缀（如果有）
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[DINOv2] 本地权重不存在，尝试在线下载...")
            self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.feature_dim = 384

    def forward(self, x):
        with torch.no_grad():
            feat = self.model.forward_features(x)
            global_feat = self.model.forward_head(feat, pre_logits=True)
            return F.normalize(global_feat, dim=-1)


class MobileNetV2Student(nn.Module):
    """MobileNetV2 backbone + 投影头，输出 512 维 L2 归一化特征"""

    def __init__(self, proj_dim=512, pretrained_backbone=False):
        super().__init__()
        # 优先尝试从本地加载 MobileNetV2 权重
        local_weight_path = get_local_weight_path('mobilenetv2_100')
        if local_weight_path and pretrained_backbone:
            print(f"[MobileNetV2] 从本地加载权重：{local_weight_path}")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)
            # 加载 safetensors 格式的权重
            state_dict = load_safetensors_weight(local_weight_path)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
        elif pretrained_backbone:
            print("[MobileNetV2] 本地权重不存在，尝试在线下载...")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
        else:
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )
        self.feature_dim = proj_dim

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        return F.normalize(proj, dim=-1)


class TeacherAdapter(nn.Module):
    """训练时使用的维度适配器（384→512），推理时丢弃"""

    def __init__(self, teacher_dim=384, student_dim=512):
        super().__init__()
        self.linear = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)
    
class MobileNetV2StudentWithAttr(nn.Module):
    """带属性预测头的 MobileNetV2 学生模型

    训练时：backbone → 投影头 (512 维) + 颜色头 + 花纹头
    推理时：只用投影头，属性头丢弃（forward_emb）

    改进：
    - 添加 SE 注意力模块增强通道交互
    - 使用 BNNeck 提升度量学习效果
    - 属性头使用 MLP 增强表达能力
    """

    def __init__(self, proj_dim=512, num_colors=13, num_patterns=13,
                 pretrained_backbone=False, use_se=True, use_bnneck=True):
        """
        Args:
            proj_dim: 投影维度
            num_colors: 颜色类别数量（默认 13，包括原始 11 个 + 2 个额外类别）
            num_patterns: 花纹类别数量（默认 13，包括原始 10 个 + 3 个额外类别）
            pretrained_backbone: 是否加载预训练 backbone 权重（默认 False，直接从检查点加载）
            use_se: 是否使用 SE 注意力模块（默认 True）
            use_bnneck: 是否使用 BNNeck（默认 True）
        """
        super().__init__()
        self.use_se = use_se
        self.use_bnneck = use_bnneck

        # 优先尝试从本地加载 MobileNetV2 权重
        local_weight_path = get_local_weight_path('mobilenetv2_100')
        if local_weight_path and pretrained_backbone:
            print(f"[MobileNetV2Attr] 从本地加载预训练权重：{local_weight_path}")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)
            # 加载 safetensors 格式的权重
            state_dict = load_safetensors_weight(local_weight_path)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
        elif pretrained_backbone:
            print("[MobileNetV2Attr] 本地权重不存在，尝试在线下载...")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
        else:
            # 不加载预训练权重，直接从检查点加载训练好的权重
            print("[MobileNetV2Attr] 不加载预训练 backbone，将从检查点加载完整模型")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)

        feat_dim = 1280  # MobileNetV2 特征维度

        # SE 注意力模块
        if use_se:
            self.se_block = SEBlock(feat_dim, reduction=16)
            print("[MobileNetV2Attr] 使用 SE 注意力模块")

        # BNNeck for metric learning
        if use_bnneck:
            self.bnneck = BNNeck(feat_dim)
            print("[MobileNetV2Attr] 使用 BNNeck")

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # 属性预测头（使用 MLP 增强表达能力）
        self.color_primary_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_colors),
        )
        self.color_secondary_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_colors),
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_patterns),
        )

        self.feature_dim = proj_dim

    def forward(self, x):
        """训练用：返回特征 + 属性预测

        Returns:
            emb: (B, proj_dim) L2 归一化的特征
            color_pri: (B, num_colors) 主色预测
            color_sec: (B, num_colors) 副色预测
            pattern: (B, num_patterns) 花纹预测
        """
        feat = self.backbone(x)

        # SE 注意力
        if self.use_se:
            feat = self.se_block(feat)

        # BNNeck（训练时使用 BN 后的特征做分类）
        if self.use_bnneck:
            feat_bn = self.bnneck(feat)
        else:
            feat_bn = feat

        # 投影（使用 BN 后的特征）
        emb = F.normalize(self.projector(feat_bn), dim=-1)

        # 属性预测（使用 BN 后的特征）
        color_pri = self.color_primary_head(feat_bn)
        color_sec = self.color_secondary_head(feat_bn)
        pattern = self.pattern_head(feat_bn)

        return emb, color_pri, color_sec, pattern

    def forward_emb(self, x):
        """推理用：只返回特征向量

        注意：推理时使用 BN 前的特征，更适合度量学习
        """
        feat = self.backbone(x)

        # SE 注意力
        if self.use_se:
            feat = self.se_block(feat)

        # 推理时不使用 BNNeck，使用原始特征
        return F.normalize(self.projector(feat), dim=-1)

    def forward_feat_for_metric(self, x):
        """度量学习用：返回 BN 前的原始特征（用于对比学习）"""
        feat = self.backbone(x)

        # SE 注意力
        if self.use_se:
            feat = self.se_block(feat)

        return feat
