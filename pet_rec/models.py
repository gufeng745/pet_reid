import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv3Teacher(nn.Module):
    """冻结的 DINOv3 ViT-S teacher，输出 384 维 L2 归一化特征"""

    def __init__(self, model_name='vit_small_patch16_dinov3'):
        super().__init__()
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

    def __init__(self, proj_dim=512):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
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

    训练时：backbone → 投影头(512维) + 颜色头 + 花纹头
    推理时：只用投影头，属性头丢弃（forward_emb）
    """

    def __init__(self, proj_dim=512, num_colors=11, num_patterns=10):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )
        # 属性预测头（共享 backbone 特征）
        self.color_primary_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, num_colors)
        )
        self.color_secondary_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, num_colors)
        )
        self.pattern_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, num_patterns)
        )
        self.feature_dim = proj_dim

    def forward(self, x):
        """训练用：返回特征 + 属性预测"""
        feat = self.backbone(x)
        emb = F.normalize(self.projector(feat), dim=-1)
        return emb, self.color_primary_head(feat), self.color_secondary_head(feat), self.pattern_head(feat)

    def forward_emb(self, x):
        """推理用：只返回特征向量"""
        feat = self.backbone(x)
        return F.normalize(self.projector(feat), dim=-1)
