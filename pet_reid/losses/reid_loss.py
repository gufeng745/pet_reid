"""
Re-ID 损失函数

包含：
- TripletLoss: 三元组损失
- SupervisedContrastiveLoss: 监督对比损失
- FeatureOrthogonalityLoss: 特征正交正则化
- LabelSmoothingCE: 标签平滑交叉熵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TripletMarginLoss(nn.Module):
    """Triplet Loss with Hard Mining

    对于每个anchor，选择最难的正样本和负样本构成三元组
    这是Re-ID任务的核心Metric Loss

    Args:
        margin: 边界参数，默认0.3
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2归一化的特征
            labels: (B,) 标签 (long)

        Returns:
            loss: 标量
        """
        # 计算余弦相似度矩阵
        sim_matrix = features @ features.T  # (B, B)

        # 构建正负样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1.0 - pos_mask

        # 排除对角线
        eye_mask = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - eye_mask
        neg_mask = neg_mask * (1.0 - eye_mask)

        # 排除自身
        sim_matrix = sim_matrix - eye_mask * 1e9

        # 最难正样本：相似度最低的正样本
        pos_sim = sim_matrix * pos_mask + (1.0 - pos_mask) * (-1e9)
        hardest_pos = pos_sim.max(dim=1)[0]  # (B,)

        # 最难负样本：相似度最高的负样本
        neg_sim = sim_matrix * neg_mask + (1.0 - neg_mask) * (-1e9)
        hardest_neg = neg_sim.max(dim=1)[0]  # (B,)

        # Triplet Loss: max(0, margin - (pos_sim - neg_sim))
        loss = F.relu(self.margin - (hardest_pos - hardest_neg))

        # 只计算有效样本的损失
        valid = (hardest_pos > -1e8) & (hardest_neg > -1e8)
        if valid.sum() > 0:
            loss = loss[valid].mean()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """监督对比损失 (Supervised Contrastive Learning)

    同一身份的样本互为正样本对，不同身份的样本互为负样本对

    Args:
        temperature: 温度系数，越小分布越尖锐
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2归一化特征
            labels: (B,) 标签 (long)

        Returns:
            loss: 标量
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # 余弦相似度矩阵
        sim_matrix = features @ features.T / self.temperature  # (B, B)

        # 构建正样本掩码：同一身份的样本对
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        eye_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask = pos_mask - eye_mask.float()  # 排除自身

        # 排除自身对角线
        sim_matrix = sim_matrix.float().masked_fill(eye_mask, -1e4)

        # 对于每个anchor，计算InfoNCE
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * pos_mask).sum(dim=1)  # (B,)
        all_sim = exp_sim.sum(dim=1)  # (B,)

        # 避免除零
        pos_sim = pos_sim.clamp(min=1e-8)
        all_sim = all_sim.clamp(min=1e-8)

        loss = -torch.log(pos_sim / all_sim).mean()
        return loss


class FeatureOrthogonalityLoss(nn.Module):
    """特征正交正则化

    鼓励特征维度之间去相关，防止多个维度编码冗余信息
    最大化特征的信息容量

    Args:
        feat_dim: 特征维度（用于归一化）
    """

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2归一化的特征

        Returns:
            loss: 标量
        """
        B, D = features.shape
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # 中心化
        features_centered = features - features.mean(dim=0, keepdim=True)

        # 协方差矩阵: (D, D)
        cov = features_centered.T @ features_centered / (B - 1)

        # 标准差
        std = torch.sqrt(torch.diag(cov).clamp(min=1e-8))

        # 相关系数矩阵
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

        # 只惩罚非对角线元素（即维度间的相关性）
        mask = ~torch.eye(D, dtype=torch.bool, device=features.device)
        loss = (corr[mask] ** 2).mean()

        return loss


class LabelSmoothingCE(nn.Module):
    """Label Smoothing Cross Entropy

    防止模型对相似类别过度自信，提升泛化能力

    Args:
        smoothing: 平滑系数，默认0.1
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测logits
            target: (B,) 真实标签 (long)

        Returns:
            loss: 标量
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # 构建平滑标签
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


class CircleLoss(nn.Module):
    """Circle Loss for Metric Learning

    相比Triplet Loss和Contrastive Loss，Circle Loss具有：
    1. 更平滑的优化目标
    2. 更灵活的权重自适应
    3. 更清晰的收敛边界

    Args:
        m: 边界参数，默认0.25
        gamma: 缩放参数，默认256
    """

    def __init__(self, m: float = 0.25, gamma: float = 256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2归一化的特征
            labels: (B,) 标签 (long)

        Returns:
            loss: 标量
        """
        # 计算余弦相似度矩阵
        sim_matrix = features @ features.T  # (B, B)

        # 构建正负样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1.0 - pos_mask

        # 排除对角线
        eye_mask = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - eye_mask
        neg_mask = neg_mask

        # 正负样本相似度
        s_p = sim_matrix * pos_mask
        s_n = sim_matrix * neg_mask

        # 计算alpha权重
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)

        # 计算delta
        delta_p = 1 - self.m
        delta_n = self.m

        # Circle Loss
        logit_p = -self.gamma * alpha_p * (s_p - delta_p) * pos_mask
        logit_n = self.gamma * alpha_n * (s_n - delta_n) * neg_mask

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        ).mean()

        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss (Additive Angular Margin Loss)

    在超球面上添加角度边界，使类内更紧凑，类间更分离

    Args:
        feat_dim: 特征维度
        num_classes: 类别数
        margin: 角度边界，默认0.5
        scale: 缩放因子，默认64
    """

    def __init__(self, feat_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 可学习的分类器权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2归一化的特征
            labels: (B,) 标签 (long)

        Returns:
            loss: 标量
        """
        # L2归一化权重
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(features, weight_norm)  # (B, num_classes)

        # 计算角度
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)

        # 只对正确类别添加角度边界
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 缩放
        output *= self.scale

        # Cross Entropy Loss
        loss = F.cross_entropy(output, labels)

        return loss
