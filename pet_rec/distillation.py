import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cosine_alignment_loss(teacher_feat, student_feat):
    """特征对齐损失：1 - cosine_similarity"""
    return (1.0 - (teacher_feat * student_feat).sum(dim=-1)).mean()


def self_similarity_loss(teacher_feat, student_feat):
    """自相似性保持损失：MSE(teacher_sim_matrix, student_sim_matrix)

    维度无关——即使 teacher 和 student 输出维度不同也能用，
    因为相似性矩阵只取决于样本间关系。
    """
    s_teacher = teacher_feat @ teacher_feat.T
    s_student = student_feat @ student_feat.T
    return F.mse_loss(s_student, s_teacher)


def koleo_uniformity_loss(feat, eps=1e-8):
    """KoLeo 均匀性损失：鼓励特征在超球面上均匀分布

    防止所有 student 输出坍塌到同一个向量。
    """
    feat = F.normalize(feat, dim=-1)
    n = feat.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=feat.device)
    dist = torch.cdist(feat, feat, p=2)
    mask = ~torch.eye(n, dtype=torch.bool, device=feat.device)
    min_dist = dist[mask].reshape(n, n - 1).min(dim=-1).values
    return -torch.log(min_dist + eps).mean()


def color_separation_loss(student_feat, color_feat, tau=0.1, eps=1e-8):
    """颜色分离损失：颜色不同的样本对，其特征相似度不应太高

    原理：
    1. 计算颜色直方图的交集矩阵（颜色相似度）
    2. 计算学生特征的余弦相似度矩阵
    3. 惩罚：颜色差异大 且 特征相似度高的样本对

    Args:
        student_feat: (B, D) L2-归一化的学生特征
        color_feat: (B, C) L1-归一化的颜色直方图
        tau: 温度系数，控制颜色相似度的软硬程度
    Returns:
        loss: 标量
    """
    B = student_feat.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=student_feat.device)

    # 学生特征余弦相似度 (已 L2 归一化，内积即余弦)
    feat_sim = student_feat @ student_feat.T  # (B, B)

    # 颜色直方图交集相似度
    # intersection = sum(min(h_i, h_j)) for each pair
    color_sim = torch.zeros(B, B, device=student_feat.device)
    for i in range(B):
        # broadcast: min(color_feat[i], color_feat[j]) 对所有 j
        color_sim[i] = torch.min(color_feat[i].unsqueeze(0), color_feat).sum(dim=-1)

    # 颜色差异权重：颜色越不相似，权重越大
    # 用 (1 - color_sim) 作为惩罚权重
    color_diff = (1.0 - color_sim).clamp(min=0)  # (B, B)

    # 只惩罚颜色不同但特征相似的样本对（排除对角线）
    mask = ~torch.eye(B, dtype=torch.bool, device=student_feat.device)
    # 损失 = color_diff * feat_sim 的均值
    # 当颜色差异大(color_diff→1)且特征相似度高(feat_sim→1)时，损失最大
    # clamp feat_sim >= 0：只惩罚正相似度（负相似度已经表示不同，不需要再惩罚）
    loss = (color_diff * feat_sim.clamp(min=0) * mask).sum() / (mask.sum() + eps)

    return loss


class DistillationLoss(nn.Module):
    """组合蒸馏损失"""

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, teacher_feat, student_feat, teacher_adapter=None):
        if teacher_adapter is not None:
            adapted_teacher = teacher_adapter(teacher_feat)
        else:
            adapted_teacher = teacher_feat

        loss_align = cosine_alignment_loss(adapted_teacher, student_feat)
        loss_sim = self_similarity_loss(teacher_feat, student_feat)
        loss_uniform = koleo_uniformity_loss(student_feat)

        total = self.alpha * loss_align + self.beta * loss_sim + self.gamma * loss_uniform
        return total, {
            'align': loss_align.item(),
            'sim': loss_sim.item(),
            'uniform': loss_uniform.item(),
            'total': total.item(),
        }


class ColorAwareDistillationLoss(nn.Module):
    """颜色感知蒸馏损失 = 基础蒸馏损失 + 颜色分离损失

    在原始蒸馏损失基础上，增加颜色分离项，
    防止模型将颜色不同但品种相同的宠物映射到相近的特征空间。
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.3):
        """
        Args:
            alpha: 对齐损失权重
            beta: 自相似性损失权重
            gamma: 均匀性损失权重
            delta: 颜色分离损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.base_loss = DistillationLoss(alpha, beta, gamma)

    def forward(self, teacher_feat, student_feat, teacher_adapter=None, color_feat=None):
        """
        Args:
            teacher_feat: (B, D_t) 教师特征
            student_feat: (B, D_s) 学生特征
            teacher_adapter: 教师适配器（可选）
            color_feat: (B, C) 颜色直方图（可选，None 时退化为基础蒸馏损失）
        """
        base, details = self.base_loss(teacher_feat, student_feat, teacher_adapter)

        if color_feat is not None and self.delta > 0:
            loss_color = color_separation_loss(student_feat, color_feat)
            total = base + self.delta * loss_color
            details['color_sep'] = loss_color.item()
            details['total'] = total.item()
        else:
            details['color_sep'] = 0.0

        return total if color_feat is not None and self.delta > 0 else base, details
    
class AttributeDistillationLoss(nn.Module):
    """多属性蒸馏损失 = 基础蒸馏 + 主色CE + 副色BCE + 花纹BCE

    - color_primary: 单标签 → CrossEntropyLoss
    - color_secondary: 多标签 → BCEWithLogitsLoss（multi-hot）
    - pattern: 多标签 → BCEWithLogitsLoss（multi-hot）
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1,
                 lambda_color_pri=0.2, lambda_color_sec=0.15, lambda_pattern=0.15):
        super().__init__()
        self.lambda_color_pri = lambda_color_pri
        self.lambda_color_sec = lambda_color_sec
        self.lambda_pattern = lambda_pattern
        self.base_loss = DistillationLoss(alpha, beta, gamma)

    def forward(self, teacher_feat, student_feat, teacher_adapter,
                color_pri_logits, color_sec_logits, pattern_logits,
                color_pri_labels, color_sec_labels, pattern_labels):
        """
        Args:
            teacher_feat: (B, D_t) 教师特征
            student_feat: (B, D_s) 学生特征
            teacher_adapter: 教师适配器
            color_pri_logits: (B, num_colors) 主色预测
            color_sec_logits: (B, num_colors) 副色预测（多标签）
            pattern_logits: (B, num_patterns) 花纹预测（多标签）
            color_pri_labels: (B,) 主色标签（long）
            color_sec_labels: (B, num_colors) 副色标签（multi-hot float）
            pattern_labels: (B, num_patterns) 花纹标签（multi-hot float）
        """
        base, details = self.base_loss(teacher_feat, student_feat, teacher_adapter)

        loss_color_pri = F.cross_entropy(color_pri_logits, color_pri_labels)
        loss_color_sec = F.binary_cross_entropy_with_logits(color_sec_logits, color_sec_labels)
        loss_pattern = F.binary_cross_entropy_with_logits(pattern_logits, pattern_labels)

        total = (base
                 + self.lambda_color_pri * loss_color_pri
                 + self.lambda_color_sec * loss_color_sec
                 + self.lambda_pattern * loss_pattern)

        details['color_pri'] = loss_color_pri.item()
        details['color_sec'] = loss_color_sec.item()
        details['pattern'] = loss_pattern.item()
        details['total'] = total.item()

        return total, details


class SupervisedContrastiveLoss(nn.Module):
    """实例级监督对比损失 (InfoNCE)

    将 batch 中每个样本视为独立类别：
    - 正样本对：同一图片的两个不同增强视角 (view1, view2)
    - 负样本对：batch 中所有其他图片

    直接优化"不同宠物个体应远离"的目标，是解决不同宠物间
    相似度过高的核心损失。

    Args:
        temperature: 温度系数，越小分布越尖锐（区分度越强）
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: (B, D) 视角1的 L2 归一化特征
            feat2: (B, D) 视角2的 L2 归一化特征
        Returns:
            loss: 标量
        """
        B = feat1.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=feat1.device)

        # 合并两个视角: (2B, D)
        features = torch.cat([feat1, feat2], dim=0)

        # 余弦相似度矩阵 (已 L2 归一化，内积即余弦)
        sim_matrix = features @ features.T / self.temperature  # (2B, 2B)

        # 正样本对标签：
        # feat1[i] 的正样本是 feat2[i]，feat2[i] 的正样本是 feat1[i]
        # 即 labels[i] = i + B, labels[i + B] = i
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ]).to(feat1.device)

        # 排除自身对角线（使用 float32 避免 AMP 下溢出）
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=feat1.device)
        sim_matrix = sim_matrix.float().masked_fill(mask_self, -1e4)

        # InfoNCE: -log(exp(sim_pos) / sum(exp(sim_all)))
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class FeatureOrthogonalityLoss(nn.Module):
    """特征正交正则化：鼓励特征维度之间去相关

    防止多个维度编码冗余信息，最大化 512 维特征的信息容量，
    使模型能够利用更多维度来编码个体区分性特征。

    Args:
        feat_dim: 特征维度（用于归一化）
    """

    def __init__(self, feat_dim=512):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, features):
        """
        Args:
            features: (B, D) L2 归一化的特征
        Returns:
            loss: 标量
        """
        B, D = features.shape
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # 计算相关系数矩阵
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


# ==================== Label Smoothing ====================

class LabelSmoothingCE(nn.Module):
    """Label Smoothing Cross Entropy

    防止模型对相似类别过度自信，提升泛化能力。
    将硬标签 (one-hot) 软化为软标签，避免 logits 过大。

    Args:
        smoothing: 平滑系数，默认 0.1
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C) 预测 logits
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


class FocalLoss(nn.Module):
    """Focal Loss for Hard Example Mining

    解决类别不平衡问题，聚焦于难分类样本。
    通过降低易分类样本的权重，增加难分类样本的权重。

    Args:
        alpha: 类别权重，默认 0.25
        gamma: 聚焦参数，默认 2.0
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) 预测 logits
            targets: (B,) 真实标签 (long)
        Returns:
            loss: 标量
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==================== Circle Loss ====================

class CircleLoss(nn.Module):
    """Circle Loss for Metric Learning

    相比 Triplet Loss 和 Contrastive Loss，Circle Loss 具有：
    1. 更平滑的优化目标
    2. 更灵活的权重自适应
    3. 更清晰的收敛边界

    Args:
        m: 边界参数，默认 0.25
        gamma: 缩放参数，默认 256
    """

    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2 归一化的特征
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

        # 计算 alpha 权重
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)

        # 计算 delta
        delta_p = 1 - self.m
        delta_n = self.m

        # Circle Loss
        logit_p = -self.gamma * alpha_p * (s_p - delta_p) * pos_mask
        logit_n = self.gamma * alpha_n * (s_n - delta_n) * neg_mask

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).mean()

        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss (Additive Angular Margin Loss)

    在超球面上添加角度边界，使类内更紧凑，类间更分离。
    是 Re-ID 任务的标准损失函数之一。

    Args:
        feat_dim: 特征维度
        num_classes: 类别数
        margin: 角度边界，默认 0.5
        scale: 缩放因子，默认 64
    """

    def __init__(self, feat_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 可学习的分类器权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2 归一化的特征
            labels: (B,) 标签 (long)
        Returns:
            loss: 标量
        """
        # L2 归一化权重
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
