import torch
import torch.nn as nn
import torch.nn.functional as F


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
