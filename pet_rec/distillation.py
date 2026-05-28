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
