"""
DINOv3 自蒸馏损失函数

实现DINOv3的损失计算，包括：
- Cross-entropy loss
- Centering
- Sharpening
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """DINOv3 自蒸馏损失

    训练目标：
    - Student的输出（经过predictor）应该匹配Teacher的输出（经过EMA更新）
    - 使用Cross-entropy loss
    - Teacher输出经过centering和sharpening

    防止崩塌的机制：
    1. Centering: 减去Teacher输出的运行均值
    2. Sharpening: 使用低温度使Teacher输出更尖锐
    3. EMA更新: Teacher缓慢跟踪Student
    4. Predictor: Student的额外预测头
    """

    def __init__(self, teacher_temp: float = 0.04, student_temp: float = 0.1):
        """
        Args:
            teacher_temp: Teacher温度，越低分布越尖锐
            student_temp: Student温度
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor
    ) -> torch.Tensor:
        """计算DINOv3损失

        Args:
            student_out: Student的预测头输出 (B, num_views, D) 或 (B, D)
            teacher_out: Teacher的投影头输出（已centering）(B, num_global, D) 或 (B, D)

        Returns:
            loss: 标量损失
        """
        # 处理不同维度的输入
        if student_out.dim() == 2:
            student_out = student_out.unsqueeze(1)
        if teacher_out.dim() == 2:
            teacher_out = teacher_out.unsqueeze(1)

        B, num_views, D = student_out.shape
        _, num_global, _ = teacher_out.shape

        # Sharpening (Teacher)
        teacher_out = F.softmax(teacher_out / self.teacher_temp, dim=-1)

        # Sharpening (Student)
        student_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        # 计算Cross-entropy loss
        total_loss = 0.0
        count = 0

        for i in range(num_views):
            for j in range(num_global):
                # KL散度（等价于cross-entropy当teacher是one-hot时）
                loss = -torch.sum(
                    teacher_out[:, j] * student_out[:, i],
                    dim=-1
                ).mean()
                total_loss = total_loss + loss
                count += 1

        return total_loss / count


class DINOLossWithCenter(nn.Module):
    """带Centering的DINOv3损失

    在计算损失时同时更新center
    """

    def __init__(
        self,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', None)

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        update_center: bool = True
    ) -> torch.Tensor:
        """计算损失并更新center

        Args:
            student_out: (B, num_views, D)
            teacher_out: (B, num_global, D) - 未centering的原始输出
            update_center: 是否更新center

        Returns:
            loss: 标量
        """
        # 初始化center
        if self.center is None:
            self.center = torch.zeros(teacher_out.shape[-1],
                                     device=teacher_out.device)

        # Centering
        teacher_out = teacher_out - self.center

        # 更新center
        if update_center:
            batch_center = teacher_out.mean(dim=(0, 1))
            self.center = self.center * self.center_momentum + \
                         batch_center * (1 - self.center_momentum)

        # 计算损失
        loss_fn = DINOLoss(self.teacher_temp, self.student_temp)
        return loss_fn(student_out, teacher_out)


def cosine_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """余弦相似度损失

    用于特征对齐
    """
    return (1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1)).mean()


def self_similarity_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """自相似性保持损失

    保持样本间的相似性关系
    """
    s_student = student_feat @ student_feat.T
    s_teacher = teacher_feat @ teacher_feat.T
    return F.mse_loss(s_student, s_teacher)
