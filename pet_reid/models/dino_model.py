"""
DINOv3 自监督模型

实现DINOv3的自蒸馏机制：
- Student: CNN backbone + 投影头 + 预测头
- Teacher: CNN backbone + 投影头 (EMA更新)

支持MAE预处理：Student看遮盖后的图像，Teacher看原始图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple
from .backbone import CNNBackbone


class MAEMaskGenerator:
    """MAE 遮盖生成器

    生成随机的patch遮盖mask，
    用于DINO+MAE混合训练。

    Args:
        mask_ratio: 遮盖比例 (0-1)
        patch_size: patch大小（像素）
    """

    def __init__(self, mask_ratio: float = 0.75, patch_size: int = 16):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成遮盖mask

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            masked_x: 遮盖后的图像 (B, C, H, W)
            mask: 遮盖mask (B, num_patches)
        """
        B, C, H, W = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        num_patches = pH * pW

        # 生成随机mask
        num_masked = int(num_patches * self.mask_ratio)
        rand_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask = torch.zeros(B, num_patches, device=x.device)
        mask[:, :num_masked] = 1
        # 恢复原始顺序
        mask = mask.scatter(1, rand_indices, mask)

        # 将mask应用到图像上
        mask_2d = mask.reshape(B, 1, pH, pW)
        mask_2d = mask_2d.repeat_interleave(self.patch_size, dim=2)
        mask_2d = mask_2d.repeat_interleave(self.patch_size, dim=3)

        # 遮盖区域用0填充
        masked_x = x * (1 - mask_2d)

        return masked_x, mask


class ProjectionHead(nn.Module):
    """投影头 (Projection Head)

    将backbone特征映射到低维空间，用于自蒸馏
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """预测头 (Prediction Head)

    DINOv3的关键组件，只在Student端使用
    防止模型崩塌（所有输出相同）
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOModel(nn.Module):
    """DINOv3 自监督模型

    包含：
    - Student: backbone + projector + predictor
    - Teacher: backbone + projector (EMA)

    训练时：
    - Teacher处理全局视图，生成软标签
    - Student处理所有视图，学习预测Teacher的输出
    - Teacher通过EMA更新，缓慢跟踪Student

    支持MAE预处理：
    - Student看遮盖后的图像
    - Teacher看原始（未遮盖）图像
    """

    def __init__(
        self,
        backbone_name: str = 'mobilenetv3_large_100',
        proj_dim: int = 384,
        hidden_dim: int = 2048,
        predictor_hidden_dim: int = 1024,
        pretrained_backbone: bool = True,
        local_weight_path: str = None,
        center_momentum: float = 0.9,
        use_mae: bool = True,
        mae_mask_ratio: float = 0.75,
        mae_patch_size: int = 16
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.center_momentum = center_momentum
        self.use_mae = use_mae

        # ========== Student ==========
        self.student_backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            local_weight_path=local_weight_path
        )
        student_feat_dim = self.student_backbone.feature_dim

        self.student_projector = ProjectionHead(
            in_dim=student_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim
        )
        self.student_predictor = PredictionHead(
            in_dim=proj_dim,
            hidden_dim=predictor_hidden_dim,
            out_dim=proj_dim
        )

        # ========== Teacher (EMA) ==========
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_projector = copy.deepcopy(self.student_projector)
        # Teacher没有predictor！

        # 冻结Teacher参数
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_projector.parameters():
            param.requires_grad = False

        # ========== Centering ==========
        self.register_buffer('center', torch.zeros(proj_dim))

        # ========== MAE ==========
        if use_mae:
            self.mae_masker = MAEMaskGenerator(
                mask_ratio=mae_mask_ratio,
                patch_size=mae_patch_size
            )

        # 统计信息
        total_params = sum(p.numel() for p in self.parameters())
        student_params = sum(p.numel() for p in self.student_backbone.parameters()) + \
                        sum(p.numel() for p in self.student_projector.parameters()) + \
                        sum(p.numel() for p in self.student_predictor.parameters())
        print(f"[DINOModel] 总参数量: {total_params/1e6:.2f}M")
        print(f"[DINOModel] Student参数量: {student_params/1e6:.2f}M")
        print(f"[DINOModel] 特征维度: {proj_dim}")
        print(f"[DINOModel] MAE遮盖: {'启用' if use_mae else '禁用'}")

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA更新Teacher

        Args:
            momentum: 动量系数，通常从0.996逐渐增加到1.0
        """
        # 更新backbone
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data

        # 更新projector
        for param_s, param_t in zip(
            self.student_projector.parameters(),
            self.teacher_projector.parameters()
        ):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """更新center

        Args:
            teacher_output: Teacher的输出 (B, proj_dim) 或 (B, num_global, proj_dim)
        """
        if teacher_output.dim() == 3:
            batch_center = teacher_output.mean(dim=(0, 1))
        else:
            batch_center = teacher_output.mean(dim=0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Student前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            pred: 预测头输出 (B, proj_dim)
        """
        feat = self.student_backbone(x)
        proj = self.student_projector(feat)
        pred = self.student_predictor(proj)
        return pred

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            proj: 投影头输出 (B, proj_dim)，已centering
        """
        feat = self.teacher_backbone(x)
        proj = self.teacher_projector(feat)
        # Centering
        proj = proj - self.center
        return proj

    def forward(
        self,
        global_views: torch.Tensor,
        local_views: torch.Tensor,
        teacher_momentum: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的DINOv3前向传播

        Args:
            global_views: 全局视图 (B, num_global, 3, H, W)
            local_views: 局部视图 (B, num_local, 3, h, w)
            teacher_momentum: Teacher EMA动量
            teacher_temp: Teacher温度
            student_temp: Student温度

        Returns:
            student_out: Student输出 (B, num_views, proj_dim)
            teacher_out: Teacher输出 (B, num_global, proj_dim)
            loss: 自蒸馏损失
        """
        B = global_views.shape[0]
        num_global = global_views.shape[1]
        num_local = local_views.shape[1]

        # ========== Teacher处理全局视图（未遮盖） ==========
        teacher_out_list = []
        for i in range(num_global):
            view = global_views[:, i]  # (B, 3, H, W)
            teacher_out = self.forward_teacher(view)  # (B, proj_dim)
            teacher_out_list.append(teacher_out)

        # 合并Teacher输出
        teacher_out = torch.stack(teacher_out_list, dim=1)  # (B, num_global, proj_dim)

        # Sharpening (Teacher)
        teacher_out = F.softmax(teacher_out / teacher_temp, dim=-1)

        # ========== Student处理所有视图 ==========
        student_out_list = []

        # 处理全局视图（应用MAE遮盖）
        for i in range(num_global):
            view = global_views[:, i]
            if self.use_mae and self.training:
                view, _ = self.mae_masker(view)  # 应用遮盖
            student_out = self.forward_student(view)
            student_out_list.append(student_out)

        # 处理局部视图（应用MAE遮盖）
        for i in range(num_local):
            view = local_views[:, i]
            if self.use_mae and self.training:
                view, _ = self.mae_masker(view)  # 应用遮盖
            student_out = self.forward_student(view)
            student_out_list.append(student_out)

        # 合并Student输出
        student_out = torch.stack(student_out_list, dim=1)  # (B, num_views, proj_dim)

        # Sharpening (Student)
        student_out = F.softmax(student_out / student_temp, dim=-1)

        # ========== 计算损失 ==========
        loss = self.compute_loss(student_out, teacher_out, student_temp, teacher_temp)

        # ========== 更新Teacher和Center ==========
        self.update_teacher(teacher_momentum)
        self.update_center(teacher_out)

        return student_out, teacher_out, loss

    def compute_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04
    ) -> torch.Tensor:
        """计算DINOv3自蒸馏损失

        使用log_softmax保证数值稳定性

        Args:
            student_out: (B, num_views, proj_dim) - 已softmax
            teacher_out: (B, num_global, proj_dim) - 已softmax
            student_temp: Student温度（用于log_softmax）
            teacher_temp: Teacher温度（用于log_softmax）

        Returns:
            loss: 标量
        """
        B, num_views, D = student_out.shape
        _, num_global, _ = teacher_out.shape

        total_loss = 0.0
        count = 0

        # 局部视图(Student) -> 全局视图(Teacher)
        # 全局视图(Student) -> 全局视图(Teacher)
        for i in range(num_views):
            for j in range(num_global):
                # 使用log保证数值稳定性
                # student_out已经softmax过，直接log
                log_student = torch.log(student_out[:, i] + 1e-8)
                # Cross-entropy loss
                loss = -torch.sum(
                    teacher_out[:, j] * log_student,
                    dim=-1
                ).mean()
                total_loss = total_loss + loss
                count += 1

        return total_loss / count

    def get_student_backbone(self) -> nn.Module:
        """获取Student的backbone（用于下游任务）"""
        return self.student_backbone

    def save_pretrained(self, path: str):
        """保存预训练模型"""
        torch.save({
            'student_backbone': self.student_backbone.state_dict(),
            'student_projector': self.student_projector.state_dict(),
            'student_predictor': self.student_predictor.state_dict(),
            'teacher_backbone': self.teacher_backbone.state_dict(),
            'teacher_projector': self.teacher_projector.state_dict(),
            'center': self.center,
        }, path)
        print(f"[DINOModel] 保存预训练模型到: {path}")

    def load_pretrained(self, path: str):
        """加载预训练模型"""
        ckpt = torch.load(path, map_location='cpu')

        self.student_backbone.load_state_dict(ckpt['student_backbone'])
        self.student_projector.load_state_dict(ckpt['student_projector'])
        self.student_predictor.load_state_dict(ckpt['student_predictor'])
        self.teacher_backbone.load_state_dict(ckpt['teacher_backbone'])
        self.teacher_projector.load_state_dict(ckpt['teacher_projector'])
        self.center = ckpt['center']

        print(f"[DINOModel] 加载预训练模型: {path}")
