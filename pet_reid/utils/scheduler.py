"""
学习率调度器模块

提供Cosine Annealing with Warmup等调度策略
"""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR
)
import math


class CosineAnnealingWarmupScheduler:
    """Cosine Annealing with Warmup 学习率调度器

    前warmup_epochs个epoch线性预热
    之后按余弦曲线衰减

    Args:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        min_lr: 最小学习率
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 200,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

        # 创建调度器
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> SequentialLR:
        """创建调度器"""
        # 预热阶段：线性增加
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )

        # 余弦退火阶段
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_epochs - self.warmup_epochs,
            eta_min=self.min_lr
        )

        # 组合调度器
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        return scheduler

    def step(self):
        """更新学习率"""
        self.scheduler.step()

    def get_last_lr(self) -> list:
        """获取当前学习率"""
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """保存状态"""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict):
        """加载状态"""
        self.scheduler.load_state_dict(state_dict)


class CosineAnnealingWarmupRestartsScheduler:
    """带重启的Cosine Annealing with Warmup

    每隔restart_interval个epoch重启一次

    Args:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        restart_interval: 重启间隔
        min_lr: 最小学习率
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        restart_interval: int = 50,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.restart_interval = restart_interval
        self.min_lr = min_lr

        self.current_epoch = 0

    def step(self):
        """更新学习率"""
        self.current_epoch += 1

        # 计算在当前重启周期内的epoch
        cycle_epoch = self.current_epoch % self.restart_interval

        if cycle_epoch < self.warmup_epochs:
            # 预热阶段
            lr_scale = cycle_epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            progress = (cycle_epoch - self.warmup_epochs) / (self.restart_interval - self.warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        # 获取基础学习率
        base_lr = self.optimizer.defaults['lr']

        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.min_lr, base_lr * lr_scale)

    def get_last_lr(self) -> list:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    total_epochs: int = 200,
    min_lr: float = 1e-6
) -> CosineAnnealingWarmupScheduler:
    """获取Warmup + Cosine Annealing调度器

    Args:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        min_lr: 最小学习率

    Returns:
        scheduler: 调度器实例
    """
    return CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr=min_lr
    )
