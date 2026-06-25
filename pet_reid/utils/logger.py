"""
训练日志和可视化模块

功能：
- 保存训练日志到文件
- 绘制loss曲线
- 绘制学习率曲线
- 保存训练配置
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器


class TrainingLogger:
    """训练日志记录器

    功能：
    - 记录训练日志到文件
    - 保存训练指标（loss, lr等）
    - 绘制训练曲线
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'dino_training',
        config: Optional[Dict] = None
    ):
        """
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
            config: 训练配置字典
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 日志文件路径
        self.log_file = os.path.join(log_dir, f'{experiment_name}_{self.timestamp}.log')
        self.metrics_file = os.path.join(log_dir, f'{experiment_name}_{self.timestamp}_metrics.json')

        # 训练指标
        self.metrics = {
            'train_loss': [],
            'learning_rate': [],
            'epoch': [],
            'timestamp': []
        }

        # 初始化日志文件
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Pet Re-ID DINOv3 Training Log\n")
            f.write(f"{'='*60}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write(f"{'='*60}\n\n")

            # 保存配置
            if self.config:
                f.write("Training Configuration:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    def log(self, message: str, print_console: bool = True):
        """记录日志

        Args:
            message: 日志消息
            print_console: 是否同时打印到控制台
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        # 打印到控制台
        if print_console:
            print(message)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        learning_rate: float,
        elapsed_time: float,
        **kwargs
    ):
        """记录epoch信息

        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            train_loss: 训练loss
            learning_rate: 学习率
            elapsed_time: 耗时
            **kwargs: 其他指标
        """
        # 保存指标
        self.metrics['train_loss'].append(train_loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['epoch'].append(epoch)
        self.metrics['timestamp'].append(time.time())

        # 格式化消息
        message = (
            f"Epoch {epoch}/{total_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"LR: {learning_rate:.6f} | "
            f"Time: {elapsed_time:.1f}s"
        )

        # 添加其他指标
        for key, value in kwargs.items():
            message += f" | {key}: {value}"

        self.log(message)

        # 保存指标到文件
        self._save_metrics()

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        learning_rate: float
    ):
        """记录batch信息

        Args:
            epoch: 当前epoch
            batch_idx: 当前batch
            total_batches: 总batch数
            loss: batch loss
            learning_rate: 学习率
        """
        message = (
            f"  [{epoch}] Batch {batch_idx}/{total_batches} | "
            f"Loss: {loss:.4f} | "
            f"LR: {learning_rate:.6f}"
        )
        self.log(message, print_console=False)

    def _save_metrics(self):
        """保存指标到JSON文件"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线

        Args:
            save_path: 保存路径（默认保存到log_dir）
        """
        if len(self.metrics['train_loss']) == 0:
            self.log("警告：没有训练数据，无法绘制曲线")
            return

        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.experiment_name}_{self.timestamp}_curves.png')

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 绘制Loss曲线
        ax1 = axes[0]
        ax1.plot(self.metrics['epoch'], self.metrics['train_loss'],
                'b-', linewidth=2, label='Train Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 绘制学习率曲线
        ax2 = axes[1]
        ax2.plot(self.metrics['epoch'], self.metrics['learning_rate'],
                'r-', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"训练曲线已保存: {save_path}")
        return save_path

    def plot_loss_detail(self, save_path: Optional[str] = None):
        """绘制详细的Loss曲线（带平滑）

        Args:
            save_path: 保存路径
        """
        if len(self.metrics['train_loss']) < 2:
            return

        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.experiment_name}_{self.timestamp}_loss_detail.png')

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.metrics['epoch']
        losses = self.metrics['train_loss']

        # 原始loss
        ax.plot(epochs, losses, 'b-', alpha=0.3, label='Raw Loss')

        # 平滑loss（移动平均）
        if len(losses) >= 5:
            window_size = min(5, len(losses) // 3)
            smoothed_losses = []
            for i in range(len(losses)):
                start = max(0, i - window_size + 1)
                smoothed_losses.append(sum(losses[start:i+1]) / (i - start + 1))
            ax.plot(epochs, smoothed_losses, 'b-', linewidth=2, label='Smoothed Loss')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss (Detail)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 添加最小loss标注
        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        ax.annotate(f'Min: {min_loss:.4f}',
                   xy=(min_epoch, min_loss),
                   xytext=(min_epoch + len(epochs)*0.1, min_loss),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"详细Loss曲线已保存: {save_path}")
        return save_path

    def generate_summary(self) -> str:
        """生成训练总结

        Returns:
            summary: 总结文本
        """
        if len(self.metrics['train_loss']) == 0:
            return "No training data available."

        losses = self.metrics['train_loss']
        lrs = self.metrics['learning_rate']

        summary = f"""
{'='*60}
Training Summary
{'='*60}
Experiment: {self.experiment_name}
Duration: {len(losses)} epochs

Loss Statistics:
  Initial Loss: {losses[0]:.4f}
  Final Loss: {losses[-1]:.4f}
  Min Loss: {min(losses):.4f} (Epoch {self.metrics['epoch'][losses.index(min(losses))]})
  Max Loss: {max(losses):.4f}
  Loss Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%

Learning Rate:
  Initial LR: {lrs[0]:.6f}
  Final LR: {lrs[-1]:.6f}
  Min LR: {min(lrs):.6f}
  Max LR: {max(lrs):.6f}

Output Files:
  Log: {self.log_file}
  Metrics: {self.metrics_file}
{'='*60}
"""
        self.log(summary)
        return summary


def create_training_logger(
    log_dir: str,
    experiment_name: str = 'dino_training',
    config: Optional[Dict] = None
) -> TrainingLogger:
    """创建训练日志记录器

    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        config: 训练配置

    Returns:
        logger: TrainingLogger实例
    """
    return TrainingLogger(log_dir, experiment_name, config)
