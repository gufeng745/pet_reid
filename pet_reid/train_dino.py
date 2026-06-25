"""
DINOv3 自监督预训练脚本

基于DINOv3的自蒸馏机制训练CNN backbone
不需要标签，只需要图片

用法：
    python train_dino.py --epochs 200 --batch_size 256
    python train_dino.py --data_root ../pet_rec/reid_dataset --epochs 100
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from config import get_dino_config, DINOConfig
from models.dino_model import DINOModel
from datasets.dino_dataset import DINODataset, DINODataLoader
from losses.dino_loss import DINOLoss
from utils.scheduler import get_warmup_cosine_scheduler
from utils.logger import create_training_logger


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_dino(args):
    """DINOv3预训练主函数"""
    # 创建配置
    config = get_dino_config(**vars(args))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 设置随机种子
    set_seed(config.seed)

    # ========== 创建输出目录 ==========
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ========== 创建日志记录器 ==========
    logger = create_training_logger(
        log_dir=config.log_dir,
        experiment_name='dino_pretraining',
        config=vars(config)
    )
    logger.log(f"Device: {device}")
    logger.log(f"Output directory: {config.output_dir}")
    logger.log(f"Checkpoint directory: {config.checkpoint_dir}")
    logger.log(f"Log directory: {config.log_dir}")

    # ========== 创建数据集 ==========
    logger.log("\n" + "=" * 50)
    logger.log("创建数据集...")
    logger.log("=" * 50)

    dataset = DINODataset(
        root=config.data_root,
        species=None  # 包含cat和dog
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=DINODataLoader.collate_fn,
        drop_last=True
    )

    logger.log(f"数据集大小: {len(dataset)}")
    logger.log(f"Batch数: {len(dataloader)}")

    # ========== 创建模型 ==========
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)

    model = DINOModel(
        backbone_name=config.backbone,
        proj_dim=config.proj_dim,
        hidden_dim=config.hidden_dim,
        predictor_hidden_dim=config.predictor_hidden_dim,
        pretrained_backbone=True
    ).to(device)

    # ========== 创建损失函数和优化器 ==========
    criterion = DINOLoss(
        teacher_temp=config.teacher_temp,
        student_temp=config.student_temp
    )

    # 只优化Student的参数
    student_params = list(model.student_backbone.parameters()) + \
                    list(model.student_projector.parameters()) + \
                    list(model.student_predictor.parameters())

    optimizer = AdamW(
        student_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=config.min_lr
    )

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if config.use_amp and device.type == 'cuda' else None

    # ========== 训练循环 ==========
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (global_views, local_views) in enumerate(dataloader):
            global_views = global_views.to(device)
            local_views = local_views.to(device)

            # 计算当前的teacher momentum
            progress = epoch / config.epochs
            teacher_momentum = config.teacher_momentum_start + \
                             (config.teacher_momentum_end - config.teacher_momentum_start) * progress

            # 前向传播
            if scaler:
                with torch.amp.autocast('cuda'):
                    student_out, teacher_out, loss = model(
                        global_views,
                        local_views,
                        teacher_momentum=teacher_momentum,
                        teacher_temp=config.teacher_temp,
                        student_temp=config.student_temp
                    )

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_params, config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                student_out, teacher_out, loss = model(
                    global_views,
                    local_views,
                    teacher_momentum=teacher_momentum,
                    teacher_temp=config.teacher_temp,
                    student_temp=config.student_temp
                )

                optimizer.zero_grad()
                loss.backward()

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student_params, config.max_grad_norm)

                optimizer.step()

            # 统计
            epoch_loss += loss.item()
            num_batches += 1

            # 打印进度
            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]['lr']
                logger.log_batch(
                    epoch=epoch+1,
                    batch_idx=batch_idx+1,
                    total_batches=len(dataloader),
                    loss=avg_loss,
                    learning_rate=lr
                )

        # 更新学习率
        scheduler.step()

        # Epoch统计
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # 记录epoch日志
        logger.log_epoch(
            epoch=epoch+1,
            total_epochs=config.epochs,
            train_loss=avg_loss,
            learning_rate=lr,
            elapsed_time=elapsed,
            teacher_momentum=teacher_momentum
        )

        # 保存checkpoint
        if (epoch + 1) % config.save_interval == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                name = 'best_dino.pth'
            else:
                name = f'dino_epoch{epoch+1}.pth'

            path = os.path.join(config.checkpoint_dir, name)
            model.save_pretrained(path)
            logger.log(f"  Saved checkpoint: {path}")

        # 每10个epoch绘制一次训练曲线
        if (epoch + 1) % 10 == 0:
            logger.plot_training_curves()
            logger.plot_loss_detail()

    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, 'final_dino.pth')
    model.save_pretrained(final_path)

    # 绘制最终训练曲线
    curves_path = logger.plot_training_curves()
    loss_detail_path = logger.plot_loss_detail()

    # 生成训练总结
    summary = logger.generate_summary()

    logger.log(f"\n最终模型: {final_path}")
    logger.log(f"训练曲线: {curves_path}")
    logger.log(f"详细Loss曲线: {loss_detail_path}")

    return model


def parse_args():
    p = argparse.ArgumentParser(description='DINOv3 自监督预训练')

    # 数据
    p.add_argument('--data_root', type=str, default='../pet_rec/reid_dataset',
                   help='数据集根目录')

    # 模型
    p.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                   help='CNN backbone')
    p.add_argument('--proj_dim', type=int, default=384,
                   help='投影维度')
    p.add_argument('--hidden_dim', type=int, default=2048,
                   help='隐藏层维度')

    # 训练
    p.add_argument('--epochs', type=int, default=200,
                   help='训练轮数')
    p.add_argument('--batch_size', type=int, default=256,
                   help='批大小')
    p.add_argument('--lr', type=float, default=5e-4,
                   help='学习率')
    p.add_argument('--weight_decay', type=float, default=0.04,
                   help='权重衰减')

    # DINOv3
    p.add_argument('--teacher_momentum_start', type=float, default=0.996,
                   help='Teacher EMA动量起始值')
    p.add_argument('--teacher_temp', type=float, default=0.04,
                   help='Teacher温度')
    p.add_argument('--student_temp', type=float, default=0.1,
                   help='Student温度')

    # 其他
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--num_workers', type=int, default=4,
                   help='数据加载线程数')
    p.add_argument('--save_interval', type=int, default=20,
                   help='保存间隔')
    p.add_argument('--log_interval', type=int, default=10,
                   help='日志间隔')

    return p.parse_args()


if __name__ == '__main__':
    train_dino(parse_args())
