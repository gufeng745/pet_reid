"""
Re-ID 监督微调脚本

基于预训练的CNN backbone进行Re-ID训练
支持加载DINOv3预训练权重

用法：
    python train_reid.py --epochs 80 --P 16 --K 4
    python train_reid.py --pretrained_dino checkpoints/dino/best_dino.pth
"""

import os
import sys
import time
import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from config import get_reid_config, ReIDConfig
from models.reid_model import ReIDModel
from datasets.reid_dataset import (
    ReIDDataset,
    PKSampler,
    split_dataset_by_id,
    get_train_transform,
    get_val_transform
)
from losses.reid_loss import (
    TripletMarginLoss,
    SupervisedContrastiveLoss,
    FeatureOrthogonalityLoss
)
from utils.scheduler import get_warmup_cosine_scheduler
from utils.metrics import compute_reid_metrics


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.005, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_reid(args):
    """Re-ID训练主函数"""
    # 创建配置
    config = get_reid_config(**vars(args))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 设置随机种子
    set_seed(config.seed)

    # ========== 创建输出目录 ==========
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ========== 按ID划分数据集 ==========
    print("\n" + "=" * 50)
    print("划分数据集...")
    print("=" * 50)

    train_ids, val_ids = split_dataset_by_id(
        config.data_root,
        val_ratio=config.val_ratio,
        seed=config.seed
    )

    # ========== 创建数据集 ==========
    print("\n" + "=" * 50)
    print("创建数据集...")
    print("=" * 50)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = ReIDDataset(
        config.data_root,
        transform=train_transform,
        id_list=train_ids
    )
    val_dataset = ReIDDataset(
        config.data_root,
        transform=val_transform,
        id_list=val_ids
    )

    num_classes = train_dataset.num_classes
    config.num_classes = num_classes

    print(f"Train: {len(train_dataset)} 张图片, {train_dataset.num_classes} 个身份")
    print(f"Val: {len(val_dataset)} 张图片, {val_dataset.num_classes} 个身份")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("错误：数据集为空")
        return

    # ========== 创建数据加载器 ==========
    train_sampler = PKSampler(
        train_dataset,
        P=config.P,
        K=config.K,
        drop_last=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ========== 创建模型 ==========
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)

    model = ReIDModel(
        backbone_name=config.backbone,
        proj_dim=config.proj_dim,
        num_classes=num_classes,
        pretrained_backbone=True,
        use_gem_pool=config.use_gem_pool,
        use_se=config.use_se,
        use_bnneck=config.use_bnneck,
        se_reduction=config.se_reduction,
        pretrained_dino_path=config.pretrained_dino,
        backbone_weight_path=config.backbone_weight_path
    ).to(device)

    # ========== 创建损失函数 ==========
    criterion_id = nn.CrossEntropyLoss(label_smoothing=config.id_label_smoothing)
    criterion_triplet = TripletMarginLoss(margin=config.triplet_margin)
    criterion_contrastive = SupervisedContrastiveLoss(temperature=config.contrastive_temp)
    criterion_ortho = FeatureOrthogonalityLoss(feat_dim=config.proj_dim)

    print(f"使用 ID Loss (label_smoothing={config.id_label_smoothing})")
    print(f"使用 Triplet Loss (margin={config.triplet_margin})")
    print(f"使用 Supervised Contrastive Loss (temp={config.contrastive_temp})")
    print(f"使用 Feature Orthogonality Loss")

    # ========== 创建优化器（双学习率） ==========
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': config.lr_backbone},
        {'params': model.projector.parameters(), 'lr': config.lr_head},
        {'params': model.id_head.parameters(), 'lr': config.lr_head},
    ], weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == 'cuda' else None
    if scaler:
        print("使用混合精度训练 (AMP)")

    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=0.005,
        mode='max'
    ) if config.use_early_stopping else None

    best_rank1 = 0.0

    # ========== 训练循环 ==========
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    for epoch in range(config.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_loss_id = 0.0
        epoch_loss_triplet = 0.0
        epoch_loss_contrastive = 0.0
        epoch_loss_ortho = 0.0

        id_correct = 0
        id_total = 0

        t0 = time.time()

        nan_count = 0  # NaN批次计数

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).long()

            # 前向传播
            if scaler:
                with torch.amp.autocast('cuda'):
                    emb, id_logits = model(images)

                    # ID分类损失
                    loss_id = criterion_id(id_logits, labels)

                    # Triplet Loss
                    loss_triplet = criterion_triplet(emb, labels)

                    # 监督对比损失
                    loss_contrastive = criterion_contrastive(emb, labels)

                    # 特征正交正则化
                    loss_ortho = criterion_ortho(emb)

                    # 总损失
                    loss = (config.lambda_id * loss_id +
                           config.lambda_triplet * loss_triplet +
                           config.lambda_contrastive * loss_contrastive +
                           config.lambda_ortho * loss_ortho)

                # NaN检测：跳过这个batch
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"  [警告] batch {batch_idx+1} 检测到 NaN/Inf loss，跳过")
                    if nan_count > 10:
                        print(f"  [错误] 连续NaN过多，停止训练")
                        break
                    continue

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                emb, id_logits = model(images)

                loss_id = criterion_id(id_logits, labels)
                loss_triplet = criterion_triplet(emb, labels)
                loss_contrastive = criterion_contrastive(emb, labels)
                loss_ortho = criterion_ortho(emb)

                loss = (config.lambda_id * loss_id +
                       config.lambda_triplet * loss_triplet +
                       config.lambda_contrastive * loss_contrastive +
                       config.lambda_ortho * loss_ortho)

                # NaN检测：跳过这个batch
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"  [警告] batch {batch_idx+1} 检测到 NaN/Inf loss，跳过")
                    if nan_count > 10:
                        print(f"  [错误] 连续NaN过多，停止训练")
                        break
                    continue

                optimizer.zero_grad()
                loss.backward()

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()

            # NaN检测通过，重置计数
            nan_count = 0

            # 统计
            epoch_loss += loss.item()
            epoch_loss_id += loss_id.item()
            epoch_loss_triplet += loss_triplet.item()
            epoch_loss_contrastive += loss_contrastive.item()
            epoch_loss_ortho += loss_ortho.item()

            # 身份准确率
            id_pred = id_logits.argmax(dim=1)
            id_correct += (id_pred == labels).sum().item()
            id_total += labels.size(0)

            # 打印进度
            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                id_acc = id_correct / id_total
                print(f"  [{epoch+1}/{config.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg_loss:.4f} id={epoch_loss_id/(batch_idx+1):.4f} "
                      f"triplet={epoch_loss_triplet/(batch_idx+1):.4f} "
                      f"con={epoch_loss_contrastive/(batch_idx+1):.4f} "
                      f"ortho={epoch_loss_ortho/(batch_idx+1):.4f} "
                      f"id_acc={id_acc:.3f}")

        # 更新学习率
        scheduler.step()

        # Epoch统计
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        id_acc = id_correct / id_total

        print(f"Epoch {epoch+1}/{config.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f} | id_acc={id_acc:.3f}")

        # ========== 验证 ==========
        val_loss, val_rank1 = validate(model, val_loader, config, device)
        print(f"  Val loss: {val_loss:.4f} | Val Rank-1: {val_rank1:.3f}")

        # ========== 保存checkpoint ==========
        if (epoch + 1) % config.save_interval == 0 or val_rank1 > best_rank1:
            if val_rank1 > best_rank1:
                best_rank1 = val_rank1
                name = 'best_reid.pth'
            else:
                name = f'reid_epoch{epoch+1}.pth'

            path = os.path.join(config.checkpoint_dir, name)
            torch.save({
                'epoch': epoch + 1,
                'student': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
                'rank1': val_rank1,
                'num_classes': num_classes,
                'args': vars(args),
            }, path)
            print(f"  Saved: {path}")

        # 早停检查
        if early_stopping and early_stopping(val_rank1):
            print(f"\n早停触发：验证Rank-1在{config.patience}个epoch内未改善")
            break

    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, 'final_reid.pth')
    torch.save({
        'student': model.state_dict(),
        'num_classes': num_classes,
        'args': vars(args),
    }, final_path)

    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证Rank-1: {best_rank1:.4f}")
    print(f"最终模型: {final_path}")
    print("=" * 50)

    return model


@torch.no_grad()
def validate(model, val_loader, config, device):
    """验证（计算loss和Rank-1准确率）"""
    model.eval()

    total_loss = 0.0
    count = 0

    all_features = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
            emb, id_logits = model(images)

        # 计算损失（float32保证精度）
        loss_id = F.cross_entropy(id_logits.float(), labels, label_smoothing=config.id_label_smoothing)
        loss = config.lambda_id * loss_id

        # NaN检测
        if torch.isnan(loss):
            continue

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

        # 收集特征
        all_features.append(emb.cpu().float())
        all_labels.append(labels.cpu())

    if count == 0:
        return float('inf'), 0.0

    # 计算Rank-1
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 余弦相似度
    sim_matrix = all_features @ all_features.T

    # 排除自身
    eye_mask = torch.eye(len(all_labels), dtype=torch.bool)
    sim_matrix = sim_matrix - eye_mask.float() * 1e9

    # 找最相似的
    top1_indices = sim_matrix.argmax(dim=1)

    # 计算Rank-1
    rank1_correct = (all_labels[top1_indices] == all_labels).sum().item()
    rank1_acc = rank1_correct / len(all_labels)

    model.train()
    return total_loss / max(count, 1), rank1_acc


def parse_args():
    p = argparse.ArgumentParser(description='Re-ID 监督微调')

    # 数据
    p.add_argument('--data_root', type=str, default='../pet_rec/reid_dataset',
                   help='数据集根目录')
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='验证集比例')

    # 模型
    p.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                   help='CNN backbone')
    p.add_argument('--proj_dim', type=int, default=512,
                   help='投影维度')
    p.add_argument('--pretrained_dino', type=str, default=None,
                   help='DINOv3预训练权重路径')

    # 训练
    p.add_argument('--epochs', type=int, default=80,
                   help='训练轮数')
    p.add_argument('--batch_size', type=int, default=64,
                   help='批大小')
    p.add_argument('--lr_backbone', type=float, default=5e-4,
                   help='backbone学习率')
    p.add_argument('--lr_head', type=float, default=1e-3,
                   help='head学习率')

    # PK Sampler
    p.add_argument('--P', type=int, default=16,
                   help='每个batch的ID数')
    p.add_argument('--K', type=int, default=4,
                   help='每个ID的样本数')

    # 损失权重
    p.add_argument('--lambda_id', type=float, default=0.5,
                   help='ID Loss权重')
    p.add_argument('--lambda_triplet', type=float, default=0.3,
                   help='Triplet Loss权重')
    p.add_argument('--lambda_contrastive', type=float, default=0.2,
                   help='Contrastive Loss权重')
    p.add_argument('--lambda_ortho', type=float, default=0.05,
                   help='Ortho Loss权重')

    # 其他
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--patience', type=int, default=25,
                   help='早停容忍epoch数')
    p.add_argument('--save_interval', type=int, default=10,
                   help='保存间隔')
    p.add_argument('--log_interval', type=int, default=5,
                   help='日志间隔')

    return p.parse_args()


if __name__ == '__main__':
    train_reid(parse_args())
