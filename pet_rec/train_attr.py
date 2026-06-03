"""多属性感知宠物模型训练脚本

使用属性标注数据集（color_primary, color_secondary, pattern）训练学生模型，
通过多任务学习迫使 backbone 编码颜色和花纹信息。

用法：
    python train_attr.py --epochs 50 --batch_size 64
"""

import os
import sys
import time
import argparse
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import DINOv3Teacher, DINOv2Teacher, MobileNetV2StudentWithAttr, TeacherAdapter
from distillation import AttributeDistillationLoss


# ==================== 标签编码器 ====================

class LabelEncoder:
    """将文本标签编码为数字，支持单标签和多标签"""

    def __init__(self, name, classes, multi_label=False):
        self.name = name
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.multi_label = multi_label
        self.num_classes = len(classes)

    def encode(self, text):
        """编码单个标签

        Args:
            text: 标签文本，多标签用逗号分隔，如 "tabby,bicolor"
        Returns:
            单标签: int
            多标签: float tensor (num_classes,) multi-hot
        """
        if not text or text.strip() == '':
            if self.multi_label:
                return torch.zeros(self.num_classes, dtype=torch.float32)
            return self.class_to_idx.get('unknown', 0)

        if self.multi_label:
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for item in text.split(','):
                item = item.strip()
                if item in self.class_to_idx:
                    vec[self.class_to_idx[item]] = 1.0
            return vec
        else:
            return self.class_to_idx.get(text.strip(), 0)


def build_label_encoders(csv_path):
    """从 CSV 文件读取所有标签，构建编码器

    Returns:
        encoders: dict of LabelEncoder
        rows: list of dict (CSV 行)
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 收集各类别的所有值
    color_primary_vals = set()
    color_secondary_vals = set()
    pattern_vals = set()

    for row in rows:
        color_primary_vals.add(row['color_primary'].strip())
        for item in row['color_secondary'].split(','):
            item = item.strip()
            if item:
                color_secondary_vals.add(item)
        for item in row['pattern'].split(','):
            item = item.strip()
            if item:
                pattern_vals.add(item)

    # 统一颜色类别（主色和副色用同一套类别）
    all_colors = sorted(color_primary_vals | color_secondary_vals)
    patterns = sorted(pattern_vals)

    print(f"颜色类别 ({len(all_colors)}): {all_colors}")
    print(f"花纹类别 ({len(patterns)}): {patterns}")

    encoders = {
        'color_primary': LabelEncoder('color_primary', all_colors, multi_label=False),
        'color_secondary': LabelEncoder('color_secondary', all_colors, multi_label=True),
        'pattern': LabelEncoder('pattern', patterns, multi_label=True),
    }
    return encoders, rows


# ==================== 数据集 ====================

class AttrPetDataset(Dataset):
    """属性标注宠物数据集

    对同一张图生成两个不同增强视角（用于蒸馏），同时返回属性标签。
    """

    def __init__(self, image_dir, rows, encoders, transform1=None, transform2=None):
        self.image_dir = image_dir
        self.rows = rows
        self.encoders = encoders
        self.transform1 = transform1 or get_dino_augmentation()
        self.transform2 = transform2 or get_dino_augmentation()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.image_dir, row['filename'])

        img = Image.open(img_path).convert('RGB')
        view1 = self.transform1(img)
        view2 = self.transform2(img)

        color_pri = self.encoders['color_primary'].encode(row['color_primary'])
        color_sec = self.encoders['color_secondary'].encode(row['color_secondary'])
        pattern = self.encoders['pattern'].encode(row['pattern'])

        return view1, view2, color_pri, color_sec, pattern


def get_dino_augmentation(crop_scale=(0.4, 1.0)):
    """DINO 风格数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_image_dir():
    """查找图片目录（支持 train 子目录）"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'cat_dog_attr')
    train_dir = os.path.join(base, 'train')
    if os.path.isdir(train_dir):
        return train_dir
    return base


# ==================== 训练 ====================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === 加载标签 ===
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations..csv')
    if not os.path.exists(csv_path):
        print(f"错误：找不到标注文件 {csv_path}")
        return

    encoders, rows = build_label_encoders(csv_path)
    num_colors = encoders['color_primary'].num_classes
    num_patterns = encoders['pattern'].num_classes
    print(f"样本数: {len(rows)}, 颜色类: {num_colors}, 花纹类: {num_patterns}")

    # === 划分 train/val ===
    val_size = int(len(rows) * 0.1)
    train_size = len(rows) - val_size
    train_rows, val_rows = random_split(
        rows, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # === 数据集 ===
    image_dir = find_image_dir()
    train_dataset = AttrPetDataset(image_dir, list(train_rows), encoders)
    val_dataset = AttrPetDataset(image_dir, list(val_rows), encoders)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # === 模型 ===
    try:
        teacher = DINOv3Teacher()
        print("Teacher: DINOv3 ViT-S (384-dim)")
    except Exception as e:
        print(f"DINOv3 加载失败 ({e}), 回退到 DINOv2")
        teacher = DINOv2Teacher()
        print("Teacher: DINOv2 ViT-S (384-dim)")
    teacher = teacher.to(device)
    teacher.eval()

    student = MobileNetV2StudentWithAttr(
        proj_dim=args.proj_dim, num_colors=num_colors, num_patterns=num_patterns
    ).to(device)
    adapter = TeacherAdapter(teacher_dim=384, student_dim=args.proj_dim).to(device)

    print(f"Student: MobileNetV2 ({sum(p.numel() for p in student.parameters())/1e6:.1f}M params)")
    print(f"Feature dim: {args.proj_dim}")

    # === 损失函数 ===
    criterion = AttributeDistillationLoss(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        lambda_color_pri=args.lambda_color_pri,
        lambda_color_sec=args.lambda_color_sec,
        lambda_pattern=args.lambda_pattern,
    )

    # === 优化器（双学习率） ===
    optimizer = AdamW([
        {'params': student.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': student.projector.parameters(), 'lr': args.lr_head},
        {'params': student.color_primary_head.parameters(), 'lr': args.lr_head},
        {'params': student.color_secondary_head.parameters(), 'lr': args.lr_head},
        {'params': student.pattern_head.parameters(), 'lr': args.lr_head},
        {'params': adapter.parameters(), 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    # === LR Schedule ===
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    # === Checkpoint 目录 ===
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float('inf')

    # === 训练循环 ===
    for epoch in range(args.epochs):
        student.train()
        adapter.train()
        loss_keys = ['total', 'align', 'sim', 'uniform', 'color_pri', 'color_sec', 'pattern']
        epoch_losses = {k: 0.0 for k in loss_keys}
        t0 = time.time()

        for batch_idx, (view1, view2, color_pri, color_sec, pattern) in enumerate(train_loader):
            view1 = view1.to(device)
            view2 = view2.to(device)
            color_pri = color_pri.to(device).long()
            color_sec = color_sec.to(device)
            pattern = pattern.to(device)

            # Teacher features
            with torch.no_grad():
                t1 = teacher(view1)
                t2 = teacher(view2)

            # Student features + attribute predictions
            emb1, cp1, cs1, pa1 = student(view1)
            emb2, cp2, cs2, pa2 = student(view2)

            # Loss (双向平均)
            loss1, d1 = criterion(t1, emb1, adapter, cp1, cs1, pa1, color_pri, color_sec, pattern)
            loss2, d2 = criterion(t2, emb2, adapter, cp2, cs2, pa2, color_pri, color_sec, pattern)
            loss = (loss1 + loss2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += (d1[k] + d2[k]) / 2

            if (batch_idx + 1) % args.log_interval == 0:
                avg = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg['total']:.4f} align={avg['align']:.4f} "
                      f"sim={avg['sim']:.4f} uniform={avg['uniform']:.4f} "
                      f"color_pri={avg['color_pri']:.4f} color_sec={avg['color_sec']:.4f} "
                      f"pattern={avg['pattern']:.4f}")

        scheduler.step()
        avg_loss = epoch_losses['total'] / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f}")

        # === 验证 ===
        val_loss = validate(student, adapter, teacher, val_loader, criterion, device)
        print(f"  Val loss: {val_loss:.4f}")

        # === 保存 checkpoint ===
        if (epoch + 1) % args.save_interval == 0 or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                name = 'best_student_attr.pth'
            else:
                name = f'student_attr_epoch{epoch+1}.pth'
            path = os.path.join(ckpt_dir, name)
            torch.save({
                'epoch': epoch + 1,
                'student': student.state_dict(),
                'adapter': adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
                'encoders': {
                    'color_classes': encoders['color_primary'].classes,
                    'pattern_classes': encoders['pattern'].classes,
                },
            }, path)
            print(f"  Saved: {path}")

    # 保存最终模型（只保留推理需要的权重）
    final_path = os.path.join(ckpt_dir, 'final_student_attr.pth')
    torch.save(student.state_dict(), final_path)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Final student: {final_path}")
    return student


@torch.no_grad()
def validate(student, adapter, teacher, val_loader, criterion, device):
    """验证"""
    student.eval()
    adapter.eval()
    total_loss = 0.0
    count = 0

    for view1, view2, color_pri, color_sec, pattern in val_loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        color_pri = color_pri.to(device).long()
        color_sec = color_sec.to(device)
        pattern = pattern.to(device)

        t1 = teacher(view1)
        emb1, cp1, cs1, pa1 = student(view1)
        loss, _ = criterion(t1, emb1, adapter, cp1, cs1, pa1, color_pri, color_sec, pattern)
        total_loss += loss.item() * view1.size(0)
        count += view1.size(0)

    student.train()
    adapter.train()
    return total_loss / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description='多属性感知宠物模型训练')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr_backbone', type=float, default=5e-4)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.04)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--proj_dim', type=int, default=512)
    p.add_argument('--alpha', type=float, default=1.0, help='alignment loss weight')
    p.add_argument('--beta', type=float, default=0.5, help='self-similarity loss weight')
    p.add_argument('--gamma', type=float, default=0.1, help='uniformity loss weight')
    p.add_argument('--lambda_color_pri', type=float, default=0.2, help='主色分类损失权重')
    p.add_argument('--lambda_color_sec', type=float, default=0.15, help='副色分类损失权重（多标签）')
    p.add_argument('--lambda_pattern', type=float, default=0.15, help='花纹分类损失权重（多标签）')
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
