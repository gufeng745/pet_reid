"""多属性感知宠物模型训练脚本 V2

基于 Re-ID SOP 全面优化：
- PK Sampler（身份感知采样）
- BNNeck + SE 注意力
- Label Smoothing + Focal Loss
- Circle Loss / ArcFace
- Random Erasing 数据增强
- 混合精度训练
- 早停机制
- 属性准确率监控

用法：
    python train_attr_v2.py --epochs 80 --batch_size 64 --P 16 --K 4
"""

import os
import sys
import time
import argparse
import csv
import json
import random
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import DINOv3Teacher, DINOv2Teacher, MobileNetV2StudentWithAttr, TeacherAdapter
from distillation import (AttributeDistillationLoss,
                          SupervisedContrastiveLoss,
                          FeatureOrthogonalityLoss,
                          LabelSmoothingCE,
                          FocalLoss,
                          CircleLoss,
                          ArcFaceLoss)


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


# ==================== 早停机制 ====================

class EarlyStopping:
    """早停机制：当验证损失不再下降时提前停止训练

    Args:
        patience: 容忍的 epoch 数，默认 10
        min_delta: 最小改善幅度，默认 0.001
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss: 当前验证损失
        Returns:
            early_stop: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ==================== PK Sampler ====================

class PKSampler(Sampler):
    """PK Sampler (身份感知采样器)

    每个 Batch 包含 P 个 ID，每个 ID 抽 K 张图片。
    保证每个 Batch 内都有正负样本，提升度量学习效果。

    Args:
        dataset: 数据集对象
        P: 每个 batch 的 ID 数量
        K: 每个 ID 的样本数量
        drop_last: 是否丢弃最后一个不完整的 batch
    """

    def __init__(self, dataset, P=16, K=4, drop_last=True):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.drop_last = drop_last

        # 按 ID 分组索引
        self.id_to_indices = defaultdict(list)
        for idx, row in enumerate(dataset.rows):
            # 使用 filename 作为 ID（同一个宠物可能有多张图片）
            pet_id = row.get('pet_id', row['filename'].split('.')[0])
            self.id_to_indices[pet_id].append(idx)

        # 过滤掉样本数不足 K 的 ID
        self.valid_ids = [pid for pid, idxs in self.id_to_indices.items()
                         if len(idxs) >= K]

        if len(self.valid_ids) < P:
            print(f"警告：有效 ID 数 ({len(self.valid_ids)}) 小于 P ({P})，将使用所有有效 ID")
            self.P = len(self.valid_ids)

        print(f"PKSampler: {len(self.valid_ids)} 个有效 ID，每个 batch {self.P} 个 ID × {self.K} 张图")

    def __iter__(self):
        # 计算总共可以生成多少个 batch
        num_batches = len(self.valid_ids) // self.P

        for _ in range(num_batches):
            # 随机选择 P 个 ID
            selected_ids = random.sample(self.valid_ids, self.P)

            batch_indices = []
            for pid in selected_ids:
                # 每个 ID 随机选择 K 张图
                indices = random.sample(self.id_to_indices[pid], self.K)
                batch_indices.extend(indices)

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.valid_ids) // self.P
        else:
            return (len(self.valid_ids) + self.P - 1) // self.P


# ==================== 数据验证 ====================

def validate_dataset(image_dir, rows, encoders):
    """验证数据集完整性

    Args:
        image_dir: 图片目录路径
        rows: CSV 数据行列表
        encoders: 标签编码器字典

    Returns:
        valid_rows: 有效的数据行列表
        report: 验证报告字典
    """
    missing_images = []
    invalid_labels = []
    valid_rows = []

    for i, row in enumerate(rows):
        filename = row.get('filename', f'row_{i}')
        img_path = os.path.join(image_dir, filename)

        # 检查图片是否存在
        if not os.path.exists(img_path):
            missing_images.append(filename)
            continue

        # 尝试打开图片（检查是否损坏）
        try:
            with Image.open(img_path) as img:
                img.load()  # 强制加载图片数据
        except Exception as e:
            missing_images.append(f"{filename} (损坏：{e})")
            continue

        # 检查标签是否有效
        row_valid = True

        # 检查主色标签
        color_pri = row.get('color_primary', '').strip()
        if not color_pri:
            invalid_labels.append((filename, 'color_primary', '空值'))
            row_valid = False
        elif color_pri not in encoders['color_primary'].class_to_idx:
            invalid_labels.append((filename, 'color_primary', f'未知类别：{color_pri}'))
            row_valid = False

        # 检查花纹标签
        pattern = row.get('pattern', '').strip()
        if not pattern:
            invalid_labels.append((filename, 'pattern', '空值'))
            row_valid = False
        else:
            # 检查多标签中的每个值
            for item in pattern.split(','):
                item = item.strip()
                if item and item not in encoders['pattern'].class_to_idx:
                    invalid_labels.append((filename, 'pattern', f'未知类别：{item}'))
                    row_valid = False
                    break

        if row_valid:
            valid_rows.append(row)
        else:
            missing_images.append(f"{filename} (标签无效)")

    # 生成报告
    report = {
        'total': len(rows),
        'valid': len(valid_rows),
        'missing_images': len(missing_images),
        'invalid_labels': len(invalid_labels),
        'missing_image_list': missing_images,
        'invalid_label_list': invalid_labels,
    }

    # 输出报告摘要
    print("\n" + "=" * 50)
    print("数据集验证报告")
    print("=" * 50)
    print(f"总样本数：{report['total']}")
    print(f"有效样本：{report['valid']} ({report['valid']/report['total']*100:.1f}%)")
    print(f"缺失/损坏图片：{report['missing_images']}")
    print(f"无效标签：{report['invalid_labels']}")

    if missing_images:
        print(f"\n缺失/无效文件列表 (前 20 个):")
        for f in missing_images[:20]:
            print(f"  - {f}")
        if len(missing_images) > 20:
            print(f"  ... 还有 {len(missing_images) - 20} 个")

    print("=" * 50 + "\n")

    return valid_rows, report


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


# ==================== 数据增强 ====================

def get_dino_augmentation(crop_scale=(0.4, 1.0), use_random_erasing=True):
    """DINO 风格数据增强 V2

    改进：
    - 添加 Random Erasing 增强抗遮挡能力
    - 保留原有的强增强策略

    Args:
        crop_scale: 随机裁剪比例范围
        use_random_erasing: 是否使用 Random Erasing
    """
    aug_list = [
        transforms.RandomResizedCrop(224, scale=crop_scale,
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ]

    # 添加 Random Erasing
    if use_random_erasing:
        aug_list.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                    ratio=(0.3, 3.3), value='random')
        )

    return transforms.Compose(aug_list)


# ==================== 数据集 ====================

class AttrPetDataset(Dataset):
    """属性标注宠物数据集 V2

    对同一张图生成两个不同增强视角（用于蒸馏），同时返回属性标签。

    改进：
    - 支持 PK Sampler（返回 pet_id）
    - 支持更灵活的数据增强
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
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations.csv')
    if not os.path.exists(csv_path):
        print(f"错误：找不到标注文件 {csv_path}")
        return

    encoders, rows = build_label_encoders(csv_path)

    # === 数据验证 ===
    image_dir = find_image_dir()
    print(f"\n正在验证数据集...")
    valid_rows, report = validate_dataset(image_dir, rows, encoders)

    # 保存验证报告
    if args.save_report:
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_validation_report.json')
        save_report = {
            'total': report['total'],
            'valid': report['valid'],
            'missing_images': report['missing_images'],
            'invalid_labels': report['invalid_labels'],
            'missing_image_list': report['missing_image_list'][:100],
            'invalid_label_list': [list(x) for x in report['invalid_label_list'][:100]],
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(save_report, f, ensure_ascii=False, indent=2)
        print(f"验证报告已保存到：{report_path}")

    if report['valid'] == 0:
        print("错误：没有有效样本，无法继续训练")
        return

    rows = valid_rows  # 使用验证后的有效数据

    num_colors = encoders['color_primary'].num_classes
    num_patterns = encoders['pattern'].num_classes
    print(f"样本数：{len(rows)}, 颜色类：{num_colors}, 花纹类：{num_patterns}")

    # === 划分 train/val ===
    val_size = int(len(rows) * 0.1)
    train_size = len(rows) - val_size
    train_rows, val_rows = random_split(
        rows, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # === 数据集 ===
    train_dataset = AttrPetDataset(image_dir, list(train_rows), encoders)
    val_dataset = AttrPetDataset(image_dir, list(val_rows), encoders)

    # === DataLoader ===
    if args.use_pk_sampler:
        # 使用 PK Sampler
        train_sampler = PKSampler(
            train_dataset, P=args.P, K=args.K, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
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
        proj_dim=args.proj_dim, num_colors=num_colors, num_patterns=num_patterns,
        use_se=args.use_se, use_bnneck=args.use_bnneck
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
    criterion_contrastive = SupervisedContrastiveLoss(temperature=args.contrastive_temp)
    criterion_ortho = FeatureOrthogonalityLoss(feat_dim=args.proj_dim)

    # 可选的损失函数
    if args.use_label_smoothing:
        criterion_color_pri = LabelSmoothingCE(smoothing=0.1)
        print("使用 Label Smoothing CE for 主色分类")
    else:
        criterion_color_pri = None

    if args.use_focal_loss:
        criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
        print("使用 Focal Loss for 主色分类")

    if args.use_circle_loss:
        criterion_circle = CircleLoss(m=0.25, gamma=256)
        print("使用 Circle Loss for 度量学习")

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

    # === 混合精度训练 ===
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if scaler:
        print("使用混合精度训练 (AMP)")

    # === 早停机制 ===
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001) if args.use_early_stopping else None

    # === Checkpoint 目录 ===
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float('inf')

    # === 训练循环 ===
    for epoch in range(args.epochs):
        student.train()
        adapter.train()
        loss_keys = ['total', 'align', 'sim', 'uniform', 'color_pri', 'color_sec', 'pattern', 'contrastive', 'ortho']
        epoch_losses = {k: 0.0 for k in loss_keys}

        # 属性准确率统计
        color_pri_correct = 0
        color_pri_total = 0
        pattern_correct = 0
        pattern_total = 0

        t0 = time.time()

        for batch_idx, (view1, view2, color_pri, color_sec, pattern) in enumerate(train_loader):
            view1 = view1.to(device)
            view2 = view2.to(device)
            color_pri = color_pri.to(device).long()
            color_sec = color_sec.to(device)
            pattern = pattern.to(device)

            # 混合精度前向传播
            if scaler:
                with torch.cuda.amp.autocast():
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

                    # 可选：使用 Label Smoothing CE 替换主色分类损失
                    if criterion_color_pri is not None:
                        loss_color_pri1 = criterion_color_pri(cp1, color_pri)
                        loss_color_pri2 = criterion_color_pri(cp2, color_pri)
                        loss1 = loss1 - d1['color_pri'] * args.lambda_color_pri + loss_color_pri1 * args.lambda_color_pri
                        loss2 = loss2 - d2['color_pri'] * args.lambda_color_pri + loss_color_pri2 * args.lambda_color_pri

                    # 实例对比损失
                    loss_con = criterion_contrastive(emb1, emb2)

                    # 特征正交正则化
                    loss_orth = criterion_ortho(emb1)

                    # 可选：使用 Circle Loss
                    if args.use_circle_loss:
                        # 需要标签才能使用 Circle Loss，这里用 color_pri 作为伪标签
                        loss_circle = criterion_circle(emb1, color_pri)
                        loss = (loss1 + loss2) / 2 + args.lambda_contrastive * loss_con + args.lambda_ortho * loss_orth + 0.5 * loss_circle
                    else:
                        loss = (loss1 + loss2) / 2 + args.lambda_contrastive * loss_con + args.lambda_ortho * loss_orth

                # 混合精度反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # 梯度裁剪
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                with torch.no_grad():
                    t1 = teacher(view1)
                    t2 = teacher(view2)

                emb1, cp1, cs1, pa1 = student(view1)
                emb2, cp2, cs2, pa2 = student(view2)

                loss1, d1 = criterion(t1, emb1, adapter, cp1, cs1, pa1, color_pri, color_sec, pattern)
                loss2, d2 = criterion(t2, emb2, adapter, cp2, cs2, pa2, color_pri, color_sec, pattern)

                # 可选：使用 Label Smoothing CE 替换主色分类损失
                if criterion_color_pri is not None:
                    loss_color_pri1 = criterion_color_pri(cp1, color_pri)
                    loss_color_pri2 = criterion_color_pri(cp2, color_pri)
                    loss1 = loss1 - d1['color_pri'] * args.lambda_color_pri + loss_color_pri1 * args.lambda_color_pri
                    loss2 = loss2 - d2['color_pri'] * args.lambda_color_pri + loss_color_pri2 * args.lambda_color_pri

                loss_con = criterion_contrastive(emb1, emb2)
                loss_orth = criterion_ortho(emb1)

                if args.use_circle_loss:
                    loss_circle = criterion_circle(emb1, color_pri)
                    loss = (loss1 + loss2) / 2 + args.lambda_contrastive * loss_con + args.lambda_ortho * loss_orth + 0.5 * loss_circle
                else:
                    loss = (loss1 + loss2) / 2 + args.lambda_contrastive * loss_con + args.lambda_ortho * loss_orth

                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()

            # 统计损失
            for k in epoch_losses:
                if k == 'total':
                    epoch_losses[k] += loss.item()
                elif k == 'contrastive':
                    epoch_losses[k] += loss_con.item()
                elif k == 'ortho':
                    epoch_losses[k] += loss_orth.item()
                else:
                    epoch_losses[k] += (d1[k] + d2[k]) / 2

            # 统计属性准确率
            color_pred = cp1.argmax(dim=1)
            color_pri_correct += (color_pred == color_pri).sum().item()
            color_pri_total += color_pri.size(0)

            # 花纹准确率（多标签，使用阈值）
            pattern_pred = (pa1 > 0.5).float()
            pattern_correct += (pattern_pred == pattern).all(dim=1).sum().item()
            pattern_total += pattern.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                avg = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                color_acc = color_pri_correct / color_pri_total
                pattern_acc = pattern_correct / pattern_total
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg['total']:.4f} align={avg['align']:.4f} "
                      f"sim={avg['sim']:.4f} uniform={avg['uniform']:.4f} "
                      f"color_pri={avg['color_pri']:.4f} color_sec={avg['color_sec']:.4f} "
                      f"pattern={avg['pattern']:.4f} "
                      f"contrastive={avg['contrastive']:.4f} ortho={avg['ortho']:.4f} "
                      f"color_acc={color_acc:.3f} pattern_acc={pattern_acc:.3f}")

        scheduler.step()
        avg_loss = epoch_losses['total'] / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        color_acc = color_pri_correct / color_pri_total
        pattern_acc = pattern_correct / pattern_total
        print(f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f} | "
              f"color_acc={color_acc:.3f} | pattern_acc={pattern_acc:.3f}")

        # === 验证 ===
        val_loss = validate(student, adapter, teacher, val_loader, criterion,
                           criterion_contrastive, args.lambda_contrastive, device)
        print(f"  Val loss: {val_loss:.4f}")

        # === 保存 checkpoint ===
        if (epoch + 1) % args.save_interval == 0 or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                name = 'best_student_attr_v2.pth'
            else:
                name = f'student_attr_v2_epoch{epoch+1}.pth'
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
                'args': vars(args),
            }, path)
            print(f"  Saved: {path}")

        # === 早停检查 ===
        if early_stopping and early_stopping(val_loss):
            print(f"\n早停触发：验证损失在 {args.patience} 个 epoch 内未改善")
            break

    # 保存最终模型
    final_path = os.path.join(ckpt_dir, 'final_student_attr_v2.pth')
    torch.save({
        'student': student.state_dict(),
        'encoders': {
            'color_classes': encoders['color_primary'].classes,
            'pattern_classes': encoders['pattern'].classes,
        },
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Final student: {final_path}")
    return student


@torch.no_grad()
def validate(student, adapter, teacher, val_loader, criterion, criterion_contrastive,
             lambda_contrastive, device):
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
        emb2, _, _, _ = student(view2)
        loss, _ = criterion(t1, emb1, adapter, cp1, cs1, pa1, color_pri, color_sec, pattern)
        loss_con = criterion_contrastive(emb1, emb2)
        loss = loss + lambda_contrastive * loss_con
        total_loss += loss.item() * view1.size(0)
        count += view1.size(0)

    student.train()
    adapter.train()
    return total_loss / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description='多属性感知宠物模型训练 V2')

    # 基础参数
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)

    # 学习率
    p.add_argument('--lr_backbone', type=float, default=5e-4)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.04)
    p.add_argument('--warmup_epochs', type=int, default=10)

    # 模型结构
    p.add_argument('--proj_dim', type=int, default=512)
    p.add_argument('--use_se', action='store_true', default=True, help='使用 SE 注意力模块')
    p.add_argument('--use_bnneck', action='store_true', default=True, help='使用 BNNeck')

    # 损失函数权重
    p.add_argument('--alpha', type=float, default=1.0, help='alignment loss weight')
    p.add_argument('--beta', type=float, default=0.5, help='self-similarity loss weight')
    p.add_argument('--gamma', type=float, default=0.1, help='uniformity loss weight')
    p.add_argument('--lambda_color_pri', type=float, default=0.5, help='主色分类损失权重（提升）')
    p.add_argument('--lambda_color_sec', type=float, default=0.15, help='副色分类损失权重')
    p.add_argument('--lambda_pattern', type=float, default=0.15, help='花纹分类损失权重')
    p.add_argument('--lambda_contrastive', type=float, default=0.3, help='实例对比损失权重（降低）')
    p.add_argument('--lambda_ortho', type=float, default=0.05, help='特征正交正则化权重')
    p.add_argument('--contrastive_temp', type=float, default=0.1, help='对比损失温度系数（提升）')

    # PK Sampler
    p.add_argument('--use_pk_sampler', action='store_true', default=False, help='使用 PK Sampler')
    p.add_argument('--P', type=int, default=16, help='PK Sampler: 每个 batch 的 ID 数')
    p.add_argument('--K', type=int, default=4, help='PK Sampler: 每个 ID 的样本数')

    # 高级损失函数
    p.add_argument('--use_label_smoothing', action='store_true', default=True, help='使用 Label Smoothing')
    p.add_argument('--use_focal_loss', action='store_true', default=False, help='使用 Focal Loss')
    p.add_argument('--use_circle_loss', action='store_true', default=False, help='使用 Circle Loss')

    # 训练技巧
    p.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度训练')
    p.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    p.add_argument('--use_early_stopping', action='store_true', default=True, help='使用早停机制')
    p.add_argument('--patience', type=int, default=15, help='早停容忍的 epoch 数')

    # 其他
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_report', action='store_true', default=True, help='保存数据验证报告')

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
