"""宠物 Re-ID 身份识别训练脚本 V2

修复了以下致命问题：
1. 数据划分：按 ID 划分 train/val，防止数据泄露
2. BNNeck：正确使用 BNNeck（训练时 BN 后用于 ID Loss，BN 前用于 Metric Loss）
3. PK Sampler：默认启用
4. Metric Loss：添加 Triplet Loss
5. Contrastive Loss：使用监督对比（需要 labels）
6. GeM Pooling：使用 GeM 替代 GAP
7. id_head：简化为单层 Linear
8. 蒸馏对齐：使用 Cosine Similarity Loss

用法：
    python train_reid.py --epochs 80 --P 16 --K 4
"""

import os
import sys
import time
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import (DINOv3Teacher, DINOv2Teacher, TeacherAdapter,
                    SEBlock, BNNeck, GeMPooling, get_local_weight_path, load_safetensors_weight)
from distillation import (SupervisedContrastiveLoss,
                          FeatureOrthogonalityLoss,
                          CircleLoss)


# ==================== 早停机制 ====================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
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


# ==================== Triplet Loss ====================

class TripletMarginLoss(nn.Module):
    """Triplet Loss with Hard Mining

    对于每个 anchor，选择最难的正样本和负样本构成三元组。
    这是 Re-ID 任务的核心 Metric Loss。

    Args:
        margin: 边界参数，默认 0.3
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2 归一化的特征
            labels: (B,) 标签 (long)
        Returns:
            loss: 标量
        """
        # 计算余弦相似度矩阵
        sim_matrix = features @ features.T  # (B, B)

        # 构建正负样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1.0 - pos_mask

        # 排除对角线
        eye_mask = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - eye_mask
        neg_mask = neg_mask * (1.0 - eye_mask)

        # 对于每个 anchor，选择最难的正样本（相似度最低的正样本）
        # 和最难的负样本（相似度最高的负样本）
        sim_matrix = sim_matrix - eye_mask * 1e9  # 排除自身

        # 最难正样本：相似度最低的正样本
        pos_sim = sim_matrix * pos_mask + (1.0 - pos_mask) * (-1e9)
        hardest_pos = pos_sim.max(dim=1)[0]  # (B,)

        # 最难负样本：相似度最高的负样本
        neg_sim = sim_matrix * neg_mask + (1.0 - neg_mask) * (-1e9)
        hardest_neg = neg_sim.max(dim=1)[0]  # (B,)

        # Triplet Loss: max(0, margin - (pos_sim - neg_sim))
        loss = F.relu(self.margin - (hardest_pos - hardest_neg))

        # 只计算有效样本的损失
        valid = (hardest_pos > -1e8) & (hardest_neg > -1e8)
        if valid.sum() > 0:
            loss = loss[valid].mean()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss


# ==================== PK Sampler ====================

class PKSampler(Sampler):
    """PK Sampler (身份感知采样器)

    每个 Batch 包含 P 个 ID，每个 ID 抽 K 张图片。
    保证每个 Batch 内都有正负样本，提升度量学习效果。
    """

    def __init__(self, dataset, P=16, K=4, drop_last=True):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.drop_last = drop_last

        # 按 ID 分组索引
        self.id_to_indices = defaultdict(list)
        for idx, (pet_id, _) in enumerate(dataset.samples):
            self.id_to_indices[pet_id].append(idx)

        # 过滤掉样本数不足 K 的 ID
        self.valid_ids = [pid for pid, idxs in self.id_to_indices.items()
                         if len(idxs) >= K]

        if len(self.valid_ids) < P:
            print(f"警告：有效 ID 数 ({len(self.valid_ids)}) 小于 P ({P})，将使用所有有效 ID")
            self.P = len(self.valid_ids)

        print(f"PKSampler: {len(self.valid_ids)} 个有效 ID，每个 batch {self.P} 个 ID × {self.K} 张图")

    def __iter__(self):
        num_batches = len(self.valid_ids) // self.P

        for _ in range(num_batches):
            selected_ids = random.sample(self.valid_ids, self.P)
            batch_indices = []
            for pid in selected_ids:
                indices = random.sample(self.id_to_indices[pid], self.K)
                batch_indices.extend(indices)
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.valid_ids) // self.P
        else:
            return (len(self.valid_ids) + self.P - 1) // self.P


# ==================== 数据集 ====================

class ReIDDataset(Dataset):
    """Re-ID 身份识别数据集

    目录结构：
        root/cat/{id}/image1.jpg
        root/dog/{id}/image1.jpg

    每个子文件夹是一个身份 ID。
    """

    def __init__(self, root, transform=None, species=None, id_list=None):
        """
        Args:
            root: 数据集根目录 (reid_dataset/)
            transform: 数据增强
            species: 'cat', 'dog', 或 None (两者都包含)
            id_list: 指定使用的 ID 列表（用于 train/val 划分）
        """
        self.root = root
        self.transform = transform or self._default_transform()
        self.samples = []  # [(pet_id, image_path), ...]
        self.id_to_label = {}  # pet_id -> label (连续整数)
        self.label_to_id = {}

        # 收集所有样本
        pet_id_counter = 0

        if species is None or species == 'cat':
            cat_dir = os.path.join(root, 'cat')
            if os.path.isdir(cat_dir):
                for pet_id in sorted(os.listdir(cat_dir), key=lambda x: int(x) if x.isdigit() else x):
                    if id_list is not None and f"cat_{pet_id}" not in id_list:
                        continue
                    pet_dir = os.path.join(cat_dir, pet_id)
                    if not os.path.isdir(pet_dir):
                        continue
                    full_id = f"cat_{pet_id}"
                    self.id_to_label[full_id] = pet_id_counter
                    self.label_to_id[pet_id_counter] = full_id
                    for img_name in os.listdir(pet_dir):
                        img_path = os.path.join(pet_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((full_id, img_path))
                    pet_id_counter += 1

        if species is None or species == 'dog':
            dog_dir = os.path.join(root, 'dog')
            if os.path.isdir(dog_dir):
                for pet_id in sorted(os.listdir(dog_dir), key=lambda x: int(x) if x.isdigit() else x):
                    if id_list is not None and f"dog_{pet_id}" not in id_list:
                        continue
                    pet_dir = os.path.join(dog_dir, pet_id)
                    if not os.path.isdir(pet_dir):
                        continue
                    full_id = f"dog_{pet_id}"
                    self.id_to_label[full_id] = pet_id_counter
                    self.label_to_id[pet_id_counter] = full_id
                    for img_name in os.listdir(pet_dir):
                        img_path = os.path.join(pet_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((full_id, img_path))
                    pet_id_counter += 1

        self.num_classes = pet_id_counter
        print(f"ReIDDataset: {len(self.samples)} 张图片, {self.num_classes} 个身份")

    def _default_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pet_id, img_path = self.samples[idx]
        label = self.id_to_label[pet_id]

        img = Image.open(img_path).convert('RGB')
        view1 = self.transform(img)
        view2 = self.transform(img)  # 第二个增强视角

        return view1, view2, label


# ==================== 数据增强 ====================

def get_augmentation(crop_scale=(0.4, 1.0)):
    """训练数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=crop_scale,
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    """验证数据增强"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


# ==================== 模型 ====================

class MobileNetV2StudentForReID(nn.Module):
    """Re-ID 用 MobileNetV2 学生模型 (V2)

    修复：
    1. 使用 GeM Pooling 替代 GAP
    2. BNNeck 放在 proj_dim 维度（512），不是 feat_dim（1280）
    3. 训练时：emb 使用 BN 前的特征（用于 Metric Loss），id_logits 使用 BN 后的特征
    4. 推理时：只返回 BN 前的特征
    5. id_head 简化为单层 Linear（无偏置）
    """

    def __init__(self, proj_dim=512, num_classes=100, use_se=True, use_bnneck=True):
        super().__init__()
        import timm

        self.use_se = use_se
        self.use_bnneck = use_bnneck

        # 加载 MobileNetV2 backbone（不使用默认的分类头和池化）
        local_weight_path = get_local_weight_path('mobilenetv2_100')
        if local_weight_path:
            print(f"[MobileNetV2ReID] 从本地加载预训练权重：{local_weight_path}")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)
            state_dict = load_safetensors_weight(local_weight_path)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print("[MobileNetV2ReID] 本地权重不存在，使用随机初始化")
            self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)

        feat_dim = 1280  # MobileNetV2 特征维度

        # GeM Pooling（替代 GAP）
        self.gem_pool = GeMPooling(p=3.0)
        print("[MobileNetV2ReID] 使用 GeM Pooling")

        # SE 注意力模块
        if use_se:
            self.se_block = SEBlock(feat_dim, reduction=16)
            print("[MobileNetV2ReID] 使用 SE 注意力模块")

        # 投影头（1280 -> 512）
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # BNNeck（放在 proj_dim 维度，用于 ID Loss）
        if use_bnneck:
            self.bnneck = BNNeck(proj_dim)
            print("[MobileNetV2ReID] 使用 BNNeck (proj_dim)")

        # 身份分类头（简化为单层 Linear，无偏置）
        self.id_head = nn.Linear(proj_dim, num_classes, bias=False)
        print(f"[MobileNetV2ReID] id_head: Linear({proj_dim}, {num_classes}, bias=False)")

        self.feature_dim = proj_dim

    def forward(self, x):
        """训练用：返回特征 + 身份预测

        Returns:
            emb: (B, proj_dim) L2 归一化的特征（BN 前，用于 Metric Loss）
            id_logits: (B, num_classes) 身份预测（BN 后，用于 ID Loss）
        """
        # 提取特征图
        feat = self.backbone.forward_features(x)  # (B, 1280, H, W)

        # GeM Pooling
        feat = self.gem_pool(feat).flatten(1)  # (B, 1280)

        # SE 注意力
        if self.use_se:
            feat = self.se_block(feat)

        # 投影到 512 维
        emb = self.projector(feat)  # (B, 512)

        # BN 前的特征用于 Metric Loss 和推理
        emb_norm = F.normalize(emb, dim=-1)

        # BN 后的特征用于 ID Loss
        if self.use_bnneck:
            feat_bn = self.bnneck(emb)  # (B, 512)
        else:
            feat_bn = emb

        id_logits = self.id_head(feat_bn)

        return emb_norm, id_logits

    def forward_emb(self, x):
        """推理用：只返回 BN 前的特征向量"""
        feat = self.backbone.forward_features(x)
        feat = self.gem_pool(feat).flatten(1)

        if self.use_se:
            feat = self.se_block(feat)

        emb = self.projector(feat)
        return F.normalize(emb, dim=-1)


# ==================== 数据集划分 ====================

def split_dataset_by_id(data_root, val_ratio=0.1, seed=42):
    """按 ID 划分训练集和验证集

    Args:
        data_root: 数据集根目录
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        train_ids: 训练集 ID 集合
        val_ids: 验证集 ID 集合
    """
    # 收集所有 ID
    all_ids = []

    cat_dir = os.path.join(data_root, 'cat')
    if os.path.isdir(cat_dir):
        for pet_id in os.listdir(cat_dir):
            if os.path.isdir(os.path.join(cat_dir, pet_id)):
                all_ids.append(f"cat_{pet_id}")

    dog_dir = os.path.join(data_root, 'dog')
    if os.path.isdir(dog_dir):
        for pet_id in os.listdir(dog_dir):
            if os.path.isdir(os.path.join(dog_dir, pet_id)):
                all_ids.append(f"dog_{pet_id}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(all_ids)

    # 划分
    val_size = max(1, int(len(all_ids) * val_ratio))
    val_ids = set(all_ids[:val_size])
    train_ids = set(all_ids[val_size:])

    print(f"按 ID 划分数据集：")
    print(f"  总身份数: {len(all_ids)}")
    print(f"  训练集: {len(train_ids)} 个身份")
    print(f"  验证集: {len(val_ids)} 个身份")

    return train_ids, val_ids


# ==================== 训练 ====================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === 数据集路径 ===
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reid_dataset')
    if not os.path.isdir(data_root):
        print(f"错误：找不到数据集目录 {data_root}")
        return

    # === 按 ID 划分数据集 ===
    train_ids, val_ids = split_dataset_by_id(
        data_root, val_ratio=args.val_ratio, seed=args.seed
    )

    # === 创建数据集 ===
    train_transform = get_augmentation()
    val_transform = get_val_transform()

    train_dataset = ReIDDataset(data_root, transform=train_transform, id_list=train_ids)
    val_dataset = ReIDDataset(data_root, transform=val_transform, id_list=val_ids)

    num_classes = train_dataset.num_classes
    print(f"Train: {len(train_dataset)} 张图片, {train_dataset.num_classes} 个身份")
    print(f"Val: {len(val_dataset)} 张图片, {val_dataset.num_classes} 个身份")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("错误：数据集为空")
        return

    # === DataLoader ===
    # PK Sampler 用于训练集
    train_sampler = PKSampler(
        train_dataset, P=args.P, K=args.K, drop_last=True
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
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

    student = MobileNetV2StudentForReID(
        proj_dim=args.proj_dim, num_classes=num_classes,
        use_se=args.use_se, use_bnneck=args.use_bnneck
    ).to(device)
    adapter = TeacherAdapter(teacher_dim=384, student_dim=args.proj_dim).to(device)

    print(f"Student: MobileNetV2 ({sum(p.numel() for p in student.parameters())/1e6:.1f}M params)")
    print(f"Feature dim: {args.proj_dim}")

    # === 损失函数 ===
    # ID Loss（Label Smoothing）
    criterion_id = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Metric Loss: Triplet Loss
    criterion_triplet = TripletMarginLoss(margin=0.3)
    print("使用 Triplet Loss (margin=0.3)")

    # 监督对比损失（使用 labels）
    criterion_contrastive = SupervisedContrastiveLoss(temperature=args.contrastive_temp)

    # 特征正交正则化
    criterion_ortho = FeatureOrthogonalityLoss(feat_dim=args.proj_dim)

    # === 优化器（双学习率） ===
    optimizer = AdamW([
        {'params': student.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': student.gem_pool.parameters(), 'lr': args.lr_head},
        {'params': student.projector.parameters(), 'lr': args.lr_head},
        {'params': student.id_head.parameters(), 'lr': args.lr_head},
        {'params': adapter.parameters(), 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    # === LR Schedule ===
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    # === 混合精度训练 ===
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
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

        epoch_loss = 0.0
        epoch_loss_id = 0.0
        epoch_loss_triplet = 0.0
        epoch_loss_contrastive = 0.0
        epoch_loss_ortho = 0.0
        epoch_loss_align = 0.0

        id_correct = 0
        id_total = 0

        t0 = time.time()

        for batch_idx, (view1, view2, labels) in enumerate(train_loader):
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device).long()

            if scaler:
                with torch.amp.autocast('cuda'):
                    # Teacher features
                    with torch.no_grad():
                        t1 = teacher(view1)
                        t2 = teacher(view2)

                    # Student features + identity predictions
                    emb1, id1 = student(view1)
                    emb2, id2 = student(view2)

                    # 蒸馏对齐损失（Cosine Similarity）
                    t1_adapted = adapter(t1)
                    t2_adapted = adapter(t2)
                    loss_align = ((1 - F.cosine_similarity(emb1, t1_adapted, dim=-1)).mean() +
                                  (1 - F.cosine_similarity(emb2, t2_adapted, dim=-1)).mean()) / 2

                    # 身份分类损失（使用 BN 后的特征，通过 id_logits）
                    loss_id = (criterion_id(id1, labels) + criterion_id(id2, labels)) / 2

                    # Triplet Loss（使用 BN 前的特征）
                    all_embs = torch.cat([emb1, emb2], dim=0)
                    all_labels = torch.cat([labels, labels], dim=0)
                    loss_triplet = criterion_triplet(all_embs, all_labels)

                    # 监督对比损失（使用 labels）
                    loss_con = criterion_contrastive(all_embs, all_labels)

                    # 特征正交正则化
                    loss_orth = criterion_ortho(emb1)

                    # 总损失
                    loss = (args.lambda_align * loss_align +
                            args.lambda_id * loss_id +
                            args.lambda_triplet * loss_triplet +
                            args.lambda_contrastive * loss_con +
                            args.lambda_ortho * loss_orth)

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    t1 = teacher(view1)
                    t2 = teacher(view2)

                emb1, id1 = student(view1)
                emb2, id2 = student(view2)

                t1_adapted = adapter(t1)
                t2_adapted = adapter(t2)
                loss_align = ((1 - F.cosine_similarity(emb1, t1_adapted, dim=-1)).mean() +
                              (1 - F.cosine_similarity(emb2, t2_adapted, dim=-1)).mean()) / 2

                loss_id = (criterion_id(id1, labels) + criterion_id(id2, labels)) / 2

                all_embs = torch.cat([emb1, emb2], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                loss_triplet = criterion_triplet(all_embs, all_labels)
                loss_con = criterion_contrastive(all_embs, all_labels)
                loss_orth = criterion_ortho(emb1)

                loss = (args.lambda_align * loss_align +
                        args.lambda_id * loss_id +
                        args.lambda_triplet * loss_triplet +
                        args.lambda_contrastive * loss_con +
                        args.lambda_ortho * loss_orth)

                optimizer.zero_grad()
                loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_loss_id += loss_id.item()
            epoch_loss_triplet += loss_triplet.item()
            epoch_loss_contrastive += loss_con.item()
            epoch_loss_ortho += loss_orth.item()
            epoch_loss_align += loss_align.item()

            # 身份准确率（基于 id_logits）
            id_pred = id1.argmax(dim=1)
            id_correct += (id_pred == labels).sum().item()
            id_total += labels.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                id_acc = id_correct / id_total
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg_loss:.4f} id={epoch_loss_id/(batch_idx+1):.4f} "
                      f"triplet={epoch_loss_triplet/(batch_idx+1):.4f} "
                      f"con={epoch_loss_contrastive/(batch_idx+1):.4f} "
                      f"ortho={epoch_loss_ortho/(batch_idx+1):.4f} "
                      f"align={epoch_loss_align/(batch_idx+1):.4f} "
                      f"id_acc={id_acc:.3f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        id_acc = id_correct / id_total
        print(f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f} | id_acc={id_acc:.3f}")

        # === 验证 ===
        val_loss, val_rank1 = validate(student, adapter, teacher, val_loader, args, device)
        print(f"  Val loss: {val_loss:.4f} | Val Rank-1: {val_rank1:.3f}")

        # === 保存 checkpoint ===
        if (epoch + 1) % args.save_interval == 0 or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                name = 'best_student_reid.pth'
            else:
                name = f'student_reid_epoch{epoch+1}.pth'
            path = os.path.join(ckpt_dir, name)
            torch.save({
                'epoch': epoch + 1,
                'student': student.state_dict(),
                'adapter': adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
                'rank1': val_rank1,
                'num_classes': num_classes,
                'args': vars(args),
            }, path)
            print(f"  Saved: {path}")

        # === 早停检查 ===
        if early_stopping and early_stopping(val_loss):
            print(f"\n早停触发：验证损失在 {args.patience} 个 epoch 内未改善")
            break

    # 保存最终模型
    final_path = os.path.join(ckpt_dir, 'final_student_reid.pth')
    torch.save({
        'student': student.state_dict(),
        'num_classes': num_classes,
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Final student: {final_path}")
    return student


@torch.no_grad()
def validate(student, adapter, teacher, val_loader, args, device):
    """验证（计算 loss 和 Rank-1 准确率）"""
    student.eval()
    adapter.eval()
    total_loss = 0.0
    count = 0

    # 用于计算 Rank-1
    all_features = []
    all_labels = []

    for view1, view2, labels in val_loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device).long()

        t1 = teacher(view1)
        emb1, id1 = student(view1)

        t1_adapted = adapter(t1)
        loss_align = (1 - F.cosine_similarity(emb1, t1_adapted, dim=-1)).mean()
        loss_id = F.cross_entropy(id1, labels, label_smoothing=0.1)

        loss = args.lambda_align * loss_align + args.lambda_id * loss_id

        total_loss += loss.item() * view1.size(0)
        count += view1.size(0)

        # 收集特征用于计算 Rank-1
        all_features.append(emb1.cpu())
        all_labels.append(labels.cpu())

    # 计算 Rank-1 准确率
    all_features = torch.cat(all_features, dim=0)  # (N, D)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    # 计算余弦相似度
    sim_matrix = all_features @ all_features.T  # (N, N)

    # 对每个样本，排除自身后找最相似的
    eye_mask = torch.eye(len(all_labels), dtype=torch.bool)
    sim_matrix = sim_matrix - eye_mask * 1e9

    # 找到每个样本最相似的索引
    top1_indices = sim_matrix.argmax(dim=1)  # (N,)

    # 计算 Rank-1：最相似的样本是否与 query 同一身份
    rank1_correct = (all_labels[top1_indices] == all_labels).sum().item()
    rank1_acc = rank1_correct / len(all_labels)

    student.train()
    adapter.train()
    return total_loss / max(count, 1), rank1_acc


def parse_args():
    p = argparse.ArgumentParser(description='宠物 Re-ID 身份识别训练 V2')

    # 基础参数
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=64)  # 用于验证
    p.add_argument('--num_workers', type=int, default=4)

    # 数据集划分
    p.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例（按 ID 划分）')
    p.add_argument('--seed', type=int, default=42, help='随机种子')

    # 学习率
    p.add_argument('--lr_backbone', type=float, default=5e-4)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.04)
    p.add_argument('--warmup_epochs', type=int, default=10)

    # 模型结构
    p.add_argument('--proj_dim', type=int, default=512)
    p.add_argument('--use_se', action='store_true', default=True)
    p.add_argument('--use_bnneck', action='store_true', default=True)

    # 损失函数权重
    p.add_argument('--lambda_align', type=float, default=1.0, help='蒸馏对齐损失权重')
    p.add_argument('--lambda_id', type=float, default=0.5, help='身份分类损失权重')
    p.add_argument('--lambda_triplet', type=float, default=0.3, help='Triplet Loss 权重')
    p.add_argument('--lambda_contrastive', type=float, default=0.2, help='对比学习损失权重')
    p.add_argument('--lambda_ortho', type=float, default=0.05, help='特征正交正则化权重')
    p.add_argument('--contrastive_temp', type=float, default=0.1, help='对比损失温度系数')

    # PK Sampler（默认启用）
    p.add_argument('--P', type=int, default=16, help='PK Sampler: 每个 batch 的 ID 数')
    p.add_argument('--K', type=int, default=4, help='PK Sampler: 每个 ID 的样本数')

    # 训练技巧
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--use_early_stopping', action='store_true', default=True)
    p.add_argument('--patience', type=int, default=15)

    # 其他
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=5)

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
