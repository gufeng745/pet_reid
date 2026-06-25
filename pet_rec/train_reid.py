"""宠物 Re-ID 身份识别训练脚本

基于 train_attr_v2.py 改造，适配 reid_dataset 目录结构：
- reid_dataset/cat/{id}/  — 猫的 Re-ID 数据
- reid_dataset/dog/{id}/  — 狗的 Re-ID 数据

训练目标：
- 身份分类（CrossEntropyLoss）
- 特征对齐（蒸馏损失）
- 对比学习（SupervisedContrastiveLoss）
- 特征正交正则化

用法：
    python train_reid.py --epochs 80 --batch_size 64 --P 16 --K 4
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
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import DINOv3Teacher, DINOv2Teacher, MobileNetV2Student, TeacherAdapter
from distillation import (SupervisedContrastiveLoss,
                          FeatureOrthogonalityLoss)


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


# ==================== PK Sampler ====================

class PKSampler(Sampler):
    """PK Sampler (身份感知采样器)

    每个 Batch 包含 P 个 ID，每个 ID 抽 K 张图片。
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

    def __init__(self, root, transform=None, species=None):
        """
        Args:
            root: 数据集根目录 (reid_dataset/)
            transform: 数据增强
            species: 'cat', 'dog', 或 None (两者都包含)
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
    """Re-ID 用 MobileNetV2 学生模型

    训练时：backbone → 投影头 (512 维) + 身份分类头
    推理时：只用投影头
    """

    def __init__(self, proj_dim=512, num_classes=100, use_se=True, use_bnneck=True):
        super().__init__()
        from models import SEBlock, BNNeck, GeMPooling, get_local_weight_path, load_safetensors_weight
        import timm

        self.use_se = use_se
        self.use_bnneck = use_bnneck

        # 加载 MobileNetV2 backbone
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

        feat_dim = 1280

        # SE 注意力模块
        if use_se:
            self.se_block = SEBlock(feat_dim, reduction=16)

        # BNNeck
        if use_bnneck:
            self.bnneck = BNNeck(feat_dim)

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # 身份分类头
        self.id_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

        self.feature_dim = proj_dim

    def forward(self, x):
        """训练用：返回特征 + 身份预测"""
        feat = self.backbone(x)

        if self.use_se:
            feat = self.se_block(feat)

        if self.use_bnneck:
            feat_bn = self.bnneck(feat)
        else:
            feat_bn = feat

        emb = F.normalize(self.projector(feat_bn), dim=-1)
        id_logits = self.id_head(feat_bn)

        return emb, id_logits

    def forward_emb(self, x):
        """推理用：只返回特征向量"""
        feat = self.backbone(x)
        if self.use_se:
            feat = self.se_block(feat)
        return F.normalize(self.projector(feat), dim=-1)


# ==================== 训练 ====================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === 数据集路径 ===
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reid_dataset')
    if not os.path.isdir(data_root):
        print(f"错误：找不到数据集目录 {data_root}")
        return

    # === 创建数据集 ===
    train_transform = get_augmentation()
    val_transform = get_val_transform()

    full_dataset = ReIDDataset(data_root, transform=train_transform)
    num_classes = full_dataset.num_classes

    if len(full_dataset) == 0:
        print("错误：数据集为空")
        return

    # === 划分 train/val ===
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 验证集使用不同的 transform
    val_dataset_obj = ReIDDataset(data_root, transform=val_transform)

    # 从 full_dataset 中获取 val 索引对应的样本
    val_indices = val_dataset.indices
    val_samples = [full_dataset.samples[i] for i in val_indices]
    val_dataset_obj.samples = val_samples
    val_dataset_obj.id_to_label = full_dataset.id_to_label
    val_dataset_obj.label_to_id = full_dataset.label_to_id

    print(f"Train: {train_size}, Val: {val_size}, Classes: {num_classes}")

    # === DataLoader ===
    if args.use_pk_sampler:
        train_sampler = PKSampler(
            full_dataset, P=args.P, K=args.K, drop_last=True
        )
        # PKSampler 需要从 full_dataset 中采样，所以需要特殊处理
        # 创建一个只包含 train 索引的子集
        train_indices = train_dataset.indices
        train_subset_dataset = ReIDDataset(data_root, transform=train_transform)
        train_subset_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        train_subset_dataset.id_to_label = full_dataset.id_to_label
        train_subset_dataset.label_to_id = full_dataset.label_to_id

        train_sampler = PKSampler(
            train_subset_dataset, P=args.P, K=args.K, drop_last=True
        )
        train_loader = DataLoader(
            train_subset_dataset, batch_sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset_obj, batch_size=args.batch_size, shuffle=False,
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
    criterion_id = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_contrastive = SupervisedContrastiveLoss(temperature=args.contrastive_temp)
    criterion_ortho = FeatureOrthogonalityLoss(feat_dim=args.proj_dim)

    # 特征对齐损失（MSE）
    criterion_align = nn.MSELoss()

    # === 优化器（双学习率） ===
    optimizer = AdamW([
        {'params': student.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': student.projector.parameters(), 'lr': args.lr_head},
        {'params': student.id_head.parameters(), 'lr': args.lr_head},
        {'params': adapter.parameters(), 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    # === LR Schedule ===
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    # === 混合精度训练 ===
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
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
        epoch_loss_align = 0.0
        epoch_loss_id = 0.0
        epoch_loss_contrastive = 0.0
        epoch_loss_ortho = 0.0

        id_correct = 0
        id_total = 0

        t0 = time.time()

        for batch_idx, (view1, view2, labels) in enumerate(train_loader):
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device).long()

            if scaler:
                with torch.cuda.amp.autocast():
                    # Teacher features
                    with torch.no_grad():
                        t1 = teacher(view1)
                        t2 = teacher(view2)

                    # Student features + identity predictions
                    emb1, id1 = student(view1)
                    emb2, id2 = student(view2)

                    # 特征对齐损失
                    t1_adapted = adapter(t1)
                    t2_adapted = adapter(t2)
                    loss_align = (criterion_align(emb1, t1_adapted) + criterion_align(emb2, t2_adapted)) / 2

                    # 身份分类损失
                    loss_id = (criterion_id(id1, labels) + criterion_id(id2, labels)) / 2

                    # 对比学习损失
                    loss_con = criterion_contrastive(emb1, emb2)

                    # 特征正交正则化
                    loss_orth = criterion_ortho(emb1)

                    # 总损失
                    loss = (args.alpha * loss_align +
                            args.lambda_id * loss_id +
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
                loss_align = (criterion_align(emb1, t1_adapted) + criterion_align(emb2, t2_adapted)) / 2

                loss_id = (criterion_id(id1, labels) + criterion_id(id2, labels)) / 2
                loss_con = criterion_contrastive(emb1, emb2)
                loss_orth = criterion_ortho(emb1)

                loss = (args.alpha * loss_align +
                        args.lambda_id * loss_id +
                        args.lambda_contrastive * loss_con +
                        args.lambda_ortho * loss_orth)

                optimizer.zero_grad()
                loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_loss_align += loss_align.item()
            epoch_loss_id += loss_id.item()
            epoch_loss_contrastive += loss_con.item()
            epoch_loss_ortho += loss_orth.item()

            # 身份准确率
            id_pred = id1.argmax(dim=1)
            id_correct += (id_pred == labels).sum().item()
            id_total += labels.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                id_acc = id_correct / id_total
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg_loss:.4f} align={epoch_loss_align/(batch_idx+1):.4f} "
                      f"id={epoch_loss_id/(batch_idx+1):.4f} "
                      f"contrastive={epoch_loss_contrastive/(batch_idx+1):.4f} "
                      f"ortho={epoch_loss_ortho/(batch_idx+1):.4f} "
                      f"id_acc={id_acc:.3f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        id_acc = id_correct / id_total
        print(f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f} | id_acc={id_acc:.3f}")

        # === 验证 ===
        val_loss = validate(student, adapter, teacher, val_loader,
                           criterion_align, criterion_id, criterion_contrastive,
                           args, device)
        print(f"  Val loss: {val_loss:.4f}")

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
def validate(student, adapter, teacher, val_loader,
             criterion_align, criterion_id, criterion_contrastive,
             args, device):
    """验证"""
    student.eval()
    adapter.eval()
    total_loss = 0.0
    count = 0

    for view1, view2, labels in val_loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device).long()

        t1 = teacher(view1)
        emb1, id1 = student(view1)
        emb2, _ = student(view2)

        t1_adapted = adapter(t1)
        loss_align = criterion_align(emb1, t1_adapted)
        loss_id = criterion_id(id1, labels)
        loss_con = criterion_contrastive(emb1, emb2)

        loss = (args.alpha * loss_align +
                args.lambda_id * loss_id +
                args.lambda_contrastive * loss_con)

        total_loss += loss.item() * view1.size(0)
        count += view1.size(0)

    student.train()
    adapter.train()
    return total_loss / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description='宠物 Re-ID 身份识别训练')

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
    p.add_argument('--use_se', action='store_true', default=True)
    p.add_argument('--use_bnneck', action='store_true', default=True)

    # 损失函数权重
    p.add_argument('--alpha', type=float, default=1.0, help='特征对齐损失权重')
    p.add_argument('--lambda_id', type=float, default=0.5, help='身份分类损失权重')
    p.add_argument('--lambda_contrastive', type=float, default=0.3, help='对比学习损失权重')
    p.add_argument('--lambda_ortho', type=float, default=0.05, help='特征正交正则化权重')
    p.add_argument('--contrastive_temp', type=float, default=0.1, help='对比损失温度系数')

    # PK Sampler
    p.add_argument('--use_pk_sampler', action='store_true', default=False)
    p.add_argument('--P', type=int, default=16, help='PK Sampler: 每个 batch 的 ID 数')
    p.add_argument('--K', type=int, default=4, help='PK Sampler: 每个 ID 的样本数')

    # 训练技巧
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--use_early_stopping', action='store_true', default=True)
    p.add_argument('--patience', type=int, default=15)

    # 其他
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=10)

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
