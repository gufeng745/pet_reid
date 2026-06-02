import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import DINOv3Teacher, DINOv2Teacher, MobileNetV2Student, TeacherAdapter
from distillation import DistillationLoss, ColorAwareDistillationLoss
from prepare_data import create_dataloaders


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === Teacher ===
    try:
        teacher = DINOv3Teacher()
        print("Teacher: DINOv3 ViT-S (384-dim)")
    except Exception as e:
        print(f"DINOv3 加载失败 ({e}), 回退到 DINOv2")
        teacher = DINOv2Teacher()
        print("Teacher: DINOv2 ViT-S (384-dim)")
    teacher = teacher.to(device)
    teacher.eval()

    # === Student + Adapter ===
    student = MobileNetV2Student(proj_dim=args.proj_dim).to(device)
    adapter = TeacherAdapter(teacher_dim=384, student_dim=args.proj_dim).to(device)

    print(f"Student: MobileNetV2 ({sum(p.numel() for p in student.parameters())/1e6:.1f}M params)")
    print(f"Feature dim: {args.proj_dim}")

    # === Loss ===
    if args.color_aware:
        criterion = ColorAwareDistillationLoss(
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta
        )
        print(f"Color-aware training enabled (delta={args.delta})")
    else:
        criterion = DistillationLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    # === Optimizer (双学习率) ===
    optimizer = AdamW([
        {'params': student.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': student.projector.parameters(), 'lr': args.lr_head},
        {'params': adapter.parameters(), 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    # === LR Schedule: warmup + cosine ===
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    # === Data ===
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_loader, eval_loader = create_dataloaders(
        dataset_root, batch_size=args.batch_size, use_trimap=args.color_aware
    )
    print(f"Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"Eval:  {len(eval_loader.dataset)} images, {len(eval_loader)} batches")

    # === Checkpoint dir ===
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float('inf')

    # === Training Loop ===
    for epoch in range(args.epochs):
        student.train()
        adapter.train()
        loss_keys = ['total', 'align', 'sim', 'uniform'] + (['color_sep'] if args.color_aware else [])
        epoch_losses = {k: 0 for k in loss_keys}
        t0 = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            if args.color_aware:
                view1, view2, color_feat = batch_data
                color_feat = color_feat.to(device)
            else:
                view1, view2, _ = batch_data
                color_feat = None

            view1 = view1.to(device)
            view2 = view2.to(device)

            # Teacher features (no grad)
            with torch.no_grad():
                t1 = teacher(view1)
                t2 = teacher(view2)

            # Student features
            s1 = student(view1)
            s2 = student(view2)

            # Loss (双向平均)
            if args.color_aware:
                loss1, details1 = criterion(t1, s1, adapter, color_feat=color_feat)
                loss2, details2 = criterion(t2, s2, adapter, color_feat=color_feat)
            else:
                loss1, details1 = criterion(t1, s1, adapter)
                loss2, details2 = criterion(t2, s2, adapter)
            loss = (loss1 + loss2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += (details1[k] + details2[k]) / 2

            if (batch_idx + 1) % args.log_interval == 0:
                avg = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                msg = (f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(train_loader)} "
                       f"loss={avg['total']:.4f} align={avg['align']:.4f} "
                       f"sim={avg['sim']:.4f} uniform={avg['uniform']:.4f}")
                if args.color_aware:
                    msg += f" color={avg['color_sep']:.4f}"
                print(msg)

        scheduler.step()
        avg_loss = epoch_losses['total'] / len(train_loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.1f}s | "
              f"loss={avg_loss:.4f} | lr={lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                name = 'best_student.pth'
            else:
                name = f'student_epoch{epoch+1}.pth'
            path = os.path.join(ckpt_dir, name)
            torch.save({
                'epoch': epoch + 1,
                'student': student.state_dict(),
                'adapter': adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, path)
            print(f"  Saved: {path}")

    # Save final
    final_path = os.path.join(ckpt_dir, 'final_student.pth')
    torch.save(student.state_dict(), final_path)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Final student: {final_path}")
    return student


def parse_args():
    p = argparse.ArgumentParser(description='DINOv3 → MobileNetV2 distillation')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr_backbone', type=float, default=5e-4)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.04)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--proj_dim', type=int, default=512)
    p.add_argument('--alpha', type=float, default=1.0, help='alignment loss weight')
    p.add_argument('--beta', type=float, default=0.5, help='self-similarity loss weight')
    p.add_argument('--gamma', type=float, default=0.1, help='uniformity loss weight')
    p.add_argument('--color_aware', action='store_true', help='启用颜色感知训练（需要 trimap 标注）')
    p.add_argument('--delta', type=float, default=0.3, help='color separation loss weight')
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
