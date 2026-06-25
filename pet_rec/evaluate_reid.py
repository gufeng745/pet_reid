"""Re-ID 模型评估脚本

使用标准 Re-ID 评估指标：
- Rank-1/5/10 准确率
- mAP (mean Average Precision)
- CMC 曲线

评估策略：
1. 严格身份分离：同一身份的所有图片只出现在 train 或 test 中
2. Query-Gallery 模式：从每个测试身份中随机选一张作为 query，其余作为 gallery
3. 多次随机评估取平均

用法：
    python evaluate_reid.py --model checkpoints/best_student_reid.pth --num_trials 10
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from train_reid import MobileNetV2StudentForReID


def load_model(model_path, device='cpu'):
    """加载 Re-ID 模型"""
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    # 从 checkpoint 获取参数
    num_classes = ckpt.get('num_classes', 82)
    args = ckpt.get('args', {})
    proj_dim = args.get('proj_dim', 512)
    use_se = args.get('use_se', True)
    use_bnneck = args.get('use_bnneck', True)

    print(f"模型参数: num_classes={num_classes}, proj_dim={proj_dim}")

    # 创建模型
    model = MobileNetV2StudentForReID(
        proj_dim=proj_dim,
        num_classes=num_classes,
        use_se=use_se,
        use_bnneck=use_bnneck
    )

    # 加载权重
    if 'student' in ckpt:
        model.load_state_dict(ckpt['student'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def get_transform():
    """推理用数据增强"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


def load_dataset(data_root, species=None):
    """加载数据集，返回 {identity: [image_paths]}"""
    identity_images = defaultdict(list)

    if species is None or species == 'cat':
        cat_dir = os.path.join(data_root, 'cat')
        if os.path.isdir(cat_dir):
            for pet_id in sorted(os.listdir(cat_dir), key=lambda x: int(x) if x.isdigit() else x):
                pet_dir = os.path.join(cat_dir, pet_id)
                if not os.path.isdir(pet_dir):
                    continue
                full_id = f"cat_{pet_id}"
                for img_name in os.listdir(pet_dir):
                    img_path = os.path.join(pet_dir, img_name)
                    if os.path.isfile(img_path):
                        identity_images[full_id].append(img_path)

    if species is None or species == 'dog':
        dog_dir = os.path.join(data_root, 'dog')
        if os.path.isdir(dog_dir):
            for pet_id in sorted(os.listdir(dog_dir), key=lambda x: int(x) if x.isdigit() else x):
                pet_dir = os.path.join(dog_dir, pet_id)
                if not os.path.isdir(pet_dir):
                    continue
                full_id = f"dog_{pet_id}"
                for img_name in os.listdir(pet_dir):
                    img_path = os.path.join(pet_dir, img_name)
                    if os.path.isfile(img_path):
                        identity_images[full_id].append(img_path)

    return dict(identity_images)


def extract_features(model, image_paths, transform, device, batch_size=32):
    """批量提取特征"""
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Warning: 无法加载 {img_path}: {e}")
                # 用零向量替代
                batch_images.append(torch.zeros(3, 224, 224))

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                feat = model.forward_emb(batch_tensor)
                feat = F.normalize(feat, dim=-1)
            features.append(feat.cpu())

    if features:
        return torch.cat(features, dim=0)
    return torch.tensor([])


def compute_reid_metrics(query_features, gallery_features, query_labels, gallery_labels):
    """计算 Re-ID 指标

    Args:
        query_features: (N_q, D) 查询特征
        gallery_features: (N_g, D) 画廊特征
        query_labels: (N_q,) 查询标签
        gallery_labels: (N_g,) 画廊标签

    Returns:
        metrics: dict with rank-1/5/10 and mAP
    """
    # 计算余弦相似度
    similarity = query_features @ gallery_features.T  # (N_q, N_g)

    # 对每个 query 计算排名
    rank_list = []
    ap_list = []

    for i in range(len(query_labels)):
        # 获取当前 query 的相似度和标签
        sim_scores = similarity[i]
        query_label = query_labels[i]

        # 按相似度降序排序
        sorted_indices = torch.argsort(sim_scores, descending=True)
        sorted_labels = gallery_labels[sorted_indices.numpy()]

        # 计算 rank-k 准确率
        matches = torch.tensor(sorted_labels == query_label)
        rank_list.append(matches)

        # 计算 AP (Average Precision)
        true_positives = matches.float()
        cumulative_tp = torch.cumsum(true_positives, dim=0)
        precision_at_k = cumulative_tp / (torch.arange(len(true_positives)).float() + 1)
        ap = (precision_at_k * true_positives).sum() / max(true_positives.sum(), 1)
        ap_list.append(ap.item())

    # 计算 Rank-k 准确率
    rank_tensor = torch.stack(rank_list)  # (N_q, N_g)
    num_gallery = rank_tensor.shape[1]

    rank_1 = rank_tensor[:, :1].any(dim=1).float().mean().item()
    rank_5 = rank_tensor[:, :5].any(dim=1).float().mean().item()
    rank_10 = rank_tensor[:, :10].any(dim=1).float().mean().item()

    # 处理 gallery 数量不足的情况
    if num_gallery < 5:
        rank_5 = rank_1
    if num_gallery < 10:
        rank_10 = rank_5

    # 计算 mAP
    mAP = np.mean(ap_list)

    return {
        'rank-1': rank_1,
        'rank-5': rank_5,
        'rank-10': rank_10,
        'mAP': mAP,
    }


def evaluate_reid(model, data_root, device, num_trials=10, min_images=3, species=None):
    """评估 Re-ID 模型

    Args:
        model: Re-ID 模型
        data_root: 数据集根目录
        device: 设备
        num_trials: 评估次数
        min_images: 每个身份最少图片数（用于划分 query/gallery）
        species: 'cat', 'dog', 或 None
    """
    transform = get_transform()

    # 加载数据集
    identity_images = load_dataset(data_root, species)

    # 过滤图片数量不足的身份
    valid_identities = {k: v for k, v in identity_images.items() if len(v) >= min_images}

    print(f"总身份数: {len(identity_images)}")
    print(f"有效身份数 (>= {min_images} 张图片): {len(valid_identities)}")

    if len(valid_identities) < 2:
        print("错误：有效身份数不足，无法评估")
        return None

    # 统计信息
    total_images = sum(len(imgs) for imgs in valid_identities.values())
    avg_images = total_images / len(valid_identities)
    print(f"总图片数: {total_images}, 平均每身份: {avg_images:.1f} 张")

    # 提取所有图片的特征
    print("\n提取特征中...")
    all_features = {}
    all_labels = {}
    identity_list = list(valid_identities.keys())

    for idx, (identity, image_paths) in enumerate(valid_identities.items()):
        features = extract_features(model, image_paths, transform, device)
        if features.shape[0] > 0:
            all_features[identity] = features
            all_labels[identity] = [identity] * len(image_paths)

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(valid_identities)} 个身份")

    print(f"特征提取完成，共 {len(all_features)} 个身份")

    # 多次随机评估
    all_metrics = []

    for trial in range(num_trials):
        # 随机划分 query 和 gallery
        query_features_list = []
        gallery_features_list = []
        query_labels_list = []
        gallery_labels_list = []

        for identity, features in all_features.items():
            num_images = features.shape[0]

            if num_images < min_images:
                continue

            # 随机选择一张作为 query
            query_idx = random.randint(0, num_images - 1)
            gallery_indices = [j for j in range(num_images) if j != query_idx]

            query_features_list.append(features[query_idx:query_idx + 1])
            gallery_features_list.append(features[gallery_indices])
            query_labels_list.extend([identity])
            gallery_labels_list.extend([identity] * len(gallery_indices))

        if not query_features_list:
            continue

        query_features = torch.cat(query_features_list, dim=0)
        gallery_features = torch.cat(gallery_features_list, dim=0)
        query_labels = np.array(query_labels_list)
        gallery_labels = np.array(gallery_labels_list)

        # 计算指标
        metrics = compute_reid_metrics(
            query_features, gallery_features,
            query_labels, gallery_labels
        )
        all_metrics.append(metrics)

    if not all_metrics:
        print("错误：无法计算指标")
        return None

    # 计算平均指标
    avg_metrics = {
        'rank-1': np.mean([m['rank-1'] for m in all_metrics]),
        'rank-5': np.mean([m['rank-5'] for m in all_metrics]),
        'rank-10': np.mean([m['rank-10'] for m in all_metrics]),
        'mAP': np.mean([m['mAP'] for m in all_metrics]),
    }

    std_metrics = {
        'rank-1': np.std([m['rank-1'] for m in all_metrics]),
        'rank-5': np.std([m['rank-5'] for m in all_metrics]),
        'rank-10': np.std([m['rank-10'] for m in all_metrics]),
        'mAP': np.std([m['mAP'] for m in all_metrics]),
    }

    return avg_metrics, std_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Re-ID 模型评估')
    p.add_argument('--model', type=str, default='checkpoints/best_student_reid.pth',
                   help='模型路径')
    p.add_argument('--data_root', type=str, default='reid_dataset',
                   help='数据集根目录')
    p.add_argument('--num_trials', type=int, default=10,
                   help='评估次数')
    p.add_argument('--min_images', type=int, default=3,
                   help='每个身份最少图片数')
    p.add_argument('--species', type=str, default=None, choices=['cat', 'dog', None],
                   help='评估物种')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = load_model(args.model, device)
    model = model.to(device)

    # 评估
    print(f"\n开始评估 ({args.num_trials} 次随机划分)...")
    result = evaluate_reid(
        model, args.data_root, device,
        num_trials=args.num_trials,
        min_images=args.min_images,
        species=args.species
    )

    if result:
        avg_metrics, std_metrics = result

        print("\n" + "=" * 50)
        print("Re-ID 评估结果")
        print("=" * 50)
        print(f"Rank-1 准确率: {avg_metrics['rank-1']*100:.2f}% ± {std_metrics['rank-1']*100:.2f}%")
        print(f"Rank-5 准确率: {avg_metrics['rank-5']*100:.2f}% ± {std_metrics['rank-5']*100:.2f}%")
        print(f"Rank-10 准确率: {avg_metrics['rank-10']*100:.2f}% ± {std_metrics['rank-10']*100:.2f}%")
        print(f"mAP: {avg_metrics['mAP']*100:.2f}% ± {std_metrics['mAP']*100:.2f}%")
        print("=" * 50)
