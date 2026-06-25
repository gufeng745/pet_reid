"""
Re-ID 模型评估脚本

使用标准Re-ID评估指标：
- Rank-1/5/10 准确率
- mAP (mean Average Precision)

用法：
    python evaluate.py --model checkpoints/reid/best_reid.pth --num_trials 10
    python evaluate.py --model checkpoints/reid/best_reid.pth --species cat
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

from models.reid_model import ReIDModel
from utils.metrics import compute_reid_metrics


def get_transform():
    """推理用数据增强"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_dataset(data_root, species=None):
    """加载数据集，返回 {identity: [image_paths]}"""
    identity_images = defaultdict(list)

    if species is None or species == 'cat':
        cat_dir = os.path.join(data_root, 'cat')
        if os.path.isdir(cat_dir):
            for pet_id in sorted(os.listdir(cat_dir),
                                key=lambda x: int(x) if x.isdigit() else x):
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
            for pet_id in sorted(os.listdir(dog_dir),
                                key=lambda x: int(x) if x.isdigit() else x):
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


def evaluate_reid(model, data_root, device, num_trials=10, min_images=3, species=None):
    """评估Re-ID模型

    Args:
        model: Re-ID模型
        data_root: 数据集根目录
        device: 设备
        num_trials: 评估次数
        min_images: 每个身份最少图片数
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
        # 随机划分query和gallery
        query_features_list = []
        gallery_features_list = []
        query_labels_list = []
        gallery_labels_list = []

        for identity, features in all_features.items():
            num_images = features.shape[0]

            if num_images < min_images:
                continue

            # 随机选择一张作为query
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
    p.add_argument('--model', type=str, default='checkpoints/reid/best_reid.pth',
                   help='模型路径')
    p.add_argument('--data_root', type=str, default='../pet_rec/reid_dataset',
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
    model = ReIDModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

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
