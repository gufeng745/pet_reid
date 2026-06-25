"""
Re-ID 推理脚本

支持：
- 单张图片特征提取
- 批量特征提取
- 相似度计算
- 图片检索

用法：
    python inference.py --model checkpoints/reid/best_reid.pth --image test.png
    python inference.py --model checkpoints/reid/best_reid.pth --query query.png --gallery gallery/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models.reid_model import ReIDModel


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


class PetReIDInference:
    """宠物Re-ID推理类

    用于提取特征、计算相似度、图片检索
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备 ('cuda', 'cpu', 或 None自动选择)
        """
        # 设置设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = ReIDModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据增强
        self.transform = get_transform()

        print("模型加载完成")

    @torch.no_grad()
    def extract_feature(self, image_path: str) -> torch.Tensor:
        """提取单张图片的特征

        Args:
            image_path: 图片路径

        Returns:
            feature: (D,) 特征向量
        """
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 提取特征
        feature = self.model.forward_emb(img_tensor)
        feature = F.normalize(feature, dim=-1)

        return feature.squeeze(0).cpu()

    @torch.no_grad()
    def extract_features_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """批量提取特征

        Args:
            image_paths: 图片路径列表
            batch_size: 批大小

        Returns:
            features: (N, D) 特征矩阵
        """
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Warning: 无法加载 {img_path}: {e}")
                    # 用零向量替代
                    batch_images.append(torch.zeros(3, 224, 224))

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                features = self.model.forward_emb(batch_tensor)
                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu())

        if all_features:
            return torch.cat(all_features, dim=0)
        return torch.tensor([])

    def compute_similarity(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor
    ) -> float:
        """计算两个特征的余弦相似度

        Args:
            feature1: (D,) 特征向量
            feature2: (D,) 特征向量

        Returns:
            similarity: 余弦相似度
        """
        return (feature1 @ feature2).item()

    def compute_similarity_matrix(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """计算特征矩阵的相似度

        Args:
            features1: (N1, D) 特征矩阵
            features2: (N2, D) 特征矩阵

        Returns:
            similarity: (N1, N2) 相似度矩阵
        """
        return features1 @ features2.T

    def search(
        self,
        query_path: str,
        gallery_paths: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """图片检索

        Args:
            query_path: 查询图片路径
            gallery_paths: 图库图片路径列表
            top_k: 返回前k个结果

        Returns:
            results: [(image_path, similarity), ...]
        """
        # 提取查询特征
        query_feature = self.extract_feature(query_path)

        # 提取图库特征
        gallery_features = self.extract_features_batch(gallery_paths)

        # 计算相似度
        similarities = (query_feature.unsqueeze(0) @ gallery_features.T).squeeze(0)

        # 排序
        top_k = min(top_k, len(gallery_paths))
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append((gallery_paths[idx], similarities[idx].item()))

        return results

    def find_most_similar(
        self,
        query_path: str,
        gallery_paths: List[str]
    ) -> Tuple[str, float]:
        """找到最相似的图片

        Args:
            query_path: 查询图片路径
            gallery_paths: 图库图片路径列表

        Returns:
            best_path: 最相似的图片路径
            best_similarity: 相似度
        """
        results = self.search(query_path, gallery_paths, top_k=1)
        return results[0]


def parse_args():
    p = argparse.ArgumentParser(description='Re-ID 推理')

    p.add_argument('--model', type=str, default='checkpoints/reid/best_reid.pth',
                   help='模型路径')
    p.add_argument('--device', type=str, default=None,
                   help='设备 (cuda/cpu)')

    # 单张图片特征提取
    p.add_argument('--image', type=str, default=None,
                   help='单张图片路径')

    # 相似度计算
    p.add_argument('--image1', type=str, default=None,
                   help='图片1路径')
    p.add_argument('--image2', type=str, default=None,
                   help='图片2路径')

    # 图片检索
    p.add_argument('--query', type=str, default=None,
                   help='查询图片路径')
    p.add_argument('--gallery', type=str, default=None,
                   help='图库目录或图片列表')
    p.add_argument('--top_k', type=int, default=5,
                   help='返回前k个结果')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 创建推理器
    inferencer = PetReIDInference(args.model, args.device)

    # 单张图片特征提取
    if args.image:
        print(f"\n提取特征: {args.image}")
        feature = inferencer.extract_feature(args.image)
        print(f"特征维度: {feature.shape}")
        print(f"特征范数: {torch.norm(feature).item():.4f}")

    # 相似度计算
    elif args.image1 and args.image2:
        print(f"\n计算相似度:")
        print(f"  图片1: {args.image1}")
        print(f"  图片2: {args.image2}")

        feature1 = inferencer.extract_feature(args.image1)
        feature2 = inferencer.extract_feature(args.image2)
        similarity = inferencer.compute_similarity(feature1, feature2)

        print(f"\n余弦相似度: {similarity:.4f}")
        print(f"是否同一身份: {'是' if similarity > 0.5 else '否'} (阈值: 0.5)")

    # 图片检索
    elif args.query and args.gallery:
        print(f"\n图片检索:")
        print(f"  查询: {args.query}")
        print(f"  图库: {args.gallery}")

        # 收集图库图片
        gallery_paths = []
        if os.path.isdir(args.gallery):
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for root, dirs, files in os.walk(args.gallery):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_extensions:
                        gallery_paths.append(os.path.join(root, file))
        else:
            gallery_paths = [args.gallery]

        print(f"  图库图片数: {len(gallery_paths)}")

        # 检索
        results = inferencer.search(args.query, gallery_paths, top_k=args.top_k)

        print(f"\nTop-{args.top_k} 结果:")
        for i, (path, sim) in enumerate(results, 1):
            print(f"  {i}. {os.path.basename(path)}: {sim:.4f}")

    else:
        print("请指定操作模式：")
        print("  1. 单张图片: --image test.png")
        print("  2. 相似度: --image1 img1.png --image2 img2.png")
        print("  3. 检索: --query query.png --gallery gallery/")
