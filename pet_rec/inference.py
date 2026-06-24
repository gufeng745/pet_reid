"""宠物重识别推理模块

基于 Re-ID SOP 实现：
- L2 归一化特征提取
- 余弦相似度计算
- 多查询融合
- 置信度分级
- Re-ranking（可选）

用法：
    python inference.py --model_path checkpoints/best_student_attr_v2.pth \
                        --query_img query.jpg \
                        --gallery_dir gallery/
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models import MobileNetV2StudentWithAttr


# ==================== 特征提取器 ====================

class PetReIDInference:
    """宠物重识别推理类

    实现完整的 Re-ID 推理流程：
    1. 图像预处理
    2. 特征提取 + L2 归一化
    3. 余弦相似度计算
    4. 多查询融合（可选）
    5. 置信度分级
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: 模型权重路径
            device: 推理设备 ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        print(f"Model loaded. Device: {self.device}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location='cpu')

        # 从 checkpoint 中获取模型参数
        args = checkpoint.get('args', {})
        encoders = checkpoint.get('encoders', {})

        # 获取类别数
        num_colors = len(encoders.get('color_classes', []))
        num_patterns = len(encoders.get('pattern_classes', []))

        # 创建模型
        model = MobileNetV2StudentWithAttr(
            proj_dim=args.get('proj_dim', 512),
            num_colors=num_colors if num_colors > 0 else 13,
            num_patterns=num_patterns if num_patterns > 0 else 13,
            use_se=args.get('use_se', True),
            use_bnneck=args.get('use_bnneck', True),
        )

        # 加载权重
        model.load_state_dict(checkpoint['student'])
        return model

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """提取单张图片的特征向量

        Args:
            image: PIL Image

        Returns:
            features: (1, D) L2 归一化的特征向量
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model.forward_emb(img_tensor)

        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        return features

    @torch.no_grad()
    def extract_features_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """批量提取特征向量

        Args:
            images: PIL Image 列表

        Returns:
            features: (N, D) L2 归一化的特征向量
        """
        img_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        features = self.model.forward_emb(img_tensors)

        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        return features

    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算两个特征向量的余弦相似度

        Args:
            feat1: (1, D) 特征向量
            feat2: (1, D) 特征向量

        Returns:
            similarity: 余弦相似度 ([-1, 1])
        """
        return F.cosine_similarity(feat1, feat2).item()

    def multi_query_fusion(self, query_images: List[Image.Image]) -> torch.Tensor:
        """多查询融合：将多张图片的特征向量平均

        Args:
            query_images: 同一只宠物的多张图片

        Returns:
            fused_features: (1, D) 融合后的特征向量
        """
        features_list = [self.extract_features(img) for img in query_images]
        fused = torch.mean(torch.stack(features_list), dim=0)

        # 重新 L2 归一化
        fused = F.normalize(fused, p=2, dim=1)
        return fused

    def match_single(self, query_img: Image.Image,
                     gallery_imgs: List[Image.Image],
                     gallery_labels: Optional[List[str]] = None) -> List[Tuple[int, float, str]]:
        """单查询匹配

        Args:
            query_img: 查询图片
            gallery_imgs: 图库图片列表
            gallery_labels: 图库标签列表（可选）

        Returns:
            results: [(index, similarity, label), ...] 按相似度降序排列
        """
        query_feat = self.extract_features(query_img)
        gallery_feats = self.extract_features_batch(gallery_imgs)

        # 计算余弦相似度
        similarities = F.cosine_similarity(query_feat, gallery_feats)

        # 置信度分级
        results = []
        for idx, sim in enumerate(similarities):
            label = self._get_confidence_label(sim.item())
            gallery_label = gallery_labels[idx] if gallery_labels else str(idx)
            results.append((idx, sim.item(), label, gallery_label))

        # 按相似度降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def match_multi_query(self, query_imgs: List[Image.Image],
                          gallery_imgs: List[Image.Image],
                          gallery_labels: Optional[List[str]] = None) -> List[Tuple[int, float, str]]:
        """多查询匹配（更准确）

        Args:
            query_imgs: 同一只宠物的多张查询图片
            gallery_imgs: 图库图片列表
            gallery_labels: 图库标签列表（可选）

        Returns:
            results: [(index, similarity, label, gallery_label), ...] 按相似度降序排列
        """
        query_feat = self.multi_query_fusion(query_imgs)
        gallery_feats = self.extract_features_batch(gallery_imgs)

        # 计算余弦相似度
        similarities = F.cosine_similarity(query_feat, gallery_feats)

        # 置信度分级
        results = []
        for idx, sim in enumerate(similarities):
            label = self._get_confidence_label(sim.item())
            gallery_label = gallery_labels[idx] if gallery_labels else str(idx)
            results.append((idx, sim.item(), label, gallery_label))

        # 按相似度降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _get_confidence_label(similarity: float) -> str:
        """置信度分级

        Args:
            similarity: 余弦相似度

        Returns:
            label: 置信度标签
        """
        if similarity > 0.85:
            return "高置信度匹配"
        elif similarity > 0.4:
            return "疑似匹配，需人工确认"
        else:
            return "不匹配"

    def is_same_pet(self, img1: Image.Image, img2: Image.Image,
                    threshold: float = 0.85) -> Tuple[bool, float, str]:
        """判断两张图片是否为同一只宠物

        Args:
            img1: 图片 1
            img2: 图片 2
            threshold: 高置信度阈值

        Returns:
            is_same: 是否为同一只宠物
            similarity: 余弦相似度
            confidence: 置信度标签
        """
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        similarity = self.compute_similarity(feat1, feat2)
        confidence = self._get_confidence_label(similarity)

        # 使用阈值判断
        is_same = similarity > threshold

        return is_same, similarity, confidence


# ==================== Re-ranking ====================

class ReRanker:
    """k-reciprocal encoding 重排序

    在 1vN 检索场景下，对初步排名进行二次优化。
    通过计算 Jaccard 距离，利用近邻信息提升检索精度。
    """

    def __init__(self, k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
        """
        Args:
            k1: 第一阶段近邻数
            k2: 第二阶段近邻数
            lambda_value: 原始距离权重
        """
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

    def re_rank(self, query_feats: torch.Tensor,
                gallery_feats: torch.Tensor) -> torch.Tensor:
        """重排序

        Args:
            query_feats: (M, D) 查询特征
            gallery_feats: (N, D) 图库特征

        Returns:
            re_ranked_dist: (M, N) 重排序后的距离矩阵
        """
        # 计算原始余弦距离
        original_dist = 1 - torch.mm(query_feats, gallery_feats.t())
        original_dist = original_dist.cpu().numpy()

        # 计算图库内部的距离
        gallery_dist = 1 - torch.mm(gallery_feats, gallery_feats.t())
        gallery_dist = gallery_dist.cpu().numpy()

        # 计算 k-reciprocal 近邻
        query_num = query_feats.shape[0]
        gallery_num = gallery_feats.shape[0]

        # 合并距离矩阵
        all_dist = np.concatenate([original_dist, gallery_dist], axis=1)
        all_num = query_num + gallery_num

        # 计算 Jaccard 距离
        V = np.zeros_like(all_dist)
        for i in range(query_num):
            # 获取 k1 近邻
            forward_k1_idx = np.argsort(all_dist[i])[:self.k1 + 1]
            backward_k1_idx = np.argsort(all_dist[:, i])[:self.k1 + 1]

            # 计算 k-reciprocal 近邻
            k_reciprocal_idx = np.intersect1d(forward_k1_idx, backward_k1_idx)

            # 计算 Jaccard 距离
            k_reciprocal_expansion_idx = k_reciprocal_idx
            for j in k_reciprocal_idx:
                candidate_forward_k2_idx = np.argsort(all_dist[j])[:self.k2 + 1]
                candidate_backward_k2_idx = np.argsort(all_dist[:, j])[:self.k2 + 1]
                candidate_k2_idx = np.intersect1d(candidate_forward_k2_idx, candidate_backward_k2_idx)

                if len(np.intersect1d(candidate_k2_idx, k_reciprocal_idx)) > 2 / 3 * len(candidate_k2_idx):
                    k_reciprocal_expansion_idx = np.union1d(k_reciprocal_expansion_idx, candidate_k2_idx)

            # 计算 V 矩阵
            weight = np.exp(-all_dist[i, k_reciprocal_expansion_idx])
            V[i, k_reciprocal_expansion_idx] = weight / np.sum(weight)

        # 计算重排序距离
        re_rank_dist = np.zeros_like(original_dist)
        for i in range(query_num):
            for j in range(gallery_num):
                re_rank_dist[i, j] = self.lambda_value * original_dist[i, j] + \
                                     (1 - self.lambda_value) * np.sum(V[i, :query_num] * V[j, query_num:])

        return torch.from_numpy(re_rank_dist)


# ==================== 工具函数 ====================

def load_image(image_path: str) -> Image.Image:
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def load_images_from_dir(dir_path: str) -> Tuple[List[Image.Image], List[str]]:
    """从目录加载所有图片"""
    images = []
    filenames = []

    for filename in sorted(os.listdir(dir_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(dir_path, filename)
            try:
                img = load_image(img_path)
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    return images, filenames


def print_results(results: List[Tuple[int, float, str, str]], top_k: int = 10):
    """打印匹配结果"""
    print("\n" + "=" * 60)
    print("匹配结果")
    print("=" * 60)

    for i, (idx, sim, confidence, label) in enumerate(results[:top_k]):
        print(f"{i+1:2d}. 索引: {idx:4d} | 相似度: {sim:.4f} | "
              f"置信度: {confidence:12s} | 标签: {label}")

    print("=" * 60)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='宠物重识别推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--query_img', type=str, required=True,
                       help='查询图片路径')
    parser.add_argument('--gallery_dir', type=str, required=True,
                       help='图库目录路径')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='高置信度阈值')
    parser.add_argument('--top_k', type=int, default=10,
                       help='显示前 K 个结果')
    parser.add_argument('--device', type=str, default='auto',
                       help='推理设备 (auto/cpu/cuda)')
    parser.add_argument('--use_rerank', action='store_true',
                       help='使用重排序')

    args = parser.parse_args()

    # 初始化推理器
    inferencer = PetReIDInference(args.model_path, args.device)

    # 加载查询图片
    query_img = load_image(args.query_img)
    print(f"\nQuery: {args.query_img}")

    # 加载图库图片
    gallery_imgs, gallery_labels = load_images_from_dir(args.gallery_dir)
    print(f"Gallery: {len(gallery_imgs)} images from {args.gallery_dir}")

    if len(gallery_imgs) == 0:
        print("错误：图库目录为空")
        return

    # 执行匹配
    results = inferencer.match_single(query_img, gallery_imgs, gallery_labels)

    # 打印结果
    print_results(results, args.top_k)

    # 统计
    high_conf = sum(1 for _, _, conf, _ in results if conf == "高置信度匹配")
    medium_conf = sum(1 for _, _, conf, _ in results if conf == "疑似匹配，需人工确认")
    low_conf = sum(1 for _, _, conf, _ in results if conf == "不匹配")

    print(f"\n统计：高置信度={high_conf}, 疑似={medium_conf}, 不匹配={low_conf}")


if __name__ == '__main__':
    main()
