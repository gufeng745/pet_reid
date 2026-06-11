# encoding: utf-8
"""
使用 FastReID 模型进行宠物特征提取和相似度匹配
集成方案：在 pet_rec 中使用 fastreid 的 FeatureExtractionDemo
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 添加 fastreid 路径
sys.path.append('../fastreid1211')

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.file_io import PathManager


def setup_cfg(config_file):
    """
    加载配置文件
    """
    import yaml
    cfg = get_cfg()
    # 使用 UTF-8 编码读取配置文件，避免 GBK 编码错误
    with open(config_file, 'r', encoding='utf-8') as f:
        loaded_dict = yaml.safe_load(f)
    # 使用 merge_from_other_cfg 直接合并字典
    from fastreid.config import CfgNode
    loaded_cfg = CfgNode(loaded_dict)
    cfg.merge_from_other_cfg(loaded_cfg)
    cfg.freeze()
    return cfg


class PetFeatureExtractor:
    """
    使用 FastReID 模型提取宠物特征
    """
    
    def __init__(self, config_file, device=None):
        """
        Args:
            config_file: FastReID 配置文件路径
            device: 计算设备，None 表示自动选择 (CPU)
        """
        self.cfg = setup_cfg(config_file)
        # 检查 CUDA 是否可用
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"使用设备：{self.device}")
        self.predictor = DefaultPredictor(self.cfg)
        
        # 获取输入尺寸
        self.input_size = tuple(self.cfg.INPUT.SIZE_TEST[::-1])  # (W, H)
        
        print(f"FastReID 模型已加载")
        print(f"输入尺寸：{self.input_size}")
        print(f"特征维度：{self.cfg.MODEL.HEADS.EMBEDDING_DIM}")
        
    def extract_feature(self, image_path):
        """
        从单张图片提取特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            feature: 归一化后的特征向量 (512,)
        """
        # 使用 OpenCV 读取图片 (BGR 格式)
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图片 {image_path}")
            return None
            
        return self._extract_from_image(img)
    
    def _extract_from_image(self, img_bgr):
        """
        从 BGR 图片数组提取特征
        
        Args:
            img_bgr: BGR 格式的图片 (H, W, 3)
            
        Returns:
            feature: 归一化后的特征向量 (512,)
        """
        # 使用 fastreid 标准的输入格式
        # 保持 BGR 格式，fastreid 内部会转换为 RGB
        img_resized = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        # 转换为 tensor (BGR -> CHW)
        img_tensor = torch.as_tensor(img_resized.astype("float32").transpose(2, 0, 1))
        img_tensor = img_tensor / 255.0  # 归一化到 [0, 1]
        
        # 应用 ImageNet 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        img_tensor = img_tensor[None].to(self.device)  # 添加 batch 维度
        
        # 使用 predictor 的模型直接提取特征
        with torch.no_grad():
            # 直接传入 tensor 作为输入
            features = self.predictor.model(img_tensor)
        
        # 获取特征 - Baseline 模型返回的是 tuple (features, losses) 或 dict
        if isinstance(features, tuple):
            feature = features[0]
        elif isinstance(features, dict):
            feature = features.get('features', list(features.values())[0])
        elif isinstance(features, list):
            feature = features[0]
        else:
            feature = features
        
        # 归一化
        feature = F.normalize(feature, p=2, dim=1)
        
        # 转换为 numpy
        feature = feature.cpu().numpy().flatten()
        
        return feature
    
    def extract_features_batch(self, image_paths, batch_size=32):
        """
        批量提取特征
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            
        Returns:
            features: 特征矩阵 (N, 512)
            valid_indices: 有效图片的索引
        """
        features = []
        valid_indices = []
        
        for i, img_path in enumerate(image_paths):
            feature = self.extract_feature(img_path)
            if feature is not None:
                features.append(feature)
                valid_indices.append(i)
        
        return np.array(features), valid_indices


def compute_similarity(query_features, gallery_features):
    """
    计算查询图片和库图片之间的相似度矩阵
    
    Args:
        query_features: 查询特征 (N, 512)
        gallery_features: 库特征 (M, 512)
        
    Returns:
        similarity_matrix: 相似度矩阵 (N, M)
    """
    # 使用余弦相似度 (因为特征已经归一化，可以直接矩阵乘法)
    similarity_matrix = query_features @ gallery_features.T
    return similarity_matrix


def evaluate_retrieval(query_features, gallery_features, gallery_labels, top_k=[1, 5, 10]):
    """
    评估检索性能
    
    Args:
        query_features: 查询特征 (N, 512)
        gallery_features: 库特征 (M, 512)
        gallery_labels: 库图片标签 (M,)
        top_k: 需要评估的 k 值列表
        
    Returns:
        metrics: 评估指标字典
    """
    similarity_matrix = compute_similarity(query_features, gallery_features)
    
    # 获取排序后的索引
    sorted_indices = np.argsort(-similarity_matrix, axis=1)  # 降序排列
    
    # 计算准确率
    metrics = {}
    num_queries = query_features.shape[0]
    
    for k in top_k:
        correct = 0
        for i in range(num_queries):
            # 获取前 k 个结果的标签
            top_k_indices = sorted_indices[i, :k]
            top_k_labels = gallery_labels[top_k_indices]
            # 如果查询图片的标签在前 k 个结果中，则正确
            if query_labels[i] in top_k_labels:
                correct += 1
        
        accuracy = correct / num_queries * 100
        metrics[f'accuracy@{k}'] = accuracy
        print(f"Accuracy@{k}: {accuracy:.2f}%")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='使用 FastReID 进行宠物特征提取和相似度匹配')
    parser.add_argument('--config', type=str, default='test_fastreid_config.yml',
                        help='FastReID 配置文件路径')
    parser.add_argument('--query-dir', type=str, default='test_dataset/query',
                        help='查询图片目录')
    parser.add_argument('--gallery-dir', type=str, default='test_dataset/gallery',
                        help='库图片目录')
    parser.add_argument('--ground-truth', type=str, default='test_dataset/ground_truth.json',
                        help='真实标签文件')
    parser.add_argument('--output-dir', type=str, default='fastreid_results',
                        help='结果输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FastReID 宠物特征提取测试")
    print("=" * 60)
    
    # 初始化特征提取器
    print("\n[1/4] 初始化特征提取器...")
    extractor = PetFeatureExtractor(args.config)
    
    # 获取图片列表
    print("\n[2/4] 加载图片列表...")
    query_dir = Path(args.query_dir)
    gallery_dir = Path(args.gallery_dir)
    
    # 使用 rglob 递归搜索所有子目录中的图片
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    query_images = []
    gallery_images = []
    
    for ext in image_extensions:
        query_images.extend(query_dir.rglob(ext))
        gallery_images.extend(gallery_dir.rglob(ext))
    
    query_images = sorted(query_images)
    gallery_images = sorted(gallery_images)
    
    print(f"查询图片数量：{len(query_images)}")
    print(f"库图片数量：{len(gallery_images)}")
    
    # 加载真实标签
    if os.path.exists(args.ground_truth):
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"已加载真实标签：{len(ground_truth)} 条")
    else:
        print(f"警告：真实标签文件不存在 {args.ground_truth}")
        ground_truth = {}
    
    # 提取特征
    print("\n[3/4] 提取特征...")
    start_time = time.time()
    
    # 提取查询图片特征
    query_features_list = []
    query_valid_indices = []
    for i, img_path in enumerate(query_images):
        feature = extractor.extract_feature(str(img_path))
        if feature is not None:
            query_features_list.append(feature)
            query_valid_indices.append(i)
        if (i + 1) % 50 == 0:
            print(f"  查询图片进度：{i + 1}/{len(query_images)}")
    
    query_features = np.array(query_features_list)
    print(f"查询特征形状：{query_features.shape}")
    
    # 提取库图片特征
    gallery_features_list = []
    gallery_valid_indices = []
    for i, img_path in enumerate(gallery_images):
        feature = extractor.extract_feature(str(img_path))
        if feature is not None:
            gallery_features_list.append(feature)
            gallery_valid_indices.append(i)
        if (i + 1) % 50 == 0:
            print(f"  库图片进度：{i + 1}/{len(gallery_images)}")
    
    gallery_features = np.array(gallery_features_list)
    print(f"库特征形状：{gallery_features.shape}")
    
    extract_time = time.time() - start_time
    print(f"特征提取耗时：{extract_time:.2f} 秒")
    
    # 保存特征
    print("\n[4/4] 保存结果...")
    
    # 保存查询特征
    query_features_path = output_dir / 'query_features.npy'
    np.save(query_features_path, query_features)
    print(f"查询特征已保存：{query_features_path}")
    
    # 保存库特征
    gallery_features_path = output_dir / 'gallery_features.npy'
    np.save(gallery_features_path, gallery_features)
    print(f"库特征已保存：{gallery_features_path}")
    
    # 保存索引映射
    index_mapping = {
        'query_valid_indices': query_valid_indices,
        'gallery_valid_indices': gallery_valid_indices,
        'query_images': [str(p.name) for p in query_images],
        'gallery_images': [str(p.name) for p in gallery_images]
    }
    index_mapping_path = output_dir / 'index_mapping.json'
    with open(index_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(index_mapping, f, ensure_ascii=False, indent=2)
    print(f"索引映射已保存：{index_mapping_path}")
    
    # 计算相似度并评估
    print("\n计算相似度矩阵...")
    similarity_matrix = compute_similarity(query_features, gallery_features)
    print(f"相似度矩阵形状：{similarity_matrix.shape}")
    
    # 保存相似度矩阵
    similarity_path = output_dir / 'similarity_matrix.npy'
    np.save(similarity_path, similarity_matrix)
    print(f"相似度矩阵已保存：{similarity_path}")
    
    # 生成匹配结果
    print("\n生成匹配结果...")
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    results = []
    for i in range(len(query_features)):
        query_name = index_mapping['query_images'][query_valid_indices[i]]
        top_matches = []
        for j in range(min(10, len(gallery_features))):
            gallery_idx = sorted_indices[i, j]
            gallery_name = index_mapping['gallery_images'][gallery_valid_indices[gallery_idx]]
            similarity = similarity_matrix[i, gallery_idx]
            top_matches.append({
                'image': gallery_name,
                'similarity': float(similarity)
            })
        results.append({
            'query': query_name,
            'top_matches': top_matches
        })
    
    results_path = output_dir / 'matching_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"匹配结果已保存：{results_path}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"结果目录：{output_dir.absolute()}")
    print(f"总耗时：{extract_time:.2f} 秒")


if __name__ == '__main__':
    main()