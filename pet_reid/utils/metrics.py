"""
Re-ID 评估指标

包含：
- Rank-1/5/10 准确率
- mAP (mean Average Precision)
- CMC 曲线
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_reid_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray
) -> Dict[str, float]:
    """计算Re-ID指标

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

    # 对每个query计算排名
    rank_list = []
    ap_list = []

    for i in range(len(query_labels)):
        # 获取当前query的相似度和标签
        sim_scores = similarity[i]
        query_label = query_labels[i]

        # 按相似度降序排序
        sorted_indices = torch.argsort(sim_scores, descending=True)
        sorted_labels = gallery_labels[sorted_indices.numpy()]

        # 计算rank-k准确率
        matches = torch.tensor(sorted_labels == query_label)
        rank_list.append(matches)

        # 计算AP (Average Precision)
        true_positives = matches.float()
        cumulative_tp = torch.cumsum(true_positives, dim=0)
        precision_at_k = cumulative_tp / (torch.arange(len(true_positives)).float() + 1)
        ap = (precision_at_k * true_positives).sum() / max(true_positives.sum(), 1)
        ap_list.append(ap.item())

    # 计算Rank-k准确率
    rank_tensor = torch.stack(rank_list)  # (N_q, N_g)
    num_gallery = rank_tensor.shape[1]

    rank_1 = rank_tensor[:, :1].any(dim=1).float().mean().item()
    rank_5 = rank_tensor[:, :5].any(dim=1).float().mean().item()
    rank_10 = rank_tensor[:, :10].any(dim=1).float().mean().item()

    # 处理gallery数量不足的情况
    if num_gallery < 5:
        rank_5 = rank_1
    if num_gallery < 10:
        rank_10 = rank_5

    # 计算mAP
    mAP = np.mean(ap_list)

    return {
        'rank-1': rank_1,
        'rank-5': rank_5,
        'rank-10': rank_10,
        'mAP': mAP,
    }


def compute_cmc_curve(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 50
) -> np.ndarray:
    """计算CMC曲线

    Args:
        query_features: (N_q, D) 查询特征
        gallery_features: (N_g, D) 画廊特征
        query_labels: (N_q,) 查询标签
        gallery_labels: (N_g,) 画廊标签
        max_rank: 最大rank

    Returns:
        cmc: (max_rank,) CMC曲线
    """
    # 计算余弦相似度
    similarity = query_features @ gallery_features.T

    # 对每个query计算排名
    num_correct = np.zeros(max_rank)

    for i in range(len(query_labels)):
        sim_scores = similarity[i]
        query_label = query_labels[i]

        # 按相似度降序排序
        sorted_indices = torch.argsort(sim_scores, descending=True)
        sorted_labels = gallery_labels[sorted_indices.numpy()]

        # 找到正确匹配的位置
        matches = (sorted_labels == query_label)

        # 计算CMC
        for rank in range(max_rank):
            if rank < len(matches) and matches[:rank+1].any():
                num_correct[rank] += 1

    # 归一化
    cmc = num_correct / len(query_labels)

    return cmc


def compute_average_precision(
    similarity: torch.Tensor,
    query_label: str,
    gallery_labels: np.ndarray
) -> float:
    """计算单个query的Average Precision

    Args:
        similarity: (N_g,) 相似度分数
        query_label: 查询标签
        gallery_labels: (N_g,) 画廊标签

    Returns:
        ap: Average Precision
    """
    # 按相似度降序排序
    sorted_indices = torch.argsort(similarity, descending=True)
    sorted_labels = gallery_labels[sorted_indices.numpy()]

    # 计算precision和recall
    matches = (sorted_labels == query_label).float()
    cumulative_matches = torch.cumsum(matches, dim=0)
    precision = cumulative_matches / (torch.arange(len(matches)).float() + 1)

    # 计算AP
    ap = (precision * matches).sum() / max(matches.sum(), 1)

    return ap.item()


def evaluate_reid_model(
    model,
    query_loader,
    gallery_loader,
    device: torch.device
) -> Dict[str, float]:
    """评估Re-ID模型

    Args:
        model: Re-ID模型
        query_loader: 查询数据加载器
        gallery_loader: 画廊数据加载器
        device: 设备

    Returns:
        metrics: 评估指标
    """
    model.eval()

    # 提取查询特征
    query_features = []
    query_labels = []

    with torch.no_grad():
        for images, labels in query_loader:
            images = images.to(device)
            features = model.forward_emb(images)
            query_features.append(features.cpu())
            query_labels.extend(labels.numpy())

    query_features = torch.cat(query_features, dim=0)
    query_labels = np.array(query_labels)

    # 提取画廊特征
    gallery_features = []
    gallery_labels = []

    with torch.no_grad():
        for images, labels in gallery_loader:
            images = images.to(device)
            features = model.forward_emb(images)
            gallery_features.append(features.cpu())
            gallery_labels.extend(labels.numpy())

    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_labels = np.array(gallery_labels)

    # 计算指标
    metrics = compute_reid_metrics(
        query_features,
        gallery_features,
        query_labels,
        gallery_labels
    )

    return metrics
