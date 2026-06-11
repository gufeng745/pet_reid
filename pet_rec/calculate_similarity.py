#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征相似度计算程序
计算 data_offline 文件夹中所有 512 维特征向量之间的余弦相似度
"""

import os
import re
import numpy as np
from pathlib import Path


def parse_feature_file(filepath):
    """
    解析特征文件，提取 512 维特征向量
    
    文件格式示例:
    features Tensor:
    {
    tensor dim:2,  Original shape:[1 512]
    tensor data:
    -0.057314  -0.047426  0.069226  ...
    }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到 "tensor data:" 的位置，只提取其后的数据
    data_marker = "tensor data:"
    marker_pos = content.find(data_marker)
    if marker_pos == -1:
        raise ValueError(f"文件 {filepath} 中未找到 'tensor data:' 标记")
    
    # 提取 "tensor data:" 之后的内容
    data_content = content[marker_pos + len(data_marker):]
    
    # 找到结束标记 "}" 的位置
    end_pos = data_content.find('}')
    if end_pos != -1:
        data_content = data_content[:end_pos]
    
    # 使用正则表达式提取所有浮点数
    # 匹配正负浮点数，包括科学计数法
    pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    values = re.findall(pattern, data_content)
    
    # 转换为浮点数
    features = np.array([float(v) for v in values], dtype=np.float32)
    
    # 验证特征维度
    if len(features) != 512:
        print(f"警告：文件 {filepath} 的特征维度为 {len(features)}，期望 512")
        print(f"  实际提取了 {len(features)} 个值")
    
    return features


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
    
    Returns:
        余弦相似度值，范围 [-1, 1]
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算向量模长
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


def calculate_all_similarities(features_dict):
    """
    计算所有特征对之间的相似度
    
    Args:
        features_dict: 字典，键为文件名，值为特征向量
    
    Returns:
        相似度矩阵和文件名列表
    """
    filenames = list(features_dict.keys())
    n = len(filenames)
    
    # 创建相似度矩阵
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 计算每对特征的相似度
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0  # 自身相似度为 1
            elif j > i:
                # 只计算上三角，下三角对称
                sim = cosine_similarity(features_dict[filenames[i]], 
                                        features_dict[filenames[j]])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    
    return similarity_matrix, filenames


def find_top_similar_pairs(similarity_matrix, filenames, top_k=5):
    """
    找出相似度最高的 top_k 对特征
    
    Args:
        similarity_matrix: 相似度矩阵
        filenames: 文件名列表
        top_k: 返回前 k 对
    
    Returns:
        最相似的对列表，每项为 (文件名 1, 文件名 2, 相似度)
    """
    n = len(filenames)
    pairs = []
    
    # 遍历上三角矩阵
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((filenames[i], filenames[j], similarity_matrix[i, j]))
    
    # 按相似度降序排序
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return pairs[:top_k]


def print_similarity_matrix(similarity_matrix, filenames):
    """
    打印格式化的相似度矩阵
    """
    n = len(filenames)
    
    # 打印表头
    header = " " * 12
    for i, fname in enumerate(filenames):
        short_name = Path(fname).stem[:8]
        header += f"{short_name:>10}"
    print(header)
    print("-" * len(header))
    
    # 打印每一行
    for i, fname in enumerate(filenames):
        short_name = Path(fname).stem[:8]
        row = f"{short_name:>12}"
        for j in range(n):
            row += f"{similarity_matrix[i, j]:>10.4f}"
        print(row)


def main():
    # 特征文件目录
    feature_dir = Path(r"data_offline\test_imgs")
    
    # 检查目录是否存在
    if not feature_dir.exists():
        print(f"错误：目录 '{feature_dir}' 不存在")
        print("请确保将特征文件放在 data_offline 文件夹中")
        return
    
    # 获取所有 .txt 文件
    txt_files = list(feature_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"错误：在 '{feature_dir}' 中未找到 .txt 文件")
        return
    
    print(f"找到 {len(txt_files)} 个特征文件\n")
    
    # 读取所有特征
    features_dict = {}
    for filepath in txt_files:
        print(f"读取：{filepath.name}")
        features_dict[filepath.name] = parse_feature_file(filepath)
    
    print()
    
    # 计算相似度
    print("计算相似度矩阵...")
    similarity_matrix, filenames = calculate_all_similarities(features_dict)
    
    # 打印相似度矩阵
    print("\n" + "=" * 80)
    print("相似度矩阵 (Cosine Similarity):")
    print("=" * 80)
    print_similarity_matrix(similarity_matrix, filenames)
    
    # 找出最相似的对
    print("\n" + "=" * 80)
    print("最相似的 5 对特征:")
    print("=" * 80)
    top_pairs = find_top_similar_pairs(similarity_matrix, filenames, top_k=5)
    for rank, (file1, file2, sim) in enumerate(top_pairs, 1):
        print(f"{rank}. {file1} & {file2}: {sim:.4f}")
    
    # 找出不相似的对
    print("\n" + "=" * 80)
    print("最不相似的 5 对特征:")
    print("=" * 80)
    least_similar = find_top_similar_pairs(similarity_matrix, filenames, top_k=len(filenames)*(len(filenames)-1)//2)
    least_similar = least_similar[-5:]
    for rank, (file1, file2, sim) in enumerate(reversed(least_similar), 1):
        print(f"{rank}. {file1} & {file2}: {sim:.4f}")
    
    # 保存结果到文件
    output_file = "data_offline/similarity_results.csv"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        header = "file," + ",".join(filenames)
        f.write(header + "\n")
        # 写入数据
        for i, fname in enumerate(filenames):
            row = fname + "," + ",".join(f"{similarity_matrix[i, j]:.6f}" for j in range(len(filenames)))
            f.write(row + "\n")
    
    print(f"\n相似度矩阵已保存到：{output_file}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    # 计算平均相似度（不包括对角线）
    upper_triangular = similarity_matrix[np.triu_indices(len(filenames), k=1)]
    print(f"平均相似度：{np.mean(upper_triangular):.4f}")
    print(f"最大相似度：{np.max(upper_triangular):.4f}")
    print(f"最小相似度：{np.min(upper_triangular):.4f}")
    print(f"相似度标准差：{np.std(upper_triangular):.4f}")


if __name__ == "__main__":
    main()