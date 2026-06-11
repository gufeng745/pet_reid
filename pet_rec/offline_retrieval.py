#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线宠物检索系统

功能:
1. 提取宠物图片的 512 维特征向量
2. 计算图片间的相似度
3. 支持批量提取和对比
4. 支持文件夹级别的特征提取和对比

依赖:
    pip install onnxruntime opencv-python numpy
"""

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class OfflinePetRetrieval:
    """离线宠物检索系统"""
    
    def __init__(self, model_path=None, use_int8=False, device='cpu'):
        """
        初始化检索系统
        
        Args:
            model_path: ONNX 模型路径，默认使用 pet_mobilenetv2.onnx
            use_int8: 是否使用 INT8 量化模型
            device: 运行设备 ('cpu' 或 'gpu')
        """
        if model_path is None:
            # 默认模型路径
            if use_int8:
                model_path = 'pet_mobilenetv2_attr.onnx'
            else:
                model_path = 'pet_mobilenetv2_attr.onnx'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        # 创建 ONNX 推理会话
        providers = ['CPUExecutionProvider']
        if device.lower() == 'gpu':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_image(self, image, size=224):
        """
        预处理单张图像
        
        Args:
            image: OpenCV 读取的 BGR 图像
            size: 目标尺寸
        
        Returns:
            预处理后的图像张量
        """
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 调整大小
        image = cv2.resize(image, (size, size))
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        # 添加 batch 维度
        image = np.expand_dims(image, 0)
        return image
    
    def extract_features_single(self, image_input, size=224):
        """
        提取单张图片的特征（单视角）
        
        Args:
            image_input: 图片路径 (str) 或已加载的图片 (numpy array)
            size: 输入尺寸
        
        Returns:
            512 维特征向量
        """
        # 判断输入是路径还是已加载的图片
        if isinstance(image_input, (str, pathlib.Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"无法读取图片：{image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError(f"不支持的输入类型：{type(image_input)}")
        
        input_tensor = self.preprocess_image(image, size)
        features = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        return features[0]
    
    def extract_features(self, image_path, use_tta=True):
        """
        提取图片特征（支持 TTA 多视角平均）
        
        TTA 策略：
        - 中心裁剪
        - 四角裁剪
        - 上述 5 种裁剪 × 2(原图 + 水平翻转) = 10 个视角
        
        Args:
            image_path: 图片路径
            use_tta: 是否使用测试时增强
        
        Returns:
            512 维 L2 归一化特征向量
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")
        
        h, w = image.shape[:2]
        target_size = 256
        crop_size = 224
        
        # 计算缩放比例
        scale = target_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 5 种裁剪位置：中心 + 四角
        crops = []
        margin_h = new_h - crop_size
        margin_w = new_w - crop_size
        
        positions = [
            (0, 0),  # 左上
            (margin_w, 0),  # 右上
            (0, margin_h),  # 左下
            (margin_w, margin_h),  # 右下
            (margin_w // 2, margin_h // 2),  # 中心
        ]
        
        features_list = []
        for y, x in positions:
            # 原图裁剪
            crop = resized[y:y+crop_size, x:x+crop_size]
            features_list.append(self.extract_features_single(crop, crop_size))
            
            # 水平翻转裁剪
            if use_tta:
                crop_flipped = cv2.flip(crop, 1)
                features_list.append(self.extract_features_single(crop_flipped, crop_size))
        
        # 平均特征
        features = np.mean(features_list, axis=0)
        
        # L2 归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def compute_similarity(self, feat1, feat2):
        """
        计算两个特征向量的余弦相似度
        
        由于特征已 L2 归一化，余弦相似度 = 点积
        
        Args:
            feat1: 512 维特征向量
            feat2: 512 维特征向量
        
        Returns:
            相似度值 (0~1)
        """
        return np.dot(feat1, feat2)
    
    def batch_extract(self, image_paths, use_tta=True, show_progress=True):
        """
        批量提取特征
        
        Args:
            image_paths: 图片路径列表
            use_tta: 是否使用 TTA
            show_progress: 是否显示进度
        
        Returns:
            dict: {图片路径：特征向量}
        """
        features = {}
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if show_progress:
                print(f"\r提取特征：{i+1}/{total} ({path.name})", end='')
            
            try:
                features[str(path)] = self.extract_features(path, use_tta)
            except Exception as e:
                print(f"\n警告：无法处理 {path}: {e}")
        
        if show_progress:
            print()
        
        return features
    
    def find_similar(self, query_path, gallery_paths, top_k=5, use_tta=True):
        """
        在图库中查找最相似的图片
        
        Args:
            query_path: 查询图片路径
            gallery_paths: 图库图片路径列表
            top_k: 返回最相似的 k 张图片
            use_tta: 是否使用 TTA
        
        Returns:
            list: [(图片路径，相似度), ...]
        """
        # 提取查询图片特征
        query_feat = self.extract_features(query_path, use_tta)
        
        # 计算与所有图库图片的相似度
        similarities = []
        for path in gallery_paths:
            try:
                gallery_feat = self.extract_features(path, use_tta)
                sim = self.compute_similarity(query_feat, gallery_feat)
                similarities.append((str(path), sim))
            except Exception as e:
                print(f"警告：无法处理 {path}: {e}")
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def compare_all(self, image_paths, use_tta=True):
        """
        计算所有图片的两两相似度
        
        Args:
            image_paths: 图片路径列表
            use_tta: 是否使用 TTA
        
        Returns:
            features: dict {路径：特征向量}
            similarity_matrix: 相似度矩阵
        """
        # 提取所有特征
        features = self.batch_extract(image_paths, use_tta)
        paths = list(features.keys())
        n = len(paths)
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = self.compute_similarity(features[paths[i]], features[paths[j]])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return features, similarity_matrix, paths
    
    def get_top_k_similar(self, features, paths, similarity_matrix, top_k=5):
        """
        获取每张图片的 top_k 相似图片
        
        Args:
            features: 特征字典
            paths: 路径列表
            similarity_matrix: 相似度矩阵
            top_k: 返回最相似的 k 张图片
        
        Returns:
            dict: {图片路径：[(相似图片路径，相似度), ...]}
        """
        results = {}
        n = len(paths)
        
        for i, path in enumerate(paths):
            # 获取该图片与其他图片的相似度
            sims = similarity_matrix[i]
            # 获取 top_k 索引（排除自己）
            top_indices = np.argsort(sims)[::-1][1:top_k+1]
            
            similar_list = []
            for idx in top_indices:
                if idx != i:
                    similar_list.append((paths[idx], float(sims[idx])))
            
            results[path] = similar_list
        
        return results
    
    def get_image_paths(self, folder_path, recursive=True):
        """
        获取文件夹下所有图片路径
        
        Args:
            folder_path: 文件夹路径
            recursive: 是否递归子文件夹
        
        Returns:
            list: 图片路径列表
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在：{folder_path}")
        
        image_paths = []
        
        if recursive:
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(folder_path.rglob(f'*{ext}'))
                image_paths.extend(folder_path.rglob(f'*{ext.upper()}'))
        else:
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(folder_path.glob(f'*{ext}'))
                image_paths.extend(folder_path.glob(f'*{ext.upper()}'))
        
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        
        return image_paths
    
    def extract_folder(self, folder_path, output_file=None, use_tta=True, recursive=True):
        """
        提取文件夹下所有图片的特征向量
        
        Args:
            folder_path: 图片文件夹路径
            output_file: 输出文件路径 (.npy 或 .json)，None 则不保存
            use_tta: 是否使用 TTA
            recursive: 是否递归子文件夹
        
        Returns:
            dict: {图片路径：特征向量}
        """
        # 获取所有图片路径
        image_paths = self.get_image_paths(folder_path, recursive)
        
        if not image_paths:
            print(f"警告：在 {folder_path} 中未找到图片")
            return {}
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 提取特征
        features = self.batch_extract(image_paths, use_tta)
        
        # 保存结果
        if output_file:
            self.save_features(features, output_file)
        
        return features
    
    def compare_folder(self, folder_path, top_k=5, output_file=None, 
                       use_tta=True, recursive=True):
        """
        提取文件夹下所有图片特征并两两对比
        
        Args:
            folder_path: 图片文件夹路径
            top_k: 每张图片返回最相似的 k 张图片
            output_file: 输出文件路径 (.csv 或 .json)，None 则不保存
            use_tta: 是否使用 TTA
            recursive: 是否递归子文件夹
        
        Returns:
            features: 特征字典
            top_k_results: top_k 相似结果
            similarity_matrix: 相似度矩阵
        """
        # 获取所有图片路径
        image_paths = self.get_image_paths(folder_path, recursive)
        
        if len(image_paths) < 2:
            print(f"警告：需要至少 2 张图片进行对比")
            return {}, {}, None
        
        print(f"找到 {len(image_paths)} 张图片，开始对比...")
        
        # 计算相似度
        features, similarity_matrix, paths = self.compare_all(image_paths, use_tta)
        
        # 获取 top_k 结果
        top_k_results = self.get_top_k_similar(features, paths, similarity_matrix, top_k)
        
        # 保存结果
        if output_file:
            self.save_comparison(top_k_results, output_file)
        
        return features, top_k_results, similarity_matrix
    
    def compare_two_folders(self, folder1, folder2, top_k=5, output_file=None,
                            use_tta=True, recursive=True):
        """
        对比两个文件夹的图片
        
        Args:
            folder1: 第一个文件夹路径
            folder2: 第二个文件夹路径
            top_k: 每张图片返回最相似的 k 张图片
            output_file: 输出文件路径，None 则不保存
            use_tta: 是否使用 TTA
            recursive: 是否递归子文件夹
        
        Returns:
            dict: {folder1 图片路径：[(folder2 图片路径，相似度), ...]}
        """
        # 获取两个文件夹的图片路径
        paths1 = self.get_image_paths(folder1, recursive)
        paths2 = self.get_image_paths(folder2, recursive)
        
        if not paths1 or not paths2:
            print("警告：至少一个文件夹中没有找到图片")
            return {}
        
        print(f"文件夹 1: {len(paths1)} 张图片")
        print(f"文件夹 2: {len(paths2)} 张图片")
        
        # 提取特征
        features1 = self.batch_extract(paths1, use_tta)
        features2 = self.batch_extract(paths2, use_tta)
        
        # 对比
        results = {}
        for path1, feat1 in features1.items():
            similarities = []
            for path2, feat2 in features2.items():
                sim = self.compute_similarity(feat1, feat2)
                similarities.append((path2, sim))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            results[path1] = similarities[:top_k]
        
        # 保存结果
        if output_file:
            self.save_comparison(results, output_file)
        
        return results
    
    def save_features(self, features, output_file):
        """
        保存特征向量
        
        Args:
            features: 特征字典
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        
        if output_path.suffix == '.npy':
            # 保存为 numpy 格式
            paths = list(features.keys())
            feature_array = np.array([features[p] for p in paths])
            np.savez(output_path, paths=paths, features=feature_array)
            print(f"特征已保存至：{output_file}")
        
        elif output_path.suffix == '.json':
            # 保存为 JSON 格式（特征转为列表）
            features_list = {k: v.tolist() for k, v in features.items()}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features_list, f, ensure_ascii=False, indent=2)
            print(f"特征已保存至：{output_file}")
        
        else:
            raise ValueError("不支持的文件格式，请使用 .npy 或 .json")
    
    def save_comparison(self, results, output_file):
        """
        保存对比结果
        
        Args:
            results: 对比结果字典
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        
        if output_path.suffix == '.csv':
            # 保存为 CSV 格式
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'rank', 'similar_image', 'similarity'])
                
                for image, similar_list in results.items():
                    for rank, (similar_img, sim) in enumerate(similar_list, 1):
                        writer.writerow([image, rank, similar_img, f"{sim:.4f}"])
            print(f"对比结果已保存至：{output_file}")
        
        elif output_path.suffix == '.json':
            # 保存为 JSON 格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"对比结果已保存至：{output_file}")
        
        else:
            raise ValueError("不支持的文件格式，请使用 .csv 或 .json")
    
    def print_top_k_results(self, results, max_display=10):
        """
        打印 top_k 相似结果
        
        Args:
            results: 对比结果字典
            max_display: 最多显示多少张图片的结果
        """
        print("\n" + "="*60)
        print("相似度对比结果")
        print("="*60)
        
        count = 0
        for image, similar_list in results.items():
            if count >= max_display:
                break
            
            # 显示文件名而非完整路径
            img_name = Path(image).name
            print(f"\n{img_name}:")
            
            for rank, (similar_img, sim) in enumerate(similar_list, 1):
                sim_name = Path(similar_img).name
                bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
                print(f"  {rank}. {sim_name}: {sim:.4f} {bar}")
            
            count += 1
        
        if len(results) > max_display:
            print(f"\n... 还有 {len(results) - max_display} 张图片的结果")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='离线宠物检索系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 提取单张图片特征
  python offline_retrieval.py --extract data/1.png
  
  # 在图库中检索相似图片
  python offline_retrieval.py --query data/1.png --gallery data_cat --top_k 5
  
  # 提取文件夹下所有图片的特征
  python offline_retrieval.py --folder data_cat --output features.npy
  
  # 提取并对比文件夹内所有图片
  python offline_retrieval.py --folder data_cat --compare --top_k 5 --output results.csv
  
  # 对比两个文件夹
  python offline_retrieval.py --folder1 data_cat --folder2 data_offline/test_imgs --output comparison.csv
        """
    )
    
    # 模型选项
    parser.add_argument('--model', type=str, default=None,
                        help='ONNX 模型路径 (默认：pet_mobilenetv2.onnx)')
    parser.add_argument('--int8', action='store_true',
                        help='使用 INT8 量化模型')
    
    # 功能选项
    parser.add_argument('--extract', type=str, default=None,
                        help='提取单张图片特征并打印维度')
    parser.add_argument('--query', type=str, default=None,
                        help='查询图片路径')
    parser.add_argument('--gallery', type=str, default=None,
                        help='图库文件夹路径')
    parser.add_argument('--folder', type=str, default=None,
                        help='提取/对比的文件夹路径')
    parser.add_argument('--folder1', type=str, default=None,
                        help='第一个文件夹路径（用于两文件夹对比）')
    parser.add_argument('--folder2', type=str, default=None,
                        help='第二个文件夹路径（用于两文件夹对比）')
    parser.add_argument('--compare', action='store_true',
                        help='启用文件夹内对比模式')
    
    # 输出选项
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--top_k', '-k', type=int, default=5,
                        help='返回最相似的 k 张图片 (默认：5)')
    
    # 其他选项
    parser.add_argument('--no-tta', action='store_true',
                        help='禁用测试时增强 (TTA)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='不递归子文件夹')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    # 创建检索系统
    try:
        retriever = OfflinePetRetrieval(
            model_path=args.model,
            use_int8=args.int8
        )
        print(f"模型加载成功：{args.model or ('pet_mobilenetv2_int8.onnx' if args.int8 else 'pet_mobilenetv2.onnx')}")
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)
    
    use_tta = not args.no_tta
    recursive = not args.no_recursive
    
    # 提取单张图片特征
    if args.extract:
        feat = retriever.extract_features(args.extract, use_tta)
        print(f"\n特征维度：{feat.shape}")
        print(f"L2 范数：{np.linalg.norm(feat):.6f}")
        if args.verbose:
            print(f"前 10 个值：{feat[:10]}")
    
    # 查询 + 图库检索
    elif args.query and args.gallery:
        gallery_paths = retriever.get_image_paths(args.gallery, recursive)
        print(f"图库中共有 {len(gallery_paths)} 张图片")
        
        results = retriever.find_similar(
            args.query, gallery_paths, 
            top_k=args.top_k, use_tta=use_tta
        )
        
        print(f"\n查询图片：{Path(args.query).name}")
        print(f"最相似的 {args.top_k} 张图片:")
        for rank, (path, sim) in enumerate(results, 1):
            print(f"  {rank}. {Path(path).name}: {sim:.4f}")
    
    # 文件夹提取/对比
    elif args.folder:
        if args.compare:
            # 文件夹内对比
            features, results, matrix = retriever.compare_folder(
                args.folder, 
                top_k=args.top_k,
                output_file=args.output,
                use_tta=use_tta,
                recursive=recursive
            )
            retriever.print_top_k_results(results)
        else:
            # 仅提取特征
            features = retriever.extract_folder(
                args.folder,
                output_file=args.output,
                use_tta=use_tta,
                recursive=recursive
            )
            print(f"\n共提取 {len(features)} 张图片的特征")
    
    # 两文件夹对比
    elif args.folder1 and args.folder2:
        results = retriever.compare_two_folders(
            args.folder1, args.folder2,
            top_k=args.top_k,
            output_file=args.output,
            use_tta=use_tta,
            recursive=recursive
        )
        retriever.print_top_k_results(results)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()