"""
测试数据集生成脚本
从现有的 data_cat 目录创建标准的测试数据集结构
"""

import os
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def get_image_files(folder: str) -> List[str]:
    """获取文件夹中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    files = []
    for f in os.listdir(folder):
        if Path(f).suffix.lower() in image_extensions:
            files.append(f)
    return sorted(files)


def group_images_by_class(source_dir: str) -> Dict[str, List[str]]:
    """
    从源目录按类别分组图片
    支持两种格式:
    1. data_cat/1/img.png (子目录格式)
    2. data_cat/1-img.png (文件名格式，如 1-1.png)
    
    Returns: {class_id: [(relative_path, full_path), ...]}
    """
    class_images = {}
    
    # 遍历源目录
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        # 情况 1: 子目录 (如 data_cat/1/)
        if os.path.isdir(item_path) and item.isdigit():
            class_id = item
            for img in get_image_files(item_path):
                full_path = os.path.join(item_path, img)
                rel_path = f"{item}/{img}"
                if class_id not in class_images:
                    class_images[class_id] = []
                class_images[class_id].append((rel_path, full_path))
        
        # 情况 2: 文件名格式 (如 data_cat/2-1.png)
        elif Path(item).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
            # 尝试解析 class_id-filename 格式
            name_without_ext = Path(item).stem  # 如 "2-1"
            parts = name_without_ext.split('-')
            if len(parts) >= 2 and parts[0].isdigit():
                class_id = parts[0]
                full_path = item_path
                rel_path = item
                if class_id not in class_images:
                    class_images[class_id] = []
                class_images[class_id].append((rel_path, full_path))
    
    return class_images


def create_test_dataset(
    source_dir: str = "data_cat",
    output_dir: str = "test_dataset",
    gallery_ratio: float = 0.7,
    seed: int = 42
):
    """
    创建测试数据集
    
    Args:
        source_dir: 源数据目录 (data_cat)
        output_dir: 输出目录 (test_dataset)
        gallery_ratio: 放入 gallery 的图片比例 (0.7 = 70% gallery, 30% query)
        seed: 随机种子
    """
    random.seed(seed)
    
    # 创建输出目录结构
    gallery_dir = os.path.join(output_dir, "gallery")
    query_dir = os.path.join(output_dir, "query")
    
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    
    print(f"创建测试数据集：{output_dir}")
    print(f"源目录：{source_dir}")
    print(f"Gallery 比例：{gallery_ratio * 100:.0f}%")
    print("-" * 50)
    
    # 存储标注信息
    annotations = {
        "dataset_info": {
            "name": "pet_recognition_test",
            "version": "1.0",
            "source": source_dir
        },
        "gallery": [],
        "query": [],
        "pet_registry": {}
    }
    
    # 简化版 ground truth 列表
    ground_truth_list = []
    
    # 按类别分组图片
    class_images = group_images_by_class(source_dir)
    class_folders = sorted(class_images.keys(), key=int)
    
    total_gallery = 0
    total_query = 0
    query_counter = 0
    
    for class_id in class_folders:
        images_list = class_images[class_id]
        images = [rel_path for rel_path, _ in images_list]
        
        if len(images) < 2:
            print(f"类别 {class_id}: 图片数量不足 ({len(images)})，跳过")
            continue
        
        # 打乱图片顺序
        random.shuffle(images)
        
        # 分割 gallery 和 query
        split_idx = int(len(images) * gallery_ratio)
        gallery_images = images[:split_idx]
        query_images = images[split_idx:]
        
        if len(query_images) == 0:
            # 至少保留一张作为 query
            query_images = [gallery_images[-1]]
            gallery_images = gallery_images[:-1]
        
        if len(gallery_images) == 0:
            print(f"类别 {class_id}: 无法分配 gallery 图片，跳过")
            continue
        
        # 创建 pet_id (使用 class_id 作为宠物 ID)
        pet_id = f"pet_{class_id}"
        
        # 创建 gallery 目录
        pet_gallery_dir = os.path.join(gallery_dir, pet_id)
        os.makedirs(pet_gallery_dir, exist_ok=True)
        
        gallery_count = 0
        for rel_path in gallery_images:
            # 获取完整路径
            if os.path.exists(rel_path):
                src = rel_path
            else:
                src = os.path.join(source_dir, rel_path)
            
            # 获取原始文件的扩展名
            ext = Path(rel_path).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                ext = '.png'  # 默认扩展名
            
            # 重命名为 {类别 ID}-{序号}.{扩展名} 格式
            new_filename = f"{class_id}-{gallery_count + 1}{ext}"
            dst = os.path.join(pet_gallery_dir, new_filename)
            shutil.copy2(src, dst)
            
            # 记录 gallery 信息
            annotations["gallery"].append({
                "id": f"gallery_{pet_id}_{gallery_count:03d}",
                "image_path": f"gallery/{pet_id}/{new_filename}",
                "pet_id": pet_id
            })
            gallery_count += 1
            total_gallery += 1
        
        # 记录宠物信息
        annotations["pet_registry"][pet_id] = {
            "class_id": class_id,
            "gallery_count": gallery_count
        }
        
        # 复制 query 图片
        query_idx = 0
        for rel_path in query_images:
            # 获取完整路径
            if os.path.exists(rel_path):
                src = rel_path
            else:
                src = os.path.join(source_dir, rel_path)
            
            # 获取原始文件的扩展名
            ext = Path(rel_path).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                ext = '.png'  # 默认扩展名
            
            # 重命名为 {类别 ID}-{序号}.{扩展名} 格式
            new_filename = f"{class_id}-{query_idx + 1}{ext}"
            dst = os.path.join(query_dir, new_filename)
            shutil.copy2(src, dst)
            
            query_counter += 1
            total_query += 1
            query_idx += 1
            
            # 记录 query 信息
            annotations["query"].append({
                "id": f"query_{query_counter:04d}",
                "image_path": f"query/{new_filename}",
                "ground_truth_pet_id": pet_id
            })
            
            # 简化版 ground truth
            ground_truth_list.append(f"query/{new_filename},{pet_id}")
        
        print(f"类别 {class_id}: {len(gallery_images)} gallery, {len(query_images)} query")
    
    # 保存 annotations.json
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"\n保存：{annotations_path}")
    
    # 保存 ground_truth.txt
    ground_truth_path = os.path.join(output_dir, "ground_truth.txt")
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        f.write("# 格式：query_image_path,pet_id\n")
        for line in ground_truth_list:
            f.write(line + "\n")
    print(f"保存：{ground_truth_path}")
    
    # 保存 gallery_list.txt (可选，加速加载)
    gallery_list_path = os.path.join(output_dir, "gallery_list.txt")
    with open(gallery_list_path, 'w', encoding='utf-8') as f:
        for item in annotations["gallery"]:
            f.write(f"{item['image_path']},{item['pet_id']}\n")
    print(f"保存：{gallery_list_path}")
    
    # 保存 query_list.txt
    query_list_path = os.path.join(output_dir, "query_list.txt")
    with open(query_list_path, 'w', encoding='utf-8') as f:
        for item in annotations["query"]:
            f.write(f"{item['image_path']},{item['ground_truth_pet_id']}\n")
    print(f"保存：{query_list_path}")
    
    print("\n" + "=" * 50)
    print(f"测试数据集创建完成!")
    print(f"Gallery 图片总数：{total_gallery}")
    print(f"Query 图片总数：{total_query}")
    print(f"宠物类别数：{len(annotations['pet_registry'])}")
    print("=" * 50)
    
    return annotations


def create_test_dataset_from_files(
    image_dir: str = "data_cat",
    output_dir: str = "test_dataset_manual",
    gallery_files: List[str] = None,
    query_files: List[str] = None
):
    """
    手动指定文件创建测试数据集
    
    Args:
        image_dir: 图片所在目录
        output_dir: 输出目录
        gallery_files: 作为 gallery 的文件列表 (相对于 image_dir)
        query_files: 作为 query 的文件列表 (相对于 image_dir)
    """
    os.makedirs(os.path.join(output_dir, "gallery"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "query"), exist_ok=True)
    
    annotations = {
        "dataset_info": {
            "name": "pet_recognition_test_manual",
            "version": "1.0"
        },
        "gallery": [],
        "query": [],
        "pet_registry": {}
    }
    
    ground_truth_list = []
    
    # 处理 gallery 文件
    if gallery_files:
        for rel_path in gallery_files:
            # 解析路径：class_id/filename -> pet_id
            parts = Path(rel_path).parts
            if len(parts) >= 2:
                class_id = parts[0]
                pet_id = f"pet_{class_id}"
                
                # 创建目录
                pet_gallery_dir = os.path.join(output_dir, "gallery", pet_id)
                os.makedirs(pet_gallery_dir, exist_ok=True)
                
                # 复制文件
                src = os.path.join(image_dir, rel_path)
                dst = os.path.join(pet_gallery_dir, Path(rel_path).name)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    
                    annotations["gallery"].append({
                        "id": f"gallery_{pet_id}_{len(annotations['gallery']):03d}",
                        "image_path": f"gallery/{pet_id}/{Path(rel_path).name}",
                        "pet_id": pet_id
                    })
                    
                    if pet_id not in annotations["pet_registry"]:
                        annotations["pet_registry"][pet_id] = {"gallery_count": 0}
                    annotations["pet_registry"][pet_id]["gallery_count"] += 1
    
    # 处理 query 文件
    if query_files:
        for idx, rel_path in enumerate(query_files):
            parts = Path(rel_path).parts
            if len(parts) >= 2:
                class_id = parts[0]
                pet_id = f"pet_{class_id}"
                
                src = os.path.join(image_dir, rel_path)
                dst = os.path.join(output_dir, "query", f"q_{idx:04d}_{Path(rel_path).name}")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    
                    query_filename = Path(dst).name
                    annotations["query"].append({
                        "id": f"query_{idx:04d}",
                        "image_path": f"query/{query_filename}",
                        "ground_truth_pet_id": pet_id
                    })
                    ground_truth_list.append(f"query/{query_filename},{pet_id}")
    
    # 保存文件
    with open(os.path.join(output_dir, "annotations.json"), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "ground_truth.txt"), 'w', encoding='utf-8') as f:
        f.write("# 格式：query_image_path,pet_id\n")
        for line in ground_truth_list:
            f.write(line + "\n")
    
    print(f"手动测试数据集创建完成：{output_dir}")
    print(f"Gallery: {len(annotations['gallery'])}, Query: {len(annotations['query'])}")


if __name__ == "__main__":
    print("=" * 50)
    print("测试数据集生成工具")
    print("=" * 50)
    
    # 自动创建测试数据集
    create_test_dataset(
        source_dir="data_cat",
        output_dir="test_dataset",
        gallery_ratio=0.7,  # 70% 图片放入 gallery
        seed=42
    )
    
    print("\n")
    
    # 示例：手动指定文件创建测试数据集
    # create_test_dataset_from_files(
    #     image_dir="data_cat",
    #     output_dir="test_dataset_manual",
    #     gallery_files=["1/1-1.png", "1/1-2.png", "2/2-1.png"],
    #     query_files=["1/1-3.png", "2/2-2.png"]
    # )