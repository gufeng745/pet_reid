#!/usr/bin/env python3
"""
Linux 环境离线评估脚本 - 调用 simulate_offline.sh 生成特征并评估
适用于 SGS-IPU SDK 环境

使用方法:
    python3 run_txt_evaluation.py \\
        --image-dir ./test_dataset/ \\
        --script-path ./simulate_offline.sh \\
        --output-dir ./evaluation_results \\
        --top-k 10
"""

import argparse
import json
import numpy as np
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_txt_features(txt_file, feature_dim=512):
    """
    从 SGS-IPU SDK 输出的 txt 文件读取 512 维特征
    格式：包含多行浮点数，用空格分隔
    
    Args:
        txt_file: txt 文件路径
        feature_dim: 特征维度，默认 512
    
    Returns:
        numpy 数组，形状为 (feature_dim,)
    """
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取所有浮点数
    pattern = r'-?\d+\.\d+'
    values = re.findall(pattern, content)
    
    if len(values) < feature_dim:
        raise ValueError(f"Expected at least {feature_dim} features, got {len(values)} in {txt_file}")
    
    # 取前 feature_dim 个值
    features = np.array([float(v) for v in values[:feature_dim]], dtype=np.float32)
    return features


def run_simulate_offline(script_path, image_dir, output_dir, model_path=None, preprocess_path=None):
    """
    调用 simulate_offline.sh 脚本生成特征文件
    
    Args:
        script_path: simulate_offline.sh 脚本路径
        image_dir: 输入图片目录
        output_dir: 输出特征目录
        model_path: 模型路径（可选，覆盖脚本默认值）
        preprocess_path: 预处理脚本路径（可选）
    
    Returns:
        是否执行成功
    """
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False
    
    # 构建命令
    cmd = [
        "bash", str(script_path),
        "--image-dir", str(image_dir),
        "--output-dir", str(output_dir)
    ]
    
    if model_path:
        cmd.extend(["--model-path", model_path])
    if preprocess_path:
        cmd.extend(["--preprocess-path", preprocess_path])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 小时超时
        )
        
        if result.returncode != 0:
            print(f"Script failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False
        
        if result.stdout:
            print(result.stdout)
        return True
        
    except subprocess.TimeoutExpired:
        print("Error: Script execution timed out")
        return False
    except Exception as e:
        print(f"Error running script: {e}")
        return False


def run_simulate_offline_direct(simulator_path, image_dir, model_path, preprocess_path=None, 
                                 num_process=8, soc_version="388g"):
    """
    直接调用 simulator.py 生成特征文件（当 simulate_offline.sh 不支持参数时）
    
    simulator.py 会创建 ./log/output/ 目录存放特征文件
    
    Args:
        simulator_path: simulator.py 路径
        image_dir: 输入图片目录
        model_path: 模型路径
        preprocess_path: 预处理脚本路径
        num_process: 进程数
        soc_version: SoC 版本
    
    Returns:
        是否执行成功，以及特征输出目录路径
    """
    # 获取项目根目录（脚本所在目录）
    project_root = Path(__file__).parent.absolute()
    original_dir = os.getcwd()
    
    # 切换到项目根目录运行
    os.chdir(project_root)
    
    try:
        # 构建相对于项目根目录的路径
        rel_image_dir = Path(image_dir).absolute().relative_to(project_root)
        rel_model_file = Path(model_path).absolute().relative_to(project_root)
        rel_preprocess = Path(preprocess_path).absolute().relative_to(project_root) if preprocess_path else Path("preprocess_attr.py")
        
        # 构建 simulator.py 命令
        cmd = [
            "python3", str(Path(simulator_path).absolute()),
            "-i", str(rel_image_dir),
            "-m", str(rel_model_file),
            "-c", "Unknown",
            "-t", "Offline",
            "-n", str(rel_preprocess),
            "--num_process", str(num_process),
            "--soc_version", soc_version
        ]
        
        print(f"Running simulator from project root: {project_root}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(project_root)  # 在项目根目录执行
        )
        
        if result.returncode != 0:
            print(f"Simulator failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            return False, None
        
        if result.stdout:
            print(result.stdout)
        
        # simulator.py 输出到 ./log/output/ 目录
        feature_output_dir = project_root / "log" / "output"
        return True, feature_output_dir
        
    except subprocess.TimeoutExpired:
        print("Error: Simulator execution timed out")
        return False, None
    except Exception as e:
        print(f"Error running simulator: {e}")
        return False, None
    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)


def load_dataset_lists(dataset_dir):
    """
    加载 test_dataset 中的 gallery 和 query 列表
    
    Args:
        dataset_dir: test_dataset 目录路径
    
    Returns:
        gallery_list: [(relative_path, pet_id), ...]
        query_list: [(relative_path, pet_id), ...]
    """
    dataset_dir = Path(dataset_dir)
    
    # 加载 gallery 列表
    gallery_file = dataset_dir / "gallery_list.txt"
    gallery_list = []
    if gallery_file.exists():
        with open(gallery_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        path = parts[0].strip()
                        pet_id = parts[1].strip()
                        gallery_list.append((path, pet_id))
    
    # 加载 query 列表
    query_file = dataset_dir / "query_list.txt"
    query_list = []
    if query_file.exists():
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        path = parts[0].strip()
                        pet_id = parts[1].strip()
                        query_list.append((path, pet_id))
    
    print(f"Loaded {len(gallery_list)} gallery images and {len(query_list)} query images")
    return gallery_list, query_list


def find_txt_feature(txt_dir, image_name):
    """
    在 txt 特征目录中查找对应图片的特征文件
    
    SGS-IPU SDK 输出的文件名格式：unknown_{model_name}_{image_name}.txt
    
    Args:
        txt_dir: txt 特征目录
        image_name: 图片文件名（如 2-1.png）
    
    Returns:
        txt 文件路径，如果未找到返回 None
    """
    txt_dir = Path(txt_dir)
    image_base = Path(image_name).stem  # 去掉扩展名
    
    # 尝试多种可能的文件名模式
    patterns = [
        f"unknown_*_{image_base}.txt",
        f"unknown_*_{image_name}.txt",
        f"*{image_base}.txt",
        f"*{image_name}.txt",
    ]
    
    for pattern in patterns:
        matches = list(txt_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def compute_similarity(query_feat, gallery_features):
    """
    计算查询特征与所有 gallery 特征的余弦相似度
    
    Args:
        query_feat: 查询特征，形状 (feature_dim,)
        gallery_features: gallery 特征，形状 (num_gallery, feature_dim)
    
    Returns:
        相似度数组，形状 (num_gallery,)
    """
    # L2 归一化
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
    
    # 余弦相似度
    similarities = np.dot(gallery_norm, query_norm)
    return similarities


def run_evaluation(txt_dir, dataset_dir, output_dir, top_k=10, feature_dim=512):
    """
    运行评估
    
    Args:
        txt_dir: txt 特征文件目录（SGS-IPU SDK 输出）
        dataset_dir: test_dataset 目录
        output_dir: 结果输出目录
        top_k: 返回前 K 个匹配结果
        feature_dim: 特征维度
    """
    txt_dir = Path(txt_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集列表
    gallery_list, query_list = load_dataset_lists(dataset_dir)
    
    if len(query_list) == 0:
        print(f"Error: No query images found in {dataset_dir}/query_list.txt")
        sys.exit(1)
    
    if len(gallery_list) == 0:
        print(f"Error: No gallery images found in {dataset_dir}/gallery_list.txt")
        sys.exit(1)
    
    # 检查 txt 特征目录
    if not txt_dir.exists():
        print(f"Error: TXT feature directory does not exist: {txt_dir}")
        sys.exit(1)
    
    txt_files = list(txt_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} txt feature files in {txt_dir}")
    
    if len(txt_files) == 0:
        print(f"Error: No txt feature files found in {txt_dir}")
        sys.exit(1)
    
    # 加载 gallery 特征
    print("Loading gallery features...")
    gallery_features = []
    gallery_names = []
    gallery_pet_ids = []
    
    for rel_path, pet_id in gallery_list:
        # 提取图片文件名
        image_name = Path(rel_path).name
        
        # 查找对应的 txt 文件
        txt_file = find_txt_feature(txt_dir, image_name)
        
        if txt_file is None:
            print(f"Warning: No txt feature found for gallery image: {image_name}")
            continue
        
        try:
            feat = parse_txt_features(txt_file, feature_dim)
            gallery_features.append(feat)
            gallery_names.append(image_name)
            gallery_pet_ids.append(pet_id)
        except Exception as e:
            print(f"Warning: Failed to load {txt_file}: {e}")
    
    if len(gallery_features) == 0:
        print("Error: No valid gallery features loaded!")
        sys.exit(1)
    
    gallery_features = np.array(gallery_features, dtype=np.float32)
    print(f"Gallery features shape: {gallery_features.shape}")
    
    # 计算相似度并评估
    print("Computing similarities and evaluating...")
    results = []
    all_ap_scores = []
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    
    for i, (rel_path, query_pet_id) in enumerate(query_list):
        image_name = Path(rel_path).name
        
        # 查找对应的 txt 文件
        txt_file = find_txt_feature(txt_dir, image_name)
        
        if txt_file is None:
            print(f"Warning: No txt feature found for query image: {image_name}")
            continue
        
        try:
            query_feat = parse_txt_features(txt_file, feature_dim)
            similarities = compute_similarity(query_feat, gallery_features)
            
            # 获取 top_k 结果（按相似度降序）
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            top_k_names = [gallery_names[idx] for idx in top_indices]
            top_k_pet_ids = [gallery_pet_ids[idx] for idx in top_indices]
            top_k_scores = [float(similarities[idx]) for idx in top_indices]
            
            result = {
                "query": image_name,
                "query_pet_id": query_pet_id,
                "rank1": top_k_names[0] if top_k_names else None,
                "rank1_pet_id": top_k_pet_ids[0] if top_k_pet_ids else None,
                "top_k": top_k_names,
                "top_k_pet_ids": top_k_pet_ids,
                "top_k_scores": top_k_scores
            }
            results.append(result)
            
            # 基于 pet_id 计算准确率
            # Rank-1 准确率
            if top_k_pet_ids and top_k_pet_ids[0] == query_pet_id:
                rank1_correct += 1
            
            # Rank-5 准确率
            if len(top_k_pet_ids) >= 5 and query_pet_id in top_k_pet_ids[:5]:
                rank5_correct += 1
            
            # Rank-10 准确率
            if len(top_k_pet_ids) >= 10 and query_pet_id in top_k_pet_ids[:10]:
                rank10_correct += 1
            
            # 计算 AP (Average Precision)
            # 查找所有匹配项的位置
            match_positions = []
            for rank, pet_id in enumerate(top_k_pet_ids):
                if pet_id == query_pet_id:
                    match_positions.append(rank + 1)
            
            if match_positions:
                ap = sum((rank + 1) / pos for rank, pos in enumerate(match_positions)) / len(match_positions)
                all_ap_scores.append(ap)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(query_list)} queries")
                
        except Exception as e:
            print(f"Warning: Failed to process {txt_file}: {e}")
            continue
    
    if not results:
        print("Error: No valid results generated!")
        sys.exit(1)
    
    # 保存详细结果
    results_file = output_dir / "matching_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 计算评估指标
    mAP = np.mean(all_ap_scores) if all_ap_scores else 0.0
    
    num_queries = len(results)
    rank1_acc = rank1_correct / num_queries if num_queries > 0 else 0.0
    rank5_acc = rank5_correct / num_queries if num_queries > 0 else 0.0
    rank10_acc = rank10_correct / num_queries if num_queries > 0 else 0.0
    
    # 保存评估报告
    report = {
        "num_queries": num_queries,
        "num_gallery": len(gallery_features),
        "feature_dim": feature_dim,
        "top_k": top_k,
        "mAP": float(mAP),
        "rank1_accuracy": float(rank1_acc),
        "rank5_accuracy": float(rank5_acc),
        "rank10_accuracy": float(rank10_acc),
        "rank1_correct": rank1_correct,
        "rank5_correct": rank5_correct,
        "rank10_correct": rank10_correct,
        "results_file": str(results_file),
        "txt_dir": str(txt_dir),
        "dataset_dir": str(dataset_dir)
    }
    
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印评估结果
    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"{'='*60}")
    print(f"查询数量：{num_queries}")
    print(f"Gallery 数量：{len(gallery_features)}")
    print(f"{'='*60}")
    print(f"Rank-1 准确率：{rank1_acc:.4f} ({rank1_correct}/{num_queries})")
    print(f"Rank-5 准确率：{rank5_acc:.4f} ({rank5_correct}/{num_queries})")
    print(f"Rank-10 准确率：{rank10_acc:.4f} ({rank10_correct}/{num_queries})")
    print(f"mAP: {mAP:.4f}")
    print(f"{'='*60}")
    print(f"结果保存至：{output_dir.absolute()}")
    print(f"  - 详细匹配结果：{results_file}")
    print(f"  - 评估报告：{report_file}")
    print(f"{'='*60}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Linux 环境离线特征评估脚本 - 调用 simulate_offline.sh 生成特征并评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基本用法（txt 特征已生成）
    python3 run_txt_evaluation.py \\
        --txt-dir ./log/output/ \\
        --dataset-dir ./test_dataset/ \\
        --output-dir ./evaluation_results
    
    # 指定 top-k 和特征维度
    python3 run_txt_evaluation.py \\
        --txt-dir ./log/output/ \\
        --dataset-dir ./test_dataset/ \\
        --output-dir ./evaluation_results \\
        --top-k 20 \\
        --feature-dim 512
    
    # 先运行 simulate_offline.sh 生成特征，再评估
    python3 run_txt_evaluation.py \\
        --image-dir ./test_dataset/ \\
        --script-path ./simulate_offline.sh \\
        --model-path ./model/pet_mobilenetv2_attr.img \\
        --output-dir ./evaluation_results \\
        --generate-features
        """
    )
    
    # 输入参数组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--txt-dir",
        help="TXT 特征文件目录（SGS-IPU SDK 输出）"
    )
    input_group.add_argument(
        "--image-dir",
        help="输入图片目录（用于调用 simulate_offline.sh 生成特征）"
    )
    
    parser.add_argument(
        "--dataset-dir",
        default="./test_dataset",
        help="测试数据集目录，包含 gallery_list.txt 和 query_list.txt（默认：./test_dataset）"
    )
    parser.add_argument(
        "--script-path",
        default="./simulate_offline.sh",
        help="simulate_offline.sh 脚本路径（默认：./simulate_offline.sh）"
    )
    parser.add_argument(
        "--model-path",
        help="模型文件路径（用于 generate-features 模式）"
    )
    parser.add_argument(
        "--preprocess-path",
        help="预处理脚本路径（用于 generate-features 模式）"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="结果输出目录（默认：./evaluation_results）"
    )
    parser.add_argument(
        "--feature-output-dir",
        default="./log/output",
        help="特征输出目录（用于 generate-features 模式，默认：./log/output）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回前 K 个匹配结果（默认：10）"
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=512,
        help="特征维度（默认：512）"
    )
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="先生成特征再评估（需要先运行 simulate_offline.sh）"
    )
    parser.add_argument(
        "--num-process",
        type=int,
        default=8,
        help="simulator.py 的进程数（默认：8）"
    )
    
    args = parser.parse_args()
    
    # 验证数据集目录
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)
    
    gallery_file = dataset_dir / "gallery_list.txt"
    query_file = dataset_dir / "query_list.txt"
    
    if not gallery_file.exists():
        print(f"Error: gallery_list.txt not found in {dataset_dir}")
        sys.exit(1)
    
    if not query_file.exists():
        print(f"Error: query_list.txt not found in {dataset_dir}")
        sys.exit(1)
    
    # 如果选择生成特征模式
    if args.generate_features:
        if not args.image_dir:
            print("Error: --image-dir is required when using --generate-features")
            sys.exit(1)
        
        if not args.model_path:
            print("Error: --model-path is required when using --generate-features")
            sys.exit(1)
        
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Image directory does not exist: {image_dir}")
            sys.exit(1)
        
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file does not exist: {model_path}")
            sys.exit(1)
        
        script_path = Path(args.script_path)
        if not script_path.exists():
            print(f"Error: Script does not exist: {script_path}")
            sys.exit(1)
        
        # 创建特征输出目录
        feature_output_dir = Path(args.feature_output_dir)
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating features using simulate_offline.sh...")
        print(f"Image directory: {image_dir}")
        print(f"Model path: {model_path}")
        print(f"Output directory: {feature_output_dir}")
        
        # 尝试直接调用 simulator.py
        # 首先读取 simulate_offline.sh 获取 simulator.py 路径
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # 提取 simulator.py 路径
        simulator_match = re.search(r'python3\s+([^\s]+simulator\.py)', script_content)
        if simulator_match:
            simulator_path = simulator_match.group(1)
            print(f"Found simulator.py: {simulator_path}")
            
            # 提取模型路径
            model_match = re.search(r'-m\s+([^\s]+)', script_content)
            if model_match:
                default_model = model_match.group(1)
                print(f"Default model in script: {default_model}")
            
            # 提取预处理脚本路径
            preprocess_match = re.search(r'-n\s+([^\s]+)', script_content)
            if preprocess_match:
                default_preprocess = preprocess_match.group(1)
                print(f"Default preprocess in script: {default_preprocess}")
            
            # 运行 simulator
            success, feature_output_dir = run_simulate_offline_direct(
                simulator_path=simulator_path,
                image_dir=image_dir,
                model_path="./model/pet_mobilenetv2_attr.img",
                preprocess_path=args.preprocess_path,
                num_process=args.num_process
            )
            
            if not success:
                print("Failed to generate features!")
                sys.exit(1)
            
            print("Features generated successfully!")
            txt_dir = feature_output_dir
        else:
            # 尝试运行脚本
            print("Could not parse simulator.py path, trying to run script directly...")
            success = run_simulate_offline(
                script_path=script_path,
                image_dir=image_dir,
                output_dir=feature_output_dir,
                model_path=args.model_path,
                preprocess_path=args.preprocess_path
            )
            
            if not success:
                print("Failed to generate features!")
                sys.exit(1)
            
            print("Features generated successfully!")
            txt_dir = feature_output_dir
    else:
        # 直接使用已有的 txt 特征
        if not args.txt_dir:
            print("Error: --txt-dir is required when not using --generate-features")
            sys.exit(1)
        
        txt_dir = Path(args.txt_dir)
        if not txt_dir.exists():
            print(f"Error: TXT feature directory does not exist: {txt_dir}")
            sys.exit(1)
    
    # 运行评估
    run_evaluation(
        txt_dir=txt_dir,
        dataset_dir=dataset_dir,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        feature_dim=args.feature_dim
    )


if __name__ == "__main__":
    main()