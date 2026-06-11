"""
宠物识别评估脚本
支持多种评估模式和指标计算
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import onnxruntime as ort


def load_test_dataset(dataset_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    加载测试数据集
    
    Returns:
        gallery: {pet_id: [image_paths]}
        query: {query_path: pet_id}
        annotations: 完整标注信息
    """
    # 加载 annotations.json
    annotations_path = os.path.join(dataset_dir, "annotations.json")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 组织 gallery 数据
    gallery = {}
    for item in annotations["gallery"]:
        pet_id = item["pet_id"]
        if pet_id not in gallery:
            gallery[pet_id] = []
        gallery[pet_id].append(item["image_path"])
    
    # 组织 query 数据
    query = {}
    for item in annotations["query"]:
        query[item["image_path"]] = item["ground_truth_pet_id"]
    
    return gallery, query, annotations


def extract_image_features(model, image_paths: List[str], device, transform=None, batch_size=1, use_attr_model=False, onnx_session=None):
    """提取图片特征"""
    all_feats = []
    valid_paths = []
    
    if onnx_session is not None:
        # 使用 ONNX Runtime 进行推理（逐张处理，因为 ONNX 模型使用固定 batch_size=1）
        print("使用 ONNX Runtime 提取特征...")
        input_name = onnx_session.get_inputs()[0].name
        
        # 获取 ONNX 输出形状信息
        output_cfg = onnx_session.get_outputs()[0]
        print(f"ONNX 输出配置：{output_cfg.name}, shape={output_cfg.shape}, type={output_cfg.type}")
        
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                if transform:
                    img = transform(img)
                
                # 单张图像推理 - 添加 batch 维度 [1, C, H, W]
                input_numpy = img.numpy().astype(np.float32)
                # transform 后是 [C, H, W]，需要添加 batch 维度变成 [1, C, H, W]
                if input_numpy.ndim == 3:
                    input_numpy = input_numpy[np.newaxis, :]
                
                outputs = onnx_session.run(None, {input_name: input_numpy})
                # ONNX 输出是 [1, feat_dim]，需要去掉 batch 维度
                output_arr = outputs[0]
                
                # 如果是 [1, N] 形状，去掉 batch 维度
                if output_arr.ndim == 2 and output_arr.shape[0] == 1:
                    feats = torch.from_numpy(output_arr[0])
                # 如果是 [1, N, 1, 1] 形状 (经过全局池化后)，flatten 成 [N]
                elif output_arr.ndim >= 2:
                    feats = torch.from_numpy(output_arr).reshape(output_arr.shape[0], -1)[0]
                else:
                    feats = torch.from_numpy(output_arr.flatten())
                
                # 确保每个特征是 2 维的 [1, 512]，以便正确连接成 [N, 512]
                if feats.dim() == 1:
                    feats = feats.unsqueeze(0)  # [512] -> [1, 512]
                
                all_feats.append(feats)
                valid_paths.append(path)
            except Exception as e:
                print(f"警告：无法加载图片 {path}: {e}")
    else:
        # 使用 PyTorch 进行推理
        model.eval()
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_imgs = []
                
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        if transform:
                            img = transform(img)
                        batch_imgs.append(img)
                    except Exception as e:
                        print(f"警告：无法加载图片 {path}: {e}")
                
                if batch_imgs:
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    # 如果是属性模型，使用 forward_emb 只获取特征
                    if use_attr_model and hasattr(model, 'forward_emb'):
                        feats = model.forward_emb(batch_tensor)
                    else:
                        feats = model(batch_tensor)
                    all_feats.append(feats.cpu())
                    valid_paths.extend(batch_paths)
    
    if all_feats:
        return torch.cat(all_feats, dim=0), valid_paths
    return torch.empty(0, 0), []


def compute_similarity_matrix(query_feats, gallery_feats, pet_to_gallery_idx):
    """
    计算 query 和 gallery 的相似度矩阵
    
    Args:
        query_feats: [num_query, feat_dim]
        gallery_feats: [num_gallery, feat_dim]
        pet_to_gallery_idx: {pet_id: [gallery_indices]}
    
    Returns:
        sim_matrix: [num_query, num_gallery]
    """
    query_norm = F.normalize(query_feats, dim=-1)
    gallery_norm = F.normalize(gallery_feats, dim=-1)
    return query_norm @ gallery_norm.T


def evaluate_matching(
    sim_matrix: torch.Tensor,
    query_gt: List[str],  # [pet_id, ...]
    gallery_pet_ids: List[str],  # [pet_id, ...]
    strategies: List[str] = None
) -> Dict:
    """
    评估匹配结果
    
    Args:
        sim_matrix: [num_query, num_gallery]
        query_gt: query 的真实 pet_id 列表
        gallery_pet_ids: gallery 的 pet_id 列表
        strategies: 评估策略列表
    
    Returns:
        评估结果字典
    """
    if strategies is None:
        strategies = ["top1", "topk", "weighted"]
    
    num_query, num_gallery = sim_matrix.shape
    results = {}
    
    # Top-1 准确率
    if "top1" in strategies:
        top1_preds = sim_matrix.argmax(dim=1).cpu().tolist()
        correct = sum(1 for i, pred_idx in enumerate(top1_preds) 
                     if gallery_pet_ids[pred_idx] == query_gt[i])
        results["top1_accuracy"] = correct / num_query
    
    # Top-K Recall
    if "topk" in strategies:
        top_k_list = [1, 3, 5, 10]
        results["topk_recall"] = {}
        
        for k in top_k_list:
            topk_indices = sim_matrix.topk(min(k, num_gallery), dim=1).indices.cpu().tolist()
            correct = 0
            for i, topk_idx in enumerate(topk_indices):
                gt_pet = query_gt[i]
                if any(gallery_pet_ids[idx] == gt_pet for idx in topk_idx):
                    correct += 1
            results["topk_recall"][f"Recall@{k}"] = correct / num_query
    
    # 加权融合策略 (考虑相似度分数)
    if "weighted" in strategies:
        # 计算每个 query 的匹配置信度
        confidences = sim_matrix.max(dim=1).values.cpu().numpy()
        results["mean_confidence"] = float(confidences.mean())
        results["confidence_std"] = float(confidences.std())
        
        # 高置信度准确率 (置信度 > 0.5)
        high_conf_mask = confidences > 0.5
        if high_conf_mask.sum() > 0:
            high_conf_correct = 0
            top1_preds = sim_matrix.argmax(dim=1).cpu().tolist()
            for i in np.where(high_conf_mask)[0]:
                pred_idx = top1_preds[i]
                if gallery_pet_ids[pred_idx] == query_gt[i]:
                    high_conf_correct += 1
            results["high_conf_accuracy"] = high_conf_correct / high_conf_mask.sum()
    
    # 混淆矩阵相关统计
    results["num_query"] = num_query
    results["num_gallery"] = num_gallery
    
    return results


def analyze_errors(
    sim_matrix: torch.Tensor,
    query_paths: List[str],
    query_gt: List[str],
    gallery_pet_ids: List[str],
    gallery_paths: List[str],
    output_dir: str,
    top_n: int = 10
):
    """
    分析错误案例并生成可视化
    
    Args:
        sim_matrix: [num_query, num_gallery]
        query_paths: query 图片路径列表
        query_gt: query 的真实 pet_id
        gallery_pet_ids: gallery 的 pet_id
        gallery_paths: gallery 图片路径
        output_dir: 输出目录
        top_n: 分析前 N 个错误案例
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_query = sim_matrix.shape[0]
    top1_preds = sim_matrix.argmax(dim=1).cpu().tolist()
    
    # 找出错误案例
    errors = []
    for i in range(num_query):
        pred_pet = gallery_pet_ids[top1_preds[i]]
        gt_pet = query_gt[i]
        if pred_pet != gt_pet:
            sim_score = sim_matrix[i, top1_preds[i]].item()
            errors.append({
                "query_idx": i,
                "query_path": query_paths[i],
                "gt_pet": gt_pet,
                "pred_pet": pred_pet,
                "pred_gallery_idx": top1_preds[i],
                "pred_gallery_path": gallery_paths[top1_preds[i]],
                "similarity": sim_score
            })
    
    # 按相似度排序 (最难分的排在前面)
    errors.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 保存错误分析结果
    error_file = os.path.join(output_dir, "error_analysis.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(errors[:top_n], f, indent=2, ensure_ascii=False)
    
    print(f"\n错误案例分析已保存：{error_file}")
    print(f"总错误数：{len(errors)}, 展示前{min(top_n, len(errors))}个")
    
    # 打印错误统计
    if errors:
        print("\n错误案例统计:")
        print("-" * 60)
        for i, err in enumerate(errors[:top_n], 1):
            print(f"{i}. Query: {Path(err['query_path']).name}")
            print(f"   真实：{err['gt_pet']} -> 预测：{err['pred_pet']} (相似度：{err['similarity']:.4f})")
            print(f"   匹配到：{Path(err['pred_gallery_path']).name}")
    
    return errors


def evaluate_strategies_comparison(
    sim_matrix: torch.Tensor,
    query_gt: List[str],
    gallery_pet_ids: List[str]
) -> Dict:
    """
    对比不同匹配策略的效果
    
    策略包括:
    1. Top-1: 直接取最相似的
    2. Threshold: 设置相似度阈值
    3. Reciprocal: 互近邻匹配
    """
    results = {}
    
    # 策略 1: Top-1
    top1_preds = sim_matrix.argmax(dim=1).cpu().tolist()
    top1_correct = sum(1 for i, pred_idx in enumerate(top1_preds) 
                      if gallery_pet_ids[pred_idx] == query_gt[i])
    results["top1"] = {
        "accuracy": top1_correct / len(query_gt),
        "correct": top1_correct,
        "total": len(query_gt)
    }
    
    # 策略 2: 阈值过滤
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results["threshold"] = {}
    for thresh in thresholds:
        mask = sim_matrix.max(dim=1).values > thresh
        valid_count = mask.sum().item()
        if valid_count > 0:
            valid_correct = 0
            for i in range(len(query_gt)):
                if mask[i]:
                    pred_idx = top1_preds[i]
                    if gallery_pet_ids[pred_idx] == query_gt[i]:
                        valid_correct += 1
            results["threshold"][f"thresh_{thresh}"] = {
                "accuracy": valid_correct / valid_count,
                "coverage": valid_count / len(query_gt),
                "correct": valid_correct,
                "valid": valid_count
            }
    
    # 策略 3: 互近邻 (Reciprocal Nearest Neighbor)
    rnn_correct = 0
    for i in range(len(query_gt)):
        # Query -> Gallery 的最近邻
        q2g_idx = sim_matrix[i].argmax().item()
        # Gallery -> Query 的最近邻
        g2q_sim = sim_matrix[:, q2g_idx]
        g2q_idx = g2q_sim.argmax().item()
        
        # 如果是互近邻且类别正确
        if g2q_idx == i and gallery_pet_ids[q2g_idx] == query_gt[i]:
            rnn_correct += 1
    
    results["reciprocal_nn"] = {
        "accuracy": rnn_correct / len(query_gt),
        "correct": rnn_correct,
        "total": len(query_gt)
    }
    
    return results


def run_full_evaluation(
    model_path: str,
    dataset_dir: str = "test_dataset",
    output_dir: str = "evaluation_results",
    device: str = None,
    transform_func = None,
    use_attr_model: bool = False,
    use_onnx: bool = False
):
    """
    运行完整评估流程
    
    Args:
        model_path: 模型检查点路径 (.pth) 或 ONNX 模型路径 (.onnx)
        dataset_dir: 测试数据集目录
        output_dir: 结果输出目录
        device: 运行设备
        transform_func: 图片变换函数
        use_attr_model: 是否使用属性模型 (MobileNetV2StudentWithAttr)
        use_onnx: 是否使用 ONNX 模型
    """
    from models import MobileNetV2Student, MobileNetV2StudentWithAttr
    from torchvision import transforms
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if transform_func is None:
        transform_func = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    print("=" * 60)
    print("宠物识别系统评估")
    print("=" * 60)
    print(f"模型：{model_path}")
    print(f"数据集：{dataset_dir}")
    print(f"设备：{device}")
    print(f"模型类型：{'属性模型' if use_attr_model else '基础模型'}")
    print(f"推理引擎：{'ONNX Runtime' if use_onnx else 'PyTorch'}")
    print("-" * 60)
    
    # 加载模型
    model = None
    onnx_session = None
    
    if use_onnx:
        # 使用 ONNX 模型
        print("加载 ONNX 模型...")
        # 设置 ONNX Runtime 会话选项
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        
        # 确定执行提供程序
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            try:
                providers.insert(0, 'CUDAExecutionProvider')
            except Exception as e:
                print(f"注意：CUDA Execution Provider 不可用：{e}")
        
        onnx_session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        print(f"ONNX 模型已加载，使用提供程序：{onnx_session.get_providers()}")
        
        # 获取输入输出信息
        input_cfg = onnx_session.get_inputs()[0]
        output_cfg = onnx_session.get_outputs()[0]
        print(f"ONNX 输入形状：{input_cfg.shape}, 类型：{input_cfg.type}")
        print(f"ONNX 输出形状：{output_cfg.shape}, 类型：{output_cfg.type}")
    else:
        # 使用 PyTorch 模型（不使用预训练 backbone，避免下载问题）
        if use_attr_model:
            model = MobileNetV2StudentWithAttr(
                proj_dim=512, 
                num_colors=13,  # 检查点中是 13 个颜色类别
                num_patterns=13,  # 检查点中是 13 个花纹类别
                pretrained_backbone=False  # 不加载预训练 backbone，直接从检查点加载
            ).to(device)
        else:
            model = MobileNetV2Student(proj_dim=512, pretrained_backbone=False).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and 'student' in ckpt:
            model.load_state_dict(ckpt['student'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print("PyTorch 模型已加载")
    
    # 加载数据集
    gallery, query, annotations = load_test_dataset(dataset_dir)
    print(f"Gallery 宠物数：{len(gallery)}")
    print(f"Query 图片数：{len(query)}")
    
    # 提取 gallery 特征
    print("\n提取 Gallery 特征...")
    gallery_paths = []
    gallery_pet_ids = []
    for pet_id, paths in gallery.items():
        for path in paths:
            full_path = os.path.join(dataset_dir, path)
            gallery_paths.append(full_path)
            gallery_pet_ids.append(pet_id)
    
    gallery_feats, valid_gallery_paths = extract_image_features(model, gallery_paths, device, transform_func, use_attr_model=use_attr_model, onnx_session=onnx_session)
    print(f"Gallery 特征：{gallery_feats.shape}, 有效路径数：{len(valid_gallery_paths)}")
    
    # 提取 query 特征
    print("提取 Query 特征...")
    query_paths = [os.path.join(dataset_dir, path) for path in query.keys()]
    query_gt = list(query.values())
    
    query_feats, valid_query_paths = extract_image_features(model, query_paths, device, transform_func, use_attr_model=use_attr_model, onnx_session=onnx_session)
    print(f"Query 特征：{query_feats.shape}, 有效路径数：{len(valid_query_paths)}")
    
    # 确保特征形状正确
    if gallery_feats.dim() == 1:
        gallery_feats = gallery_feats.unsqueeze(0)
    if query_feats.dim() == 1:
        query_feats = query_feats.unsqueeze(0)
    print(f"调整后 - Gallery 特征：{gallery_feats.shape}, Query 特征：{query_feats.shape}")
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    sim_matrix = compute_similarity_matrix(query_feats, gallery_feats, {})
    
    # 评估匹配
    print("\n评估匹配结果...")
    matching_results = evaluate_matching(sim_matrix, query_gt, gallery_pet_ids)
    
    # 策略对比
    print("对比不同策略...")
    strategy_results = evaluate_strategies_comparison(sim_matrix, query_gt, gallery_pet_ids)
    
    # 错误分析
    print("分析错误案例...")
    errors = analyze_errors(
        sim_matrix, query_paths, query_gt, gallery_pet_ids, gallery_paths,
        output_dir, top_n=10
    )
    
    # 保存完整结果
    os.makedirs(output_dir, exist_ok=True)
    
    final_results = {
        "model_path": model_path,
        "dataset": dataset_dir,
        "matching_results": matching_results,
        "strategy_comparison": strategy_results,
        "dataset_info": {
            "num_gallery_pets": len(gallery),
            "num_gallery_images": len(gallery_paths),
            "num_query_images": len(query_paths)
        }
    }
    
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("评估结果总结")
    print("=" * 60)
    
    if "top1_accuracy" in matching_results:
        print(f"Top-1 准确率：{matching_results['top1_accuracy']:.4f}")
    
    if "topk_recall" in matching_results:
        print("\nTop-K Recall:")
        for k, v in matching_results["topk_recall"].items():
            print(f"  {k}: {v:.4f}")
    
    print("\n策略对比:")
    for strategy, metrics in strategy_results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"  {strategy}: {metrics['accuracy']:.4f}")
        elif isinstance(metrics, dict):
            print(f"  {strategy}:")
            for thresh, m in metrics.items():
                if "accuracy" in m:
                    print(f"    {thresh}: acc={m['accuracy']:.4f}, coverage={m['coverage']:.4f}")
    
    print(f"\n完整结果已保存：{results_file}")
    print("=" * 60)
    
    return final_results


def evaluate_student(
    student_path: str, 
    proj_dim: int = 512, 
    teacher_type: str = 'dinov3'
):
    """保留原有的评估函数用于兼容性"""
    from prepare_data import create_dataloaders, get_eval_transform
    from models import MobileNetV2Student, DINOv3Teacher, DINOv2Teacher
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load student
    student = MobileNetV2Student(proj_dim=proj_dim).to(device)
    ckpt = torch.load(student_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'student' in ckpt:
        student.load_state_dict(ckpt['student'])
    else:
        student.load_state_dict(ckpt)
    student.eval()
    
    # Load teacher
    try:
        teacher = DINOv3Teacher().to(device)
        print("Teacher: DINOv3 ViT-S")
    except Exception:
        teacher = DINOv2Teacher().to(device)
        print("Teacher: DINOv2 ViT-S")
    teacher.eval()
    
    # Data
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    _, eval_loader = create_dataloaders(dataset_root, batch_size=1)
    
    # Extract features
    print("Extracting student features...")
    student_feats, labels = extract_image_features(
        student, 
        [],  # 不使用这个参数
        device,
        transform_func=get_eval_transform(),
        batch_size=1
    )
    
    # 使用原有的提取方式
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels_batch in eval_loader:
            imgs = imgs.to(device)
            feats = student(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels_batch)
    student_feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print("Extracting teacher features...")
    all_feats = []
    with torch.no_grad():
        for imgs, _ in eval_loader:
            imgs = imgs.to(device)
            feats = teacher(imgs)
            all_feats.append(feats.cpu())
    teacher_feats = torch.cat(all_feats, dim=0)
    
    # Retrieval
    print("\n=== Student Retrieval ===")
    recalls = {}
    for k in [1, 5, 10]:
        feats_norm = F.normalize(student_feats, dim=-1)
        sim = feats_norm @ feats_norm.T
        n = student_feats.shape[0]
        correct = 0
        for i in range(n):
            topk = sim[i].topk(k + 1).indices[1:]
            if (labels[topk] == labels[i]).any():
                correct += 1
        recalls[f'Recall@{k}'] = correct / n
    
    for k, v in recalls.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n=== Teacher Retrieval ===")
    t_recalls = {}
    for k in [1, 5, 10]:
        feats_norm = F.normalize(teacher_feats, dim=-1)
        sim = feats_norm @ feats_norm.T
        n = teacher_feats.shape[0]
        correct = 0
        for i in range(n):
            topk = sim[i].topk(k + 1).indices[1:]
            if (labels[topk] == labels[i]).any():
                correct += 1
        t_recalls[f'Recall@{k}'] = correct / n
    
    for k, v in t_recalls.items():
        print(f"  {k}: {v:.4f}")
    
    return {
        'recalls': recalls,
        'teacher_recalls': t_recalls,
    }


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    import argparse
    
    parser = argparse.ArgumentParser(description='宠物识别评估脚本')
    parser.add_argument('--mode', choices=['test_dataset', 'original'], 
                       default='test_dataset', help='评估模式')
    parser.add_argument('--model', type=str, default='checkpoints/best_student.pth',
                       help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default='test_dataset',
                       help='测试数据集目录')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'test_dataset':
        run_full_evaluation(args.model, args.dataset, args.output)
    else:
        evaluate_student(args.model)