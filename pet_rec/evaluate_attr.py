"""评估带分类头的 MobileNetV2StudentWithAttr 模型在验证集上的性能

使用与训练脚本相同的数据加载和划分方式，计算准确率等指标。

用法：
    python evaluate_attr.py
"""

import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models import MobileNetV2StudentWithAttr


class LabelEncoder:
    """将文本标签编码为数字，支持单标签和多标签"""
    
    def __init__(self, name, classes, multi_label=False):
        self.name = name
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.multi_label = multi_label
        self.num_classes = len(classes)
    
    def encode(self, text):
        """编码单个标签"""
        if not text or text.strip() == '':
            if self.multi_label:
                return torch.zeros(self.num_classes, dtype=torch.float32)
            return self.class_to_idx.get('unknown', 0)
        
        if self.multi_label:
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for item in text.split(','):
                item = item.strip()
                if item in self.class_to_idx:
                    vec[self.class_to_idx[item]] = 1.0
            return vec
        else:
            return self.class_to_idx.get(text.strip(), 0)
    
    def decode(self, idx):
        """解码数字标签为文本"""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.classes[idx] if idx < len(self.classes) else 'unknown'


def build_label_encoders(csv_path):
    """从 CSV 文件读取所有标签，构建编码器"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # 收集各类别的所有值
    color_primary_vals = set()
    color_secondary_vals = set()
    pattern_vals = set()
    
    for row in rows:
        color_primary_vals.add(row['color_primary'].strip())
        for item in row['color_secondary'].split(','):
            item = item.strip()
            if item:
                color_secondary_vals.add(item)
        for item in row['pattern'].split(','):
            item = item.strip()
            if item:
                pattern_vals.add(item)
    
    # 统一颜色类别（主色和副色用同一套类别）
    all_colors = sorted(color_primary_vals | color_secondary_vals)
    patterns = sorted(pattern_vals)
    
    print(f"颜色类别 ({len(all_colors)}): {all_colors}")
    print(f"花纹类别 ({len(patterns)}): {patterns}")
    
    encoders = {
        'color_primary': LabelEncoder('color_primary', all_colors, multi_label=False),
        'color_secondary': LabelEncoder('color_secondary', all_colors, multi_label=True),
        'pattern': LabelEncoder('pattern', patterns, multi_label=True),
    }
    return encoders, rows


def find_image_dir():
    """查找图片目录"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'cat_dog_attr')
    train_dir = os.path.join(base, 'train')
    if os.path.isdir(train_dir):
        return train_dir
    return base


class AttrPetDataset(Dataset):
    """属性标注宠物数据集"""
    
    def __init__(self, image_dir, rows, encoders, transform=None):
        self.image_dir = image_dir
        self.rows = rows
        self.encoders = encoders
        self.transform = transform or get_eval_transform()
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # 获取标签
        color_pri = self.encoders['color_primary'].encode(row['color_primary'])
        color_sec = self.encoders['color_secondary'].encode(row['color_secondary'])
        pattern = self.encoders['pattern'].encode(row['pattern'])
        
        # 返回文件名用于调试
        return img, color_pri, color_sec, pattern, row['filename']


def get_eval_transform():
    """评估时使用的变换"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def calculate_multilabel_metrics(preds, targets, num_classes):
    """计算多标签分类指标
    
    Args:
        preds: 预测的 multi-hot 向量 (N, num_classes)
        targets: 真实 multi-hot 向量 (N, num_classes)
    
    Returns:
        dict of metrics
    """
    preds = (preds > 0.5).float()
    
    # 样本级指标（exact match）
    exact_match = (preds == targets).all(dim=1).float().mean().item()
    
    # 标签级指标
    tp = ((preds == 1) & (targets == 1)).sum(dim=0)
    fp = ((preds == 1) & (targets == 0)).sum(dim=0)
    fn = ((preds == 0) & (targets == 1)).sum(dim=0)
    
    # 标签级精确率、召回率、F1
    label_precision = tp / (tp + fp + 1e-8)
    label_recall = tp / (tp + fn + 1e-8)
    label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall + 1e-8)
    
    # 宏平均和微平均
    macro_precision = label_precision.mean().item()
    macro_recall = label_recall.mean().item()
    macro_f1 = label_f1.mean().item()
    
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8).item()
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8).item()
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8).item()
    
    return {
        'exact_match': exact_match,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, encoders):
    """评估模型性能"""
    model.eval()
    
    # 收集预测和标签
    all_color_pri_pred = []
    all_color_pri_target = []
    all_color_sec_pred = []
    all_color_sec_target = []
    all_pattern_pred = []
    all_pattern_target = []
    all_filenames = []
    
    for images, color_pri, color_sec, pattern, filenames in dataloader:
        images = images.to(device)
        
        # 模型输出
        _, color_pri_logits, color_sec_logits, pattern_logits = model(images)
        
        # 主色：单标签分类，取 argmax
        color_pri_pred = torch.argmax(color_pri_logits, dim=1)
        all_color_pri_pred.append(color_pri_pred.cpu())
        all_color_pri_target.append(color_pri)
        
        # 辅色：多标签分类，sigmoid + 阈值
        color_sec_pred = torch.sigmoid(color_sec_logits)
        all_color_sec_pred.append(color_sec_pred.cpu())
        all_color_sec_target.append(color_sec)
        
        # 花纹：多标签分类，sigmoid + 阈值
        pattern_pred = torch.sigmoid(pattern_logits)
        all_pattern_pred.append(pattern_pred.cpu())
        all_pattern_target.append(pattern)
        
        all_filenames.extend(filenames)
    
    # 拼接所有结果
    color_pri_pred = torch.cat(all_color_pri_pred, dim=0)
    color_pri_target = torch.cat(all_color_pri_target, dim=0)
    color_sec_pred = torch.cat(all_color_sec_pred, dim=0)
    color_sec_target = torch.cat(all_color_sec_target, dim=0)
    pattern_pred = torch.cat(all_pattern_pred, dim=0)
    pattern_target = torch.cat(all_pattern_target, dim=0)
    
    return {
        'color_primary': {
            'pred': color_pri_pred,
            'target': color_pri_target,
        },
        'color_secondary': {
            'pred': color_sec_pred,
            'target': color_sec_target,
        },
        'pattern': {
            'pred': pattern_pred,
            'target': pattern_target,
        },
        'filenames': all_filenames,
    }


def main():
    """主函数"""
    print("=" * 70)
    print("MobileNetV2StudentWithAttr 模型验证集评估")
    print("=" * 70)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")
    
    # 超参数
    batch_size = 64
    checkpoint_path = 'checkpoints/best_student_attr.pth'
    csv_path = 'annotations.csv'
    
    # 加载标签和数据
    print(f"\n加载标注文件：{csv_path}")
    encoders, rows = build_label_encoders(csv_path)
    
    # 获取图片目录
    image_dir = find_image_dir()
    print(f"图片目录：{image_dir}")
    
    # 使用与训练相同的 seed 划分验证集
    print("\n划分验证集（与训练脚本相同的 seed=42）...")
    val_size = int(len(rows) * 0.1)
    train_size = len(rows) - val_size
    train_rows, val_rows = random_split(
        rows, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_rows = list(val_rows)
    print(f"训练集：{train_size}, 验证集：{len(val_rows)}")
    
    # 创建验证数据集
    print("\n创建验证数据集...")
    val_dataset = AttrPetDataset(image_dir, val_rows, encoders)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    # 从 checkpoint 获取类别数量
    print(f"\n加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    num_colors = state_dict['color_primary_head.1.weight'].shape[0]
    num_patterns = state_dict['pattern_head.1.weight'].shape[0]
    print(f"从 checkpoint 检测到：num_colors={num_colors}, num_patterns={num_patterns}")
    
    # 加载模型
    print(f"\n加载模型：MobileNetV2StudentWithAttr")
    model = MobileNetV2StudentWithAttr(
        proj_dim=512, num_colors=num_colors, num_patterns=num_patterns
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("模型加载完成")
    
    # 评估模型
    print(f"\n开始评估...")
    results = evaluate(model, val_loader, device, encoders)
    
    # 计算指标
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    # 1. 主色准确率（单标签分类）
    color_pri_acc = accuracy_score(
        results['color_primary']['target'].numpy(),
        results['color_primary']['pred'].numpy()
    )
    color_pri_precision = precision_score(
        results['color_primary']['target'].numpy(),
        results['color_primary']['pred'].numpy(),
        average='weighted'
    )
    color_pri_recall = recall_score(
        results['color_primary']['target'].numpy(),
        results['color_primary']['pred'].numpy(),
        average='weighted'
    )
    color_pri_f1 = f1_score(
        results['color_primary']['target'].numpy(),
        results['color_primary']['pred'].numpy(),
        average='weighted'
    )
    
    print("\n【主色分类 (color_primary)】")
    print(f"  准确率 (Accuracy):  {color_pri_acc:.4f} ({color_pri_acc*100:.2f}%)")
    print(f"  精确率 (Precision): {color_pri_precision:.4f}")
    print(f"  召回率 (Recall):    {color_pri_recall:.4f}")
    print(f"  F1 分数：           {color_pri_f1:.4f}")
    
    # 主色混淆矩阵
    cm_color_pri = confusion_matrix(
        results['color_primary']['target'].numpy(),
        results['color_primary']['pred'].numpy()
    )
    num_cm_classes = cm_color_pri.shape[0]
    print(f"\n  混淆矩阵 (颜色类别，{num_cm_classes} 类):")
    for i in range(num_cm_classes):
        cls_name = encoders['color_primary'].classes[i] if i < len(encoders['color_primary'].classes) else f"class_{i}"
        row_str = "  ".join([f"{cm_color_pri[i][j]:3d}" for j in range(num_cm_classes)])
        print(f"    {cls_name:12s}: {row_str}")
    
    # 2. 辅色指标（多标签分类）
    color_sec_metrics = calculate_multilabel_metrics(
        results['color_secondary']['pred'],
        results['color_secondary']['target'],
        num_colors
    )
    
    print("\n【辅色分类 (color_secondary) - 多标签】")
    print(f"  精确匹配率 (Exact Match): {color_sec_metrics['exact_match']:.4f}")
    print(f"  宏平均精确率 (Macro Precision): {color_sec_metrics['macro_precision']:.4f}")
    print(f"  宏平均召回率 (Macro Recall):    {color_sec_metrics['macro_recall']:.4f}")
    print(f"  宏平均 F1 (Macro F1):           {color_sec_metrics['macro_f1']:.4f}")
    print(f"  微平均精确率 (Micro Precision): {color_sec_metrics['micro_precision']:.4f}")
    print(f"  微平均召回率 (Micro Recall):    {color_sec_metrics['micro_recall']:.4f}")
    print(f"  微平均 F1 (Micro F1):           {color_sec_metrics['micro_f1']:.4f}")
    
    # 3. 花纹指标（多标签分类）
    pattern_metrics = calculate_multilabel_metrics(
        results['pattern']['pred'],
        results['pattern']['target'],
        num_patterns
    )
    
    print("\n【花纹分类 (pattern) - 多标签】")
    print(f"  精确匹配率 (Exact Match): {pattern_metrics['exact_match']:.4f}")
    print(f"  宏平均精确率 (Macro Precision): {pattern_metrics['macro_precision']:.4f}")
    print(f"  宏平均召回率 (Macro Recall):    {pattern_metrics['macro_recall']:.4f}")
    print(f"  宏平均 F1 (Macro F1):           {pattern_metrics['macro_f1']:.4f}")
    print(f"  微平均精确率 (Micro Precision): {pattern_metrics['micro_precision']:.4f}")
    print(f"  微平均召回率 (Micro Recall):    {pattern_metrics['micro_recall']:.4f}")
    print(f"  微平均 F1 (Micro F1):           {pattern_metrics['micro_f1']:.4f}")
    
    # 花纹混淆矩阵（取预测概率最高的类别）
    pattern_pred_single = torch.argmax(results['pattern']['pred'], dim=1)
    pattern_target_single = torch.argmax(results['pattern']['target'], dim=1)
    cm_pattern = confusion_matrix(pattern_target_single.numpy(), pattern_pred_single.numpy())
    num_pattern_cm_classes = cm_pattern.shape[0]
    print(f"\n  混淆矩阵 (花纹类别，{num_pattern_cm_classes} 类) - 取最高概率类别:")
    for i in range(num_pattern_cm_classes):
        cls_name = encoders['pattern'].classes[i] if i < len(encoders['pattern'].classes) else f"pattern_{i}"
        row_str = "  ".join([f"{cm_pattern[i][j]:3d}" for j in range(num_pattern_cm_classes)])
        print(f"    {cls_name:12s}: {row_str}")
    
    # 保存评估结果
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    eval_results = {
        'num_samples': len(val_rows),
        'color_primary': {
            'accuracy': float(color_pri_acc),
            'precision': float(color_pri_precision),
            'recall': float(color_pri_recall),
            'f1': float(color_pri_f1),
            'confusion_matrix': cm_color_pri.tolist(),
            'classes': encoders['color_primary'].classes,
        },
        'color_secondary': {k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v for k, v in color_sec_metrics.items()},
        'pattern': {k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v for k, v in pattern_metrics.items()},
    }
    
    output_path = os.path.join(output_dir, 'attr_eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到：{output_path}")
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()