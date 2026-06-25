# 使用datasets文件夹进行DINOv3预训练

## 概述

本指南介绍如何使用 `datasets` 文件夹（25000张图片）进行DINOv3自监督预训练。

**优势：**
- ✅ 数据量大（25000张 vs 745张）
- ✅ 不需要标签
- ✅ 自监督学习，自动提取特征
- ✅ 训练更稳定，泛化能力更强

## 数据集信息

```
datasets/
├── train/ (25000张)
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   └── ...
└── test/ (12500张)
    └── ...
```

- **总图片数**: 37500张（使用train的25000张）
- **格式**: JPG图片
- **标签**: 不需要（自监督学习）
- **内容**: 猫和狗的图片

## 快速开始

### 1. 基础训练（推荐）

```bash
cd D:/claude_workspace/pet_reid

# 使用默认参数训练
python train_dino_datasets.py
```

**默认参数：**
- backbone: MobileNetV3-Large
- 输出维度: 512
- 训练轮数: 200
- 批大小: 256
- 学习率: 5e-4

### 2. 自定义参数训练

```bash
# 小规模测试（快速验证）
python train_dino_datasets.py \
    --epochs 10 \
    --batch_size 64 \
    --save_interval 5

# 标准训练
python train_dino_datasets.py \
    --epochs 200 \
    --batch_size 256 \
    --lr 5e-4

# 大规模训练（更长时间，更好效果）
python train_dino_datasets.py \
    --epochs 300 \
    --batch_size 512 \
    --lr 1e-3
```

### 3. 使用原始train_dino.py

```bash
python train_dino.py \
    --data_root ../pet_rec/datasets \
    --epochs 200 \
    --batch_size 256 \
    --proj_dim 512
```

## 训练参数说明

### 核心参数

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `--epochs` | 200 | 训练轮数 | 100-300 |
| `--batch_size` | 256 | 批大小 | 128-512 |
| `--lr` | 5e-4 | 学习率 | 1e-4 ~ 1e-3 |
| `--proj_dim` | 512 | 输出维度 | 256-512 |

### DINOv3参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_temp` | 0.04 | Teacher温度（越小越尖锐） |
| `--student_temp` | 0.1 | Student温度 |
| `--teacher_momentum_start` | 0.996 | Teacher EMA动量 |

## 输出文件

训练完成后，会生成以下文件：

```
pet_reid/
├── checkpoints/dino/
│   ├── best_dino.pth          # 最佳模型
│   ├── dino_epoch20.pth       # 定期保存
│   ├── dino_epoch40.pth
│   └── final_dino.pth         # 最终模型
│
├── logs/dino/
│   ├── dino_pretraining_*.log           # 训练日志
│   ├── dino_pretraining_*_metrics.json  # 训练指标
│   ├── dino_pretraining_*_curves.png    # 训练曲线
│   └── dino_pretraining_*_loss_detail.png  # 详细Loss曲线
│
└── outputs/dino/
    └── (其他输出)
```

## 训练监控

### 查看训练日志

```bash
# 查看最新日志
ls -lt logs/dino/*.log | head -1

# 实时监控训练
tail -f logs/dino/dino_pretraining_*.log
```

### 查看训练曲线

训练过程中会自动生成训练曲线图片：
- `*_curves.png`: Loss和学习率曲线
- `*_loss_detail.png`: 详细Loss曲线（带平滑）

### 训练指标

训练指标保存在JSON文件中，可以用Python读取：

```python
import json

with open('logs/dino/dino_pretraining_*_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"最终Loss: {metrics['train_loss'][-1]:.4f}")
print(f"最小Loss: {min(metrics['train_loss']):.4f}")
```

## 预期训练时间

根据你的硬件配置：

| 配置 | 预计时间（200 epochs） |
|------|----------------------|
| RTX 3060 | ~8小时 |
| RTX 3080 | ~5小时 |
| RTX 4090 | ~3小时 |
| CPU only | ~48小时（不推荐） |

**建议：**
- 使用GPU训练
- 如果时间有限，可以减少epochs到100
- 批大小越大，训练越快

## 训练效果

### 预期Loss曲线

```
Loss
  ^
  │
6 │*
  │ *
5 │  *
  │   **
4 │     ***
  │        ****
3 │           *****
  │                ******
2 │                      ********
  │                              ************
  └────────────────────────────────────────────> epoch
  0    50   100   150   200
```

### 预期效果

- **初始Loss**: ~6.0
- **最终Loss**: ~2.0-3.0
- **特征质量**: 优秀的宠物特征提取能力

## 使用预训练模型

### 1. 特征提取

```python
from models.reid_model import ReIDModel
import torch

# 加载预训练模型
model = ReIDModel.from_pretrained('checkpoints/dino/best_dino.pth')
model.eval()

# 提取特征
image = load_image('test.png')
feature = model.forward_emb(image)  # (512,) 特征向量
```

### 2. 相似度计算

```python
# 计算两张图片的相似度
feature1 = model.forward_emb(image1)
feature2 = model.forward_emb(image2)
similarity = torch.cosine_similarity(feature1, feature2, dim=0)
```

### 3. Re-ID微调（可选）

如果需要个体识别能力，可以继续进行Re-ID微调：

```bash
python train_reid.py \
    --data_root ../pet_rec/reid_dataset \
    --pretrained_dino checkpoints/dino/best_dino.pth \
    --epochs 80
```

## 常见问题

### Q1: 训练过程中Loss不下降？

**可能原因：**
- 学习率太小
- 批大小太小
- 数据增强太强

**解决方案：**
```bash
# 增大学习率
python train_dino_datasets.py --lr 1e-3

# 增大批大小
python train_dino_datasets.py --batch_size 512
```

### Q2: 内存不足（OOM）？

**解决方案：**
```bash
# 减小批大小
python train_dino_datasets.py --batch_size 128

# 或者使用更小的backbone
python train_dino_datasets.py --backbone mobilenetv3_small_100
```

### Q3: 训练速度太慢？

**解决方案：**
```bash
# 增加数据加载线程
python train_dino_datasets.py --num_workers 8

# 减少日志输出
python train_dino_datasets.py --log_interval 20
```

### Q4: 如何恢复训练？

目前不支持断点续训。建议：
- 使用较短的epochs进行多次训练
- 定期保存checkpoint

## 进阶用法

### 1. 使用不同的backbone

```bash
# MobileNetV3-Small (更快)
python train_dino_datasets.py --backbone mobilenetv3_small_100

# EfficientNet-B0 (更精确)
python train_dino_datasets.py --backbone efficientnet_b0
```

### 2. 调整DINOv3参数

```bash
# 更稳定的训练（更大的momentum）
python train_dino_datasets.py --teacher_momentum_start 0.999

# 更尖锐的Teacher输出
python train_dino_datasets.py --teacher_temp 0.02
```

### 3. 多GPU训练（如果可用）

```bash
# 自动使用所有可用GPU
python train_dino_datasets.py
```

## 总结

使用datasets文件夹进行DINOv3预训练的优势：

1. **数据量大**: 25000张图片，比reid_dataset多33倍
2. **无需标注**: 自监督学习，自动提取特征
3. **泛化能力强**: 学到更通用的宠物特征
4. **训练稳定**: 大数据量使训练更稳定

训练完成后，你将获得一个强大的宠物特征提取模型，可以用于：
- 特征提取
- 相似度计算
- 图片检索
- 迁移到其他任务

开始训练吧！🚀
