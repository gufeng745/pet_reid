# Pet Re-ID 宠物重识别系统

基于DINOv3自监督学习的宠物Re-ID特征提取系统

## 📋 项目概述

本项目实现了一个完整的宠物重识别(Re-ID)系统，采用两阶段训练策略：

1. **DINOv3自监督预训练**：使用无标签数据训练CNN backbone
2. **Re-ID监督微调**：使用带标签数据进行身份识别训练

### 核心特点

- ✅ 基于DINOv3的自监督学习，无需大量标注数据
- ✅ CNN架构（MobileNetV3-Large），适合移动端部署
- ✅ 多种损失函数组合，提升特征区分度
- ✅ 支持ONNX导出和INT8量化
- ✅ 完整的训练、评估、推理流程

## 🏗️ 系统架构

```
阶段1: DINOv3自监督预训练 (无标签)
┌─────────────────────────────────────────────────────────────┐
│  全局视图 (224×224)    │    局部视图 (96×96)                │
│         ↓              │         ↓                         │
│  EMA Teacher           │    CNN Student                    │
│  (不更新梯度)           │    (更新梯度)                      │
│         ↓              │         ↓                         │
│  投影头 (384维)         │  投影头 + 预测头                   │
│         ↓              │         ↓                         │
│         └──────────────┼─────────┘                         │
│                        ↓                                   │
│              自蒸馏损失 (Cross-Entropy)                      │
└─────────────────────────────────────────────────────────────┘

阶段2: Re-ID监督微调 (需要ID标签)
┌─────────────────────────────────────────────────────────────┐
│  预训练的CNN backbone                                      │
│  + GeM Pooling + SE注意力 + BNNeck                         │
│  + ID分类头                                                │
│                                                           │
│  损失函数:                                                  │
│  - ID分类损失 (CrossEntropy)                                │
│  - Triplet Loss (度量学习)                                  │
│  - 监督对比损失                                             │
│  - 特征正交正则化                                           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
pet_reid/
├── README.md                    # 项目文档
├── requirements.txt             # 依赖包
├── config.py                   # 全局配置
│
├── datasets/
│   ├── dino_dataset.py         # DINOv3多视图数据集
│   └── reid_dataset.py         # Re-ID数据集
│
├── models/
│   ├── backbone.py             # CNN backbone (MobileNetV3)
│   ├── dino_model.py           # DINOv3模型
│   └── reid_model.py           # Re-ID模型
│
├── losses/
│   ├── dino_loss.py            # DINOv3损失函数
│   └── reid_loss.py            # Re-ID损失函数
│
├── train_dino.py               # DINOv3预训练脚本
├── train_reid.py               # Re-ID微调脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
└── export_onnx.py              # ONNX导出
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
cd pet_reid

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据集目录结构：
```
reid_dataset/
├── cat/
│   ├── 1/
│   │   ├── img1.png
│   │   └── img2.png
│   ├── 2/
│   │   └── ...
│   └── ...
└── dog/
    ├── 1/
    │   └── ...
    └── ...
```

### 3. 训练流程

#### 阶段1: DINOv3自监督预训练

```bash
# 基础训练
python train_dino.py --data_root ../pet_rec/reid_dataset --epochs 200

# 自定义参数
python train_dino.py \
    --data_root ../pet_rec/reid_dataset \
    --backbone mobilenetv3_large_100 \
    --epochs 200 \
    --batch_size 256 \
    --lr 5e-4 \
    --proj_dim 384
```

#### 阶段2: Re-ID监督微调

```bash
# 基础训练
python train_reid.py --data_root ../pet_rec/reid_dataset --epochs 80

# 使用DINOv3预训练权重
python train_reid.py \
    --data_root ../pet_rec/reid_dataset \
    --pretrained_dino checkpoints/dino/best_dino.pth \
    --epochs 80 \
    --P 16 \
    --K 4
```

### 4. 评估

```bash
# 评估模型
python evaluate.py \
    --model checkpoints/reid/best_reid.pth \
    --data_root ../pet_rec/reid_dataset \
    --num_trials 10
```

### 5. 推理

```bash
# 单张图片特征提取
python inference.py \
    --model checkpoints/reid/best_reid.pth \
    --image test.png

# 计算两张图片相似度
python inference.py \
    --model checkpoints/reid/best_reid.pth \
    --image1 img1.png \
    --image2 img2.png

# 图片检索
python inference.py \
    --model checkpoints/reid/best_reid.pth \
    --query query.png \
    --gallery gallery/ \
    --top_k 5
```

### 6. ONNX导出

```bash
# 导出FP32模型
python export_onnx.py \
    --model checkpoints/reid/best_reid.pth \
    --output_dir outputs/onnx

# 导出INT8量化模型
python export_onnx.py \
    --model checkpoints/reid/best_reid.pth \
    --int8
```

## ⚙️ 配置参数

### DINOv3预训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone | mobilenetv3_large_100 | CNN backbone |
| proj_dim | 384 | 投影维度 |
| epochs | 200 | 训练轮数 |
| batch_size | 256 | 批大小 |
| lr | 5e-4 | 学习率 |
| teacher_temp | 0.04 | Teacher温度 |
| student_temp | 0.1 | Student温度 |

### Re-ID微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| proj_dim | 512 | 投影维度 |
| epochs | 80 | 训练轮数 |
| P | 16 | 每batch的ID数 |
| K | 4 | 每ID的样本数 |
| lambda_id | 0.5 | ID Loss权重 |
| lambda_triplet | 0.3 | Triplet Loss权重 |
| lambda_contrastive | 0.2 | Contrastive Loss权重 |

## 📊 性能指标

### 模型对比

| 模型 | 参数量 | ImageNet Top-1 | 推理速度 |
|------|--------|----------------|----------|
| MobileNetV2 | 2.22M | 72.0% | ⚡⚡⚡ |
| **MobileNetV3-Large** | **4.20M** | **75.2%** | ⚡⚡⚡ |
| EfficientNet-B0 | 4.01M | 77.1% | ⚡⚡ |

### 预期性能

- Rank-1准确率: >85%
- mAP: >75%
- 推理速度: <10ms (GPU)

## 🔧 模型架构

### MobileNetV3-Large

```
Input (224×224×3)
    ↓
Conv2d (3→16, stride=2)
    ↓
InvertedResidual Blocks × 15
    ↓
Conv2d (160→960, kernel=1)
    ↓
AdaptiveAvgPool2d
    ↓
Linear (960→1280)
    ↓
Output (1280)
```

### Re-ID模型结构

```
Backbone (MobileNetV3-Large)
    ↓
GeM Pooling (可学习池化)
    ↓
SE Block (通道注意力)
    ↓
Projector (1280→512)
    ↓
BNNeck (BatchNorm)
    ↓
ID Head (512→82)
```

## 📚 技术细节

### DINOv3核心技巧

1. **Centering**: 减去Teacher输出的运行均值，防止崩塌
2. **Sharpening**: 使用低温度(0.04)使Teacher输出更尖锐
3. **EMA更新**: Teacher缓慢跟踪Student (momentum=0.996)
4. **Predictor**: Student的额外预测头，防止捷径学习

### 损失函数

1. **ID分类损失**: Label Smoothing Cross-Entropy
2. **Triplet Loss**: 难样本挖掘的三元组损失
3. **监督对比损失**: 同一身份样本互为正样本
4. **特征正交正则化**: 鼓励特征维度去相关

## 🐛 常见问题

### Q: 训练过程中loss不下降？

A: 检查以下几点：
- 学习率是否合适
- 数据增强是否过强
- Batch size是否足够大

### Q: 推理速度慢？

A: 可以尝试：
- 使用ONNX导出
- INT8量化
- 减小输入分辨率

### Q: 如何提升性能？

A: 建议：
- 使用DINOv3预训练
- 增加数据增强
- 调整损失函数权重
- 使用更大的backbone

## 📄 许可证

本项目采用 MIT 许可证

## 🙏 致谢

- [DINOv3](https://arxiv.org/abs/2304.07193)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [ReID](https://github.com/open-mmlab/OpenReID)
