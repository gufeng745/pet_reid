# Pet Re-ID 全流程优化指南

## 概述

基于 Re-ID SOP 对宠物重识别系统进行全面优化，解决 `color_primary` 和 `contrastive` loss 偏高的问题。

## 优化文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `models.py` | 模型结构优化 | ✅ 已优化 |
| `distillation.py` | 损失函数优化 | ✅ 已优化 |
| `train_attr_v2.py` | 训练脚本 V2 | ✅ 新建 |
| `inference.py` | 推理模块 | ✅ 新建 |
| `run_train_v2.py` | 快速启动脚本 | ✅ 新建 |

---

## 优化内容详解

### 1. 模型结构优化 (models.py)

#### 1.1 SE 注意力模块
```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    # 通过学习通道间的依赖关系，增强重要特征通道
    # 弥补 Depthwise 卷积缺乏通道交互的缺陷
```

**效果**：
- 提升对关键色彩和纹理通道的响应
- 增强细粒度特征表达能力

#### 1.2 BNNeck
```python
class BNNeck(nn.Module):
    """BatchNorm Neck for Metric Learning"""
    # 训练时：使用 BN 后的特征做分类
    # 推理时：使用 BN 前的特征做度量
```

**效果**：
- 减少类内方差
- 显著提升度量学习准确性

#### 1.3 GeM 池化
```python
class GeMPooling(nn.Module):
    """Generalized Mean Pooling"""
    # 通过可学习参数 p 控制池化行为
    # 放大特征图中的显著响应
```

---

### 2. 损失函数优化 (distillation.py)

#### 2.1 Label Smoothing CE
```python
class LabelSmoothingCE(nn.Module):
    """Label Smoothing Cross Entropy"""
    # 将硬标签软化为软标签
    # 防止模型对相似类别过度自信
```

**效果**：
- 防止过拟合
- 提升泛化能力
- 降低 color_primary loss

#### 2.2 Focal Loss
```python
class FocalLoss(nn.Module):
    """Focal Loss for Hard Example Mining"""
    # 聚焦于难分类样本
    # 解决类别不平衡问题
```

#### 2.3 Circle Loss
```python
class CircleLoss(nn.Module):
    """Circle Loss for Metric Learning"""
    # 更平滑的优化目标
    # 更灵活的权重自适应
    # 更清晰的收敛边界
```

#### 2.4 ArcFace Loss
```python
class ArcFaceLoss(nn.Module):
    """ArcFace Loss"""
    # 在超球面上添加角度边界
    # 使类内更紧凑，类间更分离
```

---

### 3. 训练脚本优化 (train_attr_v2.py)

#### 3.1 PK Sampler
```python
class PKSampler(Sampler):
    """PK Sampler (身份感知采样器)"""
    # 每个 Batch: P 个 ID × K 张图
    # 保证每个 Batch 内都有正负样本
```

**效果**：
- 提升训练效率
- 平衡正负样本
- 加速收敛

#### 3.2 Random Erasing
```python
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                        ratio=(0.3, 3.3), value='random')
```

**效果**：
- 增强抗遮挡能力
- 防止背景过拟合
- 提升泛化能力

#### 3.3 混合精度训练 (AMP)
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    # forward pass
```

**效果**：
- 训练速度提升 2-3 倍
- 显存占用降低约 30%

#### 3.4 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
```

**效果**：
- 防止梯度爆炸
- 稳定 contrastive loss 训练

#### 3.5 早停机制
```python
class EarlyStopping:
    def __call__(self, val_loss):
        # 当验证损失不再下降时提前停止
```

**效果**：
- 防止过拟合
- 节省训练时间

#### 3.6 属性准确率监控
```python
# 训练过程中监控
color_acc = color_pri_correct / color_pri_total
pattern_acc = pattern_correct / pattern_total
```

---

### 4. 推理模块 (inference.py)

#### 4.1 特征提取
```python
class PetReIDInference:
    def extract_features(self, image):
        # L2 归一化特征提取
```

#### 4.2 余弦相似度
```python
def compute_similarity(self, feat1, feat2):
    return F.cosine_similarity(feat1, feat2).item()
```

#### 4.3 多查询融合
```python
def multi_query_fusion(self, query_imgs):
    # 将多张图片的特征向量平均
    # 准确率大幅提升
```

#### 4.4 置信度分级
```python
@staticmethod
def _get_confidence_label(similarity):
    if similarity > 0.85:
        return "高置信度匹配"
    elif similarity > 0.4:
        return "疑似匹配，需人工确认"
    else:
        return "不匹配"
```

#### 4.5 Re-ranking
```python
class ReRanker:
    def re_rank(self, query_feats, gallery_feats):
        # k-reciprocal encoding 重排序
```

---

## 超参数优化

针对 `color_primary` 和 `contrastive` loss 偏高的问题，进行了以下调整：

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| `lambda_color_pri` | 0.2 | **0.5** | 提升主色分类权重 |
| `lambda_contrastive` | 0.5 | **0.3** | 降低对比损失权重 |
| `contrastive_temp` | 0.07 | **0.1** | 提升温度系数 |
| `warmup_epochs` | 5 | **10** | 延长预热期 |
| `patience` | - | **15** | 添加早停容忍度 |
| `max_grad_norm` | - | **1.0** | 添加梯度裁剪 |

---

## 使用方法

### 训练

```bash
# 方法 1：使用快速启动脚本（推荐）
python run_train_v2.py

# 方法 2：自定义参数
python train_attr_v2.py \
    --epochs 80 \
    --batch_size 64 \
    --lambda_color_pri 0.5 \
    --lambda_contrastive 0.3 \
    --contrastive_temp 0.1 \
    --use_se \
    --use_bnneck \
    --use_label_smoothing \
    --use_amp \
    --use_early_stopping \
    --patience 15
```

### 推理

```bash
# 单张图片匹配
python inference.py \
    --model_path checkpoints/best_student_attr_v2.pth \
    --query_img query.jpg \
    --gallery_dir gallery/ \
    --threshold 0.85 \
    --top_k 10
```

---

## 预期效果

| 指标 | 优化前 | 预期优化后 |
|------|--------|------------|
| color_primary loss | 0.5 | 0.2-0.3 |
| contrastive loss | 0.5 | 0.2-0.3 |
| 训练速度 | 1x | 2-3x (AMP) |
| 推理准确率 | - | +5-10% |

---

## 核心 Checklist

- [x] SE 注意力模块
- [x] BNNeck
- [x] Label Smoothing
- [x] Random Erasing
- [x] PK Sampler
- [x] 混合精度训练
- [x] 梯度裁剪
- [x] 早停机制
- [x] 属性准确率监控
- [x] L2 归一化 + 余弦相似度
- [x] 多查询融合
- [x] 置信度分级
- [x] Re-ranking

---

## 注意事项

1. **首次训练**：建议先用 `run_train_v2.py` 跑一遍，观察 loss 变化
2. **PK Sampler**：需要数据集中有 `pet_id` 字段，否则使用 filename 作为 ID
3. **混合精度**：确保 GPU 支持 FP16，否则会自动回退到 FP32
4. **早停**：如果训练不稳定，可以增大 `patience` 或关闭早停

---

## 文件结构

```
pet_rec/
├── models.py              # 模型定义（已优化）
├── distillation.py        # 损失函数（已优化）
├── train_attr_v2.py       # 训练脚本 V2（新建）
├── inference.py           # 推理模块（新建）
├── run_train_v2.py        # 快速启动脚本（新建）
├── OPTIMIZATION_GUIDE.md  # 本文档
├── annotations.csv        # 标注文件
├── checkpoints/           # 模型权重目录
└── datasets/              # 数据集目录
```
