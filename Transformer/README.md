# Transformer 学习与演示项目

这个项目帮助你理解和运行 Transformer 模型，基于论文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)。

## 项目结构

```
Transformer/
├── environment.yml           # Conda 环境配置
├── transformer_model.py      # Transformer 核心组件实现
├── main.py                   # 工作流程演示主程序
├── train_transformer.py      # 训练演示脚本
├── transformer_explained.ipynb  # Jupyter Notebook 交互式教程
└── README.md                 # 本文件
```

## 快速开始

### 1. 创建 Conda 环境

```bash
# 进入项目目录
cd Transformer

# 创建并激活环境
conda env create -f environment.yml
conda activate transformer_learn
```

### 2. 运行演示程序

```bash
# 运行主演示程序（推荐首先运行这个）
python main.py

# 运行训练演示
python train_transformer.py

# 或者打开 Jupyter Notebook 进行交互式学习
jupyter notebook transformer_explained.ipynb
```

## 程序说明

### 1. transformer_model.py

实现了 Transformer 的所有核心组件：

| 组件 | 说明 |
|------|------|
| `PositionalEncoding` | 位置编码，使用正弦/余弦函数编码序列位置 |
| `MultiHeadAttention` | 多头注意力机制，核心创新 |
| `FeedForward` | 位置无关的前馈神经网络 |
| `LayerNorm` | 层归一化 |
| `EncoderLayer` | Encoder 层（自注意力 + FFN） |
| `DecoderLayer` | Decoder 层（掩码自注意力 + 交叉注意力 + FFN） |
| `Encoder` / `Decoder` | Encoder/Decoder 堆叠 |
| `Transformer` | 完整模型 |

### 2. main.py - 工作流程演示

逐步演示 Transformer 的工作流程：

1. **词嵌入** - 将 token ID 映射为向量
2. **位置编码** - 添加位置信息
3. **自注意力机制** - 计算序列内部关系
4. **完整前向传播** - Encoder-Decoder 数据流
5. **交互式演示** - 可输入 token 序列测试模型

运行后会生成可视化图片：
- `positional_encoding.png` - 位置编码可视化
- `attention_weights.png` - 注意力权重矩阵
- `transformer_attention.png` - Transformer 注意力分析

### 3. train_transformer.py - 训练演示

通过实际训练任务理解 Transformer：

- 创建简单的序列到序列任务
- 训练小型 Transformer 模型
- 可视化训练曲线和注意力权重
- 测试模型预测效果

### 4. transformer_explained.ipynb - 交互式教程

Jupyter Notebook 形式的交互式学习材料，可以：
- 逐单元格运行，观察每个组件的输出
- 修改参数，实时看到效果
- 进行简单的训练实验

## Transformer 架构图解

```
输入序列 ──→ [词嵌入 + 位置编码] ──→ Encoder (N 层)
                                        │
                                        ↓
                                    Encoder 输出
                                        │
                                        ↓
输出序列 ──→ [词嵌入 + 位置编码] ──→ Decoder (N 层) ──→ 线性层 ──→ Softmax ──→ 预测
                    ↑                       │
                    │                       │
                    └───────────────────────┘
                          交叉注意力
```

### Encoder 层结构

```
输入
 │
 ├──→ [LayerNorm] ──→ [多头自注意力] ──→ [+ 残差] ──┐
 │                                                  │
 └──────────────────────────────────────────────────+
                                                    │
                                                    ↓
 ├──→ [LayerNorm] ──→ [前馈网络 FFN] ──→ [+ 残差] ──┐
 │                                                  │
 └──────────────────────────────────────────────────+
                                                    ↓
                                                  输出
```

### Decoder 层结构

```
输入
 │
 ├──→ [LayerNorm] ──→ [掩码多头自注意力] ──→ [+ 残差] ──┐
 │                                                      │
 └──────────────────────────────────────────────────────+
                                                        │
                                                        ↓
 Encoder 输出 ──→ [LayerNorm] ──→ [多头交叉注意力] ──→ [+ 残差] ──┐
 │                                                              │
 └──────────────────────────────────────────────────────────────+
                                                                │
                                                                ↓
 ──→ [LayerNorm] ──→ [前馈网络 FFN] ──→ [+ 残差] ──→ 输出
```

## 关键概念解释

### 1. 自注意力 (Self-Attention)

每个位置的表示由所有位置的加权组合得到：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

- **Q (Query)**: 当前查询
- **K (Key)**: 键，用于匹配
- **V (Value)**: 实际内容值

### 2. 多头注意力 (Multi-Head Attention)

将 Q, K, V 投影到多个子空间，分别计算注意力后拼接：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. 位置编码 (Positional Encoding)

由于 Transformer 没有序列顺序概念，需要显式添加位置信息：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. 掩码 (Masking)

- **Padding Mask**: 忽略填充位置
- **Causal Mask**: Decoder 中防止看到未来位置（下三角矩阵）

## 实验建议

1. **修改模型配置**：在 `SMALL_CONFIG` 中调整参数观察效果
2. **可视化注意力**：观察不同层的注意力模式
3. **训练任务**：尝试不同的序列到序列任务
4. **对比实验**：移除某个组件（如位置编码）看影响

## 参考资源

- 原论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Illustrated Transformer: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- Attention 可视化：[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
