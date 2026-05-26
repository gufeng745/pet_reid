"""
Transformer 模型核心组件实现
包含论文 "Attention Is All You Need" 中的所有关键组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)

    由于 Transformer 没有 RNN/CNN 的序列顺序概念，需要添加位置信息
    使用不同频率的正弦/余弦函数来编码位置
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算不同维度的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 正弦编码偶数维度，余弦编码奇数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 添加 batch 维度
        self.register_buffer('pe', pe)  # 不作为参数更新

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)

    核心思想：让模型能够同时关注不同位置的不同表示子空间
    将 Q, K, V 投影到多个头，分别计算注意力，然后拼接
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]

        返回：[batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # 线性投影并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 mask（用于 decoder 的因果掩码或 padding 掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)

        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性投影
        output = self.W_o(attn_output)

        return output, attn_weights


class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network)

    两个线性变换中间加一个 ReLU 激活
    对每个位置独立应用相同的 FFN
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class EncoderLayer(nn.Module):
    """
    Encoder 层

    结构：
    1. 多头自注意力 + 残差连接 + LayerNorm
    2. 前馈网络 + 残差连接 + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 (Pre-LayerNorm 结构)
        attn_input = self.norm1(x)
        attn_output, attn_weights = self.self_attn(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout(attn_output)

        # 前馈网络
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)

        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    Decoder 层

    结构：
    1. 掩码多头自注意力 + 残差 + LayerNorm
    2. 多头交叉注意力 + 残差 + LayerNorm
    3. 前馈网络 + 残差 + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        # 掩码自注意力（Decoder 内部）
        attn_input = self.norm1(x)
        attn_output, self_attn_weights = self.self_attn(attn_input, attn_input, attn_input, self_attn_mask)
        x = x + self.dropout(attn_output)

        # 交叉注意力（与 Encoder 输出）
        cross_input = self.norm2(x)
        cross_output, cross_attn_weights = self.cross_attn(cross_input, enc_output, enc_output, cross_attn_mask)
        x = x + self.dropout(cross_output)

        # 前馈网络
        ff_input = self.norm3(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)

        return x, self_attn_weights, cross_attn_weights


class Encoder(nn.Module):
    """
    Encoder 堆叠多个 Encoder 层
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list


class Decoder(nn.Module):
    """
    Decoder 堆叠多个 Decoder 层
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        self_attn_weights_list = []
        cross_attn_weights_list = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, enc_output, self_attn_mask, cross_attn_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)

        return x, self_attn_weights_list, cross_attn_weights_list


class Transformer(nn.Module):
    """
    完整的 Transformer 模型

    参数：
    - vocab_size: 词表大小
    - d_model: 嵌入维度 (论文中为 512)
    - num_heads: 注意力头数 (论文中为 8)
    - d_ff: 前馈网络隐藏层维度 (论文中为 2048)
    - num_layers: Encoder/Decoder 层数 (论文中为 6)
    - dropout: Dropout 概率
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()

        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx
        self.d_model = d_model

    def create_mask(self, seq, is_causal=False):
        """
        创建注意力掩码
        - padding mask: 忽略 padding 位置
        - causal mask: Decoder 自注意力中使用，防止看到未来位置
        """
        batch_size, seq_len = seq.shape

        if is_causal:
            # 因果掩码（下三角矩阵）
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        else:
            # padding mask
            mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

        return mask

    def forward(self, src, tgt):
        """
        src: [batch_size, src_len] - 源序列（Encoder 输入）
        tgt: [batch_size, tgt_len] - 目标序列（Decoder 输入，右移一位）
        """
        # 创建掩码
        src_mask = self.create_mask(src)
        tgt_self_mask = self.create_mask(tgt, is_causal=True)
        tgt_cross_mask = self.create_mask(src)

        # Encoder
        enc_output, enc_attn_weights = self.encoder(src, src_mask)

        # Decoder
        dec_output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            tgt, enc_output, tgt_self_mask, tgt_cross_mask
        )

        # 输出投影到词表
        output = self.output_layer(dec_output)

        return output, {
            'encoder_attn': enc_attn_weights,
            'decoder_self_attn': dec_self_attn_weights,
            'decoder_cross_attn': dec_cross_attn_weights
        }

    def encode(self, src):
        """单独编码"""
        src_mask = self.create_mask(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, enc_mask):
        """单独解码"""
        tgt_self_mask = self.create_mask(tgt, is_causal=True)
        return self.decoder(tgt, enc_output, tgt_self_mask, enc_mask)


def generate_square_subsequent_mask(sz):
    """生成方形因果掩码（用于 Decoder 自注意力）"""
    return torch.tril(torch.ones(sz, sz))


# 示例配置（论文中的 base 模型）
TRANSFORMER_CONFIG = {
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'dropout': 0.1,
}

# 小型配置（用于演示和快速训练）
SMALL_CONFIG = {
    'd_model': 128,
    'num_heads': 4,
    'd_ff': 256,
    'num_layers': 2,
    'dropout': 0.1,
}
