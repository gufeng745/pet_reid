"""
Transformer 工作流程演示主程序

这个程序将带你逐步理解 Transformer 的工作原理：
1. 输入处理和词嵌入
2. 位置编码
3. Encoder 自注意力机制
4. Decoder 交叉注意力机制
5. 输出预测

运行方式：python main.py
"""

import os
# 解决 OpenMP 运行时冲突问题（conda 环境中 PyTorch 的常见问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_model import Transformer, SMALL_CONFIG, PositionalEncoding, MultiHeadAttention

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demonstrate_token_embedding():
    """演示词嵌入过程"""
    print_section("1. 词嵌入 (Token Embedding)")

    # 简单词表
    vocab = {
        '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
        'I': 4, 'love': 5, 'deep': 6, 'learning': 7,
        'transformer': 8, 'is': 9, 'powerful': 10, '.': 11
    }

    vocab_size = len(vocab)
    d_model = 8  # 小维度便于展示

    print(f"\n词表大小：{vocab_size}")
    print(f"嵌入维度：{d_model}")
    print(f"\n词表映射：{vocab}")

    # 创建嵌入层
    embedding = nn.Embedding(vocab_size, d_model)

    # 输入句子："I love deep learning"
    sentence = [4, 5, 6, 7]  # token IDs
    input_tensor = torch.tensor([sentence])

    print(f"\n输入句子 token IDs: {sentence}")
    print(f"输入形状：{input_tensor.shape}")

    # 获取嵌入
    embedded = embedding(input_tensor)
    print(f"嵌入后形状：{embedded.shape}")
    print(f"\n前 4 个 token 的嵌入向量 (前 4 维):")
    for i, token_id in enumerate(sentence[:4]):
        token_name = list(vocab.keys())[list(vocab.values()).index(token_id)]
        vec = embedded[0, i, :4].detach().numpy()
        print(f"  '{token_name}': [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, {vec[3]:.3f}]")

    return vocab_size, d_model


def demonstrate_positional_encoding():
    """演示位置编码"""
    print_section("2. 位置编码 (Positional Encoding)")

    d_model = 16
    max_len = 20

    pe = PositionalEncoding(d_model, max_len)

    # 可视化位置编码
    pe_matrix = pe.pe[0, :max_len, :].numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(pe_matrix, cmap='coolwarm', center=0)
    plt.title('位置编码矩阵 (所有维度)')
    plt.xlabel('维度')
    plt.ylabel('位置')

    plt.subplot(1, 2, 2)
    positions = np.arange(max_len)
    plt.plot(positions, pe_matrix[:, 0], label='维度 0 (sin)')
    plt.plot(positions, pe_matrix[:, 1], label='维度 1 (cos)')
    plt.plot(positions, pe_matrix[:, 2], label='维度 2 (sin)')
    plt.plot(positions, pe_matrix[:, 3], label='维度 3 (cos)')
    plt.title('不同维度的位置编码曲线')
    plt.xlabel('位置')
    plt.ylabel('编码值')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150)
    print("位置编码可视化已保存到：positional_encoding.png")

    print(f"\n位置编码说明：")
    print(f"- 使用正弦/余弦函数编码位置信息")
    print(f"- 偶数维度用 sin，奇数维度用 cos")
    print(f"- 不同维度使用不同频率，形成独特的'位置指纹'")
    print(f"- 允许模型学习相对位置关系")

    plt.close()


def demonstrate_attention_mechanism():
    """演示注意力机制"""
    print_section("3. 自注意力机制 (Self-Attention)")

    d_model = 64
    num_heads = 4
    seq_len = 8

    # 创建注意力层
    attn = MultiHeadAttention(d_model, num_heads)

    # 随机输入（模拟嵌入后的序列）
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, d_model)

    print(f"输入序列长度：{seq_len}")
    print(f"注意力头数：{num_heads}")
    print(f"每头维度：{d_model // num_heads}")

    # 前向传播
    output, attn_weights = attn(x, x, x)

    print(f"\n注意力权重形状：{attn_weights.shape}")
    print(f"- [batch_size={attn_weights.shape[0]}, num_heads={attn_weights.shape[1]}, "
          f"seq_len={attn_weights.shape[2]}, seq_len={attn_weights.shape[3]}]")

    # 可视化注意力权重（第一个头）
    head_idx = 0
    attn_matrix = attn_weights[0, head_idx].detach().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(seq_len), yticklabels=range(seq_len))
    plt.title(f'注意力权重矩阵 (Head {head_idx})')
    plt.xlabel('Key 位置')
    plt.ylabel('Query 位置')
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    print("注意力权重可视化已保存到：attention_weights.png")

    print(f"\n注意力机制说明：")
    print(f"- 每个位置计算对所有位置的注意力权重")
    print(f"- 权重和为 1（softmax 归一化）")
    print(f"- 高权重表示两个位置之间的关系强")
    print(f"- 多头允许模型关注不同的表示子空间")

    plt.close()


def demonstrate_full_transformer():
    """演示完整的 Transformer 前向传播"""
    print_section("4. 完整 Transformer 前向传播")

    # 使用小型配置
    vocab_size = 100
    config = SMALL_CONFIG

    print(f"模型配置：")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        **config,
        max_len=100
    )

    print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    batch_size = 2
    src_len = 10
    tgt_len = 8

    torch.manual_seed(42)
    src = torch.randint(1, vocab_size, (batch_size, src_len))  # 源序列
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))  # 目标序列（右移）

    print(f"\n输入形状：")
    print(f"  源序列 (src): {src.shape}")
    print(f"  目标序列 (tgt): {tgt.shape}")

    # 前向传播
    with torch.no_grad():
        output, attn_info = model(src, tgt)

    print(f"\n输出形状：")
    print(f"  输出 logits: {output.shape}")
    print(f"  - 每个位置对应词表大小的预测分数")

    # 分析注意力权重
    print(f"\n注意力权重分析：")
    print(f"  Encoder 层数：{len(attn_info['encoder_attn'])}")
    print(f"  Decoder Self-Attn 层数：{len(attn_info['decoder_self_attn'])}")
    print(f"  Decoder Cross-Attn 层数：{len(attn_info['decoder_cross_attn'])}")

    # 可视化跨层注意力（最后一层）
    last_layer_idx = -1
    cross_attn = attn_info['decoder_cross_attn'][last_layer_idx][0].mean(0).numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    # Encoder 自注意力
    enc_attn = attn_info['encoder_attn'][last_layer_idx][0, 0].numpy()
    sns.heatmap(enc_attn[:src_len, :src_len], cmap='Blues')
    plt.title(f'Encoder 自注意力 (最后一层)')
    plt.xlabel('Key')
    plt.ylabel('Query')

    plt.subplot(1, 2, 2)
    # Decoder 交叉注意力
    sns.heatmap(cross_attn[:tgt_len, :src_len], cmap='Blues')
    plt.title(f'Decoder 交叉注意力 (最后一层)')
    plt.xlabel('Encoder 位置')
    plt.ylabel('Decoder 位置')

    plt.tight_layout()
    plt.savefig('transformer_attention.png', dpi=150)
    print("\nTransformer 注意力可视化已保存到：transformer_attention.png")

    plt.close()

    return model


def demonstrate_step_by_step():
    """逐步演示 Transformer 数据流"""
    print_section("5. 数据流逐步演示")

    vocab_size = 50
    config = SMALL_CONFIG

    model = Transformer(vocab_size, **config)
    model.eval()

    # 单样本
    src = torch.tensor([[10, 20, 30, 5, 2]])  # 以<EOS>=2 结尾
    tgt = torch.tensor([[2, 15, 25, 35]])     # 以<SOS>=2 开头

    print("\n=== Transformer 数据流 ===\n")

    # 1. Encoder 输入处理
    print("Step 1: Encoder 输入处理")
    src_embedded = model.encoder.embedding(src) * math.sqrt(config['d_model'])
    print(f"  词嵌入：{src.shape} -> {src_embedded.shape}")

    src_with_pos = model.encoder.pos_encoding(src_embedded)
    print(f"  + 位置编码：{src_with_pos.shape}")

    # 2. Encoder 处理
    print("\nStep 2: Encoder 逐层处理")
    src_mask = model.create_mask(src)
    enc_out = src_with_pos
    for i, layer in enumerate(model.encoder.layers):
        enc_out, attn = layer(enc_out, src_mask)
        print(f"  Layer {i+1}: {enc_out.shape}")

    print(f"\n  Encoder 最终输出：{enc_out.shape}")

    # 3. Decoder 输入处理
    print("\nStep 3: Decoder 输入处理")
    tgt_embedded = model.decoder.embedding(tgt) * math.sqrt(config['d_model'])
    tgt_with_pos = model.decoder.pos_encoding(tgt_embedded)
    print(f"  词嵌入 + 位置编码：{tgt_with_pos.shape}")

    # 4. Decoder 处理
    print("\nStep 4: Decoder 逐层处理")
    tgt_self_mask = model.create_mask(tgt, is_causal=True)
    tgt_cross_mask = model.create_mask(src)
    dec_out = tgt_with_pos

    for i, layer in enumerate(model.decoder.layers):
        dec_out, self_attn, cross_attn = layer(dec_out, enc_out, tgt_self_mask, tgt_cross_mask)
        print(f"  Layer {i+1}: {dec_out.shape}")

    # 5. 输出投影
    print("\nStep 5: 输出投影")
    output = model.output_layer(dec_out)
    print(f"  投影到词表：{output.shape}")
    print(f"  词表大小：{vocab_size}")

    # 6. 获取预测
    print("\nStep 6: 获取预测")
    probs = F.softmax(output[:, -1, :], dim=-1)  # 最后一个位置的预测
    top5 = probs.topk(5)
    print(f"  下一个 token 预测 (Top 5):")
    for idx, prob in zip(top5.indices[0], top5.values[0]):
        print(f"    Token {idx.item()}: {prob.item():.4f}")

    return model


def interactive_demo():
    """交互式演示"""
    print_section("6. 交互式演示")

    vocab_size = 100
    config = SMALL_CONFIG
    model = Transformer(vocab_size, **config)
    model.eval()

    print("\n这是一个简单的文本生成演示")
    print("输入一些 token ID（用空格分隔），模型将尝试续写")
    print("或者输入 'demo' 使用预设示例")
    print("输入 'quit' 退出\n")

    while True:
        user_input = input("请输入 token 序列 (或 demo/quit): ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'demo':
            tokens = [10, 20, 30, 40]
            print(f"使用示例 tokens: {tokens}")
        else:
            try:
                tokens = [int(x) for x in user_input.split()]
            except ValueError:
                print("无效输入，请输入数字 token ID")
                continue

        if len(tokens) == 0:
            continue

        # 生成 5 个 token
        generated = tokens.copy()
        print(f"\n生成过程:")
        print(f"初始：{generated}")

        with torch.no_grad():
            for _ in range(5):
                src = torch.tensor([generated])
                tgt = src.clone()  # 简单演示：用自己的历史

                output, _ = model(src, tgt)
                next_token = output[0, -1, :].argmax().item()
                generated.append(next_token)
                print(f"  -> 生成 token {next_token}")

        print(f"最终序列：{generated}")


if __name__ == "__main__":
    import math
    import torch.nn.functional as F

    print("=" * 60)
    print("  Transformer 工作原理演示程序")
    print("=" * 60)
    print("\n本程序将逐步演示 Transformer 的各个组件和工作流程")
    print("可视化结果将保存为 PNG 文件在当前目录")

    try:
        # 1. 词嵌入演示
        vocab_size, d_model = demonstrate_token_embedding()

        # 2. 位置编码演示
        demonstrate_positional_encoding()

        # 3. 注意力机制演示
        demonstrate_attention_mechanism()

        # 4. 完整模型演示
        model = demonstrate_full_transformer()

        # 5. 逐步数据流演示
        demonstrate_step_by_step()

        # 6. 交互式演示（可选）
        # interactive_demo()

        print_section("演示完成!")
        print("\n生成的可视化文件:")
        print("  - positional_encoding.png: 位置编码可视化")
        print("  - attention_weights.png: 注意力权重矩阵")
        print("  - transformer_attention.png: Transformer 注意力分析")
        print("\n运行 'python train_transformer.py' 来训练一个简单的翻译任务")

    except Exception as e:
        print(f"\n发生错误：{e}")
        import traceback
        traceback.print_exc()
