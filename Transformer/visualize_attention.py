"""
注意力可视化工具
帮助用户更直观地理解 Transformer 的注意力机制
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_model import Transformer, SMALL_CONFIG

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_attention_heatmap(attn_matrix, title="注意力权重", xlabel="Key", ylabel="Query",
                           save_path=None, show=True):
    """绘制注意力热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = sns.heatmap(attn_matrix, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                     cbar_kws={'label': '注意力权重'})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到：{save_path}")

    if show:
        plt.show()
    plt.close()


def compare_heads(attn_weights, save_path=None):
    """比较不同头的注意力模式"""
    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 3))

    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        sns.heatmap(attn_weights[h], ax=axes[h], cmap='Blues', vmin=0, vmax=1,
                    annot=True, fmt='.2f', cbar=h == num_heads - 1)
        axes[h].set_title(f'Head {h}')
        axes[h].set_xlabel('Key')
        if h == 0:
            axes[h].set_ylabel('Query')
        else:
            axes[h].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到：{save_path}")

    plt.show()
    plt.close()


def visualize_full_transformer_attention(model, src, tgt, layer_idx=-1, save_path=None):
    """可视化 Transformer 各部分的注意力"""
    model.eval()

    with torch.no_grad():
        _, attn_info = model(src, tgt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Encoder 注意力（第一层、中间层、最后一层）
    num_enc_layers = len(attn_info['encoder_attn'])

    # Encoder 第一层
    enc_first = attn_info['encoder_attn'][0][0, 0].numpy()
    im0 = sns.heatmap(enc_first, ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title(f'Encoder Layer 1 自注意力')
    axes[0, 0].set_xlabel('Key')
    axes[0, 0].set_ylabel('Query')

    # Encoder 中间层
    if num_enc_layers > 2:
        enc_mid = attn_info['encoder_attn'][num_enc_layers // 2][0, 0].numpy()
    else:
        enc_mid = enc_first
    im1 = sns.heatmap(enc_mid, ax=axes[0, 1], cmap='Blues')
    axes[0, 1].set_title(f'Encoder Layer {num_enc_layers // 2 + 1} 自注意力')
    axes[0, 1].set_xlabel('Key')
    axes[0, 1].set_ylabel('Query')

    # Encoder 最后一层
    enc_last = attn_info['encoder_attn'][layer_idx][0, 0].numpy()
    im2 = sns.heatmap(enc_last, ax=axes[0, 2], cmap='Blues')
    axes[0, 2].set_title(f'Encoder Last Layer 自注意力')
    axes[0, 2].set_xlabel('Key')
    axes[0, 2].set_ylabel('Query')

    # Decoder 自注意力（最后一层）
    dec_self = attn_info['decoder_self_attn'][layer_idx][0, 0].numpy()
    im3 = sns.heatmap(dec_self, ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title('Decoder 自注意力 (Last Layer)')
    axes[1, 0].set_xlabel('Key')
    axes[1, 0].set_ylabel('Query')

    # Decoder 交叉注意力（平均所有头）
    dec_cross = attn_info['decoder_cross_attn'][layer_idx][0].mean(0).numpy()
    im4 = sns.heatmap(dec_cross[:tgt.size(1), :src.size(1)], ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title('Decoder 交叉注意力 (Encoder-Decoder, Mean Heads)')
    axes[1, 1].set_xlabel('Encoder Position')
    axes[1, 1].set_ylabel('Decoder Position')

    # Decoder 交叉注意力（第一个头）
    dec_cross_h0 = attn_info['decoder_cross_attn'][layer_idx][0, 0].numpy()
    im5 = sns.heatmap(dec_cross_h0[:tgt.size(1), :src.size(1)], ax=axes[1, 2], cmap='Blues')
    axes[1, 2].set_title('Decoder 交叉注意力 (Head 0)')
    axes[1, 2].set_xlabel('Encoder Position')
    axes[1, 2].set_ylabel('Decoder Position')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到：{save_path}")

    plt.show()
    plt.close()


def demo_attention_patterns():
    """演示不同的注意力模式"""
    print("=" * 60)
    print("  注意力模式演示")
    print("=" * 60)

    # 创建小型模型
    vocab_size = 30
    config = SMALL_CONFIG.copy()
    config['d_model'] = 32
    config['num_heads'] = 4
    config['num_layers'] = 2

    model = Transformer(vocab_size, **config)
    model.eval()

    # 创建一些示例输入
    torch.manual_seed(42)
    src = torch.randint(1, vocab_size, (1, 6))
    tgt = torch.randint(1, vocab_size, (1, 5))

    print(f"\n源序列长度：{src.shape[1]}")
    print(f"目标序列长度：{tgt.shape[1]}")
    print(f"注意力头数：{config['num_heads']}")

    # 可视化
    visualize_full_transformer_attention(
        model, src, tgt,
        save_path='attention_patterns.png'
    )

    print("\n注意力模式说明：")
    print("1. Encoder 自注意力：每个位置可以看到所有其他位置")
    print("2. Decoder 自注意力：掩码机制，只能看到当前位置和之前的位置")
    print("3. 交叉注意力：Decoder 的每个位置关注 Encoder 的不同位置")


def demo_positional_encoding_effect():
    """演示位置编码的影响"""
    print("\n" + "=" * 60)
    print("  位置编码效果演示")
    print("=" * 60)

    from transformer_model import PositionalEncoding

    d_model = 32
    max_len = 30

    pe = PositionalEncoding(d_model, max_len)

    # 创建两个相同的输入，但位置不同
    x1 = torch.ones(1, 5, d_model)
    x2 = torch.ones(1, 5, d_model) * 2

    # 添加位置编码
    x1_pe = pe(x1)
    x2_pe = pe(x2)

    print(f"\n输入 1 (值为 1): 第一个 token 的前 5 维：{x1_pe[0, 0, :5].numpy()}")
    print(f"输入 2 (值为 2): 第一个 token 的前 5 维：{x2_pe[0, 0, :5].numpy()}")

    # 可视化位置编码随位置的变化
    pe_matrix = pe.pe[0, :max_len, :].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 热力图
    sns.heatmap(pe_matrix, ax=axes[0], cmap='coolwarm', center=0)
    axes[0].set_title('位置编码矩阵')
    axes[0].set_xlabel('维度')
    axes[0].set_ylabel('位置')

    # 不同位置的编码向量
    positions = np.arange(max_len)
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i in range(0, d_model, 4):
        axes[1].plot(positions, pe_matrix[:, i], label=f'维度 {i}', color=colors[i // 4 % 5])

    axes[1].set_title('不同维度的位置编码曲线')
    axes[1].set_xlabel('位置')
    axes[1].set_ylabel('编码值')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('positional_encoding_demo.png', dpi=150, bbox_inches='tight')
    print("\n位置编码可视化已保存到：positional_encoding_demo.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    demo_attention_patterns()
    demo_positional_encoding_effect()

    print("\n" + "=" * 60)
    print("  可视化演示完成!")
    print("=" * 60)
