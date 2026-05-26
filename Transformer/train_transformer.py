"""
Transformer 训练演示 - 通过实际训练任务理解 Transformer

这个脚本将训练一个小型 Transformer 来完成简单的翻译任务
通过观察训练过程，深入理解 Transformer 的工作原理
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from transformer_model import Transformer, SMALL_CONFIG
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_training_data(num_samples=1000, seq_len=8, vocab_size=50):
    """
    创建简单的训练数据
    任务：学习一个映射关系（如：每个 token+1）
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randint(1, vocab_size - 10, (num_samples, seq_len))
    # 简单任务：输出是输入的循环移位（模拟翻译）
    y = torch.roll(X, shifts=-1, dims=1)

    return X, y


class TransformerTrainer:
    """Transformer 训练器"""

    def __init__(self, vocab_size, config, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{self.device}")

        self.model = Transformer(vocab_size, **config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Decoder 输入是目标序列去掉最后一个 token
            tgt_input = batch_y[:, :-1]
            # 真实标签是目标序列去掉第一个 token
            tgt_true = batch_y[:, 1:]

            self.optimizer.zero_grad()
            output, _ = self.model(batch_x, tgt_input)

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_true.reshape(-1))
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                tgt_input = batch_y[:, :-1]
                tgt_true = batch_y[:, 1:]

                output, _ = self.model(batch_x, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_true.reshape(-1))
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """完整训练流程"""
        train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
        val_dataset = torch.utils.data.TensorDataset(val_data[0], val_data[1])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"\n训练配置:")
        print(f"  训练样本：{len(train_dataset)}")
        print(f"  验证样本：{len(val_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")

        print("\n开始训练...")
        print("-" * 60)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                      f"LR = {self.scheduler.get_last_lr()[0]:.6f}")

        print("-" * 60)
        print("训练完成!")

        return self.model

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('损失 (Loss)')
        plt.title('训练曲线')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('training_curves.png', dpi=150)
        print("训练曲线已保存到：training_curves.png")
        plt.close()

    def visualize_attention(self, src, tgt):
        """可视化注意力权重"""
        self.model.eval()
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        with torch.no_grad():
            _, attn_info = self.model(src, tgt)

        # 可视化最后一层的注意力
        last_layer = -1

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Encoder 自注意力
        enc_attn = attn_info['encoder_attn'][last_layer][0, 0].cpu().numpy()
        im0 = axes[0].imshow(enc_attn, cmap='Blues')
        axes[0].set_title('Encoder 自注意力')
        axes[0].set_xlabel('Key 位置')
        axes[0].set_ylabel('Query 位置')
        plt.colorbar(im0, ax=axes[0])

        # Decoder 自注意力
        dec_self = attn_info['decoder_self_attn'][last_layer][0, 0].cpu().numpy()
        im1 = axes[1].imshow(dec_self, cmap='Blues')
        axes[1].set_title('Decoder 自注意力')
        axes[1].set_xlabel('Key 位置')
        axes[1].set_ylabel('Query 位置')
        plt.colorbar(im1, ax=axes[1])

        # Decoder 交叉注意力
        dec_cross = attn_info['decoder_cross_attn'][last_layer][0].mean(0).cpu().numpy()
        im2 = axes[2].imshow(dec_cross[:tgt.size(1), :src.size(1)], cmap='Blues')
        axes[2].set_title('Decoder 交叉注意力 (Encoder-Decoder)')
        axes[2].set_xlabel('Encoder 位置')
        axes[2].set_ylabel('Decoder 位置')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig('attention_visualization.png', dpi=150)
        print("注意力可视化已保存到：attention_visualization.png")
        plt.close()


def demonstrate_training():
    """演示完整训练过程"""
    print("=" * 60)
    print("  Transformer 训练演示")
    print("=" * 60)

    # 配置
    vocab_size = 50
    seq_len = 8
    config = SMALL_CONFIG
    config['d_model'] = 64  # 更小以便快速训练
    config['d_ff'] = 128
    config['num_layers'] = 2

    # 创建数据
    print("\n1. 创建训练数据...")
    X, y = create_training_data(num_samples=500, seq_len=seq_len, vocab_size=vocab_size)

    # 划分训练/验证集
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"   训练集：{len(X_train)} 样本")
    print(f"   验证集：{len(X_val)} 样本")

    # 创建训练器
    print("\n2. 初始化模型...")
    trainer = TransformerTrainer(vocab_size, config)

    # 训练
    trainer.train((X_train, y_train), (X_val, y_val), epochs=30, batch_size=16)

    # 可视化
    print("\n3. 生成可视化...")
    trainer.plot_training_curves()

    # 测试一些样本
    print("\n4. 测试模型预测...")
    trainer.model.eval()
    test_src = X_val[:3].to(trainer.device)
    test_tgt = y_val[:3, :-1].to(trainer.device)
    test_true = y_val[:3, 1:].to(trainer.device)

    with torch.no_grad():
        output, _ = trainer.model(test_src, test_tgt)
        pred = output.argmax(dim=-1)

    print("\n样本预测对比:")
    for i in range(3):
        print(f"\n样本 {i+1}:")
        print(f"  输入：    {test_src[i].cpu().numpy().tolist()}")
        print(f"  真实输出：{test_true[i].cpu().numpy().tolist()}")
        print(f"  预测输出：{pred[i].cpu().numpy().tolist()}")
        correct = (pred[i] == test_true[i]).sum().item()
        total = len(pred[i])
        print(f"  准确率：{correct}/{total} = {100*correct/total:.1f}%")

    # 可视化注意力
    trainer.visualize_attention(test_src[:1], test_tgt[:1])

    print("\n" + "=" * 60)
    print("训练演示完成!")
    print("=" * 60)

    return trainer


if __name__ == "__main__":
    trainer = demonstrate_training()
