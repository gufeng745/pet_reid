"""快速启动训练 V2 脚本

预配置了优化后的超参数，直接运行即可开始训练。

用法：
    python run_train_v2.py
"""

import subprocess
import sys
import os


def main():
    """运行优化后的训练脚本"""

    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(current_dir, 'train_attr_v2.py')

    # 预配置的优化超参数
    args = [
        sys.executable, train_script,
        '--epochs', '80',
        '--batch_size', '64',
        '--num_workers', '4',

        # 学习率（双学习率策略）
        '--lr_backbone', '5e-4',
        '--lr_head', '1e-3',
        '--weight_decay', '0.04',
        '--warmup_epochs', '10',

        # 模型结构（启用所有优化）
        '--proj_dim', '512',
        '--use_se',  # SE 注意力
        '--use_bnneck',  # BNNeck

        # 损失函数权重（针对 color_primary 和 contrastive loss 偏高问题优化）
        '--alpha', '1.0',  # alignment loss
        '--beta', '0.5',  # self-similarity loss
        '--gamma', '0.1',  # uniformity loss
        '--lambda_color_pri', '0.5',  # 提升主色权重（从 0.2 提升到 0.5）
        '--lambda_color_sec', '0.15',
        '--lambda_pattern', '0.15',
        '--lambda_contrastive', '0.3',  # 降低对比损失权重（从 0.5 降低到 0.3）
        '--lambda_ortho', '0.05',
        '--contrastive_temp', '0.1',  # 提升温度系数（从 0.07 提升到 0.1）

        # 高级训练技巧
        '--use_label_smoothing',  # Label Smoothing
        '--use_amp',  # 混合精度训练
        '--max_grad_norm', '1.0',  # 梯度裁剪
        '--use_early_stopping',  # 早停机制
        '--patience', '15',  # 早停容忍度

        # 日志和保存
        '--save_interval', '10',
        '--log_interval', '10',
        '--save_report',
    ]

    print("=" * 60)
    print("Pet Re-ID 训练 V2 - 优化版")
    print("=" * 60)
    print("\n主要优化：")
    print("  ✓ SE 注意力模块增强通道交互")
    print("  ✓ BNNeck 提升度量学习效果")
    print("  ✓ Label Smoothing 防止过度自信")
    print("  ✓ 混合精度训练加速")
    print("  ✓ 梯度裁剪防止梯度爆炸")
    print("  ✓ 早停机制防止过拟合")
    print("  ✓ 优化损失函数权重")
    print("\n超参数优化：")
    print("  • 主色损失权重: 0.2 → 0.5 (提升)")
    print("  • 对比损失权重: 0.5 → 0.3 (降低)")
    print("  • 温度系数: 0.07 → 0.1 (提升)")
    print("\n" + "=" * 60)

    # 运行训练脚本
    subprocess.run(args)


if __name__ == '__main__':
    main()
