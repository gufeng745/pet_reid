"""
DINOv3 预训练脚本 - 使用datasets文件夹

使用datasets文件夹的37501张图片进行自监督预训练
不需要标签，只需要大量宠物图片

用法：
    python train_dino_datasets.py
    python train_dino_datasets.py --epochs 100 --batch_size 128
"""

import os
import sys
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from train_dino import train_dino


def main():
    """使用datasets文件夹进行DINOv3预训练"""
    print("=" * 60)
    print("DINOv3 自监督预训练 - 使用datasets文件夹")
    print("=" * 60)

    # 检查datasets文件夹
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'cat_dog_attr')
    if not os.path.exists(data_root):
        print(f"错误：datasets文件夹不存在: {data_root}")
        print("请确保路径正确")
        return

    # 统计图片数量
    # train_dir = os.path.join(data_root, 'train')
    train_dir = data_root
    if os.path.isdir(train_dir):
        num_images = len([f for f in os.listdir(train_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"找到 {num_images} 张图片")
    else:
        print(f"错误：train目录不存在: {train_dir}")
        return

    print(f"\n数据集路径: {data_root}")
    print(f"图片数量: {num_images}")
    print(f"训练模式: 自监督（不需要标签）")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DINOv3预训练 - 使用datasets文件夹')

    # 数据
    parser.add_argument('--data_root', type=str, default=data_root,
                       help='数据集根目录')

    # 模型
    parser.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                       help='CNN backbone')
    parser.add_argument('--proj_dim', type=int, default=512,
                       help='投影维度（输出维度）')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                       help='隐藏层维度')
    parser.add_argument('--backbone_weight_path', type=str, default='pre_weights/mobilenetv3_large_100.safetensors',
                       help='backbone 预训练权重路径 (safetensors)')

    # 训练
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                       help='权重衰减')

    # DINOv3参数
    parser.add_argument('--teacher_momentum_start', type=float, default=0.996,
                       help='Teacher EMA动量起始值')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                       help='Teacher温度')
    parser.add_argument('--student_temp', type=float, default=0.1,
                       help='Student温度')

    # MAE
    parser.add_argument('--use_mae_masking', action='store_true', default=True)
    parser.add_argument('--no_mae_masking', dest='use_mae_masking', action='store_false')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75)
    parser.add_argument('--mae_mask_patch_size', type=int, default=16)

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--save_interval', type=int, default=20, help='保存间隔')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')

    args = parser.parse_args()

    # 开始训练
    print("\n" + "=" * 60)
    print("开始DINOv3预训练...")
    print("=" * 60)

    model = train_dino(args)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 模型: checkpoints/dino/best_dino.pth")
    print(f"  - 日志: logs/dino/")
    print(f"  - 训练曲线: logs/dino/*.png")
    print(f"\n下一步:")
    print(f"  可以使用预训练模型进行Re-ID微调:")
    print(f"  python train_reid.py --pretrained_dino checkpoints/dino/best_dino.pth")


if __name__ == '__main__':
    main()
