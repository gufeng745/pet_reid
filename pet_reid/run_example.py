"""
Pet Re-ID 快速开始示例

本脚本演示完整的训练流程：
1. DINOv3自监督预训练（小规模测试）
2. Re-ID监督微调

用法：
    python run_example.py
"""

import os
import sys
import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

def main():
    print("=" * 60)
    print("Pet Re-ID 快速开始示例")
    print("=" * 60)

    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 数据集路径
    data_root = '../pet_rec/reid_dataset'
    if not os.path.exists(data_root):
        print(f"\n错误: 数据集不存在: {data_root}")
        print("请确保数据集路径正确")
        return

    print(f"\n数据集路径: {data_root}")

    # 统计数据集
    from datasets.reid_dataset import split_dataset_by_id
    train_ids, val_ids = split_dataset_by_id(data_root, val_ratio=0.1)

    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_ids)} 个身份")
    print(f"  验证集: {len(val_ids)} 个身份")

    # 选择训练模式
    print("\n" + "=" * 60)
    print("选择训练模式:")
    print("=" * 60)
    print("1. DINOv3自监督预训练 (推荐先运行)")
    print("2. Re-ID监督微调")
    print("3. 完整流程 (DINOv3 + Re-ID)")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == '1':
        run_dino_pretraining(data_root)
    elif choice == '2':
        run_reid_training(data_root)
    elif choice == '3':
        run_full_pipeline(data_root)
    else:
        print("无效选择")


def run_dino_pretraining(data_root):
    """运行DINOv3预训练"""
    print("\n" + "=" * 60)
    print("DINOv3 自监督预训练")
    print("=" * 60)

    # 小规模测试参数
    cmd = f"""
    python train_dino.py \
        --data_root {data_root} \
        --backbone mobilenetv3_large_100 \
        --epochs 10 \
        --batch_size 16 \
        --lr 5e-4 \
        --proj_dim 384 \
        --save_interval 5
    """

    print("\n运行命令:")
    print(cmd)
    print("\n开始训练...")

    os.system(cmd)


def run_reid_training(data_root):
    """运行Re-ID微调"""
    print("\n" + "=" * 60)
    print("Re-ID 监督微调")
    print("=" * 60)

    # 检查是否有DINOv3预训练权重
    dino_path = 'checkpoints/dino/best_dino.pth'
    pretrained_arg = f'--pretrained_dino {dino_path}' if os.path.exists(dino_path) else ''

    # 小规模测试参数
    cmd = f"""
    python train_reid.py \
        --data_root {data_root} \
        --backbone mobilenetv3_large_100 \
        --epochs 20 \
        --batch_size 32 \
        --lr_backbone 5e-4 \
        --lr_head 1e-3 \
        --P 8 \
        --K 4 \
        --proj_dim 512 \
        --save_interval 10 \
        {pretrained_arg}
    """

    print("\n运行命令:")
    print(cmd)
    print("\n开始训练...")

    os.system(cmd)


def run_full_pipeline(data_root):
    """运行完整流程"""
    print("\n" + "=" * 60)
    print("完整训练流程")
    print("=" * 60)

    print("\n[Step 1/2] DINOv3 自监督预训练")
    print("-" * 40)
    run_dino_pretraining(data_root)

    print("\n[Step 2/2] Re-ID 监督微调")
    print("-" * 40)
    run_reid_training(data_root)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print("\n模型保存位置:")
    print("  - DINOv3预训练: checkpoints/dino/")
    print("  - Re-ID模型: checkpoints/reid/")
    print("\n评估命令:")
    print("  python evaluate.py --model checkpoints/reid/best_reid.pth")


if __name__ == '__main__':
    main()
