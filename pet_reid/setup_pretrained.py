"""
预训练模型设置脚本

帮助用户将预训练模型放置到正确的位置

用法：
    # 从现有的checkpoints目录设置
    python setup_pretrained.py --source checkpoints/dino

    # 指定目标目录
    python setup_pretrained.py --source checkpoints/dino --target pretrained_models

    # 列出可用的预训练模型
    python setup_pretrained.py --list

    # 检查预训练模型状态
    python setup_pretrained.py --check
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pretrained import (
    get_manager,
    setup_pretrained_models,
    PRETRAINED_MODELS,
)


def list_models():
    """列出所有可用的预训练模型"""
    manager = get_manager()
    manager.list_models()


def check_models():
    """检查预训练模型状态"""
    manager = get_manager()
    status = manager.check_models()

    print("\n预训练模型状态检查:")
    print("=" * 60)

    all_ok = True
    for name, is_available in status.items():
        model_info = PRETRAINED_MODELS[name]
        status_str = "[OK] 可用" if is_available else "[FAIL] 未找到"

        print(f"\n{name}:")
        print(f"  描述: {model_info['description']}")
        print(f"  状态: {status_str}")

        if not is_available and model_info.get('source') != 'timm':
            model_dir = os.path.join('pretrained_models', model_info['dir'])
            print(f"  期望位置: {model_dir}/{model_info['file']}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("所有预训练模型都可用!")
    else:
        print("部分预训练模型缺失，请下载或复制到指定位置")

    return all_ok


def setup_from_checkpoints(source_dir: str, target_dir: str = 'pretrained_models'):
    """从现有的checkpoints目录设置预训练模型

    Args:
        source_dir: 源目录（如 checkpoints/dino）
        target_dir: 目标目录（默认 pretrained_models）
    """
    print(f"\n从 checkpoints 设置预训练模型")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")

    # 检查源目录
    if not os.path.exists(source_dir):
        print(f"\n错误: 源目录不存在: {source_dir}")
        print("请先运行训练脚本生成预训练模型")
        return False

    # 查找模型文件
    model_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pth') or file.endswith('.onnx'):
                model_files.append(os.path.join(root, file))

    if not model_files:
        print(f"\n错误: 在 {source_dir} 中未找到模型文件")
        return False

    print(f"\n找到 {len(model_files)} 个模型文件:")
    for f in model_files:
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    # 复制文件
    print(f"\n复制文件到 {target_dir}/dino/ ...")
    target_dino_dir = os.path.join(target_dir, 'dino')
    os.makedirs(target_dino_dir, exist_ok=True)

    for model_file in model_files:
        filename = os.path.basename(model_file)
        target_file = os.path.join(target_dino_dir, filename)

        shutil.copy2(model_file, target_file)
        print(f"  [OK] {filename}")

    print(f"\n设置完成!")
    print(f"预训练模型已放置到: {target_dino_dir}/")

    # 验证
    print("\n验证预训练模型:")
    check_models()

    return True


def create_directories():
    """创建预训练模型目录结构"""
    manager = get_manager()

    print("\n创建预训练模型目录结构:")
    print("=" * 60)

    dirs = [
        'pretrained_models',
        'pretrained_models/dino',
        'pretrained_models/imagenet',
        'pretrained_models/custom',
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [OK] {d}/")

    print("\n目录结构创建完成!")
    print("\n请将预训练模型文件放置到对应目录:")
    print("  - DINOv3模型: pretrained_models/dino/best_dino.pth")
    print("  - ImageNet模型: pretrained_models/imagenet/")
    print("  - 自定义模型: pretrained_models/custom/")


def main():
    parser = argparse.ArgumentParser(description='预训练模型设置工具')

    parser.add_argument('--source', type=str, default=None,
                       help='源目录（如 checkpoints/dino）')
    parser.add_argument('--target', type=str, default='pretrained_models',
                       help='目标目录（默认 pretrained_models）')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的预训练模型')
    parser.add_argument('--check', action='store_true',
                       help='检查预训练模型状态')
    parser.add_argument('--create_dirs', action='store_true',
                       help='创建预训练模型目录结构')

    args = parser.parse_args()

    # 列出模型
    if args.list:
        list_models()
        return

    # 检查状态
    if args.check:
        check_models()
        return

    # 创建目录
    if args.create_dirs:
        create_directories()
        return

    # 从checkpoints设置
    if args.source:
        setup_from_checkpoints(args.source, args.target)
    else:
        # 默认行为：创建目录并显示帮助
        create_directories()
        print("\n" + "=" * 60)
        print("使用方法:")
        print("  1. 将预训练模型放置到对应目录")
        print("  2. 运行 python setup_pretrained.py --check 验证")
        print("  3. 在代码中使用 load_pretrained_dino() 加载模型")


if __name__ == '__main__':
    main()
