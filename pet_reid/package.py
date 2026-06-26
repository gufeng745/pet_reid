"""
打包脚本

将pet_reid文件夹打包成zip文件，便于下载和分发
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path


def create_package(
    output_name='pet_reid_package.zip',
    include_models=False,
    include_logs=True
):
    """创建打包文件

    Args:
        output_name: 输出文件名
        include_models: 是否包含模型文件（checkpoint和onnx）
        include_logs: 是否包含日志文件
    """
    print("=" * 60)
    print("打包 pet_reid 文件夹")
    print("=" * 60)

    # 排除的文件和目录
    exclude_dirs = {
        '__pycache__',
        '.git',
        'outputs',
    }

    exclude_files = {
        'temp_export.py',
        '.gitignore',
    }

    # 模型文件（可选包含）
    model_patterns = [
        'checkpoints/',
        'outputs/',
        '*.pth',
        '*.onnx',
    ]

    # 日志文件（可选包含）
    log_patterns = [
        'logs/',
        '*.log',
        '*.json',
    ]

    # 创建zip文件
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        file_count = 0

        for root, dirs, files in os.walk('.'):
            # 排除目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                # 排除文件
                if file in exclude_files:
                    continue

                file_path = os.path.join(root, file)
                arc_path = os.path.join('pet_reid', file_path[2:])  # 去掉 './'

                # 检查是否是模型文件
                is_model = any(
                    pattern in file_path or file_path.endswith(pattern.replace('*', ''))
                    for pattern in model_patterns
                )

                # 检查是否是日志文件
                is_log = any(
                    pattern in file_path or file_path.endswith(pattern.replace('*', ''))
                    for pattern in log_patterns
                )

                # 根据选项决定是否包含
                if is_model and not include_models:
                    continue
                if is_log and not include_logs:
                    continue

                # 添加到zip
                zipf.write(file_path, arc_path)
                file_count += 1
                print(f"  添加: {arc_path}")

    # 获取文件大小
    size_mb = os.path.getsize(output_name) / (1024 * 1024)

    print(f"\n打包完成!")
    print(f"  文件名: {output_name}")
    print(f"  大小: {size_mb:.1f} MB")
    print(f"  文件数: {file_count}")

    return output_name


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='打包pet_reid文件夹')
    parser.add_argument('--output', type=str, default='pet_reid_package.zip',
                       help='输出文件名')
    parser.add_argument('--include_models', action='store_true',
                       help='包含模型文件（checkpoint和onnx）')
    parser.add_argument('--include_logs', action='store_true', default=True,
                       help='包含日志文件')
    parser.add_argument('--lightweight', action='store_true',
                       help='轻量级打包（不含模型和日志）')

    args = parser.parse_args()

    if args.lightweight:
        args.include_models = False
        args.include_logs = False

    create_package(
        output_name=args.output,
        include_models=args.include_models,
        include_logs=args.include_logs
    )


if __name__ == '__main__':
    main()
