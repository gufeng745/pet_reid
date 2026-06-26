"""
加载预训练模型示例

展示如何从本地加载预训练模型进行特征提取

用法：
    python example_load_pretrained.py
    python example_load_pretrained.py --image test.png
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pretrained import load_pretrained_dino, get_manager


def get_transform():
    """获取图像预处理"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def extract_feature(model, image_path, device='cpu'):
    """提取单张图片的特征

    Args:
        model: 预训练模型
        image_path: 图片路径
        device: 设备

    Returns:
        feature: (512,) 特征向量
    """
    # 加载图片
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 提取特征
    model.eval()
    with torch.no_grad():
        feature = model(image_tensor)

    return feature.squeeze(0).cpu()


def compute_similarity(feature1, feature2):
    """计算两个特征的余弦相似度

    Args:
        feature1: (D,) 特征向量
        feature2: (D,) 特征向量

    Returns:
        similarity: 余弦相似度
    """
    return torch.cosine_similarity(feature1, feature2, dim=0).item()


def demo_single_image(model, image_path):
    """演示：单张图片特征提取"""
    print(f"\n{'='*60}")
    print("演示: 单张图片特征提取")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        print("请使用 --image 参数指定图片路径")
        return

    # 提取特征
    feature = extract_feature(model, image_path)

    print(f"\n图片: {image_path}")
    print(f"特征维度: {feature.shape}")
    print(f"特征范数: {torch.norm(feature).item():.4f}")
    print(f"特征范围: [{feature.min().item():.4f}, {feature.max().item():.4f}]")
    print(f"特征均值: {feature.mean().item():.4f}")

    # 显示前10个特征值
    print(f"\n前10个特征值:")
    for i in range(10):
        print(f"  [{i:3d}] {feature[i].item():.4f}")


def demo_similarity(model, image1_path, image2_path):
    """演示：计算两张图片的相似度"""
    print(f"\n{'='*60}")
    print("演示: 计算图片相似度")
    print(f"{'='*60}")

    if not os.path.exists(image1_path):
        print(f"图片1不存在: {image1_path}")
        return
    if not os.path.exists(image2_path):
        print(f"图片2不存在: {image2_path}")
        return

    # 提取特征
    feature1 = extract_feature(model, image1_path)
    feature2 = extract_feature(model, image2_path)

    # 计算相似度
    similarity = compute_similarity(feature1, feature2)

    print(f"\n图片1: {image1_path}")
    print(f"图片2: {image2_path}")
    print(f"\n余弦相似度: {similarity:.4f}")

    # 判断
    if similarity > 0.8:
        print("判断: 非常相似 (可能是同一只宠物)")
    elif similarity > 0.5:
        print("判断: 比较相似 (可能是同一品种)")
    elif similarity > 0.3:
        print("判断: 有一定相似性")
    else:
        print("判断: 差异较大")


def demo_feature_extraction(model, image_dir):
    """演示：批量特征提取"""
    print(f"\n{'='*60}")
    print("演示: 批量特征提取")
    print(f"{'='*60}")

    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return

    # 查找图片
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []

    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(image_dir, file))

    if not image_paths:
        print(f"在 {image_dir} 中未找到图片")
        return

    print(f"\n找到 {len(image_paths)} 张图片")

    # 提取特征
    features = []
    for i, image_path in enumerate(image_paths[:5]):  # 只处理前5张
        feature = extract_feature(model, image_path)
        features.append(feature)
        print(f"  [{i+1}] {os.path.basename(image_path)}: {feature.shape}")

    # 计算相似度矩阵
    print(f"\n相似度矩阵:")
    n = len(features)
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_similarity(features[i], features[j])
            print(f"  {os.path.basename(image_paths[i])} <-> {os.path.basename(image_paths[j])}: {sim:.4f}")


def main():
    parser = argparse.ArgumentParser(description='预训练模型使用示例')

    parser.add_argument('--image', type=str, default=None,
                       help='单张图片路径')
    parser.add_argument('--image1', type=str, default=None,
                       help='图片1路径（用于相似度计算）')
    parser.add_argument('--image2', type=str, default=None,
                       help='图片2路径（用于相似度计算）')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图片目录（用于批量特征提取）')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')

    args = parser.parse_args()

    # 检查预训练模型
    print("=" * 60)
    print("预训练模型加载示例")
    print("=" * 60)

    # 检查模型状态
    manager = get_manager()
    status = manager.check_models()

    if not status.get('dino_mobilenetv3_large', False):
        print("\n错误: DINOv3预训练模型未找到!")
        print("\n请先设置预训练模型:")
        print("  1. 运行: python setup_pretrained.py --create_dirs")
        print("  2. 将预训练模型放置到: pretrained_models/dino/best_dino.pth")
        print("  3. 或从checkpoints复制: python setup_pretrained.py --source checkpoints/dino")
        return

    # 加载模型
    print("\n加载预训练模型...")
    device = args.device
    model = load_pretrained_dino(
        model_name='dino_mobilenetv3_large',
        proj_dim=512
    )
    model = model.to(device)
    model.eval()
    print("模型加载完成!")

    # 运行演示
    if args.image:
        demo_single_image(model, args.image)
    elif args.image1 and args.image2:
        demo_similarity(model, args.image1, args.image2)
    elif args.image_dir:
        demo_feature_extraction(model, args.image_dir)
    else:
        # 默认演示
        print("\n" + "=" * 60)
        print("使用方法:")
        print("=" * 60)
        print("\n1. 单张图片特征提取:")
        print("   python example_load_pretrained.py --image test.png")
        print("\n2. 计算两张图片相似度:")
        print("   python example_load_pretrained.py --image1 img1.png --image2 img2.png")
        print("\n3. 批量特征提取:")
        print("   python example_load_pretrained.py --image_dir ./images")
        print("\n4. 使用GPU:")
        print("   python example_load_pretrained.py --image test.png --device cuda")


if __name__ == '__main__':
    main()
