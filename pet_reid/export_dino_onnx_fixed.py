"""
DINOv3模型 ONNX导出脚本 (固定输入输出形状)

输入: (1, 3, 224, 224)
输出: (1, 512)

用法：
    python export_dino_onnx_fixed.py
    python export_dino_onnx_fixed.py --model checkpoints/dino/best_dino.pth
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models.backbone import CNNBackbone


class DINOFeatureExtractor(nn.Module):
    """DINOv3特征提取器（固定输出形状）

    输入: (1, 3, 224, 224)
    输出: (1, 512)
    """

    def __init__(self, backbone_name='mobilenetv3_large_100', proj_dim=512):
        super().__init__()

        # Backbone
        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=False
        )
        feat_dim = self.backbone.feature_dim

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x):
        """提取特征

        Args:
            x: (1, 3, 224, 224) 输入图像

        Returns:
            features: (1, 512) L2归一化的特征
        """
        feat = self.backbone(x)
        proj = self.projector(feat)
        # L2归一化
        proj = nn.functional.normalize(proj, p=2, dim=1)
        return proj


def export_onnx_fixed(
    model_path,
    output_path='outputs/onnx/best_dino.onnx',
    proj_dim=512,
    opset_version=11
):
    """导出固定形状的ONNX模型

    Args:
        model_path: 模型路径
        output_path: 输出路径
        proj_dim: 投影维度
        opset_version: ONNX opset版本
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载预训练权重
    print(f"Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')

    # 创建模型
    model = DINOFeatureExtractor(proj_dim=proj_dim)

    # 加载Student的backbone和projector权重
    if 'student_backbone' in ckpt:
        model.backbone.load_state_dict(ckpt['student_backbone'], strict=False)
        print("[OK] Loaded student_backbone weights")
    if 'student_projector' in ckpt:
        model.projector.load_state_dict(ckpt['student_projector'], strict=False)
        print("[OK] Loaded student_projector weights")

    model.eval()

    # 创建固定形状的dummy输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出ONNX（固定形状，不使用dynamic_axes）
    print(f"\nExporting ONNX: {output_path}")
    print(f"Input shape: (1, 3, 224, 224)")
    print(f"Output shape: (1, {proj_dim})")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['embedding'],
        # 不使用dynamic_axes，保持固定形状
    )

    # 验证ONNX模型
    print("\nVerifying ONNX model...")
    try:
        import onnxruntime as ort
        import numpy as np

        # 创建推理会话
        session = ort.InferenceSession(output_path)

        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        print(f"Input: {input_name}, shape: {input_shape}")
        print(f"Output: {output_name}, shape: {output_shape}")

        # 测试推理
        dummy_np = dummy_input.numpy()
        outputs = session.run([output_name], {input_name: dummy_np})

        print(f"\nTest inference:")
        print(f"  Input shape: {dummy_np.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output norm: {np.linalg.norm(outputs[0]):.4f}")

        # 对比PyTorch和ONNX输出
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        onnx_output = outputs[0]
        diff = np.abs(pytorch_output - onnx_output).max()
        print(f"  Max diff: {diff:.6f}")

        if diff < 1e-5:
            print("  Verification: PASSED")
        else:
            print("  Verification: WARNING (diff > 1e-5)")

    except ImportError:
        print("Skipping verification (install onnxruntime: pip install onnxruntime)")

    # 获取文件大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nExport completed!")
    print(f"Output: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Input: (1, 3, 224, 224)")
    print(f"Output: (1, {proj_dim})")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='DINOv3 ONNX Export (Fixed Shape)')

    parser.add_argument('--model', type=str, default='checkpoints/dino/best_dino.pth',
                       help='Model path')
    parser.add_argument('--output', type=str, default='outputs/onnx/best_dino.onnx',
                       help='Output path')
    parser.add_argument('--proj_dim', type=int, default=512,
                       help='Projection dimension')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')

    args = parser.parse_args()

    export_onnx_fixed(
        model_path=args.model,
        output_path=args.output,
        proj_dim=args.proj_dim,
        opset_version=args.opset_version
    )


if __name__ == '__main__':
    main()
