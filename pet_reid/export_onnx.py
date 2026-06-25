"""
ONNX 导出脚本

将Re-ID模型导出为ONNX格式，支持：
- FP32导出
- INT8量化（可选）
- 模型验证

用法：
    python export_onnx.py --model checkpoints/reid/best_reid.pth
    python export_onnx.py --model checkpoints/reid/best_reid.pth --int8
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models.reid_model import ReIDModel


def export_onnx(
    model_path: str,
    output_dir: str = 'outputs/onnx',
    opset_version: int = 11,
    simplify: bool = True,
    int8_quantize: bool = False,
    input_height: int = 224,
    input_width: int = 224
):
    """导出ONNX模型

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        opset_version: ONNX opset版本
        simplify: 是否简化模型
        int8_quantize: 是否INT8量化
        input_height: 输入高度
        input_width: 输入宽度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = ReIDModel.from_pretrained(model_path)
    model.eval()

    # 创建dummy输入
    dummy_input = torch.randn(1, 3, input_height, input_width)

    # 导出文件名
    base_name = Path(model_path).stem
    onnx_path = os.path.join(output_dir, f'{base_name}.onnx')

    # 导出ONNX
    print(f"\n导出ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )

    # 简化模型
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("简化ONNX模型...")
            onnx_model = onnx.load(onnx_path)
            onnx_model_sim, check = onnx_simplify(onnx_model)

            if check:
                onnx_path_sim = os.path.join(output_dir, f'{base_name}_sim.onnx')
                onnx.save(onnx_model_sim, onnx_path_sim)
                print(f"简化模型保存: {onnx_path_sim}")
            else:
                print("警告: 模型简化失败")
        except ImportError:
            print("跳过简化 (需要安装 onnxsim: pip install onnxsim)")

    # INT8量化
    if int8_quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            print("\nINT8量化...")
            onnx_path_int8 = os.path.join(output_dir, f'{base_name}_int8.onnx')
            quantize_dynamic(
                onnx_path,
                onnx_path_int8,
                weight_type=QuantType.QUInt8
            )
            print(f"INT8模型保存: {onnx_path_int8}")
        except ImportError:
            print("跳过INT8量化 (需要安装 onnxruntime)")

    # 验证ONNX模型
    print("\n验证ONNX模型...")
    try:
        import onnxruntime as ort

        # 创建推理会话
        session = ort.InferenceSession(onnx_path)

        # 推理测试
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # 使用dummy输入测试
        dummy_np = dummy_input.numpy()
        outputs = session.run([output_name], {input_name: dummy_np})

        print(f"ONNX推理成功!")
        print(f"  输入形状: {dummy_np.shape}")
        print(f"  输出形状: {outputs[0].shape}")

        # 对比PyTorch和ONNX输出
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        onnx_output = outputs[0]
        diff = np.abs(pytorch_output - onnx_output).max()
        print(f"  最大差异: {diff:.6f}")

        if diff < 1e-5:
            print("  验证通过 ✓")
        else:
            print("  警告: 差异较大，请检查模型")

    except ImportError:
        print("跳过验证 (需要安装 onnxruntime)")

    print("\n" + "=" * 50)
    print("导出完成!")
    print(f"ONNX模型: {onnx_path}")
    print("=" * 50)


def parse_args():
    p = argparse.ArgumentParser(description='ONNX 导出')

    p.add_argument('--model', type=str, default='checkpoints/reid/best_reid.pth',
                   help='模型路径')
    p.add_argument('--output_dir', type=str, default='outputs/onnx',
                   help='输出目录')
    p.add_argument('--opset_version', type=int, default=11,
                   help='ONNX opset版本')
    p.add_argument('--simplify', action='store_true', default=True,
                   help='是否简化模型')
    p.add_argument('--int8', action='store_true', default=False,
                   help='是否INT8量化')
    p.add_argument('--input_height', type=int, default=224,
                   help='输入高度')
    p.add_argument('--input_width', type=int, default=224,
                   help='输入宽度')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    export_onnx(
        model_path=args.model,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
        simplify=args.simplify,
        int8_quantize=args.int8,
        input_height=args.input_height,
        input_width=args.input_width
    )
