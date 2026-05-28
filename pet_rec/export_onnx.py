import os
import sys
import torch
import numpy as np
from models import MobileNetV2Student


def export_onnx(student_path, proj_dim=512, output_dir=None):
    """导出 MobileNetV2 student 到 ONNX"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Load student
    model = MobileNetV2Student(proj_dim=proj_dim)
    ckpt = torch.load(student_path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'student' in ckpt:
        model.load_state_dict(ckpt['student'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Export FP32
    fp32_path = os.path.join(output_dir, 'pet_mobilenetv2.onnx')
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={'image': {0: 'batch'}, 'features': {0: 'batch'}},
        opset_version=13,
        dynamo=False,
    )
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.1f} MB)")

    # INT8 dynamic quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, 'pet_mobilenetv2_int8.onnx')
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        print(f"INT8 ONNX:  {int8_path} ({int8_size:.1f} MB)")

        # Verify quality
        verify_quantization(fp32_path, int8_path)
    except ImportError:
        print("onnxruntime.quantization 不可用，跳过 INT8 量化")

    return fp32_path


def verify_quantization(fp32_path, int8_path, num_samples=20):
    """验证量化前后特征一致性"""
    import onnxruntime as ort

    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_int8 = ort.InferenceSession(int8_path)

    input_name = sess_fp32.get_inputs()[0].name
    cos_sims = []

    for _ in range(num_samples):
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        feat_fp32 = sess_fp32.run(None, {input_name: x})[0]
        feat_int8 = sess_int8.run(None, {input_name: x})[0]

        # L2 normalize
        feat_fp32 = feat_fp32 / (np.linalg.norm(feat_fp32, axis=-1, keepdims=True) + 1e-8)
        feat_int8 = feat_int8 / (np.linalg.norm(feat_int8, axis=-1, keepdims=True) + 1e-8)

        cos = (feat_fp32 * feat_int8).sum()
        cos_sims.append(cos)

    mean_cos = np.mean(cos_sims)
    print(f"FP32 vs INT8 cosine similarity: {mean_cos:.4f} (target > 0.99)")
    if mean_cos < 0.99:
        print("WARNING: 量化精度低于预期")
    else:
        print("量化质量验证通过")


def benchmark_latency(onnx_path, num_runs=100):
    """测量推理延迟"""
    import onnxruntime as ort
    import time

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: x})

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_runs):
        sess.run(None, {input_name: x})
    elapsed = (time.perf_counter() - t0) / num_runs * 1000

    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"{os.path.basename(onnx_path)}: {elapsed:.1f}ms/image, {model_size:.1f}MB")
    return elapsed


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_student.pth'
    fp32_path = export_onnx(ckpt_path)

    print("\n=== Latency Benchmark ===")
    benchmark_latency(fp32_path)
    int8_path = fp32_path.replace('.onnx', '_int8.onnx')
    if os.path.exists(int8_path):
        benchmark_latency(int8_path)
