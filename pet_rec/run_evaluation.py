import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from evaluate import run_full_evaluation

print("Starting evaluation with ONNX model...")
result = run_full_evaluation(
    model_path='pet_mobilenetv2.onnx',
    dataset_dir='test_dataset',
    output_dir='evaluation_results',
    use_attr_model=True,  # 使用属性模型 (MobileNetV2StudentWithAttr)
    use_onnx=True  # 使用 ONNX Runtime 进行推理
)
print(f"Evaluation complete: {result}")
