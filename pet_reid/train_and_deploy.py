"""
完整训练和部署流程

功能：
1. DINOv3自监督预训练
2. 导出ONNX模型
3. 推送到GitHub

用法：
    python train_and_deploy.py
    python train_and_deploy.py --epochs 10 --skip_github
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    return result.returncode == 0


def step1_train_dino(args):
    """步骤1: DINOv3预训练"""
    print("\n" + "=" * 60)
    print("步骤1: DINOv3 自监督预训练")
    print("=" * 60)

    cmd = f"""python train_dino_datasets.py \
        --epochs {args.epochs} \
        --batch_size {args.batch_size} \
        --lr {args.lr} \
        --save_interval {args.save_interval} \
        --log_interval {args.log_interval}"""

    success = run_command(cmd)

    if success:
        print("\n✓ DINOv3预训练完成!")
        return True
    else:
        print("\n✗ DINOv3预训练失败!")
        return False


def step2_export_onnx(args):
    """步骤2: 导出ONNX模型"""
    print("\n" + "=" * 60)
    print("步骤2: 导出ONNX模型")
    print("=" * 60)

    # 查找最佳模型
    model_path = 'checkpoints/dino/best_dino.pth'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return False

    # 修改export_onnx.py以支持DINOv3模型
    # 需要创建一个专门的导出脚本
    export_script = create_export_script()
    with open('export_dino_onnx.py', 'w', encoding='utf-8') as f:
        f.write(export_script)

    cmd = f"""python export_dino_onnx.py \
        --model {model_path} \
        --output_dir outputs/onnx \
        --proj_dim 512"""

    success = run_command(cmd)

    if success:
        print("\n✓ ONNX导出完成!")
        return True
    else:
        print("\n✗ ONNX导出失败!")
        return False


def create_export_script():
    """创建DINOv3模型的ONNX导出脚本"""
    return '''"""
DINOv3模型 ONNX导出脚本

将预训练的DINOv3 Student模型导出为ONNX格式
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models.backbone import CNNBackbone, GeMPooling


class DINOFeatureExtractor(nn.Module):
    """DINOv3特征提取器（用于ONNX导出）

    只包含backbone + projector，不包含predictor
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
            x: (B, 3, 224, 224) 输入图像

        Returns:
            features: (B, proj_dim) L2归一化的特征
        """
        feat = self.backbone(x)
        proj = self.projector(feat)
        # L2归一化
        proj = nn.functional.normalize(proj, p=2, dim=1)
        return proj


def export_onnx(model_path, output_dir, proj_dim=512, opset_version=11):
    """导出ONNX模型

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        proj_dim: 投影维度
        opset_version: ONNX opset版本
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载预训练权重
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')

    # 创建模型
    model = DINOFeatureExtractor(proj_dim=proj_dim)

    # 加载Student的backbone和projector权重
    if 'student_backbone' in ckpt:
        model.backbone.load_state_dict(ckpt['student_backbone'], strict=False)
        print("✓ 加载student_backbone权重")
    if 'student_projector' in ckpt:
        # 需要适配权重
        proj_state = ckpt['student_projector']
        model.projector.load_state_dict(proj_state, strict=False)
        print("✓ 加载student_projector权重")

    model.eval()

    # 创建dummy输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出文件名
    base_name = Path(model_path).stem
    onnx_path = os.path.join(output_dir, f'{base_name}.onnx')

    # 导出ONNX
    print(f"导出ONNX: {onnx_path}")
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

    # 验证ONNX模型
    print("验证ONNX模型...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        dummy_np = dummy_input.numpy()
        outputs = session.run([output_name], {input_name: dummy_np})

        print(f"✓ ONNX验证成功!")
        print(f"  输入形状: {dummy_np.shape}")
        print(f"  输出形状: {outputs[0].shape}")
        print(f"  特征维度: {outputs[0].shape[1]}")

        # 测试L2归一化
        feature_norm = (outputs[0] ** 2).sum() ** 0.5
        print(f"  特征范数: {feature_norm:.4f} (应该是1.0)")

    except ImportError:
        print("跳过验证 (需要安装 onnxruntime)")

    # 尝试简化
    try:
        from onnxsim import simplify as onnx_simplify
        import onnx

        print("简化ONNX模型...")
        onnx_model = onnx.load(onnx_path)
        onnx_model_sim, check = onnx_simplify(onnx_model)

        if check:
            onnx_path_sim = os.path.join(output_dir, f'{base_name}_sim.onnx')
            onnx.save(onnx_model_sim, onnx_path_sim)
            print(f"✓ 简化模型保存: {onnx_path_sim}")
    except ImportError:
        pass

    print(f"\\nONNX导出完成!")
    print(f"模型文件: {onnx_path}")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='DINOv3 ONNX导出')
    parser.add_argument('--model', type=str, default='checkpoints/dino/best_dino.pth',
                       help='模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs/onnx',
                       help='输出目录')
    parser.add_argument('--proj_dim', type=int, default=512,
                       help='投影维度')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset版本')

    args = parser.parse_args()

    export_onnx(
        model_path=args.model,
        output_dir=args.output_dir,
        proj_dim=args.proj_dim,
        opset_version=args.opset_version
    )


if __name__ == '__main__':
    main()
'''


def step3_push_to_github(args):
    """步骤3: 推送到GitHub"""
    print("\n" + "=" * 60)
    print("步骤3: 推送到GitHub")
    print("=" * 60)

    # 切换到项目根目录
    os.chdir('D:/claude_workspace')

    # 添加pet_reid文件夹
    print("添加文件到Git...")
    if not run_command('git add pet_reid/'):
        print("Git add失败")
        return False

    # 检查状态
    run_command('git status')

    # 提交
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    commit_message = f"feat: 添加pet_reid宠物Re-ID系统 (DINOv3自监督训练)"

    print(f"\n提交更改: {commit_message}")
    if not run_command(f'git commit -m "{commit_message}"'):
        print("Git commit失败")
        return False

    # 推送
    print("\n推送到GitHub...")
    if not run_command('git push origin master'):
        print("Git push失败")
        return False

    print("\n✓ GitHub推送完成!")
    return True


def create_summary(args, train_success, export_success, push_success):
    """创建训练总结"""
    print("\n" + "=" * 60)
    print("训练和部署总结")
    print("=" * 60)

    summary = f"""
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

步骤1: DINOv3预训练
  状态: {'✓ 成功' if train_success else '✗ 失败'}
  参数:
    - 数据集: ../pet_rec/datasets (25000张图片)
    - 训练轮数: {args.epochs}
    - 批大小: {args.batch_size}
    - 学习率: {args.lr}
    - 输出维度: 512

步骤2: ONNX导出
  状态: {'✓ 成功' if export_success else '✗ 失败'}
  输出: outputs/onnx/

步骤3: GitHub推送
  状态: {'✓ 成功' if push_success else '✗ 失败'}

输出文件:
  - 模型: checkpoints/dino/best_dino.pth
  - ONNX: outputs/onnx/best_dino.onnx
  - 日志: logs/dino/
  - 训练曲线: logs/dino/*.png

下一步:
  1. 查看训练日志: cat logs/dino/*.log
  2. 查看训练曲线: 打开logs/dino/*.png
  3. 测试ONNX模型: python inference.py --model checkpoints/dino/best_dino.pth
  4. (可选) Re-ID微调: python train_reid.py --pretrained_dino checkpoints/dino/best_dino.pth
"""
    print(summary)

    # 保存总结到文件
    with open('TRAINING_SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description='完整训练和部署流程')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='学习率')
    parser.add_argument('--save_interval', type=int, default=20,
                       help='保存间隔')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志间隔')

    # 流程控制
    parser.add_argument('--skip_train', action='store_true',
                       help='跳过训练（使用已有模型）')
    parser.add_argument('--skip_export', action='store_true',
                       help='跳过ONNX导出')
    parser.add_argument('--skip_github', action='store_true',
                       help='跳过GitHub推送')
    parser.add_argument('--small_test', action='store_true',
                       help='小规模测试（10 epochs）')

    args = parser.parse_args()

    # 小规模测试
    if args.small_test:
        args.epochs = 10
        args.batch_size = 64
        args.save_interval = 5
        print("小规模测试模式: epochs=10, batch_size=64")

    print("=" * 60)
    print("Pet Re-ID 完整训练和部署流程")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_success = False
    export_success = False
    push_success = False

    # 步骤1: 训练
    if not args.skip_train:
        train_success = step1_train_dino(args)
    else:
        print("\n跳过训练步骤")
        train_success = os.path.exists('checkpoints/dino/best_dino.pth')

    # 步骤2: ONNX导出
    if not args.skip_export and train_success:
        export_success = step2_export_onnx(args)
    else:
        if args.skip_export:
            print("\n跳过ONNX导出步骤")
        export_success = True

    # 步骤3: GitHub推送
    if not args.skip_github:
        push_success = step3_push_to_github(args)
    else:
        print("\n跳过GitHub推送步骤")
        push_success = True

    # 总结
    create_summary(args, train_success, export_success, push_success)

    # 返回状态码
    if train_success and export_success and push_success:
        print("\n✓ 所有步骤完成!")
        return 0
    else:
        print("\n✗ 部分步骤失败，请检查日志")
        return 1


if __name__ == '__main__':
    sys.exit(main())
