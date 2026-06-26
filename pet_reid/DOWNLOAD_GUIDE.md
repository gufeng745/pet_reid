# Pet Re-ID 下载指南

## 📦 可用版本

### 1. 轻量级版本（推荐）
- **文件**: `pet_reid_lightweight.zip`
- **大小**: 74 KB
- **位置**: GitHub仓库 `pet_reid/pet_reid_lightweight.zip`
- **包含**:
  - ✅ 完整Python代码
  - ✅ 配置文件
  - ✅ 文档（README.md, TRAINING_REPORT.md等）
  - ❌ 不包含模型文件
  - ❌ 不包含训练日志

**适用场景:**
- 想要自己训练模型
- 学习代码结构
- 快速部署

### 2. 完整版本
- **文件**: `pet_reid_full.zip`
- **大小**: 90 MB
- **位置**: 本地 `pet_reid/pet_reid_full.zip`
- **包含**:
  - ✅ 完整Python代码
  - ✅ 配置文件
  - ✅ 文档
  - ✅ PyTorch模型 (best_dino.pth, 97MB)
  - ✅ 训练日志和指标
  - ❌ 不包含ONNX模型

**适用场景:**
- 直接使用预训练模型
- 不想自己训练
- 需要查看训练过程

### 3. 单独下载模型

如果只需要模型文件，可以单独下载：

#### PyTorch模型
- **文件**: `checkpoints/dino/best_dino.pth`
- **大小**: 97 MB
- **格式**: PyTorch checkpoint
- **用途**: Python推理、继续训练

#### ONNX模型
- **文件**: `outputs/onnx/best_dino.onnx`
- **大小**: 47 MB
- **格式**: ONNX
- **用途**: 跨平台部署、C++推理、移动端

## 🚀 快速开始

### 方式1: 使用轻量级版本（自己训练）

```bash
# 1. 下载轻量级版本
# 从GitHub下载 pet_reid_lightweight.zip

# 2. 解压
unzip pet_reid_lightweight.zip

# 3. 安装依赖
cd pet_reid
pip install -r requirements.txt

# 4. 开始训练
python train_dino_datasets.py --epochs 200
```

### 方式2: 使用完整版本（直接使用）

```bash
# 1. 下载完整版本
# 从云存储下载 pet_reid_full.zip

# 2. 解压
unzip pet_reid_full.zip

# 3. 安装依赖
cd pet_reid
pip install -r requirements.txt

# 4. 直接使用预训练模型
python inference.py \
    --model checkpoints/dino/best_dino.pth \
    --image test.png
```

### 方式3: 使用ONNX模型

```bash
# 1. 下载ONNX模型
# 从云存储下载 best_dino.onnx

# 2. 使用ONNX Runtime推理
import onnxruntime as ort

session = ort.InferenceSession('best_dino.onnx')
outputs = session.run(None, {'input': image})
feature = outputs[0]  # (1, 512)
```

## 📥 下载链接

### GitHub（轻量级版本）
- 地址: https://github.com/gufeng745/claude_workspace
- 文件: `pet_reid/pet_reid_lightweight.zip`
- 大小: 74 KB

### 云存储（完整版本）
- 地址: [待上传]
- 文件: `pet_reid_full.zip`
- 大小: 90 MB

### 单独模型文件
- PyTorch: [待上传]
- ONNX: [待上传]

## 🔧 自己打包

如果需要自己打包，可以使用 `package.py` 脚本：

```bash
cd pet_reid

# 轻量级打包（不含模型和日志）
python package.py --lightweight --output my_lightweight.zip

# 完整打包（包含模型和日志）
python package.py --include_models --include_logs --output my_full.zip

# 只包含代码和日志（不含模型）
python package.py --include_logs --output my_code_logs.zip
```

### 打包选项

| 选项 | 说明 |
|------|------|
| `--lightweight` | 轻量级（不含模型和日志） |
| `--include_models` | 包含模型文件 |
| `--include_logs` | 包含日志文件 |
| `--output` | 指定输出文件名 |

## 📊 文件对比

| 版本 | 大小 | 代码 | 模型 | 日志 | 适用场景 |
|------|------|------|------|------|----------|
| 轻量级 | 74KB | ✅ | ❌ | ❌ | 自己训练 |
| 完整 | 90MB | ✅ | ✅ | ✅ | 直接使用 |
| 仅模型 | 97MB | ❌ | ✅ | ❌ | 集成部署 |

## 💡 建议

### 初学者
- 下载轻量级版本
- 按照README.md自己训练
- 理解DINOv3训练过程

### 快速使用
- 下载完整版本
- 直接使用预训练模型
- 专注于应用开发

### 生产部署
- 下载ONNX模型
- 使用ONNX Runtime部署
- 考虑INT8量化

## ❓ 常见问题

### Q: 为什么不把完整版本放在GitHub？
A: GitHub单个文件限制100MB，完整版本90MB，虽然可以上传但不推荐。建议使用云存储。

### Q: 如何获取完整版本？
A: 目前完整版本在本地，可以：
1. 自己训练（推荐）
2. 联系作者获取
3. 等待上传到云存储

### Q: 轻量级版本够用吗？
A: 如果你打算自己训练，完全够用。代码是完整的，只是没有预训练好的模型。

### Q: 模型文件可以单独下载吗？
A: 可以，PyTorch模型97MB，ONNX模型47MB，可以单独下载。

## 📞 获取帮助

如果下载或使用过程中遇到问题：
1. 查看README.md
2. 查看TRAINING_REPORT.md
3. 提交GitHub Issue

---

**最后更新**: 2026-06-26
**当前版本**: v1.0
