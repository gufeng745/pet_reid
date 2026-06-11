# 预训练权重目录

将本地下载的预训练权重文件放在此目录下。

## 需要的权重文件

### 1. DINOv3 ViT-S (推荐)
- **文件名**: `dinov3_vit_small_patch16.lvd1689m.safetensors`
- **来源**: Hugging Face timm 模型库
- **下载命令**:
  ```bash
  # 使用 timm 下载
  python -c "import timm; timm.create_model('vit_small_patch16_dinov3', pretrained=True)"
  # 然后从 ~/.cache/torch/hub 复制到此目录
  ```

### 2. DINOv2 ViT-S (备用)
- **文件名**: `dinov2_vit_small_patch14_reg4.lvd142m.safetensors`
- **来源**: Hugging Face timm 模型库
- **下载命令**:
  ```bash
  # 使用 timm 下载
  python -c "import timm; timm.create_model('vit_small_patch14_reg4_dinov2', pretrained=True)"
  # 然后从 ~/.cache/torch/hub 复制到此目录
  ```

### 3. MobileNetV2 (可选)
- **文件名**: `mobilenetv2.pth`
- **来源**: PyTorch 官方或 timm
- **下载命令**:
  ```bash
  # 从 PyTorch 官方下载
  wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -O mobilenetv2.pth
  ```

## 从本地缓存复制权重

### Windows
```powershell
# 查找缓存位置
dir C:\Users\<用户名>\.cache\torch\hub -r | findstr dinov

# 复制权重文件到此目录
copy "C:\Users\<用户名>\.cache\torch\hub\<权重文件名>" .
```

### Linux/Mac
```bash
# 查找缓存位置
find ~/.cache/torch/hub -name "*dinov*"

# 复制权重文件到此目录
cp ~/.cache/torch/hub/<权重文件名> .
```

## 目录结构示例
```
pre_weights/
├── dinov3_vit_small_patch16.lvd1689m.safetensors  # DINOv3 权重
├── dinov2_vit_small_patch14_reg4.lvd142m.safetensors  # DINOv2 权重 (备用)
└── mobilenetv2.pth  # MobileNetV2 权重 (可选)
```

## 使用说明

代码会自动检测此目录下的权重文件：
- 如果存在本地权重，优先使用本地权重
- 如果不存在，尝试在线下载
- 如果下载失败，会显示错误信息