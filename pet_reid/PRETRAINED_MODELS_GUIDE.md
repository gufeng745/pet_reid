# 预训练模型使用指南

## 📁 目录结构

预训练模型应该放在 `pretrained_models/` 目录下：

```
pet_reid/
├── pretrained_models/
│   ├── dino/
│   │   └── best_dino.pth          # DINOv3预训练模型 (96.5MB)
│   ├── imagenet/
│   │   └── (timm自动下载)
│   └── custom/
│       └── (自定义模型)
│
├── checkpoints/
│   └── dino/
│       └── best_dino.pth          # 训练生成的checkpoint
│
└── ...
```

## 🚀 快速开始

### 步骤1: 设置预训练模型

**方式A: 从现有checkpoints设置（推荐）**

```bash
cd D:/claude_workspace/pet_reid

# 从checkpoints目录复制到pretrained_models
python setup_pretrained.py --source checkpoints/dino
```

**方式B: 手动复制**

```bash
# 创建目录
mkdir -p pretrained_models/dino

# 复制模型文件
cp checkpoints/dino/best_dino.pth pretrained_models/dino/
```

**方式C: 下载模型文件**

如果模型在云存储，下载后放到对应目录：
- PyTorch模型: `pretrained_models/dino/best_dino.pth`
- ONNX模型: `pretrained_models/dino/best_dino.onnx`

### 步骤2: 验证预训练模型

```bash
# 检查预训练模型状态
python setup_pretrained.py --check
```

输出示例：
```
dino_mobilenetv3_large:
  描述: DINOv3预训练的MobileNetV3-Large (512维)
  状态: [OK] 可用
```

### 步骤3: 使用预训练模型

```bash
# 运行示例
python example_load_pretrained.py --image test.png
```

## 📝 代码使用示例

### 示例1: 加载预训练模型

```python
import sys
sys.path.insert(0, 'D:/claude_workspace/pet_reid')

from models.pretrained import load_pretrained_dino

# 加载DINOv3预训练模型
model = load_pretrained_dino(
    model_name='dino_mobilenetv3_large',
    proj_dim=512
)

print("模型加载成功!")
print(f"模型类型: {type(model)}")
```

### 示例2: 提取图片特征

```python
import torch
from PIL import Image
from torchvision import transforms
from models.pretrained import load_pretrained_dino

# 加载模型
model = load_pretrained_dino('dino_mobilenetv3_large')
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# 提取特征
def extract_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        feature = model(image_tensor)
    
    return feature.squeeze(0)

# 使用
feature = extract_feature('test.png')
print(f"特征维度: {feature.shape}")  # (512,)
print(f"特征范数: {torch.norm(feature).item():.4f}")  # ~1.0
```

### 示例3: 计算图片相似度

```python
import torch
from models.pretrained import load_pretrained_dino

# 加载模型
model = load_pretrained_dino('dino_mobilenetv3_large')
model.eval()

# 提取特征
feature1 = extract_feature('cat1.png')
feature2 = extract_feature('cat2.png')

# 计算余弦相似度
similarity = torch.cosine_similarity(feature1, feature2, dim=0)
print(f"相似度: {similarity.item():.4f}")

# 判断
if similarity > 0.8:
    print("非常相似 (可能是同一只宠物)")
elif similarity > 0.5:
    print("比较相似 (可能是同一品种)")
else:
    print("差异较大")
```

### 示例4: 图片检索

```python
import torch
from models.pretrained import load_pretrained_dino

def image_search(query_path, gallery_paths, model, top_k=5):
    """图片检索
    
    Args:
        query_path: 查询图片路径
        gallery_paths: 图库图片路径列表
        model: 预训练模型
        top_k: 返回前k个结果
    
    Returns:
        results: [(path, similarity), ...]
    """
    # 提取查询特征
    query_feature = extract_feature(query_path)
    
    # 提取图库特征
    gallery_features = []
    for path in gallery_paths:
        feature = extract_feature(path)
        gallery_features.append(feature)
    
    gallery_features = torch.stack(gallery_features)
    
    # 计算相似度
    similarities = torch.cosine_similarity(
        query_feature.unsqueeze(0),
        gallery_features,
        dim=1
    )
    
    # 排序
    top_k = min(top_k, len(gallery_paths))
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append((gallery_paths[idx], similarities[idx].item()))
    
    return results

# 使用示例
gallery = ['img1.png', 'img2.png', 'img3.png', 'img4.png']
results = image_search('query.png', gallery, model, top_k=3)

print("检索结果:")
for path, sim in results:
    print(f"  {path}: {sim:.4f}")
```

## 🔧 高级用法

### 使用不同的预训练模型

```python
from models.pretrained import load_pretrained_backbone

# 加载ImageNet预训练的backbone
backbone = load_pretrained_backbone('imagenet_mobilenetv3_large')

# 加载自定义预训练模型
backbone = load_pretrained_backbone('dino_mobilenetv3_large')
```

### 列出所有可用的预训练模型

```python
from models.pretrained import get_manager

manager = get_manager()
manager.list_models()
```

### 检查预训练模型状态

```python
from models.pretrained import get_manager

manager = get_manager()
status = manager.check_models()

for name, is_available in status.items():
    print(f"{name}: {'可用' if is_available else '不可用'}")
```

## 📋 命令行工具

### 设置预训练模型

```bash
# 从checkpoints设置
python setup_pretrained.py --source checkpoints/dino

# 创建目录结构
python setup_pretrained.py --create_dirs

# 列出可用模型
python setup_pretrained.py --list

# 检查模型状态
python setup_pretrained.py --check
```

### 运行示例

```bash
# 单张图片特征提取
python example_load_pretrained.py --image test.png

# 计算相似度
python example_load_pretrained.py --image1 img1.png --image2 img2.png

# 批量特征提取
python example_load_pretrained.py --image_dir ./images

# 使用GPU
python example_load_pretrained.py --image test.png --device cuda
```

## 🎯 预训练模型说明

### DINOv3预训练模型

- **模型名称**: `dino_mobilenetv3_large`
- **Backbone**: MobileNetV3-Large (4.20M参数)
- **特征维度**: 512
- **训练数据**: datasets文件夹 (25000张宠物图片)
- **训练方式**: DINOv3自监督学习
- **文件大小**: 96.5 MB
- **文件格式**: PyTorch checkpoint

### 模型性能

- **训练Loss**: 5.89 → 1.03 (下降82.5%)
- **特征质量**: 优秀的宠物特征表示
- **适用场景**: 
  - 宠物图片特征提取
  - 宠物图片相似度计算
  - 宠物图片检索
  - 迁移到其他宠物相关任务

### 模型结构

```
MobileNetV3-Large (4.20M参数)
    ↓
GeM Pooling
    ↓
Projector (1280→2048→2048→512)
    ↓
L2 Normalization
    ↓
Output: 512维特征向量
```

## ❓ 常见问题

### Q1: 预训练模型放在哪里？

**A**: 放在 `pretrained_models/dino/` 目录下：
```
pet_reid/pretrained_models/dino/best_dino.pth
```

### Q2: 如何设置预训练模型？

**A**: 使用设置脚本：
```bash
python setup_pretrained.py --source checkpoints/dino
```

### Q3: 如何验证预训练模型是否可用？

**A**: 运行检查命令：
```bash
python setup_pretrained.py --check
```

### Q4: 加载模型时报错怎么办？

**A**: 检查以下几点：
1. 模型文件是否存在
2. 文件路径是否正确
3. 文件是否损坏
4. PyTorch版本是否兼容

### Q5: 可以使用其他预训练模型吗？

**A**: 可以！在 `models/pretrained.py` 中注册新的模型：
```python
PRETRAINED_MODELS['my_model'] = {
    'file': 'my_model.pth',
    'dir': 'custom',
    'backbone': 'mobilenetv3_large_100',
    'proj_dim': 512,
}
```

### Q6: 如何在其他项目中使用？

**A**: 复制以下文件到你的项目：
- `models/pretrained.py`
- `pretrained_models/dino/best_dino.pth`

然后在代码中：
```python
from models.pretrained import load_pretrained_dino
model = load_pretrained_dino('dino_mobilenetv3_large')
```

## 📚 相关文档

- **README.md**: 项目概述
- **README_DATASETS.md**: datasets使用说明
- **TRAINING_REPORT.md**: 训练报告
- **DOWNLOAD_GUIDE.md**: 下载指南

## 🔗 快速链接

- 预训练模型: `pretrained_models/dino/best_dino.pth`
- 设置脚本: `setup_pretrained.py`
- 使用示例: `example_load_pretrained.py`

---

**最后更新**: 2026-06-26
**模型版本**: v1.0 (DINOv3预训练)
