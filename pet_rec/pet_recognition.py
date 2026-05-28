import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import json
import os
import sys
import glob as glob_module

def get_data_path():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.abspath(os.path.join(script_dir, 'data'))

DATA_PATH = get_data_path()


# ==================== 特征提取器 ====================

class PetFeatureExtractor(nn.Module):
    """EfficientNet-B4 特征提取器

    相比 ResNet50 的优势：
    - ImageNet Top-1 准确率更高 (83.0% vs 76.1%)
    - MBConv 架构在细粒度特征提取上优于传统卷积
    - 1792 维特征向量，信息更丰富
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import EfficientNet_B4_Weights
        net = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        # 去掉最后的分类头，保留特征提取部分
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.feature_dim = 1792

        # EfficientNet-B4 标准预处理
        self.transform = transforms.Compose([
            transforms.Resize(380, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        # 输出 shape: (batch, 1792, 1, 1) → squeeze → (batch, 1792)
        return self.features(x).squeeze(-1).squeeze(-1)


class MobileNetV2FeatureExtractor:
    """DINOv3 蒸馏的 MobileNetV2 特征提取器（ONNX 推理）"""

    def __init__(self, onnx_path=None):
        import onnxruntime as ort
        if onnx_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            onnx_path = os.path.join(base, 'pet_mobilenetv2.onnx')
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(base, 'pet_mobilenetv2_int8.onnx')
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.feature_dim = 512

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img_or_tensor):
        if isinstance(img_or_tensor, Image.Image):
            tensor = self.transform(img_or_tensor).unsqueeze(0).numpy()
        else:
            tensor = img_or_tensor.numpy() if isinstance(img_or_tensor, torch.Tensor) else img_or_tensor
        feat = self.session.run(None, {self.input_name: tensor})[0]
        # L2 归一化
        feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
        return feat


# ==================== 多裁剪 TTA ====================

def get_five_crops(img, crop_size=380):
    """5 裁剪策略：中心 + 四角

    先将图片短边 resize 到 crop_size，长边等比放大，
    然后从 5 个位置各裁剪出 crop_size×crop_size 的区域。
    """
    w, h = img.size
    scale = crop_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 如果长边刚好等于 crop_size，只做中心裁剪
    if new_w == crop_size and new_h == crop_size:
        return [img]

    cx, cy = new_w // 2, new_h // 2
    half = crop_size // 2
    crops = [
        img.crop((cx - half, cy - half, cx + half, cy + half)),  # 中心
        img.crop((0, 0, crop_size, crop_size)),                   # 左上
        img.crop((new_w - crop_size, 0, new_w, crop_size)),       # 右上
        img.crop((0, new_h - crop_size, crop_size, new_h)),       # 左下
        img.crop((new_w - crop_size, new_h - crop_size,
                  new_w, new_h)),                                  # 右下
    ]
    return crops


# ==================== 核心系统 ====================

class PetRecognitionSystem:
    def __init__(self, feature_dim=None, index_path=None, model_type='mobilenetv2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        if model_type == 'mobilenetv2':
            self.model = MobileNetV2FeatureExtractor()
            self.feature_dim = 512
        else:
            self.model = PetFeatureExtractor().to(self.device)
            self.model.eval()
            self.feature_dim = 1792

        if feature_dim is not None:
            self.feature_dim = feature_dim

        self.index = faiss.IndexFlatIP(self.feature_dim)
        self.metadata = []

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def _extract_single(self, img):
        """提取单张 PIL Image 的特征"""
        if isinstance(self.model, MobileNetV2FeatureExtractor):
            return self.model(img).flatten()
        tensor = self.model.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.cpu().numpy().flatten()

    def extract_features(self, image_path, use_tta=True):
        """提取图片特征向量

        TTA 策略：5 裁剪 × 2（原图 + 水平翻转）= 10 个视角取平均
        """
        img = Image.open(image_path).convert('RGB')

        if not use_tta:
            feat = self._extract_single(img)
            return np.ascontiguousarray(feat.reshape(1, -1), dtype=np.float32)

        crop_size = 224 if self.model_type == 'mobilenetv2' else 380
        crops = get_five_crops(img, crop_size=crop_size)
        feats = []
        for crop in crops:
            feats.append(self._extract_single(crop))
            feats.append(self._extract_single(crop.transpose(Image.FLIP_LEFT_RIGHT)))

        avg_feat = np.mean(feats, axis=0)
        return np.ascontiguousarray(avg_feat.reshape(1, -1), dtype=np.float32)

    def normalize_features(self, features):
        faiss.normalize_L2(features)
        return features

    def add_pet_to_database(self, image_path, pet_id, use_tta=True):
        features = self.extract_features(image_path, use_tta=use_tta)
        features = self.normalize_features(features)
        self.index.add(features)
        self.metadata.append({'pet_id': pet_id, 'path': image_path})

    def find_similar_pets(self, query_path, top_k=5, use_tta=True):
        """基础相似度查询"""
        query_features = self.extract_features(query_path, use_tta=use_tta)
        query_features = self.normalize_features(query_features)
        return self._search(query_features, top_k)

    def find_similar_pets_aqe(self, query_path, top_k=5, aqe_k=3, use_tta=True):
        """带 AQE (Average Query Expansion) 的相似度查询

        原理：首次查询取 top-K → 将查询特征与 top-K 结果特征平均 → 重新归一化 → 再次查询
        实践中能提升 3-8% 的检索准确率
        """
        query_features = self.extract_features(query_path, use_tta=use_tta)
        query_features = self.normalize_features(query_features)

        # 第一次查询
        first_k = min(top_k + aqe_k, self.index.ntotal)
        if first_k == 0:
            return []

        distances, indices = self.index.search(query_features, first_k)

        # AQE：用 top-aqe_k 个结果的特征扩展查询
        expanded = query_features.copy().flatten()
        count = 1
        for i in range(min(aqe_k, first_k)):
            idx = indices[0][i]
            if idx != -1:
                expanded += self.index.reconstruct(int(idx))
                count += 1

        expanded = expanded / count
        expanded = np.ascontiguousarray(expanded.reshape(1, -1), dtype=np.float32)
        self.normalize_features(expanded)

        # 第二次查询
        return self._search(expanded, top_k)

    def _search(self, query_features, top_k):
        """内部搜索方法"""
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_features, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    'pet_id': self.metadata[idx]['pet_id'],
                    'path': self.metadata[idx]['path'],
                    'similarity': float(distances[0][i])
                })
        return results

    def save_index(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        meta_path = path.replace('.index', '_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"索引已保存: {path} (共 {self.index.ntotal} 条)")

    def load_index(self, path):
        self.index = faiss.read_index(path)
        meta_path = path.replace('.index', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"索引已加载: {self.index.ntotal} 条记录")


# ==================== 测试入口 ====================

def test_similarity(data_path, model_type='mobilenetv2'):
    """测试 data 目录下所有图片的两两相似度（基础查询 vs AQE 查询对比）"""
    system = PetRecognitionSystem(model_type=model_type)

    data_imgs = sorted(
        glob_module.glob(os.path.join(data_path, "*.png"))
        + glob_module.glob(os.path.join(data_path, "*.jpg"))
    )

    if len(data_imgs) < 2:
        print("图片数量不足，至少需要 2 张")
        return

    model_name = f"MobileNetV2-DINOv3 ({system.feature_dim}维)" if model_type == 'mobilenetv2' else f"EfficientNet-B4 ({system.feature_dim}维)"
    print(f"在 {data_path} 中找到 {len(data_imgs)} 张图片")
    print(f"模型: {model_name}")
    print(f"TTA: 5裁剪 x 水平翻转 = 10视角平均\n")

    for img_path in data_imgs:
        pet_id = os.path.basename(img_path)
        system.add_pet_to_database(img_path, pet_id=pet_id, use_tta=True)

    n = len(data_imgs)

    # === 基础查询 ===
    print("=" * 50)
    print(f"基础查询（{model_name} + 10视角 TTA）")
    print("=" * 50)
    for img_path in data_imgs:
        name = os.path.basename(img_path)
        print(f"--- {name} ---")
        results = system.find_similar_pets(img_path, top_k=n)
        for r in results:
            marker = " *" if r["pet_id"] == name else ""
            print(f"  {r['pet_id']:>15s}  {r['similarity']:.4f}{marker}")
        print()

    # === AQE 查询 ===
    print("=" * 50)
    print("AQE 查询（+ 平均查询扩展重排序）")
    print("=" * 50)
    for img_path in data_imgs:
        name = os.path.basename(img_path)
        print(f"--- {name} ---")
        results = system.find_similar_pets_aqe(img_path, top_k=n, aqe_k=3)
        for r in results:
            marker = " *" if r["pet_id"] == name else ""
            print(f"  {r['pet_id']:>15s}  {r['similarity']:.4f}{marker}")
        print()

    system.save_index("pet_features_v2.index")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mobilenetv2', 'efficientnet_b4'], default='mobilenetv2')
    args = parser.parse_args()
    test_similarity(DATA_PATH, model_type=args.model)
