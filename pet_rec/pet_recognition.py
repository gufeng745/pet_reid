import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import faiss
import json
import sys
import glob as glob_module


# ==================== 宠物区域分割 ====================

class PetSegmenter:
    """轻量宠物前景分割器

    使用 torchvision 内置的 DeepLabV3-MobileNetV3 做语义分割，
    将人、猫、狗等前景目标从背景中分离出来。
    MobileNetV3-Large 推理速度很快，适合做前处理。
    """

    # COCO Stuff / VOC 类别中属于宠物/动物的 ID
    # DeepLabV3-MobileNetV3 (COCO) 的 person=0, cat=8, dog=15, horse=17, ...
    PET_CLASSES = {0, 8, 15, 17, 18}  # person + 常见宠物

    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=True, pretrained_backbone=True
        ).to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def segment(self, img):
        """对 PIL Image 做语义分割，返回前景 mask

        Returns:
            mask: (H, W) bool tensor, True = 前景
        """
        orig_w, orig_h = img.size
        inp = self.transform(img).unsqueeze(0).to(self.device)
        out = self.model(inp)['out']  # (1, 21, H', W')
        pred = out.argmax(dim=1).squeeze(0)  # (H', W')

        # 只保留宠物/动物类别的像素
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for cls_id in self.PET_CLASSES:
            mask |= (pred == cls_id)

        # 形态学处理：先膨胀再腐蚀，填补前景内的空洞
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

        # 缩放回原图尺寸
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask_resized).bool()

    def mask_image(self, img, bg_color=(255, 255, 255)):
        """将背景替换为纯色，返回新的 PIL Image"""
        mask = self.segment(img)
        arr = np.array(img)
        bg = np.full_like(arr, bg_color, dtype=np.uint8)
        mask_3d = mask.unsqueeze(-1).cpu().numpy().astype(np.uint8)
        result = arr * mask_3d + bg * (1 - mask_3d)
        return Image.fromarray(result)


def extract_color_histogram_from_masked_image(img, mask=None, bins=32):
    """从图像（或 masked 图像）提取 HSV 颜色直方图

    如果提供 mask，只在 mask=True 的前景像素上计算。

    Args:
        img: PIL Image (RGB)
        mask: (H, W) bool tensor 或 None
        bins: 每通道直方图 bin 数
    Returns:
        color_feat: (bins*3,) 归一化颜色直方图
    """
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if mask is not None:
        mask_np = mask.cpu().numpy().astype(bool)
        if mask_np.sum() < 50:
            mask_np = np.ones(mask_np.shape, dtype=bool)  # 前景太少退回全图
        h = hsv[:, :, 0][mask_np]
        s = hsv[:, :, 1][mask_np]
        v = hsv[:, :, 2][mask_np]
    else:
        h = hsv[:, :, 0].flatten()
        s = hsv[:, :, 1].flatten()
        v = hsv[:, :, 2].flatten()

    hist_h = np.histogram(h, bins=bins, range=(0, 180))[0].astype(np.float32)
    hist_s = np.histogram(s, bins=bins, range=(0, 256))[0].astype(np.float32)
    hist_v = np.histogram(v, bins=bins, range=(0, 256))[0].astype(np.float32)

    color_feat = np.concatenate([hist_h, hist_s, hist_v])
    color_feat = color_feat / (color_feat.sum() + 1e-8)  # L1 归一化
    return color_feat

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
    def __init__(self, feature_dim=None, index_path=None, model_type='mobilenetv2',
                 enable_color_rerank=False):
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
        self.enable_color_rerank = enable_color_rerank
        self.segmenter = None  # 延迟加载，节省内存

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

    def _get_segmenter(self):
        """延迟加载分割器"""
        if self.segmenter is None:
            self.segmenter = PetSegmenter(device=self.device)
        return self.segmenter

    def _extract_pet_color(self, image_path, use_segmentation=True):
        """提取宠物前景颜色特征

        Args:
            image_path: 图片路径
            use_segmentation: 是否用分割模型去除背景
        Returns:
            color_feat: (96,) 归一化 HSV 颜色直方图
        """
        img = Image.open(image_path).convert('RGB')
        if use_segmentation:
            seg = self._get_segmenter()
            mask = seg.segment(img)
        else:
            mask = None
        return extract_color_histogram_from_masked_image(img, mask=mask)

    def find_similar_pets_color_rerank(self, query_path, top_k=5, color_weight=0.3,
                                        retrieval_k=None, use_tta=True,
                                        use_segmentation=True):
        """带颜色感知重排序的相似度检索

        流程：
        1. 用特征向量做初筛（召回 retrieval_k 个候选）
        2. 用宠物前景颜色直方图对候选重排序
        3. 按 feature_sim * alpha + color_sim * (1-alpha) 综合排序

        Args:
            query_path: 查询图片路径
            top_k: 最终返回数量
            color_weight: 颜色相似度权重 (0~1)，越大越看重颜色区分
            retrieval_k: 初筛召回数量，默认 top_k * 5
            use_tta: 是否用 TTA
            use_segmentation: 是否用分割模型去除背景（推荐）
        """
        if retrieval_k is None:
            retrieval_k = min(top_k * 5, self.index.ntotal)

        # 1. 特征向量初筛
        query_results = self.find_similar_pets(query_path, top_k=retrieval_k, use_tta=use_tta)
        if not query_results:
            return []

        # 2. 提取查询图的宠物前景颜色
        query_color = self._extract_pet_color(query_path, use_segmentation=use_segmentation)

        # 3. 对每个候选提取颜色并计算综合得分
        feature_weight = 1.0 - color_weight
        for r in query_results:
            try:
                cand_color = self._extract_pet_color(r['path'], use_segmentation=use_segmentation)
                # 颜色相似度：用直方图交集度量（越大越相似）
                color_sim = np.minimum(query_color, cand_color).sum()
            except Exception:
                color_sim = 0.0
            r['color_similarity'] = float(color_sim)
            r['combined_score'] = feature_weight * r['similarity'] + color_weight * color_sim

        # 4. 按综合得分排序
        query_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return query_results[:top_k]

    def add_pet_to_database_with_color(self, image_path, pet_id, use_tta=True,
                                        use_segmentation=True):
        """入库时同时存储颜色特征（用于颜色重排序）"""
        self.add_pet_to_database(image_path, pet_id, use_tta=use_tta)
        color_feat = self._extract_pet_color(image_path, use_segmentation=use_segmentation)
        self.metadata[-1]['color_hist'] = color_feat.tolist()

    def find_similar_pets_color_rerank_fast(self, query_path, top_k=5, color_weight=0.3,
                                             retrieval_k=None, use_tta=True,
                                             use_segmentation=True):
        """快速颜色重排序版本（颜色特征从 metadata 读取，不需要重复提取）

        前提：入库时使用了 add_pet_to_database_with_color()
        """
        if retrieval_k is None:
            retrieval_k = min(top_k * 5, self.index.ntotal)

        query_results = self.find_similar_pets(query_path, top_k=retrieval_k, use_tta=use_tta)
        if not query_results:
            return []

        query_color = self._extract_pet_color(query_path, use_segmentation=use_segmentation)

        feature_weight = 1.0 - color_weight
        for r in query_results:
            cand_color = np.array(r.get('color_hist', []))
            if len(cand_color) > 0:
                color_sim = float(np.minimum(query_color, cand_color).sum())
            else:
                color_sim = 0.0
            r['color_similarity'] = color_sim
            r['combined_score'] = feature_weight * r['similarity'] + color_weight * color_sim

        query_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return query_results[:top_k]

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


def compare_folder(folder_path, model_type='mobilenetv2', top_k=5,
                    color_rerank=False, color_weight=0.3):
    """对指定文件夹下的图片进行特征提取、两两比对和检索

    Args:
        color_rerank: 是否启用颜色感知重排序
        color_weight: 颜色相似度权重 (0~1)
    """
    system = PetRecognitionSystem(model_type=model_type, enable_color_rerank=color_rerank)

    imgs = sorted(
        glob_module.glob(os.path.join(folder_path, "*.png"))
        + glob_module.glob(os.path.join(folder_path, "*.jpg"))
        + glob_module.glob(os.path.join(folder_path, "*.jpeg"))
        + glob_module.glob(os.path.join(folder_path, "*.bmp"))
    )

    if len(imgs) < 2:
        print(f"图片数量不足（找到 {len(imgs)} 张），至少需要 2 张")
        return

    model_name = f"MobileNetV2-DINOv3 ({system.feature_dim}维)" if model_type == 'mobilenetv2' else f"EfficientNet-B4 ({system.feature_dim}维)"
    print(f"文件夹: {folder_path}")
    print(f"找到 {len(imgs)} 张图片")
    print(f"模型: {model_name}")
    print(f"TTA: 5裁剪 x 水平翻转 = 10视角平均\n")

    # 提取特征并入库
    names = []
    feats = []
    for img_path in imgs:
        name = os.path.basename(img_path)
        names.append(name)
        feat = system.extract_features(img_path, use_tta=True)
        feats.append(feat.flatten())
        system.add_pet_to_database(img_path, pet_id=name, use_tta=True)

    # === 特征向量信息 ===
    print("=" * 60)
    print("特征向量信息")
    print("=" * 60)
    for name, feat in zip(names, feats):
        print(f"  {name:>20s}  dim={len(feat)}  L2 norm={np.linalg.norm(feat):.4f}  "
              f"mean={feat.mean():.4f}  std={feat.std():.4f}")

    # === 两两余弦相似度 ===
    n = len(imgs)
    print(f"\n{'=' * 60}")
    print("两两余弦相似度矩阵")
    print("=" * 60)
    # 表头
    header = f"{'':>20s}" + "".join(f"{name:>12s}" for name in names)
    print(header)
    for i, name_a in enumerate(names):
        row = f"{name_a:>20s}"
        for j, name_b in enumerate(names):
            cos = np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]) + 1e-8)
            row += f"{cos:>12.4f}"
        print(row)

    # === 基础检索 ===
    print(f"\n{'=' * 60}")
    print(f"基础检索（{model_name} + 10视角 TTA）")
    print("=" * 60)
    for img_path in imgs:
        name = os.path.basename(img_path)
        print(f"--- 查询: {name} ---")
        results = system.find_similar_pets(img_path, top_k=min(top_k, n), use_tta=True)
        for r in results:
            marker = " <<<" if r["pet_id"] == name else ""
            print(f"  {r['pet_id']:>20s}  相似度={r['similarity']:.4f}{marker}")

    # === AQE 检索 ===
    print(f"\n{'=' * 60}")
    print("AQE 检索（+ 平均查询扩展重排序）")
    print("=" * 60)
    for img_path in imgs:
        name = os.path.basename(img_path)
        print(f"--- 查询: {name} ---")
        results = system.find_similar_pets_aqe(img_path, top_k=min(top_k, n), aqe_k=3, use_tta=True)
        for r in results:
            marker = " <<<" if r["pet_id"] == name else ""
            print(f"  {r['pet_id']:>20s}  相似度={r['similarity']:.4f}{marker}")

    # === 颜色感知重排序 ===
    if color_rerank:
        print(f"\n{'=' * 60}")
        print(f"颜色感知重排序（color_weight={color_weight}）")
        print("  - 使用 DeepLabV3 分割前景，只在宠物区域提取颜色")
        print("  - 综合得分 = 特征相似度 × {:.1f} + 颜色相似度 × {:.1f}".format(
            1 - color_weight, color_weight))
        print("=" * 60)
        for img_path in imgs:
            name = os.path.basename(img_path)
            print(f"--- 查询: {name} ---")
            results = system.find_similar_pets_color_rerank(
                img_path, top_k=min(top_k, n),
                color_weight=color_weight, use_tta=True, use_segmentation=True,
            )
            for r in results:
                marker = " <<<" if r["pet_id"] == name else ""
                print(f"  {r['pet_id']:>20s}  特征={r['similarity']:.4f}  "
                      f"颜色={r['color_similarity']:.4f}  "
                      f"综合={r['combined_score']:.4f}{marker}")

    # 保存索引
    index_path = os.path.join(folder_path, "pet_features.index")
    system.save_index(index_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='宠物图片特征提取与比对')
    parser.add_argument('--folder', type=str, default=None, help='图片文件夹路径（默认 data 目录）')
    parser.add_argument('--model', choices=['mobilenetv2', 'efficientnet_b4'], default='mobilenetv2')
    parser.add_argument('--top_k', type=int, default=5, help='检索返回数量')
    parser.add_argument('--color_rerank', action='store_true', help='启用颜色感知重排序（需要安装 opencv-python）')
    parser.add_argument('--color_weight', type=float, default=0.3, help='颜色相似度权重 (0~1)，默认 0.3')
    args = parser.parse_args()

    folder = args.folder if args.folder else DATA_PATH
    compare_folder(folder, model_type=args.model, top_k=args.top_k,
                   color_rerank=args.color_rerank, color_weight=args.color_weight)
