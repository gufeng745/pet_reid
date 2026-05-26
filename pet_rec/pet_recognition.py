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
import glob as glob_module  # 避免命名冲突

def get_data_path():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.abspath(os.path.join(script_dir, 'data'))

DATA_PATH = get_data_path()

class PetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(PetFeatureExtractor, self).__init__()
        from torchvision.models import ResNet50_Weights
        # 建议使用 ResNet101 或 EfficientNet 以获得更好效果，这里保持 ResNet50 兼容
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        
        # 【关键修正】推理Transform必须是确定性的，不能包含 RandomFlip 或 ColorJitter
        self.infer_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.feature_extractor(x).squeeze(-1).squeeze(-1)

class PetRecognitionSystem:
    def __init__(self, feature_dim=2048, index_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PetFeatureExtractor().to(self.device)
        self.model.eval()

        self.index = faiss.IndexFlatIP(feature_dim)
        self.feature_dim = feature_dim
        self.metadata = []

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def _get_single_feature(self, image_tensor):
        """内部方法：获取单张Tensor的特征"""
        with torch.no_grad():
            feat = self.model(image_tensor.to(self.device))
        return feat.cpu().numpy()

    def extract_features(self, image_path, use_tta=True):
        """
        提取特征，支持 TTA (测试时增强)
        TTA 策略：原图 + 水平翻转图 + 五角裁剪平均 (可选)
        这能显著提高对姿态变化的鲁棒性
        """
        img = Image.open(image_path).convert('RGB')
        
        # 1. 基础变换
        base_tensor = self.model.infer_transform(img).unsqueeze(0)
        base_feat = self._get_single_feature(base_tensor)
        
        if not use_tta:
            return np.ascontiguousarray(base_feat, dtype=np.float32)

        # 2. TTA: 水平翻转
        h_flip_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0), # 强制翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        flip_tensor = h_flip_transform(img).unsqueeze(0)
        flip_feat = self._get_single_feature(flip_tensor)

        # 3. 特征平均 (Feature Averaging)
        # 平均后的特征比单一视角更稳定
        avg_feat = (base_feat + flip_feat) / 2.0
        
        return np.ascontiguousarray(avg_feat, dtype=np.float32)

    def normalize_features(self, features):
        faiss.normalize_L2(features)
        return features

    def add_pet_to_database(self, image_path, pet_id, use_tta=True):
        features = self.extract_features(image_path, use_tta=use_tta)
        features = self.normalize_features(features)
        
        self.index.add(features)
        self.metadata.append({'pet_id': pet_id, 'path': image_path})
        # 静默模式，避免刷屏，或者保留print
        # print(f"✅ 已添加: {pet_id}")

    def find_similar_pets(self, query_path, top_k=5, use_tta=True):
        query_features = self.extract_features(query_path, use_tta=use_tta)
        query_features = self.normalize_features(query_features)

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
        print(f"💾 索引已保存: {path} (共 {self.index.ntotal} 条)")

    def load_index(self, path):
        self.index = faiss.read_index(path)
        meta_path = path.replace('.index', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"📂 索引已加载: {self.index.ntotal} 条记录")

if __name__ == "__main__":
    # 1. 初始化
    system = PetRecognitionSystem()

    # 2. 批量导入 data 目录下的图片
    # 注意：glob 返回的是绝对路径列表
    data_imgs = glob_module.glob(os.path.join(DATA_PATH, "*.png"))
    # 也可以加上 jpg/jpeg
    data_imgs += glob_module.glob(os.path.join(DATA_PATH, "*.jpg"))
    
    print(f"📊 在 {DATA_PATH} 中找到 {len(data_imgs)} 张图片")
    
    for idx, img_path in enumerate(data_imgs):
        try:
            # 使用文件名作为 ID，或者自定义逻辑
            pet_id = f"pet_{os.path.basename(img_path)}"
            system.add_pet_to_database(img_path, pet_id=pet_id, use_tta=True)
        except Exception as e:
            print(f"❌ 处理失败 {img_path}: {e}")

    # 3. 保存
    system.save_index("pet_features_v2.index")

    # 4. 验证查询
    # 重新加载以模拟真实场景
    search_system = PetRecognitionSystem(index_path="pet_features_v2.index")
    
    # 假设根目录下有测试图片
    test_files = ["test_1.png", "test_2.png"] 
    for t_file in test_files:
        if os.path.exists(t_file):
            print(f"\n🔍 查询: {t_file}")
            results = search_system.find_similar_pets(t_file, top_k=3, use_tta=True)
            for r in results:
                print(f"  -> ID: {r['pet_id']} | Sim: {r['similarity']:.4f}")
        else:
            print(f"⚠️ 未找到测试文件: {t_file}")
