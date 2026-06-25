
import os
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from models.backbone import CNNBackbone

class DINOFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='mobilenetv3_large_100', proj_dim=512):
        super().__init__()
        self.backbone = CNNBackbone(model_name=backbone_name, pretrained=False)
        feat_dim = self.backbone.feature_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        proj = nn.functional.normalize(proj, p=2, dim=1)
        return proj

print('Loading model...')
ckpt = torch.load('checkpoints/dino/best_dino.pth', map_location='cpu')

model = DINOFeatureExtractor(proj_dim=512)

if 'student_backbone' in ckpt:
    model.backbone.load_state_dict(ckpt['student_backbone'], strict=False)
    print('Loaded student_backbone')
if 'student_projector' in ckpt:
    model.projector.load_state_dict(ckpt['student_projector'], strict=False)
    print('Loaded student_projector')

model.eval()

os.makedirs('outputs/onnx', exist_ok=True)

dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = 'outputs/onnx/best_dino.onnx'

print(f'Exporting to {onnx_path}...')
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
)

print('Export completed!')
print(f'ONNX model: {onnx_path}')
