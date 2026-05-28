import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from prepare_data import create_dataloaders, get_eval_transform
from models import MobileNetV2Student, DINOv3Teacher, DINOv2Teacher


def extract_all_features(model, dataloader, device):
    """提取整个数据集的特征和标签"""
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def retrieval_recall(feats, labels, top_k_list=(1, 5, 10)):
    """同品种检索 Recall@K"""
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T
    n = feats.shape[0]
    results = {}
    for k in top_k_list:
        correct = 0
        for i in range(n):
            topk = sim[i].topk(k + 1).indices[1:]  # 排除自身
            if (labels[topk] == labels[i]).any():
                correct += 1
        results[f'Recall@{k}'] = correct / n
    return results


def cross_architecture_agreement(feats_teacher, feats_student, sample_size=500):
    """Teacher-Student 排序相关性 (Spearman rho)"""
    n = feats_teacher.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        feats_teacher = feats_teacher[idx]
        feats_student = feats_student[idx]

    feats_teacher = F.normalize(feats_teacher, dim=-1)
    feats_student = F.normalize(feats_student, dim=-1)

    sim_teacher = (feats_teacher @ feats_teacher.T).numpy()
    sim_student = (feats_student @ feats_student.T).numpy()

    mask = ~np.eye(sim_teacher.shape[0], dtype=bool)
    rho, pval = spearmanr(sim_teacher[mask], sim_student[mask])
    return rho, pval


def feature_quality_check(feats, labels):
    """特征质量检查：同类/异类相似度、特征方差"""
    feats = F.normalize(feats, dim=-1)
    sim = (feats @ feats.T).numpy()
    labels = labels.numpy()

    same_mask = (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)
    diff_mask = (labels[:, None] != labels[None, :])

    same_mean = sim[same_mask].mean() if same_mask.any() else 0
    diff_mean = sim[diff_mask].mean() if diff_mask.any() else 0
    feat_std = feats.std(dim=0).mean().item()

    return {
        'same_class_sim': same_mean,
        'diff_class_sim': diff_mean,
        'sim_gap': same_mean - diff_mean,
        'feature_std': feat_std,
    }


def evaluate_student(student_path, proj_dim=512, teacher_type='dinov3'):
    """完整评估流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load student
    student = MobileNetV2Student(proj_dim=proj_dim).to(device)
    ckpt = torch.load(student_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'student' in ckpt:
        student.load_state_dict(ckpt['student'])
    else:
        student.load_state_dict(ckpt)
    student.eval()

    # Load teacher
    try:
        teacher = DINOv3Teacher().to(device)
        print("Teacher: DINOv3 ViT-S")
    except Exception:
        teacher = DINOv2Teacher().to(device)
        print("Teacher: DINOv2 ViT-S")
    teacher.eval()

    # Data
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    _, eval_loader = create_dataloaders(dataset_root, batch_size=32)

    # Extract features
    print("Extracting student features...")
    student_feats, labels = extract_all_features(student, eval_loader, device)
    print("Extracting teacher features...")
    teacher_feats, _ = extract_all_features(teacher, eval_loader, device)

    # Retrieval
    print("\n=== Student Retrieval ===")
    recalls = retrieval_recall(student_feats, labels)
    for k, v in recalls.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Teacher Retrieval ===")
    t_recalls = retrieval_recall(teacher_feats, labels)
    for k, v in t_recalls.items():
        print(f"  {k}: {v:.4f}")

    # Cross-architecture agreement
    print("\n=== Cross-Architecture Agreement ===")
    rho, pval = cross_architecture_agreement(teacher_feats, student_feats)
    print(f"  Spearman rho: {rho:.4f} (p={pval:.2e})")

    # Feature quality
    print("\n=== Feature Quality ===")
    sq_student = feature_quality_check(student_feats, labels)
    sq_teacher = feature_quality_check(teacher_feats, labels)
    print(f"  Student: same={sq_student['same_class_sim']:.4f} "
          f"diff={sq_student['diff_class_sim']:.4f} "
          f"gap={sq_student['sim_gap']:.4f} std={sq_student['feature_std']:.4f}")
    print(f"  Teacher: same={sq_teacher['same_class_sim']:.4f} "
          f"diff={sq_teacher['diff_class_sim']:.4f} "
          f"gap={sq_teacher['sim_gap']:.4f} std={sq_teacher['feature_std']:.4f}")

    return {
        'recalls': recalls,
        'teacher_recalls': t_recalls,
        'spearman_rho': rho,
        'student_quality': sq_student,
        'teacher_quality': sq_teacher,
    }


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_student.pth'
    evaluate_student(ckpt_path)
