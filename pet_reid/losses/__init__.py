"""Pet Re-ID 损失函数模块"""

from .dino_loss import DINOLoss
from .reid_loss import (
    TripletMarginLoss,
    SupervisedContrastiveLoss,
    FeatureOrthogonalityLoss,
    LabelSmoothingCE
)

__all__ = [
    'DINOLoss',
    'TripletMarginLoss',
    'SupervisedContrastiveLoss',
    'FeatureOrthogonalityLoss',
    'LabelSmoothingCE'
]
