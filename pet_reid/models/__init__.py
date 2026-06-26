"""Pet Re-ID 模型模块"""

from .backbone import CNNBackbone
from .dino_model import DINOModel
from .reid_model import ReIDModel
from .pretrained import (
    PretrainedModelManager,
    get_manager,
    load_pretrained_backbone,
    load_pretrained_dino,
    setup_pretrained_models,
    PRETRAINED_MODELS,
)

__all__ = [
    'CNNBackbone',
    'DINOModel',
    'ReIDModel',
    'PretrainedModelManager',
    'get_manager',
    'load_pretrained_backbone',
    'load_pretrained_dino',
    'setup_pretrained_models',
    'PRETRAINED_MODELS',
]
