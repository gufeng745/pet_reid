"""Pet Re-ID 模型模块"""

from .backbone import CNNBackbone
from .dino_model import DINOModel
from .reid_model import ReIDModel

__all__ = ['CNNBackbone', 'DINOModel', 'ReIDModel']
