"""Regression heads (spatial maps or pooled vectors) and the bay classifier."""

from .cls import build_classifier, soft_ce, topk_soft_argmax
from .dwconv import DwConvHead
from .factory import RegressionHead
from .flatten import FlattenHead
from .mlp import MLPHead
from .squeeze import SqueezeHead

__all__ = [
    'DwConvHead', 'FlattenHead', 'MLPHead', 'RegressionHead', 'SqueezeHead',
    'build_classifier', 'soft_ce', 'topk_soft_argmax',
]
