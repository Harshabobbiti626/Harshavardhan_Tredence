"""
AQUA Platform - Reverse-Mode Autodiff Engine
Built from scratch in pure Python & NumPy.
"""

from engine.tensor import Tensor
from engine.ops import (
    relu,
    gelu,
    tanh,
    sigmoid,
    softmax_cross_entropy,
)

__all__ = [
    "Tensor",
    "relu",
    "gelu",
    "tanh",
    "sigmoid",
    "softmax_cross_entropy",
]
