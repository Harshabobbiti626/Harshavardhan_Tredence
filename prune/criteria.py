"""
Importance criteria for connection pruning in self-pruning neural networks.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from engine.tensor import Tensor


class ImportanceCriterion:
    """Base class for connection importance criteria."""

    def compute_importance(self, param: Tensor) -> np.ndarray:
        raise NotImplementedError

    def update_gradient(self, param: Tensor) -> None:
        pass


class MagnitudeCriterion(ImportanceCriterion):
    """
    Baseline Magnitude Pruning:
      I_ij = |W_ij|
    Assumes connection importance is purely proportional to parameter weight magnitude.
    """

    def compute_importance(self, param: Tensor) -> np.ndarray:
        return np.abs(param.data)


class TaylorSaliencyCriterion(ImportanceCriterion):
    """
    First-Order Taylor Expansion Saliency Pruning:
      I_ij = |W_ij * g_ij|
    
    Derivation:
      Let Delta W_ij = -W_ij (removing connection ij by setting it to 0).
      Taylor expansion of loss change around current weights:
        L(W + Delta W) - L(W) = sum_ij (dL/dW_ij * Delta W_ij) + O(||Delta W||^2)
        Delta L_ij approx - dL/dW_ij * W_ij
      Taking the absolute value yields the expected loss perturbation:
        I_ij = |W_ij * (dL/dW_ij)|
    
    To reduce stochastic mini-batch gradient variance, this criterion maintains
    an Exponential Moving Average (EMA) of gradients over training steps:
      g_bar_t = gamma * g_bar_{t-1} + (1 - gamma) * g_t
    """

    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self.grad_ema: Dict[int, np.ndarray] = {}

    def update_gradient(self, param: Tensor) -> None:
        if param.grad is None:
            return
        pid = id(param)
        g = param.grad
        if pid not in self.grad_ema:
            self.grad_ema[pid] = np.abs(g).copy()
        else:
            self.grad_ema[pid] = self.ema_decay * self.grad_ema[pid] + (1.0 - self.ema_decay) * np.abs(g)

    def compute_importance(self, param: Tensor) -> np.ndarray:
        pid = id(param)
        if pid in self.grad_ema:
            grad_val = self.grad_ema[pid]
        elif param.grad is not None:
            grad_val = np.abs(param.grad)
        else:
            grad_val = np.ones_like(param.data)

        return np.abs(param.data * grad_val)


class MovementCriterion(ImportanceCriterion):
    """
    Movement Pruning Criterion (Sanh et al., 2020):
      I_ij = - W_ij * g_ij
    Connections shrinking toward zero during SGD/Adam are pruned first.
    """

    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self.grad_ema: Dict[int, np.ndarray] = {}

    def update_gradient(self, param: Tensor) -> None:
        if param.grad is None:
            return
        pid = id(param)
        g = param.grad
        if pid not in self.grad_ema:
            self.grad_ema[pid] = g.copy()
        else:
            self.grad_ema[pid] = self.ema_decay * self.grad_ema[pid] + (1.0 - self.ema_decay) * g

    def compute_importance(self, param: Tensor) -> np.ndarray:
        pid = id(param)
        if pid in self.grad_ema:
            g = self.grad_ema[pid]
        elif param.grad is not None:
            g = param.grad
        else:
            g = np.zeros_like(param.data)

        # Movement: if W > 0 and g > 0, weight is being pushed downward (toward 0).
        # We assign higher importance to weights that resist pruning.
        return - (param.data * g)
