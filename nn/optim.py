"""
Custom optimizers (SGD and Adam) implemented from scratch in pure NumPy.
Engineered with strict masked-weight isolation to prevent momentum leakage and zombie drift.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from engine.tensor import Tensor


class Optimizer:
    """Base optimizer class."""

    def __init__(self, params: List[Tensor], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict] = {}

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def reset_state(self, p: Tensor) -> None:
        """Reset internal optimizer state for a parameter."""
        pid = id(p)
        if pid in self.state:
            del self.state[pid]


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional Nesterov/standard momentum.
    Strictly isolates masked connections.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        sanitize_moments: bool = True,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.sanitize_moments = sanitize_moments

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Enforce 0 gradient for masked weights
            if self.sanitize_moments and p.mask is not None:
                grad = grad * p.mask

            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p.data

            pid = id(p)
            if self.momentum > 0.0:
                if pid not in self.state:
                    self.state[pid] = {"v": np.zeros_like(p.data)}
                v = self.state[pid]["v"]
                v = self.momentum * v + grad

                # Sanitize momentum buffer for masked weights
                if self.sanitize_moments and p.mask is not None:
                    v = v * p.mask
                self.state[pid]["v"] = v
                step_direction = v
            else:
                step_direction = grad

            p.data = p.data - self.lr * step_direction

            # Enforce masked weights remain 0
            if self.sanitize_moments and p.mask is not None:
                p.data = p.data * p.mask


class Adam(Optimizer):
    """
    Adam optimizer from scratch with bias correction and masked-weight isolation.

    Mathematical Formulation:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
      m_hat_t = m_t / (1 - beta1 ** t)
      v_hat_t = v_t / (1 - beta2 ** t)
      theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    Masked-Weight Correctness:
      If M_ij == 0:
        1. Incoming g_t[i, j] = 0
        2. Optimizer moments m_t[i, j] and v_t[i, j] are zeroed:
           m_t = m_t * M, v_t = v_t * M.
           This prevents exponential decay leakage (beta1 * m_{t-1}) from silently
           drifting dormant weights while they contribute nothing to forward pass.
        3. theta_t[i, j] remains strictly 0.0.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sanitize_moments: bool = True,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.sanitize_moments = sanitize_moments

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Enforce gradient is strictly zero for masked weights
            if self.sanitize_moments and p.mask is not None:
                grad = grad * p.mask

            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p.data

            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {
                    "step": 0,
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            state = self.state[pid]
            state["step"] += 1
            t = state["step"]

            m = state["m"]
            v = state["v"]

            # Update biased 1st and 2nd moment estimates
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            # CRITICAL OPTIMIZER ISOLATION:
            # If masked and sanitization enabled, zero out moments to prevent momentum leakage or zombie drift
            if self.sanitize_moments and p.mask is not None:
                m = m * p.mask
                v = v * p.mask

            state["m"] = m
            state["v"] = v

            # Compute bias-corrected moments
            # (Note: beta ** t handles numerical underflow smoothly)
            bias_corr1 = 1.0 - (self.beta1 ** t)
            bias_corr2 = 1.0 - (self.beta2 ** t)

            m_hat = m / bias_corr1
            v_hat = v / bias_corr2

            # Compute Adam step
            denom = np.sqrt(v_hat) + self.eps
            step_val = (m_hat / denom)

            p.data = p.data - self.lr * step_val

            # Enforce exact zero on masked weights
            if self.sanitize_moments and p.mask is not None:
                p.data = p.data * p.mask
