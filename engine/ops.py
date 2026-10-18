"""
Differentiable operations for reverse-mode automatic differentiation.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union, List
import numpy as np
from engine.tensor import Tensor


class Op:
    """Base class for all differentiable operations."""

    def __init__(self, parents: List[Tensor]):
        self.parents = parents

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


def _ensure_tensor(x: Union[Tensor, float, int, np.ndarray], dtype=None) -> Tensor:
    from engine.tensor import Tensor
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False, dtype=dtype)


def _prepare_operands(a, b):
    from engine.tensor import Tensor
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a, b
    elif isinstance(a, Tensor):
        return a, _ensure_tensor(b, dtype=a.dtype)
    elif isinstance(b, Tensor):
        return _ensure_tensor(a, dtype=b.dtype), b
    else:
        return _ensure_tensor(a), _ensure_tensor(b)


class Add(Op):
    @classmethod
    def apply(cls, a: Union[Tensor, float, int, np.ndarray], b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        a_t, b_t = _prepare_operands(a, b)
        op = cls([a_t, b_t])
        out_data = a_t.data + b_t.data
        requires_grad = a_t.requires_grad or b_t.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, creator=op if requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (grad, grad)


class Sub(Op):
    @classmethod
    def apply(cls, a: Union[Tensor, float, int, np.ndarray], b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        a_t, b_t = _prepare_operands(a, b)
        op = cls([a_t, b_t])
        out_data = a_t.data - b_t.data
        requires_grad = a_t.requires_grad or b_t.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, creator=op if requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (grad, -grad)


class Mul(Op):
    @classmethod
    def apply(cls, a: Union[Tensor, float, int, np.ndarray], b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        a_t, b_t = _prepare_operands(a, b)
        op = cls([a_t, b_t])
        out_data = a_t.data * b_t.data
        requires_grad = a_t.requires_grad or b_t.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, creator=op if requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.parents
        return (grad * b.data, grad * a.data)


class Div(Op):
    @classmethod
    def apply(cls, a: Union[Tensor, float, int, np.ndarray], b: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        a_t, b_t = _prepare_operands(a, b)
        op = cls([a_t, b_t])
        out_data = a_t.data / b_t.data
        requires_grad = a_t.requires_grad or b_t.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, creator=op if requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.parents
        grad_a = grad / b.data
        grad_b = -grad * a.data / (b.data ** 2)
        return (grad_a, grad_b)


class Neg(Op):
    @classmethod
    def apply(cls, a: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a])
        out_data = -a.data
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (-grad,)


class MatMul(Op):
    @classmethod
    def apply(cls, a: Tensor, b: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a, b])
        out_data = np.matmul(a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, creator=op if requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.parents
        # Y = A @ B
        # dL/dA = grad @ B^T
        # dL/dB = A^T @ grad
        b_t = np.swapaxes(b.data, -1, -2)
        a_t = np.swapaxes(a.data, -1, -2)
        grad_a = np.matmul(grad, b_t)
        grad_b = np.matmul(a_t, grad)
        return (grad_a, grad_b)


class Sum(Op):
    def __init__(self, parents: List[Tensor], axis=None, keepdims=False, input_shape=()):
        super().__init__(parents)
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = input_shape

    @classmethod
    def apply(cls, a: Tensor, axis=None, keepdims=False) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a], axis=axis, keepdims=keepdims, input_shape=a.shape)
        out_data = np.sum(a.data, axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        if not self.keepdims and self.axis is not None:
            axis = (self.axis,) if isinstance(self.axis, int) else self.axis
            shape = list(self.input_shape)
            for a in sorted(axis):
                shape[a] = 1
            grad = grad.reshape(shape)
        return (np.broadcast_to(grad, self.input_shape).copy(),)


class Mean(Op):
    def __init__(self, parents: List[Tensor], axis=None, keepdims=False, input_shape=()):
        super().__init__(parents)
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = input_shape

    @classmethod
    def apply(cls, a: Tensor, axis=None, keepdims=False) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a], axis=axis, keepdims=keepdims, input_shape=a.shape)
        out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        if self.axis is None:
            count = np.prod(self.input_shape) if len(self.input_shape) > 0 else 1
            expanded_grad = np.broadcast_to(grad, self.input_shape)
        else:
            axis = (self.axis,) if isinstance(self.axis, int) else self.axis
            count = np.prod([self.input_shape[a] for a in axis])
            shape = list(self.input_shape)
            if not self.keepdims:
                for a in sorted(axis):
                    shape[a] = 1
                grad = grad.reshape(shape)
            expanded_grad = np.broadcast_to(grad, self.input_shape)

        return ((expanded_grad / count).copy(),)


class ReLU(Op):
    @classmethod
    def apply(cls, a: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a])
        out_data = np.maximum(0.0, a.data)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.parents[0]
        grad_a = grad * (a.data > 0.0)
        return (grad_a,)


class GELU(Op):
    """
    Gaussian Error Linear Unit (accurate tanh approximation):
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    @classmethod
    def apply(cls, a: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a])
        c = np.sqrt(2.0 / np.pi)
        inner = c * (a.data + 0.044715 * (a.data ** 3))
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * a.data * (1.0 + tanh_inner)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.parents[0]
        x = a.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * (x ** 3))
        tanh_inner = np.tanh(inner)
        sech2 = 1.0 - (tanh_inner ** 2)
        d_inner = c * (1.0 + 3.0 * 0.044715 * (x ** 2))
        d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
        return (grad * d_gelu,)


class Tanh(Op):
    @classmethod
    def apply(cls, a: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a])
        out_data = np.tanh(a.data)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.parents[0]
        tanh_val = np.tanh(a.data)
        grad_a = grad * (1.0 - tanh_val ** 2)
        return (grad_a,)


class Sigmoid(Op):
    @classmethod
    def apply(cls, a: Tensor) -> Tensor:
        from engine.tensor import Tensor
        op = cls([a])
        # Numerically stable sigmoid
        positive_mask = a.data >= 0
        neg_mask = ~positive_mask
        out_data = np.empty_like(a.data)
        out_data[positive_mask] = 1.0 / (1.0 + np.exp(-a.data[positive_mask]))
        exp_val = np.exp(a.data[neg_mask])
        out_data[neg_mask] = exp_val / (1.0 + exp_val)
        return Tensor(out_data, requires_grad=a.requires_grad, creator=op if a.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.parents[0]
        # Recompute sigmoid safely
        positive_mask = a.data >= 0
        neg_mask = ~positive_mask
        s = np.empty_like(a.data)
        s[positive_mask] = 1.0 / (1.0 + np.exp(-a.data[positive_mask]))
        exp_val = np.exp(a.data[neg_mask])
        s[neg_mask] = exp_val / (1.0 + exp_val)
        grad_a = grad * s * (1.0 - s)
        return (grad_a,)


class SoftmaxCrossEntropy(Op):
    """
    Fused Softmax + Cross-Entropy Loss for maximum numerical stability.
    Uses log-sum-exp trick to prevent overflow/underflow.
    """
    def __init__(self, parents: List[Tensor], targets: np.ndarray, log_probs: np.ndarray):
        super().__init__(parents)
        self.targets = targets
        self.log_probs = log_probs

    @classmethod
    def apply(cls, logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
        from engine.tensor import Tensor
        if isinstance(targets, Tensor):
            targets_arr = targets.data
        else:
            targets_arr = np.asarray(targets)

        # Ensure 2D logits: (N, C)
        assert logits.ndim == 2, f"Logits must be 2D (N, C), got shape {logits.shape}"
        N, C = logits.shape

        # Log-Sum-Exp Trick: m = max(z, axis=-1, keepdims=True)
        # log(sum(exp(z))) = m + log(sum(exp(z - m)))
        max_logits = np.max(logits.data, axis=-1, keepdims=True)
        shifted = logits.data - max_logits
        log_sum_exp = max_logits + np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = logits.data - log_sum_exp

        # Cross-Entropy Loss: L = - 1/N * sum(log_probs[i, target_i])
        if targets_arr.ndim == 1:
            # Integer class targets
            row_indices = np.arange(N)
            nll = -log_probs[row_indices, targets_arr.astype(int)]
            loss_val = np.mean(nll)
        else:
            # One-hot targets
            loss_val = -np.mean(np.sum(targets_arr * log_probs, axis=-1))

        op = cls([logits], targets=targets_arr, log_probs=log_probs)
        return Tensor(loss_val, requires_grad=logits.requires_grad, creator=op if logits.requires_grad else None)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        logits = self.parents[0]
        N, C = logits.shape
        probs = np.exp(self.log_probs)

        if self.targets.ndim == 1:
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), self.targets.astype(int)] = 1.0
        else:
            one_hot = self.targets

        # Gradient: dL/dz = (probs - one_hot) / N * grad
        grad_logits = (probs - one_hot) / N * grad
        return (grad_logits,)


# Convenient functional wrappers
def relu(x: Tensor) -> Tensor:
    return ReLU.apply(x)


def gelu(x: Tensor) -> Tensor:
    return GELU.apply(x)


def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)


def softmax_cross_entropy(logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
    return SoftmaxCrossEntropy.apply(logits, targets)
