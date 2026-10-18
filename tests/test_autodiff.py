"""
Finite-difference numerical gradient checking test suite for reverse-mode autodiff.
"""

import pytest
import numpy as np
from engine.tensor import Tensor
from engine.ops import (
    relu,
    gelu,
    tanh,
    sigmoid,
    softmax_cross_entropy,
)


def compute_numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute numerical gradient using central finite difference."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        pos_val = float(f(x))

        x[idx] = orig_val - eps
        neg_val = float(f(x))

        x[idx] = orig_val
        grad[idx] = (pos_val - neg_val) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(grad_analytic: np.ndarray, grad_numerical: np.ndarray) -> float:
    diff = np.linalg.norm(grad_analytic - grad_numerical)
    denom = max(np.linalg.norm(grad_analytic), np.linalg.norm(grad_numerical), 1e-8)
    return float(diff / denom)


class TestAutodiffGradientChecks:

    def test_add_sub(self):
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        y_data = np.random.randn(3, 4).astype(np.float64)

        # f(x, y) = sum((x + y) - 2 * x)
        def func_x(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            ty = Tensor(y_data, requires_grad=False, dtype=np.float64)
            out = (tx + ty) - (tx * 2.0)
            return out.sum().data

        def func_y(y):
            tx = Tensor(x_data, requires_grad=False, dtype=np.float64)
            ty = Tensor(y, requires_grad=True, dtype=np.float64)
            out = (tx + ty) - (tx * 2.0)
            return out.sum().data

        # Analytic
        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        ty = Tensor(y_data, requires_grad=True, dtype=np.float64)
        loss = ((tx + ty) - (tx * 2.0)).sum()
        loss.backward()

        num_grad_x = compute_numerical_gradient(func_x, x_data.copy())
        num_grad_y = compute_numerical_gradient(func_y, y_data.copy())

        assert relative_error(tx.grad, num_grad_x) < 1e-6
        assert relative_error(ty.grad, num_grad_y) < 1e-6

    def test_mul_div(self):
        np.random.seed(42)
        x_data = np.random.uniform(0.5, 2.0, (2, 3)).astype(np.float64)
        y_data = np.random.uniform(0.5, 2.0, (2, 3)).astype(np.float64)

        def func_x(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            ty = Tensor(y_data, requires_grad=False, dtype=np.float64)
            out = (tx * ty) / (tx + ty)
            return out.sum().data

        def func_y(y):
            tx = Tensor(x_data, requires_grad=False, dtype=np.float64)
            ty = Tensor(y, requires_grad=True, dtype=np.float64)
            out = (tx * ty) / (tx + ty)
            return out.sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        ty = Tensor(y_data, requires_grad=True, dtype=np.float64)
        loss = ((tx * ty) / (tx + ty)).sum()
        loss.backward()

        num_grad_x = compute_numerical_gradient(func_x, x_data.copy())
        num_grad_y = compute_numerical_gradient(func_y, y_data.copy())

        assert relative_error(tx.grad, num_grad_x) < 1e-6
        assert relative_error(ty.grad, num_grad_y) < 1e-6

    def test_matmul(self):
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        w_data = np.random.randn(4, 5).astype(np.float64)

        def func_x(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            tw = Tensor(w_data, requires_grad=False, dtype=np.float64)
            return (tx @ tw).sum().data

        def func_w(w):
            tx = Tensor(x_data, requires_grad=False, dtype=np.float64)
            tw = Tensor(w, requires_grad=True, dtype=np.float64)
            return (tx @ tw).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        tw = Tensor(w_data, requires_grad=True, dtype=np.float64)
        loss = (tx @ tw).sum()
        loss.backward()

        num_grad_x = compute_numerical_gradient(func_x, x_data.copy())
        num_grad_w = compute_numerical_gradient(func_w, w_data.copy())

        assert relative_error(tx.grad, num_grad_x) < 1e-6
        assert relative_error(tw.grad, num_grad_w) < 1e-6

    def test_reductions_sum_mean(self):
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)

        # Mean along axis 0
        def func_mean0(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return tx.mean(axis=0).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = tx.mean(axis=0).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_mean0, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6

        # Mean along axis 1 with keepdims
        def func_mean1(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return (tx.mean(axis=1, keepdims=True) * 3.0).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = (tx.mean(axis=1, keepdims=True) * 3.0).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_mean1, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6

    def test_activations(self):
        np.random.seed(42)
        # Avoid 0 for ReLU non-differentiability point
        x_data = np.array([-2.5, -1.0, -0.2, 0.3, 1.2, 2.8], dtype=np.float64)

        # ReLU
        def func_relu(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return relu(tx).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = relu(tx).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_relu, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6

        # GELU
        def func_gelu(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return gelu(tx).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = gelu(tx).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_gelu, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-5

        # Tanh
        def func_tanh(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return tanh(tx).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = tanh(tx).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_tanh, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6

        # Sigmoid
        def func_sigmoid(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            return sigmoid(tx).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        loss = sigmoid(tx).sum()
        loss.backward()
        num_grad = compute_numerical_gradient(func_sigmoid, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6

    def test_softmax_cross_entropy(self):
        np.random.seed(42)
        logits_data = np.random.randn(4, 5).astype(np.float64)
        targets = np.array([0, 3, 4, 1])

        def func_ce(logits):
            t_logits = Tensor(logits, requires_grad=True, dtype=np.float64)
            return softmax_cross_entropy(t_logits, targets).data

        t_logits = Tensor(logits_data, requires_grad=True, dtype=np.float64)
        loss = softmax_cross_entropy(t_logits, targets)
        loss.backward()

        num_grad = compute_numerical_gradient(func_ce, logits_data.copy())
        assert relative_error(t_logits.grad, num_grad) < 1e-6

    def test_gradient_accumulation_and_diamond_graph(self):
        r"""
        Verify topological sort handles diamond graph correctly:
            x
           / \
          a   b
           \ /
            y
        """
        np.random.seed(42)
        x_data = np.random.randn(3, 3).astype(np.float64)

        def func_diamond(x):
            tx = Tensor(x, requires_grad=True, dtype=np.float64)
            a = tx * 2.0
            b = tx * 3.0
            y = a + b
            return (y * y).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        a = tx * 2.0
        b = tx * 3.0
        y = a + b
        loss = (y * y).sum()
        loss.backward()

        num_grad = compute_numerical_gradient(func_diamond, x_data.copy())
        assert relative_error(tx.grad, num_grad) < 1e-6


