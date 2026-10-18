"""
Tests verifying that backward pass correctly accumulates broadcasted gradients.
"""

import pytest
import numpy as np
from engine.tensor import Tensor
try:
    from tests.test_autodiff import compute_numerical_gradient, relative_error
except ImportError:
    from test_autodiff import compute_numerical_gradient, relative_error


class TestBroadcastingGradients:

    def test_matrix_vector_add_broadcasting(self):
        """Shape (N, D) + (D,) -> verifies bias gradient summation along axis 0."""
        np.random.seed(42)
        X_data = np.random.randn(5, 4).astype(np.float64)
        b_data = np.random.randn(4).astype(np.float64)

        def func_b(b):
            tX = Tensor(X_data, requires_grad=False, dtype=np.float64)
            tb = Tensor(b, requires_grad=True, dtype=np.float64)
            return (tX + tb).sum().data

        def func_X(X):
            tX = Tensor(X, requires_grad=True, dtype=np.float64)
            tb = Tensor(b_data, requires_grad=False, dtype=np.float64)
            return (tX + tb).sum().data

        tX = Tensor(X_data, requires_grad=True, dtype=np.float64)
        tb = Tensor(b_data, requires_grad=True, dtype=np.float64)
        loss = (tX + tb).sum()
        loss.backward()

        num_grad_b = compute_numerical_gradient(func_b, b_data.copy())
        num_grad_X = compute_numerical_gradient(func_X, X_data.copy())

        assert tb.grad.shape == b_data.shape
        assert tX.grad.shape == X_data.shape
        assert relative_error(tb.grad, num_grad_b) < 1e-6
        assert relative_error(tX.grad, num_grad_X) < 1e-6

    def test_outer_broadcasting(self):
        """Shape (5, 1) + (1, 4) -> both broadcasted."""
        np.random.seed(42)
        a_data = np.random.randn(5, 1).astype(np.float64)
        b_data = np.random.randn(1, 4).astype(np.float64)

        def func_a(a):
            ta = Tensor(a, requires_grad=True, dtype=np.float64)
            tb = Tensor(b_data, requires_grad=False, dtype=np.float64)
            return ((ta * 2.0) + (tb * 3.0)).sum().data

        def func_b(b):
            ta = Tensor(a_data, requires_grad=False, dtype=np.float64)
            tb = Tensor(b, requires_grad=True, dtype=np.float64)
            return ((ta * 2.0) + (tb * 3.0)).sum().data

        ta = Tensor(a_data, requires_grad=True, dtype=np.float64)
        tb = Tensor(b_data, requires_grad=True, dtype=np.float64)
        loss = ((ta * 2.0) + (tb * 3.0)).sum()
        loss.backward()

        num_grad_a = compute_numerical_gradient(func_a, a_data.copy())
        num_grad_b = compute_numerical_gradient(func_b, b_data.copy())

        assert ta.grad.shape == (5, 1)
        assert tb.grad.shape == (1, 4)
        assert relative_error(ta.grad, num_grad_a) < 1e-6
        assert relative_error(tb.grad, num_grad_b) < 1e-6

    def test_tensor_scalar_broadcasting(self):
        """Shape (3, 4, 5) * scalar."""
        np.random.seed(42)
        x_data = np.random.randn(3, 4, 5).astype(np.float64)
        s_data = np.array(2.5, dtype=np.float64)

        def func_s(s):
            tx = Tensor(x_data, requires_grad=False, dtype=np.float64)
            ts = Tensor(s, requires_grad=True, dtype=np.float64)
            return (tx * ts).sum().data

        tx = Tensor(x_data, requires_grad=True, dtype=np.float64)
        ts = Tensor(s_data, requires_grad=True, dtype=np.float64)
        loss = (tx * ts).sum()
        loss.backward()

        num_grad_s = compute_numerical_gradient(func_s, s_data.copy())
        assert ts.grad.shape == ()
        assert relative_error(ts.grad, num_grad_s) < 1e-6


