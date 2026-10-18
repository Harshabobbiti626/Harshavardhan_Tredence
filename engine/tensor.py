"""
Tensor class and computation graph for reverse-mode automatic differentiation.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Union
import numpy as np


class Tensor:
    """
    A Tensor wraps a NumPy array and participates in a reverse-mode
    dynamic computation graph.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray],
        requires_grad: bool = False,
        creator: Optional[Op] = None,
        dtype: Optional[Union[np.dtype, type]] = None,
    ):
        if isinstance(data, (np.ndarray, np.generic)):
            arr = np.asarray(data)
            if dtype is not None and arr.dtype != dtype:
                self.data = arr.astype(dtype)
            else:
                self.data = arr
        else:
            if dtype is None:
                self.data = np.array(data, dtype=np.float32)
            else:
                self.data = np.array(data, dtype=dtype)

        self.requires_grad = requires_grad
        self.creator = creator
        self.grad: Optional[np.ndarray] = None
        # Track mask if this tensor is a masked parameter
        self.mask: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def zero_grad(self) -> None:
        """Clear accumulated gradients."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a new detached Tensor sharing the same data."""
        return Tensor(self.data, requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Attach a binary mask to the parameter.
        Enforces that the parameter data is zero wherever mask is zero.
        """
        assert mask.shape == self.shape, f"Mask shape {mask.shape} does not match parameter shape {self.shape}"
        self.mask = mask.astype(bool)
        self.data = self.data * self.mask

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients of the root tensor with respect to all graph ancestors
        using reverse topological order.
        """
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=self.dtype)

        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

        # Build topological ordering via Depth-First Search
        topo: List[Tensor] = []
        visited = set()

        def build_topo(t: Tensor):
            if id(t) not in visited:
                visited.add(id(t))
                if t.creator is not None:
                    for parent in t.creator.parents:
                        build_topo(parent)
                topo.append(t)

        build_topo(self)

        # Backpropagate in reverse topological order
        for node in reversed(topo):
            if node.creator is None:
                continue

            assert node.grad is not None, f"Node {node} in computation graph has no gradient"
            incoming_grad = node.grad

            # Execute backward for the op
            parent_grads = node.creator.backward(incoming_grad)

            for parent, p_grad in zip(node.creator.parents, parent_grads):
                if not parent.requires_grad:
                    continue

                # Ensure shape matches parent after accounting for broadcasting
                p_grad = unbroadcast(p_grad, parent.shape)

                # If parent has an active mask, enforce that masked weights receive 0 gradient
                if parent.mask is not None:
                    p_grad = p_grad * parent.mask

                if parent.grad is None:
                    parent.grad = p_grad.copy()
                else:
                    parent.grad += p_grad

    # Operator Overloads
    def __add__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Add
        return Add.apply(self, other)

    def __radd__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Add
        return Add.apply(other, self)

    def __sub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Sub
        return Sub.apply(self, other)

    def __rsub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Sub
        return Sub.apply(other, self)

    def __mul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Mul
        return Mul.apply(other, self)

    def __truediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Div
        return Div.apply(self, other)

    def __rtruediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Div
        return Div.apply(other, self)

    def __neg__(self) -> Tensor:
        from engine.ops import Neg
        return Neg.apply(self)

    def __matmul__(self, other: Tensor) -> Tensor:
        from engine.ops import MatMul
        return MatMul.apply(self, other)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def relu(self) -> Tensor:
        from engine.ops import relu
        return relu(self)

    def gelu(self) -> Tensor:
        from engine.ops import gelu
        return gelu(self)

    def tanh(self) -> Tensor:
        from engine.ops import tanh
        return tanh(self)

    def sigmoid(self) -> Tensor:
        from engine.ops import sigmoid
        return sigmoid(self)

    def __repr__(self) -> str:
        grad_info = f", grad_fn=<{self.creator.__class__.__name__}>" if self.creator else ""
        if self.requires_grad and self.creator is None:
            grad_info = ", requires_grad=True"
        return f"Tensor({self.data}{grad_info})"


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reverse broadcasted gradient back to the target shape.
    Sums out axes that were prepended or expanded during NumPy broadcasting.
    """
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)

    # 1. Sum over extra leading dimensions if target has fewer dimensions
    if grad_ndim > target_ndim:
        axes_to_sum = tuple(range(grad_ndim - target_ndim))
        grad = np.sum(grad, axis=axes_to_sum)

    # 2. Sum over dimensions where target_shape is 1 and grad shape > 1
    axes_to_sum = []
    for i, (dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and dim > 1:
            axes_to_sum.append(i)

    if axes_to_sum:
        grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)

    return grad.reshape(target_shape)
