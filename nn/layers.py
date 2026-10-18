"""
Composable neural network layers and modules.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from engine.tensor import Tensor
from engine.ops import relu, gelu


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        super().__setattr__("_modules", [])
        super().__setattr__("_parameters", [])

    def register_parameter(self, name: str, param: Tensor) -> Tensor:
        self._parameters.append((name, param))
        return param

    def register_module(self, name: str, module: Module) -> Module:
        self._modules.append((name, module))
        return module

    def __setattr__(self, name: str, value):
        if name not in ("_modules", "_parameters"):
            if "_parameters" in self.__dict__:
                self.__dict__["_parameters"] = [(n, p) for n, p in self._parameters if n != name]
            if "_modules" in self.__dict__:
                self.__dict__["_modules"] = [(n, m) for n, m in self._modules if n != name]

            if isinstance(value, Tensor) and value.requires_grad:
                if "_parameters" not in self.__dict__:
                    self.__dict__["_parameters"] = []
                self.__dict__["_parameters"].append((name, value))
            elif isinstance(value, Module):
                if "_modules" not in self.__dict__:
                    self.__dict__["_modules"] = []
                self.__dict__["_modules"].append((name, value))

        super().__setattr__(name, value)

    def parameters(self) -> List[Tensor]:
        params = [p for _, p in self._parameters]
        for _, m in self._modules:
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, Tensor]]:
        named = []
        for name, p in self._parameters:
            full_name = f"{prefix}.{name}" if prefix else name
            named.append((full_name, p))
        for m_name, m in self._modules:
            sub_prefix = f"{prefix}.{m_name}" if prefix else m_name
            named.extend(m.named_parameters(sub_prefix))
        return named

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Fully-connected linear layer: Y = X @ W + b.
    Uses Kaiming / He normal initialization: W ~ N(0, sqrt(2 / in_features))
    and zero bias initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=np.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dtype = dtype

        # Kaiming / He Normal Initialization
        # Var(W) = 2.0 / in_features for ReLU/GELU networks to prevent vanishing/exploding variance
        std = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features).astype(dtype) * std
        self.weight = Tensor(w_data, requires_grad=True, dtype=dtype)

        if bias:
            b_data = np.zeros(out_features, dtype=dtype)
            self.bias: Optional[Tensor] = Tensor(b_data, requires_grad=True, dtype=dtype)
        else:
            self.bias = None

    def set_mask(self, mask: np.ndarray) -> None:
        """Enforce binary mask on layer weights."""
        self.weight.set_mask(mask)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLULayer(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class GELULayer(Module):
    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            self.register_module(f"layer_{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
