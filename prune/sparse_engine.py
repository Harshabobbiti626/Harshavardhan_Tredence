"""
Sparse inference engine and honest cost measurement for pruned neural networks.
Converts masked weights into Compressed Sparse Row (CSR) representation and
executes sparse forward passes to prove real FLOP reductions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import time
import numpy as np
from engine.tensor import Tensor
from nn.layers import Module, Linear


class CSRLinear:
    """
    Compressed Sparse Row (CSR) linear layer for sparse-aware inference.

    Forward Computation:
      Dense:  Y = X @ W + b        (FLOPs = 2 * N * d_in * d_out)
      Sparse: For each output neuron j:
                Y[:, j] = sum_{k in nonzeros(W[:, j])} (X[:, k] * W[k, j]) + b[j]
              (FLOPs = 2 * N * nnz(W))

    Stores non-zeros in CSR arrays:
      - data: non-zero values
      - indices: column indices of non-zero values
      - indptr: index pointers for rows
    """

    def __init__(self, linear_layer: Linear):
        mask = linear_layer.weight.mask if linear_layer.weight.mask is not None else np.ones(linear_layer.weight.shape, dtype=bool)
        W = linear_layer.weight.data * mask
        # Transpose W to shape (d_out, d_in) so each row corresponds to one output neuron
        W_T = W.T
        self.d_out, self.d_in = W_T.shape
        self.bias = linear_layer.bias.data.copy() if linear_layer.bias is not None else None

        # Build CSR representation of W_T
        self.indptr = [0]
        self.indices = []
        self.data = []

        for row in range(self.d_out):
            row_data = W_T[row]
            nonzero_cols = np.where(row_data != 0.0)[0]
            self.indices.extend(nonzero_cols)
            self.data.extend(row_data[nonzero_cols])
            self.indptr.append(len(self.indices))

        self.indptr = np.array(self.indptr, dtype=np.int32)
        self.indices = np.array(self.indices, dtype=np.int32)
        self.data = np.array(self.data, dtype=np.float32)
        self.nnz = len(self.data)
        self.total_weights = self.d_in * self.d_out
        self.sparsity = 1.0 - (self.nnz / self.total_weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Execute sparse matrix multiplication: Y = X @ W + b
        Only visits non-zero weights.
        """
        N = x.shape[0]
        out = np.zeros((N, self.d_out), dtype=np.float32)

        # Vectorized over batch N for each sparse output neuron
        for j in range(self.d_out):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            if start == end:
                continue

            cols = self.indices[start:end]
            weights = self.data[start:end]
            # out[:, j] = sum_k (x[:, cols] * weights)
            out[:, j] = np.dot(x[:, cols], weights)

        if self.bias is not None:
            out += self.bias

        return out

    def compute_flops(self, batch_size: int) -> Dict[str, float]:
        """
        Compute theoretical dense vs. sparse FLOPs for this layer.
        One multiply-accumulate (MAC) is counted as 2 FLOPs (1 multiply, 1 add).
        """
        dense_macs = batch_size * self.d_in * self.d_out
        sparse_macs = batch_size * self.nnz
        dense_flops = 2.0 * dense_macs + (batch_size * self.d_out if self.bias is not None else 0)
        sparse_flops = 2.0 * sparse_macs + (batch_size * self.d_out if self.bias is not None else 0)

        return {
            "dense_flops": dense_flops,
            "sparse_flops": sparse_flops,
            "flop_reduction_ratio": dense_flops / max(sparse_flops, 1.0),
            "theoretical_savings_pct": (1.0 - (sparse_flops / max(dense_flops, 1.0))) * 100.0,
            "nnz": self.nnz,
            "total_weights": self.total_weights,
            "sparsity": self.sparsity,
        }


class SparseSequential:
    """Sequential model where all Linear layers are converted to CSR format."""

    def __init__(self, layers: List[CSRLinear]):
        self.layers = layers

    @classmethod
    def from_dense_model(cls, model: Module) -> SparseSequential:
        sparse_layers = []
        for name, module in model._modules:
            if isinstance(module, Linear):
                sparse_layers.append(CSRLinear(module))
        return cls(sparse_layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            # ReLU activation between hidden layers
            if i < len(self.layers) - 1:
                out = np.maximum(0.0, out)
        return out

    def compute_total_flops(self, batch_size: int) -> Dict[str, float]:
        total_dense = 0.0
        total_sparse = 0.0
        total_nnz = 0
        total_weights = 0

        for layer in self.layers:
            stats = layer.compute_flops(batch_size)
            total_dense += stats["dense_flops"]
            total_sparse += stats["sparse_flops"]
            total_nnz += stats["nnz"]
            total_weights += stats["total_weights"]

        return {
            "total_dense_flops": total_dense,
            "total_sparse_flops": total_sparse,
            "theoretical_speedup": total_dense / max(total_sparse, 1.0),
            "overall_sparsity": 1.0 - (total_nnz / max(total_weights, 1)),
            "active_parameters": total_nnz,
            "total_parameters": total_weights,
        }


def benchmark_dense_vs_sparse(
    dense_model: Module,
    x_test: np.ndarray,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Benchmark wall-clock latency of dense model vs sparse-aware CSR model.
    """
    sparse_model = SparseSequential.from_dense_model(dense_model)
    batch_size = x_test.shape[0]

    # Warm-up runs
    for _ in range(5):
        _ = dense_model(Tensor(x_test, requires_grad=False))
        _ = sparse_model.forward(x_test)

    # Measure dense wall-clock time
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = dense_model(Tensor(x_test, requires_grad=False))
    t1 = time.perf_counter()
    dense_time_ms = ((t1 - t0) / num_runs) * 1000.0

    # Measure sparse wall-clock time
    t2 = time.perf_counter()
    for _ in range(num_runs):
        _ = sparse_model.forward(x_test)
    t3 = time.perf_counter()
    sparse_time_ms = ((t3 - t2) / num_runs) * 1000.0

    flop_stats = sparse_model.compute_total_flops(batch_size)

    return {
        "batch_size": batch_size,
        "dense_time_ms": dense_time_ms,
        "sparse_time_ms": sparse_time_ms,
        "dense_flops": flop_stats["total_dense_flops"],
        "sparse_flops": flop_stats["total_sparse_flops"],
        "theoretical_speedup": flop_stats["theoretical_speedup"],
        "sparsity": flop_stats["overall_sparsity"],
        "active_params": flop_stats["active_parameters"],
        "total_params": flop_stats["total_parameters"],
    }
