"""
AQUA Platform - Self-Pruning Engine, Schedulers, and Sparse Inference.
"""

from prune.criteria import (
    ImportanceCriterion,
    TaylorSaliencyCriterion,
    MagnitudeCriterion,
    MovementCriterion,
    RandomCriterion,
)
from prune.scheduler import CubicSparsityScheduler, ConstantSparsityScheduler
from prune.mask import Pruner
from prune.sparse_engine import CSRLinear, benchmark_dense_vs_sparse

__all__ = [
    "ImportanceCriterion",
    "TaylorSaliencyCriterion",
    "MagnitudeCriterion",
    "MovementCriterion",
    "RandomCriterion",
    "CubicSparsityScheduler",
    "ConstantSparsityScheduler",
    "Pruner",
    "CSRLinear",
    "benchmark_dense_vs_sparse",
]
