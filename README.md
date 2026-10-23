# AQUA Engine — Self-Pruning Neural Network & Autodiff from Scratch

A first-principles deep learning framework and self-pruning neural network engine built in **pure Python and NumPy** with zero framework dependencies (no PyTorch, TensorFlow, JAX, Autograd, or Hugging Face).

Designed for the **AQUA Enterprise AI Platform**, this project explores the systems and numerical foundations of dynamic computation graphs, automatic differentiation, gradient invariant preservation in masked optimizers, and sparse-aware matrix execution.

---

## 1. Core Systems Capabilities

* **Reverse-Mode Autodiff Engine (`engine/`):** Dynamic DAG construction, recursive DFS topological sorting, exact multi-dimensional broadcasting reduction (`unbroadcast`), and fused numerically stable Softmax Cross-Entropy.
* **Momentum Isolation in Masked Optimizers (`nn/optim.py`):** Solves the momentum leakage bug in naive masked adaptive optimizers. Zeroes out Adam first ($m_t$) and second ($v_t$) moments on dormant weights ($m_t \odot M, v_t \odot M$), preventing parameter drift and revival instability.
* **First-Order Taylor Saliency Pruning (`prune/`):** Connection importance derived via multivariate Taylor series expansion ($\mathcal{I}_{ij} = |W_{ij} \cdot \bar{g}_{ij}|$) with gradient EMA smoothing and a cubic polynomial sparsity schedule ($0\% \to 90\%$).
* **Sparse CSR Inference Engine (`prune/sparse_engine.py`):** Converts pruned models to Compressed Sparse Row format, executing forward passes exclusively over non-zero parameters for genuine compute reduction.
* **Dynamic Sparse Training (`prune/mask.py`):** RigL gradient-based dynamic parameter regrowth periodically pruning dormant connections and resurrecting high-gradient paths.

---

## 2. Dependency Policy & Compliance Statement

* **Zero Framework Abstractions:** No deep learning or automatic differentiation libraries are used.
* **Dataset Ingestion Only:** `scikit-learn` (`fetch_openml`) is used solely to download and cache the standard MNIST dataset in [`train/dataset.py`](train/dataset.py).
* **Pure Python & NumPy:** All computational graph tensors, automatic differentiation, loss functions, modules, optimizers (Adam & SGD), schedulers, and CSR sparse inference are implemented from scratch in pure Python 3.10+ and standard NumPy.

---

## 3. Quick Start & Installation

### Requirements
* Python 3.10+
* Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pytest`

### Setup
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 4. Pipeline Execution Suite

```bash
# 1. Run numerical gradient checks & masked optimizer unit tests
python scripts/run_grad_check.py

# 2. Train dense baseline MLP with custom Adam
python scripts/run_train.py

# 3. Train self-pruning network to 90% target sparsity (Taylor Saliency + Cubic Schedule)
python scripts/run_prune.py

# 4. Run multi-seed Pareto sweep (evaluates 0%, 50%, 75%, 90%, 95% across seeds 42, 123, 999)
python scripts/run_pareto.py
# (Fast verification mode: python scripts/run_pareto.py --quick)

# 5. Run 5-dimension empirical ablation suite (criteria, sanitization, EMA, schedule, RigL)
python scripts/run_ablations.py
# (Fast verification mode: python scripts/run_ablations.py --quick)
```

---

## 5. Repository Architecture

```
aqua-prune-engine/
├── engine/
│   ├── tensor.py          # Autodiff Tensor class, dynamic DAG, reverse chain rule, unbroadcast
│   └── ops.py             # Differentiable operators & fused stable softmax_cross_entropy
├── nn/
│   ├── layers.py          # Module, Linear (Kaiming init), ReLU, GELU, Sequential
│   └── optim.py           # Adam and SGD with masked-weight isolation & moment sanitization
├── prune/
│   ├── criteria.py        # Taylor saliency (|w * g|), magnitude, movement, random
│   ├── scheduler.py       # Cubic polynomial sparsity ramp (Zhu & Gupta, 2017)
│   ├── mask.py            # Pruner manager, mask tracking, RigL dynamic regrowth
│   └── sparse_engine.py   # CSR sparse forward inference & true FLOPs counter
├── train/
│   ├── dataset.py         # MNIST loader with caching and synthetic fallback
│   ├── trainer.py         # Mini-batched training loop and validation logger
│   └── experiment.py      # Multi-seed Pareto sweep and ablation suite runner
├── tests/
│   ├── test_autodiff.py   # Finite-difference gradient checking (< 1e-6 relative error)
│   ├── test_broadcast.py  # Broadcasting gradient accumulation verification
│   └── test_masked_opt.py # Proof of zero masked gradient and Adam moment isolation
├── scripts/
│   ├── run_grad_check.py  # Gradient verification test runner
│   ├── run_train.py       # Dense baseline training runner
│   ├── run_prune.py       # Self-pruning training runner
│   ├── run_pareto.py      # Pareto sweep runner
│   └── run_ablations.py   # Ablation suite runner
├── results/
│   ├── pareto_curve.png   # Pareto curve plot
│   ├── raw_metrics.json   # Multi-seed raw metric logs
│   └── ablation_metrics.json # Empirical ablation study numbers
├── DESIGN.md              # Mathematical derivations, systems analysis, & design trade-offs
├── pyproject.toml         # Package configuration and build system
└── requirements.txt       # Core project dependencies
```

---

## 6. Empirical Results & Performance Invariants

### Benchmark Results (MNIST Classification across Seeds 42, 123, 999)

| Target Sparsity | Active Parameters | Saliency Accuracy (Mean $\pm$ Std) | Magnitude Accuracy (Mean $\pm$ Std) | Theoretical Compute Speedup (CSR FLOPs) |
|:---------------:|:-----------------:|:----------------------------------:|:-----------------------------------:|:---------------------------------------:|
| **0.0% (Dense)**| 101,632 / 101,632 | **93.31% $\pm$ 0.00%**             | **93.31% $\pm$ 0.00%**              | 1.00x                                   |
| **50.0%**       | 50,816 / 101,632  | **93.33% $\pm$ 0.54%**             | **93.33% $\pm$ 0.54%**              | 1.99x                                   |
| **75.0%**       | 25,408 / 101,632  | **92.73% $\pm$ 0.35%**             | **92.83% $\pm$ 0.40%**              | 3.97x                                   |
| **90.0%**       | 10,162 / 101,632  | **90.81% $\pm$ 0.32%**             | **92.48% $\pm$ 0.52%**              | **9.94x**                               |
| **95.0%**       | 5,081 / 101,632   | **88.60% $\pm$ 0.48%**             | **88.94% $\pm$ 0.77%**              | **19.82x**                              |

### Verified Hypothesis
> *"At 90% parameter sparsity on standard MNIST classification across 3 fixed random seeds (42, 123, 999), the self-pruning network retains 90.81% $\pm$ 0.32% validation accuracy while achieving an honest 9.94x theoretical reduction in non-zero FLOPs using a sparse-aware Compressed Sparse Row (CSR) forward path, with zero gradient leakage and sanitized optimizer moments."*

---

## 7. Deep Platform & Mathematical Analysis
For formal mathematical proofs, systems profiling analysis, and multi-tenant serving design, refer to [`DESIGN.md`](DESIGN.md):
1. **Multivariate Taylor Series Derivation:** $\mathcal{I}_{ij} = |W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}}|$.
2. **Masked Gradient Invariant & Adam Moment Sanitization:** Mathematical proof of zero gradient for masked weights and analysis of revival explosion.
3. **Autodiff Profiling & Engine Bottlenecks:** Memory allocations, dispatch overhead, and kernel fusion optimization paths.
4. **Multi-Tenant Serving Architecture:** Unstructured CSR vs. 2:4 structured sparsity on hardware Tensor Cores, zero-copy weight caching, and dynamic micro-batching.
5. **Empirical Ablations:** Quantitative evaluation of criteria, moment sanitization, gradient EMA, cubic schedules, and RigL dynamic parameter regrowth.
