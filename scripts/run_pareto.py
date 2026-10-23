"""
Sparsity-Accuracy Pareto sweep across multiple random seeds.
Evaluates accuracy retention vs achieved sparsity and non-zero CSR FLOP reduction.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from train.experiment import run_pareto_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQUA Engine - Pareto Sweep")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid reproduction (1 seed, 3 sparsities, 2 epochs)")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per run (default: 8, or 2 in --quick mode)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for plots and JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("AQUA ENGINE -- SPARSITY-ACCURACY PARETO SWEEP & BENCHMARK")
    print("=" * 80)

    if args.quick:
        print("[Notice] Running in --quick mode for fast verification (~20s).")
        sparsity_levels = [0.0, 0.50, 0.90]
        seeds = [42]
        epochs = 2 if args.epochs == 8 else args.epochs
    else:
        # Benchmark sweep across multiple random seeds: 0%, 50%, 75%, 90%, 95%
        sparsity_levels = [0.0, 0.50, 0.75, 0.90, 0.95]
        seeds = [42, 123, 999]
        epochs = args.epochs

    results = run_pareto_experiment(
        sparsity_levels=sparsity_levels,
        seeds=seeds,
        epochs=epochs,
        output_dir=args.output_dir,
    )

    # Compute statistics for 90% sparsity if present
    sp_key = "0.9" if "0.9" in results["saliency"] else str(sparsity_levels[-1])
    sal_stats = [x["val_acc"] * 100.0 for x in results["saliency"].get(sp_key, [])]
    mag_stats = [x["val_acc"] * 100.0 for x in results["magnitude"].get(sp_key, [])]
    speedup = np.mean([x["theoretical_speedup"] for x in results["saliency"].get(sp_key, [])]) if sal_stats else 1.0

    sal_mean, sal_std = (np.mean(sal_stats), np.std(sal_stats)) if sal_stats else (0.0, 0.0)
    mag_mean, mag_std = (np.mean(mag_stats), np.std(mag_stats)) if mag_stats else (0.0, 0.0)


    print("\n" + "=" * 80)
    print("EMPIRICAL HYPOTHESIS VERIFICATION")
    print("=" * 80)
    print(f"At 90% target parameter sparsity on MNIST classification:")
    print(f"  - First-order Taylor Saliency pruning achieves: {sal_mean:.2f}% +/- {sal_std:.2f}% val accuracy")
    print(f"  - Baseline Magnitude pruning achieves:         {mag_mean:.2f}% +/- {mag_std:.2f}% val accuracy")
    print(f"  - Theoretical CSR compute speedup:             {speedup:.2f}x reduction in non-zero FLOPs")
    print(f"  - Saliency Delta over Magnitude:               +{sal_mean - mag_mean:+.2f}%")
    print("=" * 80)
