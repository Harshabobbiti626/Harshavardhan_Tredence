"""
Pareto sweep experiment runner, baseline comparison, and empirical verification.
Generates committed plots and raw JSON numbers across random seeds and sparsities.
"""

from __future__ import annotations
from typing import Dict, List, Any
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn.layers import Linear, ReLULayer, Sequential
from nn.optim import Adam
from prune.criteria import (
    TaylorSaliencyCriterion,
    MagnitudeCriterion,
    MovementCriterion,
    RandomCriterion,
)
from prune.scheduler import CubicSparsityScheduler, ConstantSparsityScheduler
from prune.mask import Pruner
from prune.sparse_engine import benchmark_dense_vs_sparse
from train.dataset import load_dataset, DataLoader
from train.trainer import Trainer


def build_mlp(in_features: int = 784, hidden_dim: int = 128, num_classes: int = 10, seed: int = 42) -> Sequential:
    """Construct standard MLP: 784 -> 128 -> ReLU -> 10."""
    np.random.seed(seed)
    return Sequential(
        Linear(in_features, hidden_dim),
        ReLULayer(),
        Linear(hidden_dim, num_classes),
    )


def train_single_model(
    target_sparsity: float,
    criterion_type: str = "saliency",  # "saliency", "magnitude", "movement", "random"
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    dataset_name: str = "mnist",
    allow_regrowth: bool = False,
    ema_decay: float = 0.9,
    scheduler_type: str = "cubic",  # "cubic" or "constant"
    sanitize_moments: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a single model under target sparsity, criterion, scheduler, and optimizer configuration."""
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, seed=seed, max_samples=8000)

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = build_mlp(in_features=784, hidden_dim=128, num_classes=10, seed=seed)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4, sanitize_moments=sanitize_moments)

    pruner = None
    if target_sparsity > 0.0:
        total_steps = len(train_loader) * epochs
        start_step = max(1, int(total_steps * 0.15))
        end_step = max(start_step + 1, int(total_steps * 0.75))

        if criterion_type == "saliency":
            criterion = TaylorSaliencyCriterion(ema_decay=ema_decay)
        elif criterion_type == "movement":
            criterion = MovementCriterion(ema_decay=ema_decay)
        elif criterion_type == "random":
            criterion = RandomCriterion(seed=seed)
        else:
            criterion = MagnitudeCriterion()

        if scheduler_type == "constant":
            scheduler = ConstantSparsityScheduler(target_sparsity=target_sparsity, prune_step=start_step)
        else:
            scheduler = CubicSparsityScheduler(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                start_step=start_step,
                end_step=end_step,
                prune_freq=20,
            )
        pruner = Pruner(
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            optimizer=optimizer,
            allow_regrowth=allow_regrowth,
        )

    trainer = Trainer(model=model, optimizer=optimizer, pruner=pruner)
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)

    final_loss, final_acc = trainer.evaluate(val_loader)
    achieved_sparsity = pruner.get_sparsity() if pruner is not None else 0.0

    # Benchmark true sparse inference & FLOP savings
    sparse_bench = benchmark_dense_vs_sparse(model, X_test[:100], num_runs=20)

    return {
        "target_sparsity": target_sparsity,
        "achieved_sparsity": achieved_sparsity,
        "criterion": criterion_type,
        "seed": seed,
        "final_val_loss": final_loss,
        "final_val_acc": final_acc,
        "history": history,
        "benchmark": sparse_bench,
    }


def run_pareto_experiment(
    sparsity_levels: List[float] = [0.0, 0.50, 0.75, 0.90, 0.95],
    seeds: List[int] = [42, 123, 999],
    epochs: int = 8,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Executes the Sparsity-Accuracy Pareto sweep across multiple random seeds.
    Compares Taylor Saliency vs Magnitude Baseline and produces benchmark artifacts.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {
        "sparsity_levels": sparsity_levels,
        "seeds": seeds,
        "saliency": {},
        "magnitude": {},
    }

    print("=" * 80)
    print("STARTING SPARSITY-ACCURACY PARETO SWEEP ACROSS SEEDS & CRITERIA")
    print("=" * 80)

    for sp in sparsity_levels:
        results["saliency"][str(sp)] = []
        results["magnitude"][str(sp)] = []

        print(f"\nEvaluating Target Sparsity: {sp * 100:.1f}%")
        for seed in seeds:
            # 1. Saliency Pruning
            print(f"  [Saliency] Seed {seed}...")
            res_sal = train_single_model(
                target_sparsity=sp,
                criterion_type="saliency",
                epochs=epochs,
                seed=seed,
                verbose=False,
            )
            results["saliency"][str(sp)].append({
                "seed": seed,
                "achieved_sparsity": res_sal["achieved_sparsity"],
                "val_acc": res_sal["final_val_acc"],
                "theoretical_speedup": res_sal["benchmark"]["theoretical_speedup"],
                "dense_flops": res_sal["benchmark"]["dense_flops"],
                "sparse_flops": res_sal["benchmark"]["sparse_flops"],
            })

            # 2. Magnitude Pruning Baseline
            if sp == 0.0:
                # 0% sparsity is identical for both
                results["magnitude"][str(sp)].append(results["saliency"][str(sp)][-1])
            else:
                print(f"  [Magnitude] Seed {seed}...")
                res_mag = train_single_model(
                    target_sparsity=sp,
                    criterion_type="magnitude",
                    epochs=epochs,
                    seed=seed,
                    verbose=False,
                )
                results["magnitude"][str(sp)].append({
                    "seed": seed,
                    "achieved_sparsity": res_mag["achieved_sparsity"],
                    "val_acc": res_mag["final_val_acc"],
                    "theoretical_speedup": res_mag["benchmark"]["theoretical_speedup"],
                    "dense_flops": res_mag["benchmark"]["dense_flops"],
                    "sparse_flops": res_mag["benchmark"]["sparse_flops"],
                })

    # Save raw numbers
    raw_path = os.path.join(output_dir, "raw_metrics.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifact Saved] Raw metrics written to {raw_path}")

    # Generate Pareto Plot
    plot_path = os.path.join(output_dir, "pareto_curve.png")
    _plot_pareto_curve(results, plot_path)
    print(f"[Artifact Saved] Pareto plot saved to {plot_path}")

    return results


def _plot_pareto_curve(results: Dict[str, Any], save_path: str) -> None:
    """Plot Sparsity vs Accuracy with variance error bars."""
    sparsities = results["sparsity_levels"]
    sal_acc_means = []
    sal_acc_stds = []
    mag_acc_means = []
    mag_acc_stds = []

    for sp in sparsities:
        sp_key = str(sp)
        sal_vals = [x["val_acc"] * 100.0 for x in results["saliency"][sp_key]]
        mag_vals = [x["val_acc"] * 100.0 for x in results["magnitude"][sp_key]]

        sal_acc_means.append(np.mean(sal_vals))
        sal_acc_stds.append(np.std(sal_vals))
        mag_acc_means.append(np.mean(mag_vals))
        mag_acc_stds.append(np.std(mag_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Sparsity vs Accuracy
    ax1.errorbar(
        [s * 100 for s in sparsities],
        sal_acc_means,
        yerr=sal_acc_stds,
        fmt="-o",
        linewidth=2.5,
        capsize=5,
        label="Taylor Saliency Pruning (|w * g|)",
        color="#1f77b4",
    )
    ax1.errorbar(
        [s * 100 for s in sparsities],
        mag_acc_means,
        yerr=mag_acc_stds,
        fmt="--s",
        linewidth=2.0,
        capsize=5,
        label="Magnitude Pruning Baseline (|w|)",
        color="#ff7f0e",
    )
    ax1.set_xlabel("Target Sparsity (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("AQUA Platform: Sparsity vs. Accuracy Pareto Curve", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # Subplot 2: Sparsity vs Theoretical Compute Reduction (FLOPs)
    theoretical_speedups = [
        np.mean([x["theoretical_speedup"] for x in results["saliency"][str(sp)]])
        for sp in sparsities
    ]
    ax2.plot([s * 100 for s in sparsities], theoretical_speedups, "-^", color="#2ca02c", linewidth=2.5)
    ax2.set_xlabel("Target Sparsity (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Theoretical Compute Speedup (x)", fontsize=12, fontweight="bold")
    ax2.set_title("AQUA Platform: True Compute Reduction (CSR FLOPs)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_ablation_experiments(
    output_dir: str = "results",
    epochs: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the comprehensive empirical ablation suite:
      1. Pruning Criteria: Taylor Saliency vs Magnitude vs Movement vs Random
      2. Optimizer Sanitization: Sanitized Adam (m*M, v*M) vs Unsanitized Adam (momentum leakage)
      3. Gradient EMA Smoothing: EMA (gamma=0.9) vs Raw mini-batch gradient (gamma=0.0)
      4. Sparsity Schedule: Gradual Cubic Polynomial vs Instant / Constant Step
      5. Topology Dynamics: Static Pruning vs RigL Dynamic Regrowth (at 90% and 95% sparsity)
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ablation_results: Dict[str, Any] = {
        "metadata": {
            "epochs": epochs,
            "seed": seed,
            "dataset": "mnist",
            "target_sparsity": 0.90,
        },
        "criteria_ablation": {},
        "sanitization_ablation": {},
        "ema_ablation": {},
        "schedule_ablation": {},
        "regrowth_ablation": {},
    }

    print("=" * 80)
    print("AQUA ENGINE -- COMPREHENSIVE EMPIRICAL ABLATION SUITE")
    print("=" * 80)

    # 1. Pruning Criteria Ablation (90% sparsity)
    print("\n[Ablation 1/5] Pruning Criteria Comparison (90% Target Sparsity)...")
    for crit in ["saliency", "magnitude", "movement", "random"]:
        print(f"  Running Criterion: {crit}...")
        res = train_single_model(
            target_sparsity=0.90,
            criterion_type=crit,
            epochs=epochs,
            seed=seed,
            verbose=False,
        )
        ablation_results["criteria_ablation"][crit] = {
            "val_acc": res["final_val_acc"],
            "achieved_sparsity": res["achieved_sparsity"],
            "theoretical_speedup": res["benchmark"]["theoretical_speedup"],
        }
        print(f"  -> {crit.capitalize()}: Acc={res['final_val_acc']*100:.2f}%, Sparsity={res['achieved_sparsity']*100:.2f}%")

    # 2. Optimizer Moment Sanitization Ablation (90% sparsity)
    print("\n[Ablation 2/5] Optimizer Moment Sanitization (Sanitized vs. Unsanitized Adam)...")
    for sanitize in [True, False]:
        tag = "sanitized" if sanitize else "unsanitized"
        print(f"  Running Adam: {tag}...")
        res = train_single_model(
            target_sparsity=0.90,
            criterion_type="saliency",
            epochs=epochs,
            seed=seed,
            sanitize_moments=sanitize,
            verbose=False,
        )
        ablation_results["sanitization_ablation"][tag] = {
            "val_acc": res["final_val_acc"],
            "achieved_sparsity": res["achieved_sparsity"],
            "sanitized": sanitize,
        }
        print(f"  -> Adam ({tag}): Acc={res['final_val_acc']*100:.2f}%")

    # 3. Gradient EMA Smoothing Ablation (90% sparsity)
    print("\n[Ablation 3/5] Gradient EMA Smoothing (gamma=0.9 vs. gamma=0.0)...")
    for ema in [0.9, 0.0]:
        tag = "ema_0.9" if ema > 0 else "raw_minibatch_0.0"
        print(f"  Running EMA: {tag}...")
        res = train_single_model(
            target_sparsity=0.90,
            criterion_type="saliency",
            epochs=epochs,
            seed=seed,
            ema_decay=ema,
            verbose=False,
        )
        ablation_results["ema_ablation"][tag] = {
            "val_acc": res["final_val_acc"],
            "achieved_sparsity": res["achieved_sparsity"],
            "ema_decay": ema,
        }
        print(f"  -> EMA ({tag}): Acc={res['final_val_acc']*100:.2f}%")

    # 4. Sparsity Schedule Ablation (90% sparsity)
    print("\n[Ablation 4/5] Sparsity Schedule (Cubic Polynomial vs. Constant One-Shot)...")
    for sched in ["cubic", "constant"]:
        print(f"  Running Schedule: {sched}...")
        res = train_single_model(
            target_sparsity=0.90,
            criterion_type="saliency",
            epochs=epochs,
            seed=seed,
            scheduler_type=sched,
            verbose=False,
        )
        ablation_results["schedule_ablation"][sched] = {
            "val_acc": res["final_val_acc"],
            "achieved_sparsity": res["achieved_sparsity"],
        }
        print(f"  -> Schedule ({sched}): Acc={res['final_val_acc']*100:.2f}%")

    # 5. Static vs. RigL Dynamic Regrowth Ablation (90% and 95% sparsity)
    print("\n[Ablation 5/5] Static Pruning vs. RigL Dynamic Regrowth (90% & 95% Sparsity)...")
    for sp in [0.90, 0.95]:
        sp_key = f"{int(sp*100)}%"
        ablation_results["regrowth_ablation"][sp_key] = {}
        for regrowth in [False, True]:
            mode = "rigl_regrowth" if regrowth else "static_pruning"
            print(f"  Running Sparsity={sp_key}, Mode={mode}...")
            res = train_single_model(
                target_sparsity=sp,
                criterion_type="saliency",
                epochs=epochs,
                seed=seed,
                allow_regrowth=regrowth,
                verbose=False,
            )
            ablation_results["regrowth_ablation"][sp_key][mode] = {
                "val_acc": res["final_val_acc"],
                "achieved_sparsity": res["achieved_sparsity"],
            }
            print(f"  -> Sparsity {sp_key} ({mode}): Acc={res['final_val_acc']*100:.2f}%")

    # Save to JSON
    out_path = os.path.join(output_dir, "ablation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[Artifact Saved] Ablation metrics written to {out_path}")
    print("=" * 80)
    return ablation_results

