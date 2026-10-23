"""
Comprehensive Empirical Ablation Suite Runner.
Systematically evaluates:
  1. Pruning Criteria (Taylor Saliency vs Magnitude vs Movement vs Random)
  2. Optimizer Sanitization (Sanitized Adam vs Unsanitized Adam)
  3. Gradient EMA Smoothing (gamma=0.9 vs gamma=0.0)
  4. Sparsity Schedule (Cubic Polynomial vs Constant One-Shot)
  5. Topology Dynamics (Static Pruning vs RigL Dynamic Regrowth)
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.experiment import run_ablation_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQUA Engine - Empirical Ablation Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid reproduction (~30s)")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per ablation run (default: 6)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON artifacts")
    args = parser.parse_args()

    epochs = 2 if args.quick else args.epochs

    print("=" * 80)
    print("AQUA ENGINE -- EMPIRICAL ABLATION SUITE RUNNER")
    print(f"Mode: {'Quick Verification' if args.quick else 'Full Reproduction'} (Epochs: {epochs})")
    print("=" * 80)

    results = run_ablation_experiments(
        output_dir=args.output_dir,
        epochs=epochs,
        seed=42,
    )

    print("\n" + "=" * 80)
    print("ABLATION SUITE SUMMARY TABLE")
    print("=" * 80)
    print("1. Pruning Criteria (at 90% target sparsity):")
    for k, v in results["criteria_ablation"].items():
        print(f"   - {k.capitalize():<12}: {v['val_acc']*100:.2f}% val acc | {v['achieved_sparsity']*100:.1f}% sparsity")

    print("\n2. Optimizer Sanitization (Taylor at 90% target sparsity):")
    for k, v in results["sanitization_ablation"].items():
        print(f"   - {k.capitalize():<12}: {v['val_acc']*100:.2f}% val acc")

    print("\n3. Gradient EMA Smoothing (Taylor at 90% target sparsity):")
    for k, v in results["ema_ablation"].items():
        print(f"   - {k:<18}: {v['val_acc']*100:.2f}% val acc")

    print("\n4. Sparsity Schedule (Taylor at 90% target sparsity):")
    for k, v in results["schedule_ablation"].items():
        print(f"   - {k.capitalize():<12}: {v['val_acc']*100:.2f}% val acc")

    print("\n5. RigL Dynamic Regrowth vs Static (at 90% and 95% target sparsity):")
    for sp_key, modes in results["regrowth_ablation"].items():
        print(f"   - Target Sparsity {sp_key}:")
        for mode, v in modes.items():
            print(f"       * {mode:<16}: {v['val_acc']*100:.2f}% val acc")
    print("=" * 80)
