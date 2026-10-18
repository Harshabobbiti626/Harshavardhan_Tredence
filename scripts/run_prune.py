"""
Single-command runner for Part 3: Self-pruning training to 90% target sparsity
using derived first-order Taylor saliency criterion and cubic schedule.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.experiment import train_single_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQUA Platform - Part 3 Self-Pruning Network")
    parser.add_argument("--sparsity", type=float, default=0.90, help="Target parameter sparsity (default: 0.90)")
    parser.add_argument("--criterion", type=str, default="saliency", choices=["saliency", "magnitude", "movement"], help="Pruning criterion (default: saliency)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--regrowth", action="store_true", help="Enable dynamic regrowth (RigL)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"AQUA PLATFORM - PART 3 SELF-PRUNING NETWORK ({args.sparsity*100:.1f}% TARGET SPARSITY)")
    print("=" * 80)
    print(f"Criterion: {args.criterion.capitalize()} Pruning with Gradient EMA")
    print(f"Schedule: Cubic Sparsity Ramp (0% -> {args.sparsity*100:.1f}% across 75% steps, 25% fine-tuning)")
    print("Masking: Binary mask M in {0, 1} with Adam moment sanitization")
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, Seed={args.seed}, Regrowth={args.regrowth}")
    print("-" * 80)

    results = train_single_model(
        target_sparsity=args.sparsity,
        criterion_type=args.criterion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        allow_regrowth=args.regrowth,
        verbose=True,
    )


    bench = results["benchmark"]
    print("\n" + "=" * 80)
    print("PART 3 SELF-PRUNING RUN COMPLETE")
    print(f"Target Sparsity:             {results['target_sparsity']*100:.1f}%")
    print(f"Achieved Sparsity:           {results['achieved_sparsity']*100:.2f}%")
    print(f"Final Validation Accuracy:   {results['final_val_acc']*100:.2f}%")
    print(f"Active Parameters:           {bench['active_params']} / {bench['total_params']}")
    print(f"Dense Theoretical FLOPs:     {bench['dense_flops']:.0f}")
    print(f"CSR Sparse Theoretical FLOPs:{bench['sparse_flops']:.0f}")
    print(f"Theoretical Compute Speedup: {bench['theoretical_speedup']:.2f}x")
    print("=" * 80)
