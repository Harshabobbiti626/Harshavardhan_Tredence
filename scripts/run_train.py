"""
Train dense baseline neural network with custom Adam optimizer
and report convergence metrics (loss and validation accuracy over epochs).
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.experiment import train_single_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQUA Engine - Dense Model Training")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 80)
    print("AQUA ENGINE -- DENSE BASELINE MODEL TRAINING (SCRATCH ADAM)")
    print("=" * 80)
    print("Architecture: Linear(784 -> 128) -> ReLU -> Linear(128 -> 10)")
    print("Initialization: Kaiming / He Normal (Var = 2 / in_features) to preserve variance")
    print(f"Optimizer: Adam (lr={args.lr}, beta1=0.9, beta2=0.999, eps=1e-8)")
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, Seed={args.seed}")
    print("-" * 80)

    results = train_single_model(
        target_sparsity=0.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE: Final Validation Accuracy = {results['final_val_acc']*100:.2f}%")
    print("=" * 80)

