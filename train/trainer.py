"""
Mini-batch training loop, metrics logger, and validation engine.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from engine.tensor import Tensor
from engine.ops import softmax_cross_entropy
from nn.layers import Module
from nn.optim import Optimizer
from prune.mask import Pruner
from train.dataset import DataLoader


class Trainer:
    """
    Mini-batched trainer with integrated pruning hooks and metrics logging.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        pruner: Optional[Pruner] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.pruner = pruner
        self.global_step = 0

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Compute average loss and classification accuracy over dataset."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in loader:
            logits = self.model(x_batch)
            loss = softmax_cross_entropy(logits, y_batch)

            total_loss += float(loss.data) * len(y_batch)
            preds = np.argmax(logits.data, axis=-1)
            total_correct += int(np.sum(preds == y_batch))
            total_samples += len(y_batch)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = float(total_correct / max(total_samples, 1))
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "sparsity": [],
        }

        if verbose:
            print("=" * 78)
            print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Sparsity':^10}")
            print("-" * 78)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            t0 = time.time()

            for x_batch, y_batch in train_loader:
                self.global_step += 1

                # 1. Zero gradients
                self.optimizer.zero_grad()

                # 2. Forward pass
                logits = self.model(x_batch)
                loss = softmax_cross_entropy(logits, y_batch)

                # 3. Backward pass
                loss.backward()

                # 4. Pruner hook: register gradient for importance criteria
                if self.pruner is not None:
                    self.pruner.on_step_backward()

                # 5. Optimizer step (with masked-weight isolation)
                self.optimizer.step()

                # 6. Pruner step: update masks if scheduled
                if self.pruner is not None:
                    self.pruner.step(self.global_step)

                # Track metrics
                batch_size = len(y_batch)
                epoch_loss += float(loss.data) * batch_size
                preds = np.argmax(logits.data, axis=-1)
                epoch_correct += int(np.sum(preds == y_batch))
                epoch_samples += batch_size

            train_loss = epoch_loss / epoch_samples
            train_acc = epoch_correct / epoch_samples
            val_loss, val_acc = self.evaluate(val_loader)
            current_sparsity = self.pruner.get_sparsity() if self.pruner is not None else 0.0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["sparsity"].append(current_sparsity)

            if verbose:
                print(
                    f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc*100:^9.2f}% | "
                    f"{val_loss:^10.4f} | {val_acc*100:^9.2f}% | {current_sparsity*100:^9.1f}%"
                )

        if verbose:
            print("=" * 78)

        return history
