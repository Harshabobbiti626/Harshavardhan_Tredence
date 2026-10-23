"""
Pruner manager: applies criteria, manages masks, executes schedules, and handles regrowth.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
from engine.tensor import Tensor
from nn.layers import Module, Linear
from nn.optim import Optimizer
from prune.criteria import ImportanceCriterion
from prune.scheduler import PruningScheduler


class Pruner:
    """
    Manages pruning and dynamic topology adaptation across model layers.
    """

    def __init__(
        self,
        model: Module,
        criterion: ImportanceCriterion,
        scheduler: PruningScheduler,
        optimizer: Optional[Optimizer] = None,
        allow_regrowth: bool = False,
        regrowth_ratio: float = 0.2,
    ):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.allow_regrowth = allow_regrowth
        self.regrowth_ratio = regrowth_ratio

        # Identify all prunable weight tensors (Linear layer weights)
        self.prunable_params: List[Tuple[str, Tensor]] = []
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim == 2:
                self.prunable_params.append((name, param))
                if param.mask is None:
                    # Initially fully dense (all ones)
                    param.set_mask(np.ones(param.shape, dtype=bool))

    def on_step_backward(self) -> None:
        """Call immediately after backward pass to register gradients for importance criteria."""
        for _, param in self.prunable_params:
            self.criterion.update_gradient(param)

    def step(self, current_step: int) -> Dict[str, float]:
        """
        Check schedule and update masks if required.
        Returns a dictionary of current sparsity metrics.
        """
        if self.scheduler.should_prune(current_step):
            target_sparsity = self.scheduler.get_sparsity(current_step)
            self._update_masks(target_sparsity)

        return {
            "sparsity": self.get_sparsity(),
            "active_params": self.get_active_params(),
            "total_params": self.get_total_params(),
        }

    def _update_masks(self, target_sparsity: float) -> None:
        """Update binary masks for all prunable parameters."""
        for name, param in self.prunable_params:
            if target_sparsity <= 0.0:
                continue

            current_mask = param.mask if param.mask is not None else np.ones(param.shape, dtype=bool)
            scores = self.criterion.compute_importance(param)

            # Compute target number of zero weights
            k = int(np.round(param.size * target_sparsity))
            if k >= param.size:
                k = param.size - 1

            flat_scores = scores.flatten()

            if not self.allow_regrowth or np.all(current_mask):
                # Standard progressive pruning:
                # Rank all weights, prune bottom target_sparsity fraction
                if k > 0:
                    prune_indices = np.argpartition(flat_scores, k)[:k]
                    new_mask = np.ones(param.size, dtype=bool)
                    new_mask[prune_indices] = False
                    new_mask = new_mask.reshape(param.shape)
                else:
                    new_mask = np.ones(param.shape, dtype=bool)

                param.set_mask(new_mask)

            else:
                # Dynamic Regrowth (RigL / Dynamic Sparse Training):
                # 1. Prune bottom regrowth_ratio of currently active weights
                active_indices = np.where(current_mask)
                num_active = len(active_indices[0])
                num_to_swap = int(np.round(num_active * self.regrowth_ratio))

                if num_to_swap > 0:
                    active_scores = scores[active_indices]
                    prune_order = np.argsort(active_scores)[:num_to_swap]
                    prune_rows = active_indices[0][prune_order]
                    prune_cols = active_indices[1][prune_order]

                    # 2. Revive top regrowth_ratio of dormant weights based on gradient magnitude
                    dormant_indices = np.where(~current_mask)
                    num_dormant = len(dormant_indices[0])
                    num_to_revive = min(num_to_swap, num_dormant)

                    if num_to_revive > 0:
                        grad_scores = np.abs(param.grad) if param.grad is not None else np.abs(np.random.randn(*param.shape))
                        dormant_scores = grad_scores[dormant_indices]
                        revive_order = np.argsort(dormant_scores)[-num_to_revive:]
                        revive_rows = dormant_indices[0][revive_order]
                        revive_cols = dormant_indices[1][revive_order]

                        new_mask = current_mask.copy()
                        new_mask[prune_rows, prune_cols] = False
                        new_mask[revive_rows, revive_cols] = True

                        # Reset optimizer moments for revived weights to avoid momentum contamination
                        if self.optimizer is not None:
                            pid = id(param)
                            if pid in self.optimizer.state:
                                state = self.optimizer.state[pid]
                                if "m" in state:
                                    state["m"][revive_rows, revive_cols] = 0.0
                                if "v" in state:
                                    state["v"][revive_rows, revive_cols] = 0.0

                        param.set_mask(new_mask)

    def get_sparsity(self) -> float:
        total = self.get_total_params()
        active = self.get_active_params()
        if total == 0:
            return 0.0
        return float(1.0 - (active / total))

    def get_active_params(self) -> int:
        active = 0
        for _, p in self.prunable_params:
            if p.mask is not None:
                active += int(np.sum(p.mask))
            else:
                active += p.size
        return active

    def get_total_params(self) -> int:
        return sum(p.size for _, p in self.prunable_params)
