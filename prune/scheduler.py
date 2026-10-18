"""
Pruning schedules for controlling sparsity progression during training.
"""

from __future__ import annotations


class PruningScheduler:
    """Base pruning schedule interface."""

    def get_sparsity(self, step: int) -> float:
        raise NotImplementedError

    def should_prune(self, step: int) -> bool:
        raise NotImplementedError


class ConstantSparsityScheduler(PruningScheduler):
    """Instant/constant target sparsity."""

    def __init__(self, target_sparsity: float, prune_step: int = 100):
        self.target_sparsity = target_sparsity
        self.prune_step = prune_step

    def get_sparsity(self, step: int) -> float:
        if step >= self.prune_step:
            return self.target_sparsity
        return 0.0

    def should_prune(self, step: int) -> bool:
        return step == self.prune_step


class CubicSparsityScheduler(PruningScheduler):
    """
    Gradual Polynomial / Cubic Sparsity Ramp (Zhu & Gupta, 2017):
      s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / T_prune)^3

    Phases:
      1. Warm-up (t < t_0): Sparsity = s_i (0%). Model learns dense representations.
      2. Pruning Ramp (t_0 <= t <= t_0 + T_prune): Sparsity ramps up cubically every prune_freq steps.
      3. Fine-tuning (t > t_0 + T_prune): Sparsity fixed at s_f. Network settles into sparse topology.
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.90,
        start_step: int = 100,
        end_step: int = 1000,
        prune_freq: int = 50,
    ):
        assert 0.0 <= initial_sparsity < 1.0
        assert 0.0 <= final_sparsity < 1.0
        assert end_step > start_step

        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.prune_freq = prune_freq

    def get_sparsity(self, step: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.final_sparsity

        progress = float(step - self.start_step) / float(self.end_step - self.start_step)
        # Cubic curve: rapid pruning initially, gentle pruning as target approaches
        factor = (1.0 - progress) ** 3
        sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * factor
        return float(sparsity)

    def should_prune(self, step: int) -> bool:
        if step < self.start_step:
            return False
        if step > self.end_step:
            return False
        return (step - self.start_step) % self.prune_freq == 0
