"""
Verification of masked-weight invariants:
Masked weights must contribute zero to forward pass, receive well-defined zero gradients,
and isolate optimizer state (Adam m and v moments) from momentum leakage and zombie drift.
"""

import pytest
import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam, SGD
from engine.ops import softmax_cross_entropy


class TestMaskedWeightCorrectness:

    def test_masked_forward_zero_contribution(self):
        """Verify that masked weights contribute identically zero to the forward pass."""
        np.random.seed(42)
        layer = Linear(in_features=4, out_features=3)

        # Create binary mask: zero out column 1 and specific weights
        mask = np.ones((4, 3), dtype=bool)
        mask[0, 0] = False
        mask[2, 1] = False
        mask[3, 2] = False
        layer.set_mask(mask)

        # Verify weights are strictly zero where mask is False
        assert np.all(layer.weight.data[~mask] == 0.0)

        # Forward pass with non-zero inputs
        X = Tensor(np.ones((2, 4), dtype=np.float32), requires_grad=False)
        out1 = layer(X)

        # Even if someone attempts to manually inject non-zero data into masked spots,
        # set_mask zeroes it out
        layer.weight.data[~mask] = 999.0
        layer.set_mask(mask)
        out2 = layer(X)

        np.testing.assert_allclose(out1.data, out2.data, atol=1e-7)

    def test_masked_weight_receives_zero_gradient(self):
        """Verify that backward pass computes exact 0.0 gradient for all masked connections."""
        np.random.seed(42)
        layer = Linear(in_features=4, out_features=3)
        mask = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=bool)
        layer.set_mask(mask)

        X = Tensor(np.random.randn(5, 4).astype(np.float32), requires_grad=False)
        targets = np.array([0, 1, 2, 0, 1])

        logits = layer(X)
        loss = softmax_cross_entropy(logits, targets)
        loss.backward()

        assert layer.weight.grad is not None
        # Gradients for masked weights MUST be identically 0.0
        assert np.all(layer.weight.grad[~mask] == 0.0)
        # Active weights should have non-zero gradients
        assert np.any(layer.weight.grad[mask] != 0.0)

    def test_adam_optimizer_moment_isolation(self):
        """
        Critical Correctness Test:
        Prove that Adam does NOT leak momentum into masked weights and does NOT
        cause parameter drift over multiple optimization steps.
        """
        np.random.seed(42)
        layer = Linear(in_features=5, out_features=4)
        mask = np.ones((5, 4), dtype=bool)
        # Prune half the weights
        mask[::2, :] = False
        layer.set_mask(mask)

        optimizer = Adam(layer.parameters(), lr=1e-2, weight_decay=1e-4)

        X = Tensor(np.random.randn(8, 5).astype(np.float32), requires_grad=False)
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        # Run 25 optimization steps
        for step in range(25):
            optimizer.zero_grad()
            logits = layer(X)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            # 1. Weights at masked locations MUST remain strictly 0.0
            np.testing.assert_allclose(
                layer.weight.data[~mask], 0.0, atol=1e-8,
                err_msg=f"Masked weights drifted at step {step}!"
            )

            # 2. Adam moments (m and v) at masked locations MUST remain 0.0
            pid = id(layer.weight)
            state = optimizer.state[pid]
            np.testing.assert_allclose(
                state["m"][~mask], 0.0, atol=1e-8,
                err_msg=f"Adam first moment leaked into masked weights at step {step}!"
            )
            np.testing.assert_allclose(
                state["v"][~mask], 0.0, atol=1e-8,
                err_msg=f"Adam second moment leaked into masked weights at step {step}!"
            )

    def test_prune_and_revival_stability(self):
        """
        Verify that pruning a connection and later reviving it does not corrupt
        the optimizer state or cause NaN loss spikes.
        """
        np.random.seed(42)
        layer = Linear(in_features=6, out_features=4)
        optimizer = Adam(layer.parameters(), lr=1e-2)

        X = Tensor(np.random.randn(10, 6).astype(np.float32), requires_grad=False)
        targets = np.random.randint(0, 4, size=(10,))

        # Phase 1: Train dense for 5 steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = softmax_cross_entropy(layer(X), targets)
            loss.backward()
            optimizer.step()

        # Phase 2: Prune connections (mask out row 0 and 1)
        mask = np.ones((6, 4), dtype=bool)
        mask[0:2, :] = False
        layer.set_mask(mask)

        for _ in range(5):
            optimizer.zero_grad()
            loss = softmax_cross_entropy(layer(X), targets)
            loss.backward()
            optimizer.step()
            assert np.all(layer.weight.data[0:2, :] == 0.0)

        # Phase 3: Revive row 0
        mask[0, :] = True
        layer.set_mask(mask)

        # Train further: verify stable forward and backward pass
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = softmax_cross_entropy(layer(X), targets)
            loss_val = float(loss.data)
            assert not np.isnan(loss_val), "Loss became NaN upon weight revival!"
            assert not np.isinf(loss_val), "Loss became Inf upon weight revival!"
            losses.append(loss_val)
            loss.backward()
            optimizer.step()

        # Check that revived weights now participate cleanly in training
        assert np.any(layer.weight.data[0, :] != 0.0)
        assert np.all(layer.weight.data[1, :] == 0.0)  # Row 1 remains pruned


