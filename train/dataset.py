"""
Dataset loading and mini-batch generation for AQUA Platform.
NOTE: As permitted by challenge instructions, scikit-learn is used solely
for loading the standard dataset. All training, models, and autodiff are from scratch.
"""

from __future__ import annotations
from typing import Tuple, Generator, Optional
import os
import numpy as np
from engine.tensor import Tensor


def get_synthetic_dataset(
    num_samples: int = 5000,
    in_features: int = 784,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate deterministic synthetic 10-class classification data with clustered structure.
    Used as an offline-safe fallback or fast verification dataset.
    """
    rng = np.random.RandomState(seed)
    # Create cluster centroids in a 32-dim latent space, projected to 784 dims
    latent_dim = 32
    centroids = rng.randn(num_classes, latent_dim) * 2.0
    projection = rng.randn(latent_dim, in_features) / np.sqrt(latent_dim)

    y = rng.randint(0, num_classes, size=num_samples)
    latents = centroids[y] + rng.randn(num_samples, latent_dim) * 0.8
    X = np.dot(latents, projection)

    # Normalize to [0, 1] range
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    X = X.astype(np.float32)

    # 80/20 train/test split
    n_train = int(num_samples * 0.8)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def load_dataset(
    name: str = "mnist",
    cache_dir: Optional[str] = None,
    seed: int = 42,
    max_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load classification dataset (MNIST or synthetic fallback).
    Always anchors to the project data directory regardless of CWD.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if cache_dir is None:
        cache_dir = os.path.join(repo_root, "data")
    elif not os.path.isabs(cache_dir):
        # Prefer repo_root/data if it exists, else resolve from repo_root
        candidate = os.path.join(repo_root, cache_dir)
        if os.path.exists(candidate) or not os.path.exists(cache_dir):
            cache_dir = candidate

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{name}_{max_samples}_{seed}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


    try:
        from sklearn.datasets import fetch_openml
        print(f"Fetching {name} from OpenML...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)

        # Subsample to max_samples for fast and reproducible training
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(X))[:max_samples]
        X = X[indices]
        y = y[indices]

        n_train = int(len(X) * 0.8)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"Notice: OpenML fetch failed or offline ({e}). Using deterministic synthetic benchmark dataset.")
        X_train, y_train, X_test, y_test = get_synthetic_dataset(
            num_samples=max_samples,
            in_features=784,
            num_classes=10,
            seed=seed,
        )
        np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return X_train, y_train, X_test, y_test


class DataLoader:
    """
    Mini-batch generator for NumPy tensors.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.num_samples = len(X)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Generator[Tuple[Tensor, np.ndarray], None, None]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            indices = self.rng.permutation(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_x = self.X[batch_idx]
            batch_y = self.y[batch_idx]
            # Wrap X in Tensor (requires_grad=False)
            yield Tensor(batch_x, requires_grad=False), batch_y
