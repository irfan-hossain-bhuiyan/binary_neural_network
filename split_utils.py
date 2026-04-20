import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

def split_dataset(
    x:           Tensor,
    y:           Tensor,
    train_ratio: float = 0.8,
    shuffle:     bool  = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split tensors into (x_train, y_train, x_test, y_test)."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample counts: x={x.shape[0]}, y={y.shape[0]}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=x.device)
        x, y = x[perm], y[perm]

    n_train = int(n * train_ratio)
    if n_train <= 0 or n_train >= n:
        raise ValueError("train_ratio produced an empty train or test split.")

    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]
