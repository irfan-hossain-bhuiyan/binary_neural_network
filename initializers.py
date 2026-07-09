"""Lightweight PyTorch initializers used by the binary neural network."""

import torch
import torch.nn as nn


class NormalInitWrapper:
    """Wrap ``torch.nn.init.normal_`` with a configurable mean."""

    def __init__(self, mean: float):
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=self.mean)


class ConstantInitWrapper:
    """Initialize a tensor to a constant value."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.ones_(tensor) * self.value
