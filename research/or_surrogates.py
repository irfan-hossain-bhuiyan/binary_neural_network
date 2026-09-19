"""Differentiable OR-like aggregation operators.

All operators consume contributions ``v`` in [0, 1] with shape
``(..., fan_in)`` and reduce the final dimension.  The implementation is
kept independent from the logic network so that operator semantics can be
tested without changing the historical models.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import Tensor, nn


def _zero_safe_ratio(num: Tensor, den: Tensor) -> Tensor:
    out = torch.zeros_like(num)
    return torch.where(den > 0, num / den, out)


class OrSurrogate(nn.Module):
    """Common interface for OR-like reductions."""

    name = "or_surrogate"

    def forward(self, v: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class HardMax(OrSurrogate):
    name = "hardmax"

    def forward(self, v: Tensor) -> Tensor:
        return v.max(dim=-1).values


class LehmerMean(OrSurrogate):
    def __init__(self, p: float = 1.0):
        super().__init__()
        if p <= 0:
            raise ValueError("Lehmer p must be positive")
        self.p = float(p)
        self.name = f"lehmer_p{self.p:g}"

    def forward(self, v: Tensor) -> Tensor:
        powers = v.pow(self.p)
        return _zero_safe_ratio((powers * v).sum(dim=-1), powers.sum(dim=-1))


class OddsWeightedMean(OrSurrogate):
    name = "odds"

    def forward(self, v: Tensor) -> Tensor:
        # At v=1 the mathematical odds are infinite and the aggregate is
        # exactly one.  Work on the interior separately for finite gradients.
        endpoint = (v >= 1).any(dim=-1)
        interior = v.clamp(min=0, max=1 - torch.finfo(v.dtype).eps)
        odds = interior / (1 - interior)
        out = _zero_safe_ratio((odds * interior).sum(dim=-1), odds.sum(dim=-1))
        return torch.where(endpoint, torch.ones_like(out), out)


class LogHazardWeightedMean(OrSurrogate):
    name = "log_hazard"

    def forward(self, v: Tensor) -> Tensor:
        endpoint = (v >= 1).any(dim=-1)
        interior = v.clamp(min=0, max=1 - torch.finfo(v.dtype).eps)
        hazard = -torch.log1p(-interior)
        out = _zero_safe_ratio((hazard * interior).sum(dim=-1), hazard.sum(dim=-1))
        return torch.where(endpoint, torch.ones_like(out), out)


class ProbabilisticOr(OrSurrogate):
    name = "probabilistic_or"

    def forward(self, v: Tensor) -> Tensor:
        # -expm1 is accurate when the product is close to zero.
        interior = v.clamp(min=0, max=1)
        log_product = torch.log1p(-interior).sum(dim=-1)
        return -torch.expm1(log_product)


def _einstein_pair(a: Tensor, b: Tensor) -> Tensor:
    return (a + b) / (1 + a * b)


class EinsteinOr(OrSurrogate):
    name = "einstein_or"

    def forward(self, v: Tensor) -> Tensor:
        out = v[..., 0]
        for i in range(1, v.shape[-1]):
            out = _einstein_pair(out, v[..., i])
        return out


def _hamacher_pair(a: Tensor, b: Tensor, lam: float) -> Tensor:
    numerator = a + b - (2 - lam) * a * b
    denominator = 1 - (1 - lam) * a * b
    out = _zero_safe_ratio(numerator, denominator)
    # The lambda=0 formula has a 0/0 representation at (1, 1), whose
    # continuous t-conorm limit is one.
    return torch.where((a >= 1) | (b >= 1), torch.ones_like(out), out)


class HamacherOr(OrSurrogate):
    def __init__(self, lam: float = 0.0):
        super().__init__()
        if lam < 0:
            raise ValueError("Hamacher lambda must be nonnegative")
        self.lam = float(lam)
        self.name = f"hamacher_l{self.lam:g}"

    def forward(self, v: Tensor) -> Tensor:
        out = v[..., 0]
        for i in range(1, v.shape[-1]):
            out = _hamacher_pair(out, v[..., i], self.lam)
        return out


class LukasiewiczOr(OrSurrogate):
    name = "lukasiewicz_bounded_sum"

    def forward(self, v: Tensor) -> Tensor:
        return v.sum(dim=-1).clamp(max=1)


class SoftmaxWeightedValue(OrSurrogate):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.name = f"softmax_value_a{self.alpha:g}"

    def forward(self, v: Tensor) -> Tensor:
        return (torch.softmax(self.alpha * v, dim=-1) * v).sum(dim=-1)


def candidate_factories() -> Dict[str, Callable[[], OrSurrogate]]:
    return {
        "hardmax": HardMax,
        "lehmer_p0.5": lambda: LehmerMean(0.5),
        "lehmer_p1": lambda: LehmerMean(1),
        "lehmer_p2": lambda: LehmerMean(2),
        "lehmer_p4": lambda: LehmerMean(4),
        "odds": OddsWeightedMean,
        "log_hazard": LogHazardWeightedMean,
        "probabilistic_or": ProbabilisticOr,
        "einstein_or": EinsteinOr,
        "hamacher_l0": lambda: HamacherOr(0),
        "hamacher_l0.5": lambda: HamacherOr(0.5),
        "hamacher_l1": lambda: HamacherOr(1),
        "lukasiewicz_bounded_sum": LukasiewiczOr,
        "softmax_value_a1": lambda: SoftmaxWeightedValue(1),
        "softmax_value_a4": lambda: SoftmaxWeightedValue(4),
        "softmax_value_a16": lambda: SoftmaxWeightedValue(16),
    }


def get_operator(name: str) -> OrSurrogate:
    try:
        return candidate_factories()[name]()
    except KeyError as exc:
        raise KeyError(f"unknown OR surrogate: {name}") from exc
