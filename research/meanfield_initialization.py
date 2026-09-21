"""Mean-field, Gaussian initialization utilities.

The initializer targets the *thresholded* edge-selection probability while
keeping trainable parameters Gaussian.  No Bernoulli sampling is used.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def target_selected_probability(fan_in: int) -> float:
    """Return s such that (1-s/2)**fan_in == 1/2 (stably)."""
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    return float(-2.0 * math.expm1(-math.log(2.0) / fan_in))


def normal_quantile_probability(p: float, *, dtype=torch.float64) -> torch.Tensor:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    return normal.icdf(torch.tensor(p, dtype=dtype))


def meanfield_gaussian_metadata(fan_in: int, sigma: float) -> dict[str, float]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    s = target_selected_probability(fan_in)
    z = float(normal_quantile_probability(s))
    return {
        "fan_in": float(fan_in),
        "s_target": s,
        "sigma": float(sigma),
        "mu": float(sigma * z),
        "z": z,
        "expected_selected_fan_in": float(fan_in * s),
    }


def meanfield_gaussian_edge_init_(
    raw_edge: torch.Tensor,
    fan_in: int | None = None,
    sigma: float = 4.0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Fill ``raw_edge`` with N(mu_m, sigma^2), targeting P(raw_edge>=0)=s.

    ``raw_edge`` remains an unconstrained Gaussian parameter; sigmoid is
    applied by the model during forward propagation.
    """
    m = int(fan_in if fan_in is not None else raw_edge.shape[-1])
    meta = meanfield_gaussian_metadata(m, sigma)
    with torch.no_grad():
        raw_edge.normal_(mean=meta["mu"], std=sigma, generator=generator)
    return meta


def bias_one_normal_init_(bias: torch.Tensor, std: float = 0.1, generator: torch.Generator | None = None) -> None:
    """Initialize raw bias near one for the existing leaky-clamp mapping."""
    with torch.no_grad():
        bias.normal_(mean=1.0, std=std, generator=generator)


def gaussian_bias_init_(
    bias: torch.Tensor,
    mean: float,
    std: float,
    generator: torch.Generator | None = None,
) -> None:
    """Fill a bias tensor from ``Normal(mean, std**2)`` using explicit RNG."""
    with torch.no_grad():
        z = torch.empty_like(bias).normal_(mean=0.0, std=1.0, generator=generator)
        bias.copy_(mean + std * z)
