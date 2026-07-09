"""Regularization factories for the binary neural network."""

from typing import Any, cast

import torch
from torch import Tensor

from layers import OrNorGateLayer
from stopping_utils import PlateauTracker


def regularization_factory(
    l1_lambda: float = 1e-1,
    disc_lambda: float = 1e-1,
    tau_lambda: float = 1e-1,
    patience: int = 10,
    min_err: float = 0.01,
):
    """Build a regularizer that toggles on/off when training plateaus.

    The regularizer encourages:
      * High tau (via ``exp(-tau)``).
      * Binary weights/biases near 0 or 1 (discretization error).
      * Positive weights (L1 on ``w.relu()``).
    """
    is_regularization = True
    plateau_check = PlateauTracker(patience, min_err)

    def regularization(module: Any, epoch: int, avg_error: float) -> Tensor:
        nonlocal is_regularization
        if plateau_check.update(epoch, avg_error):
            is_regularization = not is_regularization
            print(f"Plateau found, regularization toggled. active={is_regularization}")

        if not is_regularization:
            return torch.tensor(0.0)

        reg = torch.tensor(0.0, device=next(module.parameters()).device)
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            w = layer.weight
            b = layer.bias

            l1_error = w.relu().mean()
            disc_error_w = (0.5 - (w - 0.5).abs()).relu().mean()
            disc_error_b = (0.5 - (b - 0.5).abs()).relu().mean()
            disc_error = disc_error_w + disc_error_b
            tau_err = torch.exp(-layer.tau)

            reg += (disc_lambda * disc_error) + (tau_lambda * tau_err) + (l1_lambda * l1_error)
        return reg

    return regularization


def regularization_factory2(
    disc_lambda: float = 1e-1,
    tau_lambda: float = 1e-1,
    patience: int = 15,
    min_err: float = 0.01,
    isolate_on_plateau: bool = True,
):
    """Alternative regularizer that toggles between two discretization losses.

    When active, it penalizes values in the uncertain middle region
    ``(0.5 - |w - 0.5|).relu()``.  When toggled off, it instead pulls values
    toward 0.5 from below via ``w.clamp(0.0, 0.5).mean()``.
    """
    reg_active = True
    min_err = min_err if isolate_on_plateau else 0.0
    plateau_check = PlateauTracker(patience, min_err)

    def regularization(module: Any, epoch: int, avg_error: float) -> Tensor:
        nonlocal reg_active
        if plateau_check.update(epoch, avg_error):
            reg_active = not reg_active
            print(f"Plateau found, regularization toggled. active={reg_active}")

        reg = torch.tensor(0.0, device=next(module.parameters()).device)
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            w = layer.weight
            b = layer.bias

            if reg_active:
                disc_error_w = (0.5 - (w - 0.5).abs()).relu().mean()
                disc_error_b = (0.5 - (b - 0.5).abs()).relu().mean()
            else:
                disc_error_w = w.clamp(0.0, 0.5).mean()
                disc_error_b = b.clamp(0.0, 0.5).mean()

            disc_error = disc_error_w + disc_error_b
            tau_err = torch.exp(-layer.tau)
            reg += (disc_lambda * disc_error) + (tau_lambda * tau_err)
        return reg

    return regularization
