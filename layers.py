"""Differentiable OR/NOR binary gate layers for the binary neural network."""

from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from prelude import leaky_clamp


def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise continuous XOR: a + b - 2ab."""
    return a + b - 2 * a * b


class OrNorGateLayer(nn.Module):
    """Differentiable OR/NOR layer with soft (softmax) or hard (argmax) output selection.

    Each output feature computes a weighted OR/NOR over the input features.  Weights
    and biases are constrained to [0, 1] via ``leaky_clamp``.  The continuous XOR of
    the input and bias is weighted, then either the max value is returned (hard) or
    a softmax-weighted expectation is returned (soft).

    Args:
        in_features: number of input features.
        out_features: number of output features.
        max_threshold: minimum truth value required for a single active input when
            using softmax mode.  Kept for API compatibility but no longer used to
            derive a temperature floor.
        temperature: shared or per-layer temperature.  A ``float`` creates a new
            parameter/buffer owned by this layer.  A ``torch.Tensor`` (parameter or
            buffer) is used directly, enabling temperature sharing across layers.
        learnable_tau: if True, ``temperature`` is an ``nn.Parameter``; otherwise it
            is a buffer.  Ignored when ``temperature`` is already a ``torch.Tensor``.
        use_softmax: whether to use softmax-weighted expectation (True) or hard max
            selection (False).
        weight_initialization: initializer for ``self.weight``.
        bias_initialization: initializer for ``self.bias``.
        grad_scalar: if True, scale gradients flowing through the softmax output by
            ``1 / (max_softmax_prob + 0.2)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float | nn.Parameter | torch.Tensor = 1.0,
        learnable_tau: bool = True,
        use_softmax: bool = False,
        weight_initialization: Callable[..., Any] = nn.init.normal_,
        bias_initialization: Callable[..., Any] = lambda x: nn.init.normal_(x, mean=0.5),
        grad_scalar: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax
        self.grad_scalar = grad_scalar

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features, in_features))
        weight_initialization(self.weight)
        bias_initialization(self.bias)

        self._last_input: torch.Tensor | None = None
        self._input_capture_hook: torch.utils.hooks.RemovableHandle | None = None

        if isinstance(temperature, nn.Parameter):
            self.temperature = temperature
        elif isinstance(temperature, torch.Tensor):
            self.register_buffer("temperature", temperature)
        elif learnable_tau:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)))

    @property
    def tau(self) -> torch.Tensor:
        """Inverse temperature used internally by the softmax (``1 / temperature``)."""
        return 1.0 / self.temperature.clamp_min(1e-6)

    def _capture_input_hook(
        self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        """Forward-pre hook that stores the most recent input."""
        self._last_input = inputs[0].detach()

    def enable_input_capture(self) -> None:
        """Register the forward-pre hook that captures inputs for regularization."""
        if self._input_capture_hook is None:
            self._input_capture_hook = self.register_forward_pre_hook(self._capture_input_hook)

    def disable_input_capture(self) -> None:
        """Remove the input-capture forward-pre hook, if registered."""
        if self._input_capture_hook is not None:
            self._input_capture_hook.remove()
            self._input_capture_hook = None
            self._last_input = None

    def actual_weight(self) -> torch.Tensor:
        """Return the weight constrained to [0, 1] with leaky gradients."""
        return cast(torch.Tensor, leaky_clamp(self.weight, 0, 1, 0.1))

    def actual_bias(self) -> torch.Tensor:
        """Return the bias constrained to [0, 1] with leaky gradients."""
        return cast(torch.Tensor, leaky_clamp(self.bias, 0, 1, 0.1))

    def discretize(self, threshold: float) -> None:
        """Snap raw weights and biases to 0 or 1 based on ``threshold``."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        with torch.no_grad():
            discrete_w = torch.where(
                self.weight < threshold,
                torch.clamp_max(self.weight, 0),
                torch.clamp_min(self.weight, 1),
            )
            self.weight.copy_(discrete_w)

            discrete_b = torch.where(
                self.bias < threshold,
                torch.clamp_max(self.bias, 0),
                torch.clamp_min(self.bias, 1),
            )
            self.bias.copy_(discrete_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_weight = self.actual_weight()
        actual_bias = self.actual_bias()

        x_exp = x.unsqueeze(1)
        b_exp = actual_bias.unsqueeze(0)
        # Continuous XOR between input and bias.
        x_xor_b = x_exp + b_exp - 2.0 * x_exp * b_exp

        # Weighted contribution per input.
        z = x_xor_b * actual_weight.unsqueeze(0)

        if self.use_softmax:
            z_scaled = self.tau * z
            p = F.softmax(z_scaled, dim=-1)
            s = (p * z).sum(dim=-1)

            if self.grad_scalar and s.requires_grad:
                max_p_s = p.max(dim=-1).values.detach()
                s.register_hook(lambda grad: grad / (max_p_s + 2e-1))
        else:
            s = z.max(dim=-1).values

        return s

    def to_hardmax(self) -> None:
        """Switch this layer to hard max selection."""
        self.use_softmax = False

    def batch_variance_cost(self, x: torch.Tensor) -> torch.Tensor:
        """Maximize the batch-wise variance of this layer's activations.

        Computes the variance of the layer output across the batch dimension and
        returns ``0.5**2 - variance`` so that minimizing this cost maximizes
        variance.  The minimum achievable value is ``-0.25`` when variance is
        maximal (0.5**2 for binary outputs).
        """
        with torch.enable_grad():
            out = self.forward(x)
        variance = out.var(dim=0, unbiased=False).mean()
        return (0.5**2) - variance
