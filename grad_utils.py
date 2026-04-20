import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from rich.table import Table
from core_types import LayerGradStats, VanishingGradReport
from trainer_utils import CONSOLE

class LeakyClamp(torch.autograd.Function):
    """Clamp with a small gradient leak outside the clamped range."""

    @staticmethod
    def forward(ctx, x: Tensor, min_val: float, max_val: float, leak: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        ctx.min_val, ctx.max_val, ctx.leak = min_val, max_val, leak
        return torch.clamp(x, min=float(min_val), max=float(max_val))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        in_bounds  = (x >= ctx.min_val) & (x <= ctx.max_val)
        grad_input = grad_output.clone()
        grad_input[~in_bounds] *= ctx.leak
        return grad_input, None, None, None


def leaky_clamp(x: Tensor, min_val: float, max_val: float, leak: float = 0.01) -> Tensor:
    return LeakyClamp.apply(x, min_val, max_val, leak)


def collect_grad_stats(model: nn.Module) -> Dict[str, LayerGradStats]:
    """
    Collect per-layer gradient statistics for every parameter that currently
    has a gradient.
    """
    stats: Dict[str, LayerGradStats] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        n = g.numel()
        stats[name] = LayerGradStats(
            mean_abs        = g.abs().mean(),
            norm_normalized = g.norm() / (n ** 0.5),
            max_abs         = g.abs().max(),
        )
    return stats


def detect_vanishing_gradients(
    model:           nn.Module,
    threshold:       float = 1e-6,
    ratio_threshold: float = 0.01,
) -> VanishingGradReport:
    """
    Inspect current gradients on model and return a structured report.
    """
    per_layer: Dict[str, float] = {
        name: param.grad.abs().mean().item()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    if not per_layer:
        return VanishingGradReport(
            is_vanishing=False, first_last_ratio=1.0, frozen_layers=[],
            per_layer={}, min_mean_abs=0.0, max_mean_abs=0.0,
        )

    vals         = list(per_layer.values())
    ratio        = vals[0] / (vals[-1] + 1e-12)
    frozen       = [name for name, v in per_layer.items() if v < threshold]
    is_vanishing = ratio < ratio_threshold or len(frozen) > 0

    return VanishingGradReport(
        is_vanishing     = is_vanishing,
        first_last_ratio = ratio,
        frozen_layers    = frozen,
        per_layer        = per_layer,
        min_mean_abs     = min(vals),
        max_mean_abs     = max(vals),
    )


def _print_grad_table(avg_stats: Dict[str, LayerGradStats]) -> None:
    """Print a rich table of per-layer gradient statistics."""
    table = Table(title="Gradient Stats (epoch avg)", show_header=True, header_style="bold cyan")
    table.add_column("parameter",       style="white",   overflow="fold")
    table.add_column("mean_abs",        justify="right", style="magenta")
    table.add_column("norm_normalized", justify="right", style="yellow")
    table.add_column("max_abs",         justify="right", style="red")

    if not avg_stats:
        table.add_row("(no gradients)", "-", "-", "-")
    else:
        for name, s in sorted(avg_stats.items()):
            table.add_row(name, f"{s.mean_abs.item():.4e}", f"{s.norm_normalized.item():.4e}", f"{s.max_abs.item():.4e}")
    CONSOLE.print(table)


def _print_vanishing_warning(report: VanishingGradReport, epoch: int) -> None:
    CONSOLE.print(f"[bold red]Vanishing gradient detected at epoch {epoch}![/bold red]")
    CONSOLE.print(f"   first/last ratio = [red]{report.first_last_ratio:.2e}[/red]")
    if report.frozen_layers:
        CONSOLE.print(f"   frozen layers    = [red]{report.frozen_layers}[/red]")
    CONSOLE.print(f"   min mean_abs = {report.min_mean_abs:.2e}  |  max = {report.max_mean_abs:.2e}")
