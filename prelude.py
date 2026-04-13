"""
trainer.py — Modular PyTorch training library.

Modules
-------
  device        — Device resolution
  types         — Core dataclasses (TrainConfig, HistoryEntry, Checkpoint)
  stopping      — Stop-condition factories (early_stopping, stop_on_epoch)
  scheduling    — LR scheduler factories (model-aware)
  grad          — Gradient collection, vanishing detection, visualization
  data          — Dataset splitting
  checkpointing — Save / load / merge checkpoints
  visualization — Loss curves, weight distributions, gradient animations
  testing       — Accuracy evaluation
  Trainer       — Main training orchestrator

Constraint convention
---------------------
  If your model defines an ``apply_constraints(self)`` method, the Trainer
  will call it automatically after every optimizer step — no wiring needed.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from rich.console import Console
from rich.table import Table


# ══════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════

def resolve_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE:   torch.device = resolve_device()
CONSOLE:  Console      = Console()


# ══════════════════════════════════════════════
# CORE DATACLASSES
# ══════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Snapshot of the hyperparameters used for a training run."""
    num_epochs:       int
    batch_size:       int
    optimizer_cls:    str
    optimizer_kwargs: Dict[str, Any]
    loss_fn:          str


@dataclass
class LayerGradStats:
    """
    Per-layer gradient statistics collected each epoch.

    mean_abs        — average absolute gradient value per element.
                      This is the primary vanishing-gradient signal used by
                      researchers: if it falls below ~1e-6 in early layers
                      while late layers remain high, gradients are vanishing.

    norm_normalized — L2 norm divided by sqrt(numel), making it comparable
                      across layers of different sizes.

    max_abs         — largest individual gradient magnitude; catches sparse
                      exploding gradients that mean_abs would average away.
    """
    mean_abs:        Tensor
    norm_normalized: Tensor
    max_abs:         Tensor


@dataclass
class HistoryEntry:
    """Metrics recorded at the end of one epoch."""
    epoch:              int
    avg_loss:           float
    avg_err:            float
    avg_regularization: float
    # Maps parameter name -> LayerGradStats averaged over all batches.
    grad_stats:         Dict[str, LayerGradStats]


@dataclass
class VanishingGradReport:
    """
    Result of detect_vanishing_gradients().

    is_vanishing      — True when the first/last mean_abs ratio drops below
                        ratio_threshold OR any layer's mean_abs drops below
                        threshold.  This is the actionable boolean flag.
    first_last_ratio  — mean_abs of the first parameter layer divided by that
                        of the last.  Healthy networks are close to 1.0;
                        values below 0.01 indicate severe vanishing.
    frozen_layers     — parameter names whose mean_abs is below threshold;
                        these layers are practically not learning.
    per_layer         — mean_abs value for every parameter with a gradient.
    min_mean_abs      — smallest mean_abs across all layers.
    max_mean_abs      — largest mean_abs across all layers.
    """
    is_vanishing:      bool
    first_last_ratio:  float
    frozen_layers:     List[str]
    per_layer:         Dict[str, float]
    min_mean_abs:      float
    max_mean_abs:      float


@dataclass
class Checkpoint:
    """Everything needed to resume or analyse a finished training run."""
    model:            nn.Module
    train_config:     TrainConfig
    training_history: List[HistoryEntry]

    def avg_losses(self) -> List[float]:
        return [e.avg_loss for e in self.training_history]

    def avg_errors(self) -> List[float]:
        return [e.avg_err for e in self.training_history]


# ══════════════════════════════════════════════
# STOPPING CONDITIONS
# ══════════════════════════════════════════════

@dataclass
class TrainerState:
    """Consolidated state tracking for training, including plateau detection."""
    epoch: int = 0
    avg_loss: float = float("inf")
    avg_error: float = float("inf")
    avg_regularization: float = 0.0

    # Plateau tracking properties
    best_loss: float = float("inf")
    best_error: float = float("inf")
    loss_no_improve: int = 0
    error_no_improve: int = 0
    
    # Thresholds & settings for plateau
    loss_min_delta: float = 1e-4
    error_min_delta: float = 1e-3
    patience: int = 10
    is_plateaued: bool = False

    def update(self, epoch: int, avg_loss: float, avg_error: float, avg_regularization: float):
        # If we plateaued LAST epoch, reset our tracking now before evaluating this epoch.
        if self.is_plateaued:
            self.best_loss = float("inf")
            self.loss_no_improve = 0
            self.best_error = float("inf")
            self.error_no_improve = 0
            self.is_plateaued = False

        self.epoch = epoch
        self.avg_loss = avg_loss
        self.avg_error = avg_error
        self.avg_regularization = avg_regularization
        
        # Track loss plateau
        if avg_loss < self.best_loss - self.loss_min_delta:
            self.best_loss = avg_loss
            self.loss_no_improve = 0
        else:
            self.loss_no_improve += 1
            
        # Track error plateau
        if avg_error < self.best_error - self.error_min_delta:
            self.best_error = avg_error
            self.error_no_improve = 0
        else:
            self.error_no_improve += 1
            
        if self.loss_no_improve >= self.patience or self.error_no_improve >= self.patience:
            self.is_plateaued = True



def early_stopping(max_epochs: int = 500) -> Callable[..., bool]:
    """
    Factory: plateau-based stopping.

    Stops when the state reports it has plateaued AND the epoch has reached (or exceeded) max_epochs.
    """
    def callback(epoch: int, state: "TrainerState") -> bool:
        return state.is_plateaued or epoch >= max_epochs

    return callback


def stop_on_epoch(max_epochs: int) -> Callable[..., bool]:
    """Factory: stops exactly after max_epochs epochs."""
    def callback(epoch: int) -> bool:
        return epoch >= max_epochs
class FnCallOnPlateau:
    """Custom scheduler that calls model.discretize() when plateauing, without changing LR."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,func:Callable[[nn.Module]]):
        self.model = model
        self.optimizer = optimizer
        self.func=func

    def step(self, state: "TrainerState"):
        if state.is_plateaued:
            CONSOLE.print(f"[bold yellow]Plateau reached (Epoch {state.epoch}): Discretizing model[/bold yellow]")
            self.func(self.model)
            # Reset the optimizer state since the model has changed
            from collections import defaultdict
            self.optimizer.state = defaultdict(dict)

def fn_call_on_plateau_scheduler(
    fn:Callable[[nn.Module]],
) -> Callable[[nn.Module, torch.optim.Optimizer], FnCallOnPlateau]:
    """Factory: Custom plateau scheduler that discretizes the model without changing LR."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return FnCallOnPlateau(
            model=model, optimizer=optimizer,func=fn
        )
    return factory


def fn_call_on_plateau_scheduler(
    fn:Callable[[nn.Module]],
    patience:  int   = 10,
    min_delta: float = 1e-4,
) -> Callable[[nn.Module, torch.optim.Optimizer], FnCallOnPlateau]:
    """Factory: Custom plateau scheduler that discretizes the model without changing LR."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return FnCallOnPlateau(
            model=model, optimizer=optimizer,func=fn
        )
    return factory


def plateau_scheduler(
    factor:   float = 0.5,
    patience: int   = 10,
    min_lr:   float = 1e-6,
) -> Callable[[nn.Module, torch.optim.Optimizer], torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Factory: ReduceLROnPlateau."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr,
        )
    return factory


def cosine_annealing_scheduler(
    T_max:   int,
    eta_min: float = 0.0,
) -> Callable[[nn.Module, torch.optim.Optimizer], torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Factory: CosineAnnealingLR."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return factory


def step_scheduler(
    step_size: int,
    gamma:     float = 0.1,
) -> Callable[[nn.Module, torch.optim.Optimizer], torch.optim.lr_scheduler.StepLR]:
    """Factory: StepLR."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return factory


# ══════════════════════════════════════════════
# GRADIENT UTILITIES
# ══════════════════════════════════════════════

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
    has a gradient.  Three metrics are returned — see LayerGradStats for the
    rationale behind each choice.
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

    Detection criteria (either triggers is_vanishing = True):
      1. mean_abs of first layer / mean_abs of last layer < ratio_threshold
         (relative collapse across depth).
      2. Any individual layer has mean_abs < threshold
         (absolute freeze regardless of depth).

    Call this right after loss.backward() and before optimizer.step().
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


# ══════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════

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


# ══════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════

def save_checkpoint(checkpoint: Checkpoint, filepath: Path | str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    CONSOLE.print(f"[green]Checkpoint saved -> {filepath}[/green]")


def load_checkpoint(
    filepath:     Path | str,
    map_location: str | torch.device | None = None,
) -> Checkpoint:
    ckpt = torch.load(Path(filepath), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, Checkpoint):
        raise TypeError(f"Expected Checkpoint, got {type(ckpt)}")
    return ckpt


def merge_checkpoints(ckpt1: Checkpoint, ckpt2: Checkpoint) -> Checkpoint:
    """
    Chain two checkpoints end-to-end.

    Inherits model weights, optimizer class, and batch size from ckpt2
    (the later run).  Epoch numbers in ckpt2's history are offset so the
    merged timeline is continuous.
    """
    c1, c2 = ckpt1.train_config, ckpt2.train_config

    if type(ckpt1.model) is not type(ckpt2.model):
        raise ValueError(f"Model architectures differ: {type(ckpt1.model)} vs {type(ckpt2.model)}")
    if c1.loss_fn != c2.loss_fn:
        raise ValueError("Cannot merge: loss functions differ.")
    if c1.optimizer_cls != c2.optimizer_cls:
        raise ValueError("Cannot merge: optimizers differ.")

    last_epoch     = ckpt1.training_history[-1].epoch if ckpt1.training_history else 0
    merged_history = list(ckpt1.training_history) + [
        HistoryEntry(
            epoch              = last_epoch + e.epoch,
            avg_loss           = e.avg_loss,
            avg_err            = e.avg_err,
            avg_regularization = e.avg_regularization,
            grad_stats         = e.grad_stats,
        )
        for e in ckpt2.training_history
    ]

    return Checkpoint(
        model            = ckpt2.model,
        train_config     = TrainConfig(
            num_epochs       = c1.num_epochs + c2.num_epochs,
            batch_size       = c2.batch_size,
            optimizer_cls    = c2.optimizer_cls,
            optimizer_kwargs = c2.optimizer_kwargs,
            loss_fn          = c1.loss_fn,
        ),
        training_history = merged_history,
    )


# ══════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════

def plot_training_loss(loss_history: List[float]) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="tomato", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def plot_gradient_flow(checkpoint: Checkpoint) -> None:
    """
    Plot mean_abs gradient per layer over training epochs.

    This is the standard gradient-flow plot used in research.
    Healthy training shows roughly flat bars across layers; collapsing bars
    in early layers indicate vanishing gradients.
    """
    import matplotlib.pyplot as plt

    entries = checkpoint.training_history
    if not entries or not entries[0].grad_stats:
        CONSOLE.print("[yellow]No gradient stats found in checkpoint.[/yellow]")
        return

    param_names = list(entries[0].grad_stats.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(param_names) * 1.2), 4))
    for entry in entries:
        means = [entry.grad_stats.get(n, LayerGradStats(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))).mean_abs.item() for n in param_names]
        ax.plot(param_names, means, alpha=0.3, linewidth=1, color="steelblue")

    # Highlight first and last epoch prominently.
    for idx, label in [(0, "first epoch"), (-1, "last epoch")]:
        entry = entries[idx]
        means = [entry.grad_stats.get(n, LayerGradStats(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))).mean_abs.item() for n in param_names]
        ax.plot(param_names, means, linewidth=2.5, label=f"Epoch {entry.epoch} ({label})")

    ax.set_yscale("log")
    ax.set_xlabel("Layer / parameter")
    ax.set_ylabel("Mean |gradient| (log scale)")
    ax.set_title("Gradient Flow — mean_abs per layer across training")
    ax.legend(); ax.grid(alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.show()


def plot_weight_distribution(model: nn.Module, bins: int = 50, min_elements: int = 2) -> None:
    import matplotlib.pyplot as plt
    params = {
        name: param.detach().cpu().flatten()
        for name, param in model.named_parameters()
        if param.numel() > min_elements
    }
    if not params:
        CONSOLE.print("[yellow]No parameters large enough to plot.[/yellow]")
        return

    n     = len(params)
    ncols = max(1, int(n ** 0.5))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes  = [axes] if n == 1 else list(axes.flatten())

    for idx, (name, weights) in enumerate(params.items()):
        ax = axes[idx]
        ax.hist(weights.numpy(), bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{name}\nmu={weights.mean():.3f}  sigma={weights.std():.3f}", fontsize=9)
        ax.set_xlabel("Weight"); ax.set_ylabel("Count")

    for ax in axes[n:]:
        fig.delaxes(ax)
    plt.tight_layout(); plt.show()


def plot_checkpoints(checkpoints: List[Checkpoint], title: str = "Checkpoint Comparison") -> None:
    import matplotlib.pyplot as plt
    if not checkpoints:
        return

    n   = len(checkpoints)
    fig = plt.figure(figsize=(12, 6 + 4 * n))
    gs  = fig.add_gridspec(n + 1, 1)

    ax_loss = fig.add_subplot(gs[0, 0])
    for i, ckpt in enumerate(checkpoints):
        c     = ckpt.train_config
        label = f"Ckpt {i+1} (opt={c.optimizer_cls}, loss={c.loss_fn})"
        ax_loss.plot(range(1, len(ckpt.avg_losses()) + 1), ckpt.avg_losses(), linewidth=2, label=label)
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title=f"{title} - Loss Curves")
    ax_loss.grid(alpha=0.3); ax_loss.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    for i, ckpt in enumerate(checkpoints):
        ax = fig.add_subplot(gs[i + 1, 0])
        with torch.no_grad():
            all_w = torch.cat([p.detach().cpu().flatten() for p in ckpt.model.parameters() if p.numel() > 1])
        ax.hist(all_w.numpy(), bins=100, color=f"C{i}", alpha=0.7)
        ax.set_title(f"Ckpt {i+1} weights - mu={all_w.mean():.3f}  sigma={all_w.std():.3f}")
        ax.set(xlabel="Weight", ylabel="Count"); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()


def animate_gradient_flow(checkpoint: Checkpoint) -> None:
    """
    Animate mean_abs gradient per layer across training epochs.
    Each frame is one epoch; bar height is mean_abs on a log scale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    entries = checkpoint.training_history
    if not entries or not entries[0].grad_stats:
        CONSOLE.print("[yellow]No gradient stats in checkpoint.[/yellow]")
        return

    param_names = list(entries[0].grad_stats.keys())
    fig, ax     = plt.subplots(figsize=(max(8, len(param_names) * 1.2), 4))

    def animate(epoch_idx: int):
        ax.clear()
        entry = entries[epoch_idx]
        means = [entry.grad_stats.get(n, LayerGradStats(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))).mean_abs.item() for n in param_names]
        ax.bar(param_names, means, color="orange", edgecolor="black")
        ax.set_yscale("log")
        ax.set_ylabel("Mean |gradient|")
        ax.set_title(f"Gradient Flow - Epoch {entry.epoch}")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

    animation.FuncAnimation(fig, animate, frames=len(entries), interval=700, repeat=True)
    plt.show()


# ══════════════════════════════════════════════
# TESTING / EVALUATION
# ══════════════════════════════════════════════

def evaluate_accuracy(
    model:      nn.Module,
    dataset:    Tuple[Tensor, Tensor],
    threshold:  float        = 0.5,
    batch_size: int          = 200,
    device:     torch.device = DEVICE,
    sample_wise_comparison: bool = False,
) -> float:
    """
    Bit-accuracy for binary / multi-label classification.
    Returns the fraction of bits predicted correctly across the full dataset.
    If sample_wise_comparison is True, computes exact match accuracy per sample instead.
    """
    x_test, y_test = dataset
    model.eval()
    correct = 0.0
    total   = x_test.size(0)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb    = x_test[i : i + batch_size].to(device)
            yb    = y_test[i : i + batch_size].to(device)
            preds = (model(xb) >= threshold).float()
            
            if sample_wise_comparison:
                # Compare per sample (row-wise match)
                if yb.dim() > 1:
                    correct += (preds == yb).all(dim=-1).float().sum().item()
                else:
                    correct += (preds == yb).float().sum().item()
            else:
                # Compare every individual bit
                correct += (preds == yb).float().sum().item()

    if sample_wise_comparison:
        total_items = total
    else:
        total_items = total * (y_test.shape[-1] if y_test.dim() > 1 else 1)
        
    model.train()
    return correct / total_items


# ══════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════

def _call_matching(func: Callable, arg_dict: Dict[str, Any]) -> Any:
    """Call func forwarding only the kwargs its signature accepts."""
    valid = set(inspect.signature(func).parameters)
    return func(**{k: v for k, v in arg_dict.items() if k in valid})


def _format_peek(peek_results: Dict[str, Any]) -> str:
    return " | ".join(
        f"{k} = {v:.6f}" if isinstance(v, float) else f"{k} = {v}"
        for k, v in peek_results.items()
    )


def _accumulate_grad_stats(
    acc: Dict[str, LayerGradStats],
    new: Dict[str, LayerGradStats],
) -> Dict[str, LayerGradStats]:
    """Sum two LayerGradStats dicts element-wise (for later averaging)."""
    result = dict(acc)
    for name, s in new.items():
        if name in result:
            prev = result[name]
            result[name] = LayerGradStats(
                mean_abs        = prev.mean_abs        + s.mean_abs,
                norm_normalized = prev.norm_normalized + s.norm_normalized,
                max_abs         = prev.max_abs         + s.max_abs,
            )
        else:
            result[name] = s
    return result


def _divide_grad_stats(acc: Dict[str, LayerGradStats], n: int) -> Dict[str, LayerGradStats]:
    return {
        name: LayerGradStats(
            mean_abs        = (s.mean_abs        / n).detach().cpu(),
            norm_normalized = (s.norm_normalized / n).detach().cpu(),
            max_abs         = (s.max_abs         / n).detach().cpu(),
        )
        for name, s in acc.items()
    }


# ══════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════
from dataclasses import field
@dataclass
class Trainer:
    """
    Orchestrates model training.

    Parameters
    ----------
    model :
        The network to train.  If the model defines an apply_constraints(self)
        method it will be called automatically after every optimizer step.
    dataset :
        (x, y) tensors — the full training set.
    stop_on :
        (metrics: dict) -> bool — return True to halt.
    batch_size :
        Mini-batch size.
    loss_fn :
        Training loss (default MSELoss).
    error_fn :
        Reporting metric only, not back-propagated (default L1Loss).
    regularization_fn :
        () -> Tensor — scalar added to the loss each batch.
    checkpoint_path :
        If set, the final Checkpoint is saved here automatically.
    optimizer_kwargs :
        Forwarded to the optimizer constructor.
    optimizer_cls :
        Optimizer class (default Adam).
    lr_scheduler_factory :
        (model, optimizer) -> LRScheduler.
    device :
        Target device.
    check_grad :
        Print a per-parameter gradient-stats table each epoch.
    vanishing_grad_check_every :
        Run detect_vanishing_gradients every N epochs and warn if detected.
        Set to 0 to disable.
    constraint :
        Optional ``() -> None`` called after every optimizer step.
        Use this for constraints that live outside the model class — generic
        clamps, one-off lambdas, etc.  If the model also defines
        ``apply_constraints(self)``, both are called: constraint first, then
        the model method.
    peek :
        () -> Dict[str, Any] appended to the per-epoch console line.
    """

    model:                      nn.Module
    dataset:                    Tuple[Tensor, Tensor]
    stop_on:                    Callable[[dict], bool]
    batch_size:                 int
    loss_fn:                    nn.modules.loss._Loss                  = field(default_factory=nn.MSELoss)
    error_fn:                   nn.modules.loss._Loss                  = field(default_factory=nn.L1Loss)
    regularization_fn:          Optional[Callable[[nn.Module], Tensor]]= None
    checkpoint_path:            Optional[Path]                         = None
    optimizer_kwargs:           Dict[str, Any]                         = field(default_factory=dict)
    optimizer_cls:              Type[torch.optim.Optimizer]            = Adam
    lr_scheduler_factory:       Optional[Callable[..., Any]]           = None
    device:                     torch.device                           = field(default_factory=resolve_device)
    check_grad:                 bool                                   = False
    constraint:                 Optional[Callable[[nn.Module], None]]  = None
    peek:                       Optional[Callable[[], Dict[str, Any]]] = None
    state:                      Any                                    = field(default_factory=TrainerState)

    # ------------------------------------------------------------------
    def train(self) -> Checkpoint:
        self.model = self.model.to(self.device)

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        scheduler = (
            self.lr_scheduler_factory(self.model, optimizer)
            if self.lr_scheduler_factory is not None else None
        )


        x_train, y_train = (t.to(self.device) for t in self.dataset)
        n_samples        = x_train.shape[0]
        history: List[HistoryEntry] = []
        epoch = 0

        while True:
            epoch += 1
            perm    = torch.randperm(n_samples, device=self.device)
            x_epoch = x_train[perm]
            y_epoch = y_train[perm]

            epoch_loss = epoch_error = epoch_reg = 0.0
            acc_stats: Dict[str, LayerGradStats] = {}
            num_batches = 0

            for i in range(0, n_samples, self.batch_size):
                xb = x_epoch[i : i + self.batch_size]
                yb = y_epoch[i : i + self.batch_size]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                
                if self.regularization_fn is not None:
                    reg = _call_matching(self.regularization_fn, {"module": self.model, "model": self.model, "state": self.state})
                else:
                    reg = torch.tensor(0.0)
                
                loss   = self.loss_fn(logits, yb) + reg
                loss.backward()

                # Collect stats right after backward, before step.
                if self.check_grad:
                    acc_stats = _accumulate_grad_stats(acc_stats, collect_grad_stats(self.model))

                optimizer.step()

                with torch.no_grad():
                    epoch_error += self.error_fn(logits, yb).item()
                    if self.constraint is not None:
                        _call_matching(self.constraint, {"module": self.model, "model": self.model, "state": self.state})
                    
                    underlying_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    if hasattr(underlying_model, "apply_constraints"):
                        _call_matching(underlying_model.apply_constraints, {"module": underlying_model, "model": underlying_model, "state": self.state})
                epoch_loss += loss.item()
                epoch_reg  += reg.item()
                num_batches += 1

            # ── Epoch averages ─────────────────────────────────────────
            avg_loss  = epoch_loss  / num_batches
            avg_error = epoch_error / num_batches
            avg_reg   = epoch_reg   / num_batches
            avg_stats = _divide_grad_stats(acc_stats, num_batches) if self.check_grad else {}
            
            if hasattr(self.state, "update"):
                self.state.update(epoch, avg_loss, avg_error, avg_reg)

            # ── Console logging ────────────────────────────────────────
            peek_str = (" | " + _format_peek(self.peek())) if self.peek is not None else ""
            CONSOLE.print(
                f"Epoch [bold]{epoch:04d}[/bold] | "
                f"loss = [red]{avg_loss:.6f}[/red] | "
                f"error = [yellow]{avg_error:.6f}[/yellow] | "
                f"reg = {avg_reg:.6f}"
                + peek_str
            )

            if self.check_grad:
                _print_grad_table(avg_stats)
            
            # ── LR scheduler ───────────────────────────────────────────
            if scheduler is not None:
                _call_matching(scheduler.step, {"metrics": avg_error, "avg_error": avg_error,"avg_loss":avg_loss, "state": self.state})


            # ── History ────────────────────────────────────────────────
            history.append(HistoryEntry(
                epoch              = epoch,
                avg_loss           = avg_loss,
                avg_err            = avg_error,
                avg_regularization = avg_reg,
                grad_stats         = avg_stats,
            ))

            # ── Stopping condition ─────────────────────────────────────
            stop_metrics = {
                "epoch":              epoch,
                "avg_loss":           avg_loss,
                "avg_error":          avg_error,
                "avg_regularization": avg_reg,
                "history":            history,
                "model":              self.model,
                "state":              self.state,
            }
            if _call_matching(self.stop_on, stop_metrics):
                CONSOLE.print(f"[bold red]Stopping at epoch {epoch}.[/bold red]")
                break

        # ── Build and optionally save checkpoint ───────────────────────
        checkpoint = Checkpoint(
            model        = self.model,
            train_config = TrainConfig(
                num_epochs       = epoch,
                batch_size       = self.batch_size,
                optimizer_cls    = self.optimizer_cls.__name__,
                optimizer_kwargs = self.optimizer_kwargs,
                loss_fn          = self.loss_fn.__class__.__name__,
            ),
            training_history = history,
        )

        if self.checkpoint_path is not None:
            save_checkpoint(checkpoint, self.checkpoint_path)

        return checkpoint

    # ------------------------------------------------------------------
    def export_for_burn(self, export_dir: str | Path) -> None:
        """Export model (ONNX) + dataset (.npz) + config (.json) for burn.rs."""
        import numpy as np

        export_dir  = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        x_data, y_data = self.dataset
        dummy_input    = x_data[:1] if x_data.dim() > 1 else x_data[:1].unsqueeze(0)

        onnx_path = export_dir / "model.onnx"
        self.model.eval()
        torch.onnx.export(
            self.model.cpu(), dummy_input.cpu(), str(onnx_path),
            export_params=True, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
        )
        CONSOLE.print(f"[green]ONNX model -> {onnx_path}[/green]")

        dataset_path = export_dir / "dataset.npz"
        np.savez(dataset_path, x=x_data.cpu().numpy(), y=y_data.cpu().numpy())
        CONSOLE.print(f"[green]Dataset    -> {dataset_path}[/green]")

        config_path = export_dir / "train_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "batch_size":       self.batch_size,
                    "loss_fn":          self.loss_fn.__class__.__name__,
                    "optimizer":        self.optimizer_cls.__name__,
                    "optimizer_kwargs": self.optimizer_kwargs,
                },
                f, indent=4,
            )
        CONSOLE.print(f"[green]Config     -> {config_path}[/green]")
