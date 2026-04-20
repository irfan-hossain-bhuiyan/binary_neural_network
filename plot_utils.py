import torch
import torch.nn as nn
from typing import List
from core_types import Checkpoint, LayerGradStats
from trainer_utils import CONSOLE

def plot_training_loss(loss_history: List[float], header: str = "") -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="tomato", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    title_str = f"Training Loss - {header}" if header else "Training Loss"
    plt.title(title_str)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def plot_gradient_flow(checkpoint: Checkpoint) -> None:
    """
    Plot mean_abs gradient per layer over training epochs.
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
