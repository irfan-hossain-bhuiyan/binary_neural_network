from typing import Callable, Any
from rich.console import Console
import torch
from pathlib import Path
from dataclasses import dataclass
from torch import nn
from typing import Any, Callable, Dict, List
import matplotlib.pyplot as plt

from torch._prims_common import Tensor

@dataclass
class TrainConfig:
    num_epochs: int
    batch_size: int
    optimizer_cls: str
    optimizer_kwargs: dict[str, Any]
    loss_fn: str
    lr: float

@dataclass
class HistoryEntry:
    epoch: int
    avg_loss: float
    gradient_data:Dict[str,float]

@dataclass
class Checkpoint:
    model: nn.Module
    train_config: TrainConfig
    training_history: list[HistoryEntry]



def save_training_checkpoint(checkpoint: Checkpoint, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)

def load_training_checkpoint(filepath: str | Path, map_location: str | torch.device | None = None) -> Checkpoint:
    checkpoint = torch.load(Path(filepath), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"Expected Checkpoint object, got {type(checkpoint)}")
    return checkpoint

from rich.table import Table
from torch.optim import Adam

def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = _resolve_device()
CONSOLE = Console()

class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val, leak=0.01):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.leak = leak
        return torch.clamp(input, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Mask where the input was within bounds
        mask = (input >= ctx.min_val) & (input <= ctx.max_val)

        # Gradient is 1.0 inside, and 'leak' outside
        grad_input = grad_output.clone()
        grad_input[~mask] *= ctx.leak

        return grad_input, None, None, None

# Usage helper
def leaky_clamp(input, min_val, max_val, leak=0.01):
    return LeakyClamp.apply(input, min_val, max_val, leak)

def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    shuffle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into train and test partitions."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched samples: x={x.shape[0]}, y={y.shape[0]}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=x.device)
        x = x[perm]
        y = y[perm]

    n_train = int(n * train_ratio)
    if n_train <= 0 or n_train >= n:
        raise ValueError("train_ratio produced empty train or test split")

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]
    return x_train, y_train, x_test, y_test

def _format_grad_stats(stats: dict[str, float], max_items: int = 6) -> Table:
    table = Table(title="Gradient Norms", show_header=True, header_style="bold cyan")
    table.add_column("param", style="white", overflow="fold")
    table.add_column("norm", justify="right", style="magenta")

    if not stats:
        table.add_row("(no gradients)", "-")
        return table

    items = sorted(stats.items(), key=lambda kv: kv[0])[:max_items]
    for k, v in items:
        table.add_row(k, f"{v:.3e}")
    return table

def _optimizer_name(optimizer_cls: type[torch.optim.Optimizer]) -> str:
    return optimizer_cls.__name__

def _loss_name(loss_fn: nn.modules.loss._Loss) -> str:
    return loss_fn.__class__.__name__
from torch import nn

def train_model(
    dataset: tuple[torch.Tensor, torch.Tensor],
    num_epochs: int,
    batch_size: int,
    model: torch.nn.Module,
    loss_fn: nn.modules.loss._Loss | None = None,
    regularization_fn: Callable[[], torch.Tensor] | None = None,
    checkpoint_path: Path | None = None,
    optimizer_kwargs: Dict[str, Any] | None = None,
    optimizer_cls: type[torch.optim.Optimizer] = Adam,
    lr: float = 0.1,
    lr_schedular: Callable[..., torch.optim.lr_scheduler.LRScheduler] | Any | None = None,
    constraint: None | Callable = None,
    lr_schedular_kargs: Dict[str, Any] | None = None,
    device=DEVICE,
    test_verify_ratio=0.8,
    check_grad: bool = False
):
    model = model.to(device)

    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": lr}
       
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    loss_fn = nn.MSELoss() if loss_fn is None else loss_fn

    if lr_schedular_kargs is None:
        lr_schedular_kargs = {}

    if lr_schedular is not None:
        scheduler = lr_schedular(optimizer, **lr_schedular_kargs)
    else:
        scheduler = None
        
    x_data, y_data = dataset
    # Pass test_verify_ratio to split_dataset if accepted by the function
    x_train, y_train, x_test, y_test = split_dataset(x_data, y_data, train_ratio=test_verify_ratio)
    train_count = x_train.shape[0]

    loss_history: list[float] = []
    history: list[HistoryEntry] = []

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(train_count, device=device)
        X_epoch = x_train[perm]
        Y_epoch = y_train[perm]

        epoch_loss = 0.0
        num_batches = 0

        grad_stats_first_batch: dict[str, float] | None = None
        epoch_vanishing = False
        batch_grad_norms: dict[str, float] = {}
        
        for i in range(0, train_count, batch_size):
            xb = X_epoch[i : i + batch_size]
            yb = Y_epoch[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(xb)

            reg_loss = 0.0
            if regularization_fn is not None:
                reg_loss = regularization_fn()

            loss = loss_fn(logits, yb) + reg_loss
            loss.backward()
            
            if check_grad:
                # Calculate gradient norms for the batches
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.detach().norm(2).item()
                        batch_grad_norms[name] = batch_grad_norms.get(name, 0.0) + norm

            optimizer.step()

            if constraint is not None:
                constraint()

            epoch_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        avg_grad_norms = {}
        if check_grad:
            from rich.table import Table
            table = Table(title="Gradient Norms")
            table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
            table.add_column("Avg Grad Norm", justify="right", style="magenta")
            
            for name, val in batch_grad_norms.items():
                avg_val = val / num_batches
                avg_grad_norms[name] = avg_val
                # Use string formatting for the table
                table.add_row(name, f"{avg_val:.6f}")

        CONSOLE.print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f}")
        
        if check_grad:
            CONSOLE.print(table)

        history.append(
            HistoryEntry(
                epoch=epoch,
                avg_loss=avg_loss,
                gradient_data=avg_grad_norms
            )
        )

    checkpoint = Checkpoint(
        model=model,
        train_config=TrainConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            optimizer_cls=optimizer_cls.__name__,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn.__class__.__name__,
            lr=lr,
        ),
        training_history=history,
    )

    if checkpoint_path is not None:
        save_training_checkpoint(checkpoint, checkpoint_path)

    return checkpoint

def plot_weight_distribution(model: nn.Module, bins: int = 50, n_size: int = 1):
    params_to_plot = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.numel() > n_size:
                params_to_plot[name] = param.detach().cpu().flatten()

    num_plots = len(params_to_plot)
    if num_plots == 0:
        return

    cols = int(num_plots**0.5)
    if cols * cols < num_plots:
        cols += 1
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    # Ensure axes is iterable even if it's a single plot
    if num_plots == 1:
        axes = [axes]
    elif num_plots > 1:
        axes = axes.flatten()

    for idx, (name, all_weights) in enumerate(params_to_plot.items()):
        ax = axes[idx]
        ax.hist(all_weights.numpy(), bins=bins, color="steelblue", edgecolor="white")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name}\nmean={all_weights.mean():.3f}, std={all_weights.std():.3f}", fontsize=10)

    # Remove any extra empty subplots
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


