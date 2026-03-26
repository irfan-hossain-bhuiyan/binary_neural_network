from typing import Callable, Any
from rich.console import Console

import torch
from pathlib import Path
from dataclasses import dataclass
from torch import nn, tensor
from typing import Any, Callable, Dict, List,Tuple
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
    avg_err: float
    avg_regularization: float
    gradient_data:Dict[str,float]

@dataclass
class Checkpoint:
    model: nn.Module
    train_config: TrainConfig
    training_history: list[HistoryEntry]

    def get_avg_losses(self) -> list[float]:
        return [entry.avg_loss for entry in self.training_history]

def plot_training_loss(loss_history: list[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="tomato", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




def save_training_checkpoint(checkpoint: Checkpoint, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)

def load_training_checkpoint(filepath: str | Path, map_location: str | torch.device | None = None) -> Checkpoint:
    checkpoint = torch.load(Path(filepath), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"Expected Checkpoint object, got {type(checkpoint)}")
    return checkpoint

from rich.table import Table
from rich.console import Console
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
        return torch.clamp(input, min=float(min_val), max=float(max_val))

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

from dataclasses import dataclass

@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 1e-4
    max_epochs: int = 1000

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Tuple[torch.Tensor, torch.Tensor],
        training_type: int | EarlyStopping,
        batch_size: int,
        loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
        error_fn: nn.modules.loss._Loss = nn.L1Loss(),
        regularization_fn: Callable[[], torch.Tensor] | None = None,
        checkpoint_path: Path | None = None,
        optimizer_kwargs: Dict[str, Any] | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = Adam,
        lr: float = 0.1,
        lr_schedular: Callable[..., torch.optim.lr_scheduler.LRScheduler] | Any | None = None,
        constraint: None | Callable = None,
        lr_schedular_kargs: Dict[str, Any] = None,
        device=DEVICE,
        check_grad: bool = False,
        peek: Callable[[], Dict[str, Any]] | None = None
    ):
        if lr_schedular_kargs is None:
            lr_schedular_kargs = {}
            
        self.model = model
        self.dataset = dataset
        self.training_type = training_type
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.error_fn =error_fn
        self.regularization_fn = regularization_fn
        self.checkpoint_path = checkpoint_path
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {"lr": lr}
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.lr_schedular = lr_schedular
        self.constraint = constraint
        self.lr_schedular_kargs = lr_schedular_kargs
        self.device = device
        self.check_grad = check_grad
        self.peek = peek

    def train(self) -> Checkpoint:
        self.model = self.model.to(self.device)
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        if self.lr_schedular is not None:
            scheduler = self.lr_schedular(optimizer, **self.lr_schedular_kargs)
        else:
            scheduler = None
            
        x_data, y_data = self.dataset
        x_train = x_data.to(self.device)
        y_train = y_data.to(self.device)
        train_count = x_train.shape[0]

        if isinstance(self.training_type, int):
            num_epochs = self.training_type
            patience = None
            min_delta = 0.0
        else:
            num_epochs = self.training_type.max_epochs
            patience = self.training_type.patience
            min_delta = self.training_type.min_delta

        best_err = float('inf')
        epochs_no_improve = 0

        history: list[HistoryEntry] = []

        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(train_count, device=self.device)
            X_epoch = x_train[perm]
            Y_epoch = y_train[perm]

            epoch_loss = 0.0
            epoch_error = 0.0
            epoch_regularization = 0.0
            num_batches = 0
            batch_grad_norms: dict[str, float] = {}
            
            for i in range(0, train_count, self.batch_size):
                xb = X_epoch[i : i + self.batch_size]
                yb = Y_epoch[i : i + self.batch_size]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)

                reg_loss = tensor(0.0)
                if self.regularization_fn is not None:
                    reg_loss = self.regularization_fn()
                loss = self.loss_fn(logits, yb) + reg_loss
                loss.backward()
                
                if self.check_grad:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            norm = param.grad.detach().abs().mean().item()
                            batch_grad_norms[name] = batch_grad_norms.get(name, 0.0) + norm

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    epoch_error+=self.error_fn(logits,yb).item()
                    if self.constraint is not None:
                        self.constraint()

                epoch_loss += loss.item()
                epoch_regularization+=reg_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_error = epoch_error/num_batches
            avg_regularization = epoch_regularization/num_batches
            
            peek_info = ""
            if self.peek is not None:
                peek_results = self.peek()
                formatted_peeks = []
                for k, v in peek_results.items():
                    if isinstance(v, float):
                        formatted_peeks.append(f"{k} = {v:.6f}")
                    else:
                        formatted_peeks.append(f"{k} = {v}")
                if formatted_peeks:
                    peek_info = " | " + " | ".join(formatted_peeks)

            CONSOLE.print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f} | error = {avg_error:.6f} | regularization = {avg_regularization:.6f} {peek_info}")
            
            avg_grad_norms = {}
            if self.check_grad:
                from rich.table import Table
                table = Table(title="Gradient Norms")
                table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
                table.add_column("Avg Grad Norm", justify="right", style="magenta")
                
                for name, val in batch_grad_norms.items():
                    avg_val = val / num_batches
                    avg_grad_norms[name] = avg_val
                    table.add_row(name, f"{avg_val:.6f}")
                CONSOLE.print(table)

            if patience is not None:
                if avg_error < best_err - min_delta:
                    best_err = avg_error
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    CONSOLE.print(f"[bold red]Early stopping triggered![/bold red] No improvement for {patience} epochs.")
                    break

            history.append(HistoryEntry(epoch=epoch, avg_loss=avg_loss,avg_regularization=avg_regularization,avg_err=avg_error ,gradient_data=avg_grad_norms))

        checkpoint = Checkpoint(
            model=self.model,
            train_config=TrainConfig(
                num_epochs=num_epochs,
                batch_size=self.batch_size,
                optimizer_cls=self.optimizer_cls.__name__,
                optimizer_kwargs=self.optimizer_kwargs,
                loss_fn=self.loss_fn.__class__.__name__,
                lr=self.lr,
            ),
            training_history=history,
        )

        if self.checkpoint_path is not None:
            save_training_checkpoint(checkpoint, self.checkpoint_path)

        return checkpoint

    def export_for_burn(self, export_dir: str | Path):
        """
        Exports the model to ONNX and the training dataset to a format consumable by a Rust burn.rs program.
        We'll save the ONNX file and use `.safetensors` or standard numpy `.npz` for the dataset.
        """
        import numpy as np
        
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export model to ONNX
        dummy_x, _ = self.dataset
        # Create a tiny dummy input matching the shape of one dataset entry, or batch
        dummy_input = dummy_x[:0].unsqueeze(0) if dummy_x.dim() == 1 else dummy_x[:1]
        
        onnx_path = export_dir / "model.onnx"
        self.model.eval()
        torch.onnx.export(
            self.model.cpu(), 
            dummy_input.cpu(), 
            str(onnx_path), 
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_shapes={"x": {0: torch.export.Dim("batch_size", min=1)}}
        )
        CONSOLE.print(f"Exported ONNX model to {onnx_path}")
        
        # 2. Export dataset
        x_data, y_data = self.dataset
        dataset_path = export_dir / "dataset.npz"
        np.savez(
            dataset_path, 
            x=x_data.cpu().numpy(), 
            y=y_data.cpu().numpy()
        )
        CONSOLE.print(f"Exported dataset to {dataset_path}")
        
        # 3. Export train config
        import json
        config_path = export_dir / "train_config.json"
        
        if isinstance(self.training_type, int):
            num_epochs = self.training_type
            patience = None
        else:
            num_epochs = self.training_type.max_epochs
            patience = self.training_type.patience
            
        config = {
            "num_epochs": num_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "loss_fn": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer_cls.__name__,
            "patience": patience
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        CONSOLE.print(f"Exported training metadata to {config_path}")


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

def merge_checkpoints(ckpt1: Checkpoint, ckpt2: Checkpoint) -> Checkpoint:
    """Merges two checkpoints, chaining the training history together.
    
    The latest model parameters, optimizer, and learning rate are inherited
    from the second checkpoint (`ckpt2`).
    """
    c1 = ckpt1.train_config
    c2 = ckpt2.train_config
    
    # Validation
    if type(ckpt1.model) != type(ckpt2.model):
        raise ValueError(f"Model architectures do not match: {type(ckpt1.model)} vs {type(ckpt2.model)}")
    if c1.loss_fn != c2.loss_fn:
        raise ValueError("Cannot merge checkpoints: loss functions are missing or mismatched.")
    if c1.optimizer_cls != c2.optimizer_cls:
        raise ValueError("Cannot merge checkpoints: optimizers are mismatched.")
        
    merged_history = list(ckpt1.training_history)
    # The last recorded epoch from the first training sprint
    last_epoch = merged_history[-1].epoch if merged_history else 0
    
    # Adjust epoch numbers and append the history of the second checkpoint
    for entry in ckpt2.training_history:
        merged_history.append(
            HistoryEntry(
                epoch=last_epoch + entry.epoch,
                avg_loss=entry.avg_loss,
                
                gradient_data=entry.gradient_data
            )
        )
        
    merged_config = TrainConfig(
        num_epochs=c1.num_epochs + c2.num_epochs,
        batch_size=c2.batch_size,            # Prefer latest run configuration
        optimizer_cls=c2.optimizer_cls,      
        optimizer_kwargs=c2.optimizer_kwargs, 
        loss_fn=c1.loss_fn,
        lr=c2.lr
    )

    return Checkpoint(
        model=ckpt2.model,  # Take the weights trained up to the latest point
        train_config=merged_config,
        training_history=merged_history
    )

def plot_checkpoints(checkpoints: list[Checkpoint], title: str = "Checkpoint Comparison"):
    if not checkpoints:
        return

    num_checkpoints = len(checkpoints)

    loss_data = []
    labels = []
    
    for i, ckpt in enumerate(checkpoints):
        losses = ckpt.get_avg_losses()
        c = ckpt.train_config
        opt = c.optimizer_cls
        lr = c.lr
        loss_fn = c.loss_fn
        label = f"Ckpt {i+1} (LR={lr}, Opt={opt}, Loss={loss_fn})"
        loss_data.append(losses)
        labels.append(label)

    fig = plt.figure(figsize=(12, 6 + 4 * num_checkpoints))
    gs = fig.add_gridspec(num_checkpoints + 1, 1)

    ax_loss = fig.add_subplot(gs[0, 0])
    for i, losses in enumerate(loss_data):
        ax_loss.plot(range(1, len(losses) + 1), losses, linewidth=2, label=labels[i])
    
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{title} - Loss Curves")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for i, ckpt in enumerate(checkpoints):
        ax_weights = fig.add_subplot(gs[i + 1, 0])
        all_weights = []
        with torch.no_grad():
            for name, param in ckpt.model.named_parameters():
                if param.numel() > 1:
                    all_weights.append(param.detach().cpu().flatten())
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            ax_weights.hist(all_weights.numpy(), bins=100, color=f"C{i}", alpha=0.7)
            mean_w = all_weights.mean().item()
            std_w = all_weights.std().item()
            ax_weights.set_title(f"{labels[i]} - Weights (Mean: {mean_w:.3f}, Std: {std_w:.3f})")
            ax_weights.set_xlabel("Weight Value")
            ax_weights.set_ylabel("Frequency")
            ax_weights.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def testing(
    model: nn.Module,
    dataset:Tuple[Tensor,Tensor],
    threshold: float = 0.5,
    # threshold is for what value it will be treated as 1.
    num_samples: int = 2000,
    device: torch.device=DEVICE,
) -> float:
    x_test,y_test=dataset
    model.eval()
    with torch.no_grad():
        
        X_test = x_test.to(device)
        Y_test = y_test.to(device)
            
        # evaluate in batches to prevent OutOfMemory errors on evaluation
        batch_size = 200
        correct_bits_sum = 0.0
        total_samples = X_test.size(0)
        
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                X_batch = X_test[i:i+batch_size]
                Y_batch = Y_test[i:i+batch_size]
                
                logits = model(X_batch)
                preds = (logits >= threshold).float()
                
                correct_bits_sum += (preds == Y_batch).float().sum().item()
                
                            
        # Compute exact mean based on total bits processed
        total_bits = total_samples * (Y_test.shape[-1] if Y_test.dim() > 1 else 1)
        correct_bits = correct_bits_sum / total_bits
    model.train()
    return correct_bits


