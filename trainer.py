from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from core_types import Checkpoint, HistoryEntry, LayerGradStats, TrainConfig
from trainer_utils import _call_matching, _format_peek, _accumulate_grad_stats, _divide_grad_stats, CONSOLE
from grad_utils import collect_grad_stats
from device_utils import resolve_device
from checkpoint_utils import save_checkpoint

@dataclass
class Trainer:
    """
    Orchestrates model training.
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
    constraints:                 Callable[[nn.Module], None] | List[Callable[[nn.Module], None]] | None = None
    peek:                       Optional[Callable[[], Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    def train(self, print_terminal: bool = True) -> Checkpoint:
        self.model = self.model.to(self.device)
        unwrapped_model = getattr(self.model, "module", self.model)

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        scheduler = (
            self.lr_scheduler_factory(unwrapped_model, optimizer)
            if self.lr_scheduler_factory is not None else None
        )

        x_train, y_train = (t.to(self.device) for t in self.dataset)
        n_samples        = x_train.shape[0]
        history: List[HistoryEntry] = []
        epoch = 0
        avg_loss = float("inf")
        avg_error = float("inf")
        avg_reg = 0.0

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
                    reg = _call_matching(self.regularization_fn, {"module": unwrapped_model, "model": unwrapped_model, "epoch": epoch, "avg_loss": avg_loss, "avg_error": avg_error, "avg_reg": avg_reg})
                else:
                    reg = torch.tensor(0.0)
                
                loss   = self.loss_fn(logits, yb) + reg
                loss.backward()

                # Collect stats right after backward, before step.
                if self.check_grad:
                    acc_stats = _accumulate_grad_stats(acc_stats, collect_grad_stats(unwrapped_model))

                optimizer.step()

                with torch.no_grad():
                    epoch_error += self.error_fn(logits, yb).item()
                    if self.constraints is not None:
                        constraints = self.constraints if isinstance(self.constraints, list) else [self.constraints]
                        kwargs = {"module": unwrapped_model, "model": unwrapped_model, "epoch": epoch, "avg_loss": avg_loss, "avg_error": avg_error, "avg_reg": avg_reg}
                        for c in constraints:
                            _call_matching(c, kwargs)
                    
                epoch_loss += loss.item()
                epoch_reg  += reg.item()
                num_batches += 1

            # ── Epoch averages ─────────────────────────────────────────
            avg_loss  = epoch_loss  / num_batches
            avg_error = epoch_error / num_batches
            avg_reg   = epoch_reg   / num_batches
            avg_stats = _divide_grad_stats(acc_stats, num_batches) if self.check_grad else {}

            # ── Console logging ────────────────────────────────────────
            if print_terminal:
                peek_str = (" | " + _format_peek(self.peek())) if self.peek is not None else ""
                CONSOLE.print(
                    f"Epoch [bold]{epoch:04d}[/bold] | "
                    f"loss = [red]{avg_loss:.6f}[/red] | "
                    f"error = [yellow]{avg_error:.6f}[/yellow] | "
                    f"reg = {avg_reg:.6f}"
                    + peek_str
                )

                if self.check_grad:
                    from grad_utils import _print_grad_table
                    _print_grad_table(avg_stats)
            
            # ── LR scheduler ───────────────────────────────────────────
            if scheduler is not None:
                _call_matching(scheduler.step, {"metrics": avg_error, "epoch": epoch, "avg_error": avg_error, "avg_loss": avg_loss, "avg_reg": avg_reg})

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
                "model":              unwrapped_model,
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
