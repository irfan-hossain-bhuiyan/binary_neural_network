import torch
from pathlib import Path
from core_types import Checkpoint, HistoryEntry, TrainConfig
from trainer_utils import CONSOLE

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
