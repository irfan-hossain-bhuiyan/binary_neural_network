from typing import Callable
import torch
import torch.nn as nn
from trainer_utils import CONSOLE

class PlateauTracker:
    """Independent plateau tracker that components like schedulers or stoppers can use to track plateauing."""
    def __init__(self, patience: int = 10, loss_min_delta: float = 1e-4, error_min_delta: float = 1e-3):
        self.patience = patience
        self.loss_min_delta = loss_min_delta
        self.error_min_delta = error_min_delta
        self.reset()

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.best_error = float("inf")
        self.loss_no_improve = 0
        self.error_no_improve = 0

    def update(self, avg_loss: float, avg_error: float) -> bool:
        """Returns True if a plateau is detected, and resets the tracker."""
        if avg_loss < self.best_loss - self.loss_min_delta:
            self.best_loss = avg_loss
            self.loss_no_improve = 0
        else:
            self.loss_no_improve += 1
            
        if avg_error < self.best_error - self.error_min_delta:
            self.best_error = avg_error
            self.error_no_improve = 0
        else:
            self.error_no_improve += 1
            
        if self.loss_no_improve >= self.patience or self.error_no_improve >= self.patience:
            self.reset()
            return True
        return False


def early_stopping(max_epochs: int = 500, patience: int = 10) -> Callable[..., bool]:
    """
    Factory: plateau-based stopping.
    """
    tracker = PlateauTracker(patience=patience)
    def callback(epoch: int, avg_loss: float, avg_error: float) -> bool:
        return tracker.update(avg_loss, avg_error) or epoch >= max_epochs

    return callback


def stop_on_epoch(max_epochs: int) -> Callable[..., bool]:
    """Factory: stops exactly after max_epochs epochs."""
    def callback(epoch: int) -> bool:
        return epoch >= max_epochs
    return callback


class FnCallOnPlateau:
    """Custom scheduler that calls model.discretize() when plateauing, without changing LR."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, func: Callable[[nn.Module]], patience: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.func = func
        self.tracker = PlateauTracker(patience=patience)

    def step(self, epoch: int, avg_loss: float, avg_error: float):
        if self.tracker.update(avg_loss, avg_error):
            CONSOLE.print(f"[bold yellow]Plateau reached (Epoch {epoch}): Discretizing model[/bold yellow]")
            self.func(self.model)
            # Reset the optimizer state since the model has changed
            from collections import defaultdict
            self.optimizer.state = defaultdict(dict)

def fn_call_on_plateau_scheduler(
    fn: Callable[[nn.Module]],
    patience: int = 10,
    min_delta: float = 1e-4,
) -> Callable[[nn.Module, torch.optim.Optimizer], FnCallOnPlateau]:
    """Factory: Custom plateau scheduler that discretizes the model without changing LR."""
    def factory(model: nn.Module, optimizer: torch.optim.Optimizer):
        return FnCallOnPlateau(
            model=model, optimizer=optimizer, func=fn, patience=patience
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
