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

# Export everything previously in this file by importing from the new submodules.
# Doing this cleanly allows us to preserve the `prelude` module interface without breaking
# any currently working code.

from core_types import TrainConfig, LayerGradStats, HistoryEntry, VanishingGradReport, Checkpoint
from device_utils import resolve_device, DEVICE
from trainer_utils import CONSOLE, _call_matching, _format_peek, _accumulate_grad_stats, _divide_grad_stats
from stopping_utils import (
    PlateauTracker, early_stopping, stop_on_epoch, FnCallOnPlateau,
    fn_call_on_plateau_scheduler, plateau_scheduler, cosine_annealing_scheduler, step_scheduler
)
from grad_utils import (
    LeakyClamp, leaky_clamp, collect_grad_stats, detect_vanishing_gradients, 
    _print_grad_table, _print_vanishing_warning
)
from split_utils import split_dataset
from checkpoint_utils import save_checkpoint, load_checkpoint, merge_checkpoints
from plot_utils import plot_training_loss, plot_gradient_flow, plot_weight_distribution, plot_checkpoints, animate_gradient_flow
from eval_utils import evaluate_accuracy
from trainer import Trainer
