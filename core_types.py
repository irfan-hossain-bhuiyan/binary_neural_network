from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

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
    grad_stats:         Dict[str, LayerGradStats]

@dataclass
class VanishingGradReport:
    """Result of detect_vanishing_gradients()."""
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
