"""Baseline research entry point.

Reuses the existing training machinery (Trainer, MultiLayerLogicGateNet,
synthetic XOR data generation) and returns machine-readable metrics.

Must NOT depend on pretty terminal output: metrics are returned directly
as Python values.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python research/run_experiment.py` from the repo root: top-level
# source modules (models, trainer, ...) live in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from data_utils import generate_xor_dataset
from eval_utils import evaluate_accuracy
from initializers import NormalInitWrapper
from models import MultiLayerLogicGateNet
from prelude import Trainer, split_dataset, stop_on_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_experiment(
    experiment_name: str = "baseline",
    seed: int = 0,
    num_bits: int = 4,
    epochs: int = 20,
    num_samples: int = 20000,
    batch_size: int = 256,
    layer_dims: tuple[int, ...] = (64, 32, 32),
    start_temperature: float = 1.0,
    end_temperature: float = 0.01,
    variance_weight: float = 1e-3,
    lr: float = 0.05,
) -> dict:
    """Train the baseline BNN on synthetic XOR and return metrics dict."""
    t_start = time.time()
    seed_everything(seed)
    device = _resolve_device()

    layer_dims_full = tuple(layer_dims) + (num_bits,)

    # Synthetic data generated in-memory: no dataset .pt files needed,
    # so Kaggle runs do not depend on packaged artifacts.
    x_all, y_all = generate_xor_dataset(
        num_samples=num_samples, num_bits=num_bits, device=torch.device("cpu")
    )
    x_train, y_train, x_test, y_test = split_dataset(
        x_all, y_all, train_ratio=0.8, shuffle=True
    )

    net = MultiLayerLogicGateNet(
        input_dim=2 * num_bits,
        layer_dims=layer_dims_full,
        use_softmax=True,
        grad_scalar=True,
        odd_initialization=NormalInitWrapper(0.5),
        even_initialization=NormalInitWrapper(0.5),
        bias_initialization=NormalInitWrapper(1.0),
    ).to(device)

    def variance_regularizer(module):
        return variance_weight * MultiLayerLogicGateNet.batch_variance_regularization(
            module
        )

    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=stop_on_epoch(epochs),
        batch_size=batch_size,
        model=net,
        loss_fn=nn.MSELoss(),
        optimizer_cls=Adam,
        optimizer_kwargs={"lr": lr},
        regularization_fn=variance_regularizer,
        lr_scheduler_factory=None,
        constraints=[MultiLayerLogicGateNet.constraint],
        on_epoch=MultiLayerLogicGateNet.linear_temperature_anneal_factory(
            start_temperature, end_temperature, end_epoch=epochs
        ),
        checkpoint_path=None,
        device=device,
        check_grad=True,  # collected into history as values, not terminal tables
        peek=None,
    )

    ckpt = trainer.train(print_terminal=False)

    history = ckpt.training_history
    final_loss = history[-1].avg_loss if history else None

    # ---- continuous accuracy (thresholded soft outputs) ----
    test_set = (x_test.to(device), y_test.to(device))
    continuous_bit_acc = evaluate_accuracy(
        ckpt.model, test_set, threshold=0.5, device=device,
        sample_wise_comparison=False,
    )
    continuous_exact_acc = evaluate_accuracy(
        ckpt.model, test_set, threshold=0.5, device=device,
        sample_wise_comparison=True,
    )

    # ---- discrete Boolean accuracy (actual Boolean-mapped network) ----
    unwrapped: Any = getattr(ckpt.model, "module", ckpt.model)
    discrete_model: Any = unwrapped.to_discrete(threshold=0.5).to(device)
    discrete_model.eval()
    test_correct = 0
    test_total = x_test.shape[0]
    with torch.no_grad():
        for i in range(0, test_total, batch_size):
            xb = x_test[i : i + batch_size].to(device).to(torch.bool)
            yb = y_test[i : i + batch_size].to(device).to(torch.bool)
            preds = discrete_model(xb)
            test_correct += (preds == yb).all(dim=-1).sum().item()
    discrete_accuracy = test_correct / test_total if test_total else 0.0

    gap = continuous_exact_acc - discrete_accuracy

    # ---- gradient stats from the final epoch (machine-readable) ----
    gradient_stats: dict = {}
    if history and history[-1].grad_stats:
        for name, s in history[-1].grad_stats.items():
            gradient_stats[name] = {
                "mean_abs": _to_float(s.mean_abs.detach().cpu())
                if torch.is_tensor(s.mean_abs)
                else _to_float(s.mean_abs),
                "norm_normalized": _to_float(s.norm_normalized.detach().cpu())
                if torch.is_tensor(s.norm_normalized)
                else _to_float(s.norm_normalized),
                "max_abs": _to_float(s.max_abs.detach().cpu())
                if torch.is_tensor(s.max_abs)
                else _to_float(s.max_abs),
            }

    # ---- parameter stats: distance of mapped weights/biases from {0,1} ----
    parameter_stats: dict = {}
    with torch.no_grad():
        for i, layer in enumerate(unwrapped.expectation_layers):
            w = layer.actual_weight().detach().float().cpu()
            b = layer.actual_bias().detach().float().cpu()
            parameter_stats[f"layer_{i}"] = {
                "weight_mean": float(w.mean().item()),
                "weight_dist_to_binary": float(torch.minimum(w, 1.0 - w).mean().item()),
                "bias_mean": float(b.mean().item()),
                "bias_dist_to_binary": float(torch.minimum(b, 1.0 - b).mean().item()),
            }

    runtime_seconds = time.time() - t_start

    metrics = {
        "experiment_name": experiment_name,
        "seed": seed,
        "num_bits": num_bits,
        "epochs": epochs,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "layer_dims": list(layer_dims_full),
        "continuous_accuracy": continuous_bit_acc,
        "continuous_exact_accuracy": continuous_exact_acc,
        "discrete_accuracy": discrete_accuracy,
        "continuous_discrete_gap": gap,
        "final_loss": final_loss,
        "runtime_seconds": runtime_seconds,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "gradient_stats": gradient_stats,
        "activation_stats": None,  # not exposed cleanly yet; do not fake
        "parameter_stats": parameter_stats,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline BNN experiment")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--output", default=None,
                        help="Optional path to write metrics JSON")
    args = parser.parse_args()

    metrics = run_experiment(
        experiment_name=args.experiment_name,
        seed=args.seed,
        num_bits=args.num_bits,
        epochs=args.epochs,
        num_samples=args.num_samples,
    )
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
