"""Config-driven Boolean benchmark runner.

An experiment is defined by TaskConfig + ModelConfig + TrainingConfig + seed.
The neural-network mathematics (layers.py, models.py, ...) are used as-is;
this module only measures: continuous/discrete accuracy, truth-table
recovery, activation statistics (forward hooks), gradient summaries and
Boolean circuit-size metrics — all returned as plain Python values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from the repo root or research/: top-level source modules
# (models, trainer, ...) live in the repo root, harness modules here.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from boolean_tasks import FULL_TRUTH_TABLE, build_task
from eval_utils import evaluate_accuracy
from initializers import NormalInitWrapper
from models import MultiLayerLogicGateNet, ModernLogicGateNet
from prelude import Trainer, split_dataset, stop_on_epoch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "research" / "configs" / "baseline.json"

OPTIMIZERS = {"Adam": Adam}
LOSSES = {"MSELoss": nn.MSELoss}

# Layer-level gradient summary method (documented choice, see §9):
# layer_mean_abs_grad[i] = arithmetic mean of the final-epoch per-parameter
# `mean_abs` values over that layer's `weight` and `bias` entries.
# first_* = layer_0, last_* = layer_{num_layers-1}.
GRAD_SUMMARY_METHOD = (
    "layer_mean_abs_grad[i] = mean of final-epoch per-parameter mean_abs "
    "over that layer's weight and bias; first=layer_0, last=final layer"
)


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


def _to_float(value: Any) -> float | None:
    try:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() != 1:
                return None
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return None


def make_experiment_id(
    task_name: str,
    task_params: dict,
    seed: int,
    model_cfg: dict,
    training_cfg: dict,
    git_commit: str | None,
) -> str:
    payload = json.dumps(
        {
            "task": task_name,
            "task_params": task_params,
            "seed": seed,
            "model": model_cfg,
            "training": training_cfg,
            "git_commit": git_commit,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def init_from_spec(spec: dict):
    kind = spec.get("type", "normal")
    if kind != "normal":
        raise ValueError(f"Unsupported initializer type: {kind!r}")
    return NormalInitWrapper(float(spec.get("mean", 0.0)))


def build_model(model_cfg: dict, input_dim: int, output_dim: int) -> nn.Module:
    if model_cfg.get("architecture") == "modern":
        width = int(model_cfg.get("width", 64))
        blocks = int(model_cfg.get("num_residual_blocks", 2))
        count = 2 + 2 * blocks
        specs = model_cfg.get("layer_initializers")
        inits = [init_from_spec(s) for s in specs] if specs is not None else None
        if inits is not None and len(inits) != count:
            raise ValueError(f"modern model needs {count} explicit layer initializers")
        return ModernLogicGateNet(
            input_dim=input_dim, output_dim=output_dim, width=width,
            num_residual_blocks=blocks,
            init_temperature=float(model_cfg.get("init_temperature", 1.0)),
            learnable_tau=bool(model_cfg.get("learnable_tau", False)),
            use_softmax=bool(model_cfg.get("use_softmax", True)),
            layer_initializations=inits,
            bias_initialization=init_from_spec(model_cfg.get("bias_init", {"type": "normal", "mean": 1.0})),
            grad_scalar=bool(model_cfg.get("grad_scalar", True)),
            residual_enabled=bool(model_cfg.get("residual_enabled", True)),
        )
    hidden = [int(h) for h in model_cfg.get("hidden_dims", [64, 32, 32])]
    return MultiLayerLogicGateNet(
        input_dim=input_dim,
        layer_dims=tuple(hidden) + (output_dim,),
        init_temperature=float(model_cfg.get("init_temperature", 1.0)),
        shared_temperature=bool(model_cfg.get("shared_temperature", False)),
        learnable_tau=bool(model_cfg.get("learnable_tau", False)),
        use_softmax=bool(model_cfg.get("use_softmax", True)),
        even_initialization=init_from_spec(model_cfg.get("even_init", {"type": "normal", "mean": 0.5})),
        odd_initialization=init_from_spec(model_cfg.get("odd_init", {"type": "normal", "mean": 0.5})),
        bias_initialization=init_from_spec(model_cfg.get("bias_init", {"type": "normal", "mean": 1.0})),
        grad_scalar=bool(model_cfg.get("grad_scalar", True)),
    )


def compute_activation_stats(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
    eps: float = 1e-6,
) -> dict:
    """Per-layer activation stats via temporary forward hooks (no model change)."""
    layers: list = list(getattr(model, "expectation_layers"))
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(layers))}
    handles = []

    def _hook(idx: int):
        def fn(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            captured[idx].append(output.detach().float().cpu())
        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_hook(i)))
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = X[start:start + batch_size].to(device)
                model(xb)
    finally:
        for h in handles:
            h.remove()

    stats: dict[str, dict] = {}
    with torch.no_grad():
        for i in range(len(layers)):
            if not captured[i]:
                stats[f"layer_{i}"] = {}
                continue
            a = torch.cat([t.reshape(-1) for t in captured[i]])
            near_zero = (a <= 0.05).float().mean().item()
            near_one = (a >= 0.95).float().mean().item()
            middle = ((a > 0.25) & (a < 0.75)).float().mean().item()
            ac = a.clamp(eps, 1.0 - eps)
            entropy = (-(ac * ac.log2() + (1.0 - ac) * (1.0 - ac).log2())).mean().item()
            stats[f"layer_{i}"] = {
                "mean": float(a.mean().item()),
                "variance": float(a.var(unbiased=False).item()),
                "minimum": float(a.min().item()),
                "maximum": float(a.max().item()),
                "fraction_near_zero": near_zero,
                "fraction_near_one": near_one,
                "fraction_middle": middle,
                "mean_binary_entropy": entropy,
            }
    return stats


def compute_residual_diagnostics(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                                device: torch.device, batch_size: int = 512) -> dict:
    """Measure branch distributions, direct XOR gain, flips, and activation grads."""
    blocks = list(getattr(model, "blocks", []))
    if not blocks:
        return {}
    from collections import defaultdict
    captured: dict[int, dict[str, list[torch.Tensor]]] = {
        i: defaultdict(list) for i in range(len(blocks))
    }
    handles = []
    for i, block in enumerate(blocks):
        handles.extend([
            block.register_forward_pre_hook(lambda _m, args, i=i: captured[i]["input"].append(args[0].detach().cpu())),
            block.layer1.register_forward_hook(lambda _m, _a, out, i=i: captured[i]["h1"].append(out.detach().cpu())),
            block.layer2.register_forward_hook(lambda _m, _a, out, i=i: captured[i]["h2"].append(out.detach().cpu())),
            block.register_forward_hook(lambda _m, _a, out, i=i: captured[i]["output"].append(out.detach().cpu())),
        ])
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                model(X[start:start + batch_size].to(device))
    finally:
        for handle in handles:
            handle.remove()

    def distribution(t: torch.Tensor) -> dict:
        t = t.float().reshape(-1)
        tc = t.clamp(1e-6, 1 - 1e-6)
        return {
            "mean": t.mean().item(), "variance": t.var(unbiased=False).item(),
            "fraction_near_0": (t <= 0.05).float().mean().item(),
            "fraction_near_1": (t >= 0.95).float().mean().item(),
            "fraction_middle": ((t > 0.25) & (t < 0.75)).float().mean().item(),
            "binary_entropy": (-(tc * tc.log2() + (1-tc) * (1-tc).log2())).mean().item(),
        }

    result = {}
    for i, values in captured.items():
        merged = {k: torch.cat(v, dim=0) for k, v in values.items()}
        gain = (1 - 2 * merged["h2"]).abs().reshape(-1)
        sorted_gain = gain.sort().values
        def q(p: float) -> float:
            return float(sorted_gain[min(int(p * (len(sorted_gain)-1)), len(sorted_gain)-1)].item())
        hard_x = merged["input"] >= 0.5
        hard_f = merged["h2"] >= 0.5
        hard_y = merged["output"] >= 0.5
        result[f"block_{i}"] = {
            "activations": {k: distribution(v) for k, v in merged.items()},
            "direct_gain": {"mean": gain.mean().item(), "median": q(0.5), "p10": q(0.1),
                            "p25": q(0.25), "p75": q(0.75), "p90": q(0.9),
                            "fraction_lt_0_1": (gain < 0.1).float().mean().item(),
                            "fraction_lt_0_25": (gain < 0.25).float().mean().item(),
                            "fraction_gt_0_75": (gain > 0.75).float().mean().item(),
                            "fraction_gt_0_9": (gain > 0.9).float().mean().item()},
            "xor_flip_fraction": hard_f.float().mean().item(),
            "output_differs_from_input_fraction": (hard_x ^ hard_y).float().mean().item(),
        }

    # Actual activation gradient entering and leaving each block on a diagnostic
    # task batch, independent of the direct-derivative proxy above.
    grad_capture: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(len(blocks))}
    gh = []
    for i, block in enumerate(blocks):
        def pre(_m, args, i=i):
            v = args[0]
            if v.requires_grad:
                v.retain_grad(); grad_capture[i]["input"] = v
        def post(_m, _args, out, i=i):
            if out.requires_grad:
                out.retain_grad(); grad_capture[i]["output"] = out
        gh.extend([block.register_forward_pre_hook(pre), block.register_forward_hook(post)])
    model.zero_grad(set_to_none=True)
    xb = X[:min(len(X), batch_size)].to(device)
    yb = Y[:min(len(Y), batch_size)].to(device)
    pred = model(xb)
    nn.functional.mse_loss(pred, yb).backward()
    for h in gh:
        h.remove()
    for i in range(len(blocks)):
        vals = {}
        for side in ("input", "output"):
            act = grad_capture[i].get(side)
            g = act.grad if act is not None else None
            vals[side] = {"mean_abs_grad": float(g.abs().mean().item()),
                          "gradient_norm": float(g.norm().item())} if g is not None else None
        result[f"block_{i}"]["actual_activation_gradients"] = vals
    return result


def summarize_gradients(grad_stats: dict) -> dict:
    """Derive per-run layer-level gradient summary (method in GRAD_SUMMARY_METHOD)."""
    per_layer: dict[int, list[float]] = {}
    for name, s in grad_stats.items():
        # names look like "expectation_layers.{i}.weight|bias"
        try:
            idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        v = _to_float(getattr(s, "mean_abs", None))
        if v is not None:
            per_layer.setdefault(idx, []).append(v)
    layer_mean = {i: float(sum(v) / len(v)) for i, v in per_layer.items() if v}
    if not layer_mean:
        return {"method": GRAD_SUMMARY_METHOD}
    first = layer_mean[min(layer_mean)]
    last = layer_mean[max(layer_mean)]
    vals = list(layer_mean.values())
    return {
        "method": GRAD_SUMMARY_METHOD,
        "layer_mean_abs_grad": {f"layer_{i}": layer_mean[i] for i in sorted(layer_mean)},
        "first_layer_mean_abs_grad": first,
        "last_layer_mean_abs_grad": last,
        "first_last_gradient_ratio": first / (last + 1e-12),
        "minimum_layer_mean_abs_grad": min(vals),
        "maximum_layer_mean_abs_grad": max(vals),
    }


def summarize_modern_layer_gradients(grad_stats: dict) -> dict:
    """Group ModernLogicGateNet parameter gradients into its named graph layers."""
    groups: dict[str, list[float]] = {}
    for name, stat in grad_stats.items():
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        if name.startswith("stem."):
            label = "stem"
        elif name.startswith("head."):
            label = "head"
        elif name.startswith("blocks."):
            parts = name.split(".")
            label = f"block_{parts[1]}.{parts[2]}"
        else:
            continue
        value = _to_float(getattr(stat, "mean_abs", None))
        if value is not None:
            groups.setdefault(label, []).append(value)
    means = {k: sum(v)/len(v) for k, v in groups.items() if v}
    ordered = list(means.values())
    return {"layer_mean_abs_grad": means,
            "first_last_gradient_ratio": ordered[0]/(ordered[-1]+1e-12) if len(ordered) > 1 else None}


def compute_polarization_stats(model: nn.Module) -> dict:
    """Distance-to-binary and corner-fraction metrics on effective [0,1] params."""
    with torch.no_grad():
        w_all = []
        b_all = []
        for layer in getattr(model, "expectation_layers"):
            w_all.append(layer.actual_weight().detach().float().cpu().reshape(-1))
            b_all.append(layer.actual_bias().detach().float().cpu().reshape(-1))
        if not w_all:
            return {}
        w = torch.cat(w_all)
        b = torch.cat(b_all)

        def corner_stats(x: torch.Tensor) -> dict:
            return {
                "D": float(torch.minimum(x, 1.0 - x).mean().item()),
                "corner_01": float(((x <= 0.01) | (x >= 0.99)).float().mean().item()),
                "corner_05": float(((x <= 0.05) | (x >= 0.95)).float().mean().item()),
                "corner_10": float(((x <= 0.10) | (x >= 0.90)).float().mean().item()),
                "middle_40_60": float(((x > 0.4) & (x < 0.6)).float().mean().item()),
            }

        w_s = corner_stats(w)
        b_s = corner_stats(b)
        combined = torch.cat([w, b])
        c_s = corner_stats(combined)
        return {
            "D_w": w_s["D"],
            "D_b": b_s["D"],
            "D_combined": c_s["D"],
            "w_corner_01": w_s["corner_01"],
            "w_corner_05": w_s["corner_05"],
            "w_corner_10": w_s["corner_10"],
            "w_middle_40_60": w_s["middle_40_60"],
            "b_corner_01": b_s["corner_01"],
            "b_corner_05": b_s["corner_05"],
            "b_corner_10": b_s["corner_10"],
            "b_middle_40_60": b_s["middle_40_60"],
            "combined_corner_01": c_s["corner_01"],
            "combined_corner_05": c_s["corner_05"],
            "combined_corner_10": c_s["corner_10"],
        }


def get_temperatures(model: nn.Module) -> dict:
    with torch.no_grad():
        temps = getattr(model, "temperatures", None)
        if temps is None:
            return {}
        if isinstance(temps, (torch.Tensor, nn.Parameter)):
            v = float(temps.detach().cpu().item()) if temps.numel() == 1 else [float(x) for x in temps.detach().cpu().tolist()]
            return {"temperature_0": v} if isinstance(v, float) else {f"temperature_{i}": x for i, x in enumerate(v)}
        # list
        out = {}
        for i, t in enumerate(temps):
            try:
                out[f"temperature_{i}"] = float(t.detach().cpu().item())
            except Exception:
                out[f"temperature_{i}"] = None
        return out


def compute_circuit_stats(discrete_net: nn.Module) -> dict:
    """Boolean circuit-size metrics from the discretized network.

    weight=1 means the connection participates; for selected connections
    bias=1 means negated polarity. Bias values on unselected edges are
    NOT counted as literals.
    """
    total_selected = 0
    total_possible = 0
    total_negated = 0
    per_layer: dict[str, dict] = {}
    widths: list[int] = []
    layers: list = list(getattr(discrete_net, "expectation_layers"))
    for i, layer in enumerate(layers):
        w = layer.weight.detach().to(torch.bool)
        b = layer.bias.detach().to(torch.bool)
        selected = int(w.sum().item())
        possible = int(w.numel())
        negated = int((w & b).sum().item())
        total_selected += selected
        total_possible += possible
        total_negated += negated
        widths.append(int(w.shape[0]))
        per_layer[f"layer_{i}"] = {
            "selected_edges": selected,
            "possible_edges": possible,
            "selected_fraction": selected / possible if possible else 0.0,
            "negated_selected_edges": negated,
        }
    return {
        "number_of_layers": len(layers),
        "layer_widths": widths,
        "selected_edges": total_selected,
        "total_possible_edges": total_possible,
        "selected_edge_fraction": total_selected / total_possible if total_possible else 0.0,
        "negated_selected_edges": total_negated,
        "negated_selected_edge_fraction": total_negated / total_selected if total_selected else 0.0,
        "per_layer": per_layer,
    }


def _build_regularizer(training_cfg: dict):
    reg_spec = training_cfg.get("regularizer")
    if reg_spec is not None:
        rtype = reg_spec.get("type")
        if rtype == "regularization_factory2":
            from regularizers import regularization_factory2
            return regularization_factory2(
                disc_lambda=float(reg_spec.get("disc_lambda", 0.5)),
                tau_lambda=float(reg_spec.get("tau_lambda", 0.3)),
                patience=int(reg_spec.get("patience", 15)),
                min_err=float(reg_spec.get("min_err", 0.01)),
                isolate_on_plateau=bool(reg_spec.get("isolate_on_plateau", True)),
            )
        if rtype == "regularization_factory":
            from regularizers import regularization_factory
            return regularization_factory(
                l1_lambda=float(reg_spec.get("l1_lambda", 0.1)),
                disc_lambda=float(reg_spec.get("disc_lambda", 0.1)),
                tau_lambda=float(reg_spec.get("tau_lambda", 0.1)),
                patience=int(reg_spec.get("patience", 10)),
                min_err=float(reg_spec.get("min_err", 0.01)),
            )
        raise ValueError(f"Unknown regularizer type {rtype!r}")
    # legacy variance path
    vw = training_cfg.get("variance_weight")
    if vw is not None:
        vw = float(vw)
        def variance_regularizer(module: nn.Module):
            return vw * MultiLayerLogicGateNet.batch_variance_regularization(module)
        return variance_regularizer
    # no regularization
    return None


def _build_constraints(training_cfg: dict) -> list:
    constraints: list = [MultiLayerLogicGateNet.constraint]
    for c in training_cfg.get("constraints", []) or []:
        if isinstance(c, dict) and c.get("type") == "noise_on_plateau":
            from stopping_utils import call_fn_on_plateau
            constraints.append(
                call_fn_on_plateau(
                    MultiLayerLogicGateNet.noise_injector_factory(float(c.get("std", 0.3))),
                    patience=int(c.get("patience", 15)),
                    min_delta=float(c.get("min_delta", 0.01)),
                )
            )
        elif isinstance(c, str) and c == "clamp_weights_bias_temperature":
            pass  # already added
    return constraints


def _build_on_epoch(training_cfg: dict, epochs: int):
    # recovered regime has no annealing; legacy has start/end temperature
    if training_cfg.get("regularizer") is not None:
        return None
    st = training_cfg.get("start_temperature")
    et = training_cfg.get("end_temperature")
    if st is not None and et is not None:
        return MultiLayerLogicGateNet.linear_temperature_anneal_factory(
            float(st), float(et), end_epoch=epochs
        )
    return None


def run_single(
    task_name: str,
    task_params: dict,
    model_cfg: dict,
    training_cfg: dict,
    seed: int,
    discretization_threshold: float = 0.5,
    git_commit: str | None = None,
) -> dict:
    """Train one (task, seed) configuration; return machine-readable metrics."""
    t_start = time.time()
    seed_everything(seed)
    device = _resolve_device()
    git_commit = git_commit if git_commit is not None else get_git_commit()

    task = build_task(task_name, task_params, seed=seed)
    X_all, Y_all = task["X"], task["Y"]
    eval_mode = task["eval_mode"]

    if eval_mode == FULL_TRUTH_TABLE:
        x_train, y_train = X_all, Y_all
        x_eval, y_eval = X_all, Y_all
    else:
        train_ratio = float(task.get("train_ratio", 0.8))
        x_train, y_train, x_eval, y_eval = split_dataset(
            X_all, Y_all, train_ratio=train_ratio, shuffle=True
        )

    net = build_model(model_cfg, task["input_dim"], task["output_dim"]).to(device)
    # Preserve explicit train_ratio/threshold for reporting (even for full truth table).
    _train_ratio = float(task.get("train_ratio", 0.8)) if eval_mode != FULL_TRUTH_TABLE else 1.0

    epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 20)))
    batch_size = int(training_cfg.get("batch_size", 256))
    check_every = max(1, int(training_cfg.get("check_every", 10)))
    opt_name = training_cfg.get("optimizer", "Adam")
    loss_name = training_cfg.get("loss", "MSELoss")
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {opt_name!r}")
    if loss_name not in LOSSES:
        raise ValueError(f"Unsupported loss: {loss_name!r}")

    regularization_fn = _build_regularizer(training_cfg)
    constraints = _build_constraints(training_cfg)
    on_epoch = _build_on_epoch(training_cfg, epochs)

    # ---- Epoch-0 pre-training measurement (observational only, no grad) ----
    initial_state: dict[str, Any] = {}
    with torch.no_grad():
        pol0 = compute_polarization_stats(net)
        temps0 = get_temperatures(net)
        tau0 = {k.replace("temperature", "tau"): (1.0 / v if isinstance(v, float) and v else None) for k, v in temps0.items()}
        eval_set0 = (x_eval.to(device), y_eval.to(device))
        cont_bit0 = evaluate_accuracy(net, eval_set0, threshold=0.5, device=device, sample_wise_comparison=False)
        cont_exact0 = evaluate_accuracy(net, eval_set0, threshold=0.5, device=device, sample_wise_comparison=True)
        # discrete clone of initial model
        try:
            disc0: Any = net.to_discrete(threshold=discretization_threshold).to(device)
            disc0.eval()
            n0 = x_eval.shape[0]
            bit_c0 = 0; bit_t0 = 0; exact_c0 = 0
            with torch.no_grad():
                for i in range(0, n0, batch_size):
                    xb = x_eval[i:i+batch_size].to(device).to(torch.bool)
                    yb = y_eval[i:i+batch_size].to(device).to(torch.bool)
                    preds = disc0(xb).to(torch.bool)
                    bit_c0 += (preds == yb).sum().item()
                    bit_t0 += preds.numel()
                    exact_c0 += (preds == yb).all(dim=-1).sum().item()
            disc_bit0 = bit_c0 / bit_t0 if bit_t0 else 0.0
            disc_exact0 = exact_c0 / n0 if n0 else 0.0
        except Exception:
            disc_bit0 = disc_exact0 = None
        initial_state = {
            "epoch": 0,
            "train_ratio": _train_ratio,
            "discretization_threshold": discretization_threshold,
            "D_w": pol0.get("D_w"),
            "D_b": pol0.get("D_b"),
            "D_combined": pol0.get("D_combined"),
            "polarization": pol0,
            "temperatures": temps0,
            "taus": tau0,
            "continuous_bit_accuracy": cont_bit0,
            "continuous_exact_accuracy": cont_exact0,
            "discrete_bit_accuracy": disc_bit0,
            "discrete_exact_accuracy": disc_exact0,
        }

    # Per-epoch trajectory (instrumentation, no grad).
    trajectory: list[dict] = []
    discrete_eval_epochs = {1, 5, 10, 20, 50, 100, 150, 200, 250, 300, epochs}
    discrete_eval_epochs.update(range(check_every, epochs + 1, check_every))
    # Also include final epoch always.

    def epoch_callback(state: dict) -> None:
        epoch = state["epoch"]
        avg_loss = state["avg_loss"]
        avg_reg = state["avg_regularization"]
        avg_err = state["avg_error"]
        grad_stats = state["grad_stats"]
        model = state["model"]
        with torch.no_grad():
            pol = compute_polarization_stats(model)
            ready = bool(pol.get("D_w", float("inf")) <= 0.01 and
                         pol.get("D_b", float("inf")) <= 0.01 and
                         pol.get("w_corner_05", 0.0) >= 0.95 and
                         pol.get("b_corner_05", 0.0) >= 0.95)
            temps = get_temperatures(model)
            # gradient ratio from grad_stats
            summary = summarize_gradients(grad_stats)
            # continuous accuracy every 10 epochs or at eval epochs to limit cost
            do_cont = (epoch in discrete_eval_epochs) or (epoch == 1)
            cont_bit = cont_exact = None
            if do_cont:
                eval_set = (x_eval.to(device), y_eval.to(device))
                cont_bit = evaluate_accuracy(
                    getattr(model, "module", model) if hasattr(model, "module") else model,  # type: ignore
                    eval_set, threshold=0.5, device=device, sample_wise_comparison=False
                )
                cont_exact = evaluate_accuracy(
                    getattr(model, "module", model) if hasattr(model, "module") else model,  # type: ignore
                    eval_set, threshold=0.5, device=device, sample_wise_comparison=True
                )
            # discrete clone evaluation (never mutate training model)
            disc_bit = disc_exact = None
            if epoch in discrete_eval_epochs:
                try:
                    # to_discrete is on the continuous model; use its device copy
                    discrete_clone: Any = model.to_discrete(threshold=discretization_threshold).to(device)
                    discrete_clone.eval()
                    n_eval = x_eval.shape[0]
                    bit_correct = 0
                    bit_total = 0
                    exact_correct = 0
                    with torch.no_grad():
                        for i in range(0, n_eval, batch_size):
                            xb = x_eval[i:i + batch_size].to(device).to(torch.bool)
                            yb = y_eval[i:i + batch_size].to(device).to(torch.bool)
                            preds = discrete_clone(xb).to(torch.bool)
                            bit_correct += (preds == yb).sum().item()
                            bit_total += preds.numel()
                            exact_correct += (preds == yb).all(dim=-1).sum().item()
                    disc_bit = bit_correct / bit_total if bit_total else 0.0
                    disc_exact = exact_correct / n_eval if n_eval else 0.0
                except Exception:
                    pass
            entry: dict[str, Any] = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "avg_error": avg_err,
                "avg_regularization": avg_reg,
                "task_loss": avg_loss - avg_reg,
                "regularization_loss": avg_reg,
                "total_loss": avg_loss,
                "D_w": pol.get("D_w"),
                "D_b": pol.get("D_b"),
                "D_combined": pol.get("D_combined"),
                "polarization": pol,
                "discretization_ready": ready,
                "diagnostic_discrete_status": "ready_evaluation" if ready else "diagnostic_only",
                "temperatures": temps,
                "taus": {k.replace("temperature", "tau"): (1.0 / v if isinstance(v, float) and v else None) for k, v in temps.items()},
                "gradient_summary": summary,
            }
            if cont_bit is not None:
                entry["continuous_bit_accuracy"] = cont_bit
                entry["continuous_exact_accuracy"] = cont_exact
            if disc_bit is not None:
                entry["discrete_bit_accuracy"] = disc_bit
                entry["discrete_exact_accuracy"] = disc_exact
            trajectory.append(entry)

    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=stop_on_epoch(epochs),
        batch_size=batch_size,
        model=net,
        loss_fn=LOSSES[loss_name](),
        optimizer_cls=OPTIMIZERS[opt_name],
        optimizer_kwargs={"lr": float(training_cfg.get("lr", 0.05))},
        regularization_fn=regularization_fn,
        lr_scheduler_factory=None,
        constraints=constraints,
        on_epoch=on_epoch,
        checkpoint_path=None,
        device=device,
        check_grad=True,
        peek=None,
        epoch_callback=epoch_callback,
    )

    ckpt = trainer.train(print_terminal=False)
    history = ckpt.training_history
    epochs_completed = len(history)
    final_loss = history[-1].avg_loss if history else None

    eval_set = (x_eval.to(device), y_eval.to(device))
    cont_bit = evaluate_accuracy(ckpt.model, eval_set, threshold=0.5,
                                 device=device, sample_wise_comparison=False)
    cont_exact = evaluate_accuracy(ckpt.model, eval_set, threshold=0.5,
                                   device=device, sample_wise_comparison=True)

    unwrapped: Any = getattr(ckpt.model, "module", ckpt.model)
    discrete_model: Any = unwrapped.to_discrete(
        threshold=discretization_threshold).to(device)
    discrete_model.eval()
    n_eval = x_eval.shape[0]
    bit_correct = 0
    bit_total = 0
    exact_correct = 0
    with torch.no_grad():
        for i in range(0, n_eval, batch_size):
            xb = x_eval[i:i + batch_size].to(device).to(torch.bool)
            yb = y_eval[i:i + batch_size].to(device).to(torch.bool)
            preds = discrete_model(xb).to(torch.bool)
            bit_correct += (preds == yb).sum().item()
            bit_total += preds.numel()
            exact_correct += (preds == yb).all(dim=-1).sum().item()
    disc_bit = bit_correct / bit_total if bit_total else 0.0
    disc_exact = exact_correct / n_eval if n_eval else 0.0
    function_recovery = bool(n_eval > 0 and exact_correct == n_eval)

    gradient_stats: dict = {}
    if history and history[-1].grad_stats:
        for name, s in history[-1].grad_stats.items():
            gradient_stats[name] = {
                "mean_abs": _to_float(s.mean_abs),
                "norm_normalized": _to_float(s.norm_normalized),
                "max_abs": _to_float(s.max_abs),
            }
    gradient_summary = summarize_gradients(
        history[-1].grad_stats if history else {})

    activation_stats = compute_activation_stats(unwrapped, x_eval, device)
    residual_diagnostics = compute_residual_diagnostics(unwrapped, x_eval, y_eval, device)

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
    circuit_stats = compute_circuit_stats(discrete_model)

    runtime_seconds = time.time() - t_start
    resolved_model_cfg = dict(model_cfg)
    resolved_model_cfg["layer_dims"] = (
        [int(h) for h in model_cfg.get("hidden_dims", [])] + [task["output_dim"]]
    )
    experiment_id = make_experiment_id(
        task_name, task["task_params"], seed, resolved_model_cfg,
        training_cfg, git_commit)

    metrics = {
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "task_name": task_name,
        "task_parameters": task["task_params"],
        "experiment_name": task_name,
        "seed": seed,
        "model_config": resolved_model_cfg,
        "training_config": dict(training_cfg),
        "eval_mode": eval_mode,
        "input_dim": task["input_dim"],
        "output_dim": task["output_dim"],
        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,
        # Generic accuracy keys (bit-level + whole-sample exact).
        "continuous_accuracy": cont_bit,
        "continuous_bit_accuracy": cont_bit,
        "continuous_exact_accuracy": cont_exact,
        "discrete_bit_accuracy": disc_bit,
        "discrete_accuracy": disc_exact,
        "discrete_exact_accuracy": disc_exact,
        "discrete_function_recovery": function_recovery,
        "continuous_discrete_gap": cont_exact - disc_exact,
        # Explicit truth-table recovery keys for full-table tasks.
        "truth_table_continuous_bit_accuracy": cont_bit if eval_mode == FULL_TRUTH_TABLE else None,
        "truth_table_continuous_exact_accuracy": cont_exact if eval_mode == FULL_TRUTH_TABLE else None,
        "truth_table_discrete_bit_accuracy": disc_bit if eval_mode == FULL_TRUTH_TABLE else None,
        "truth_table_discrete_exact_accuracy": disc_exact if eval_mode == FULL_TRUTH_TABLE else None,
        "truth_table_exact_match": function_recovery if eval_mode == FULL_TRUTH_TABLE else None,
        "final_loss": final_loss,
        "runtime_seconds": runtime_seconds,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gradient_stats": gradient_stats,
        "gradient_summary": gradient_summary,
        "named_layer_gradient_summary": summarize_modern_layer_gradients(
            history[-1].grad_stats if history else {}),
        "activation_stats": activation_stats,
        "residual_diagnostics": residual_diagnostics,
        "parameter_stats": parameter_stats,
        "circuit_stats": circuit_stats,
        "trajectory": trajectory,
        "initial_state": initial_state,
        "final_polarization": compute_polarization_stats(unwrapped),
        "discretization_ready": bool(
            compute_polarization_stats(unwrapped).get("D_w", float("inf")) <= 0.01 and
            compute_polarization_stats(unwrapped).get("D_b", float("inf")) <= 0.01 and
            compute_polarization_stats(unwrapped).get("w_corner_05", 0.0) >= 0.95 and
            compute_polarization_stats(unwrapped).get("b_corner_05", 0.0) >= 0.95
        ),
        "train_ratio": _train_ratio,
        "discretization_threshold": discretization_threshold,
    }
    return metrics


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
    """Legacy entry point: original bitwise-XOR baseline (values unchanged)."""
    return run_single(
        task_name="bitwise_xor",
        task_params={"bits": num_bits, "num_samples": num_samples, "train_ratio": 0.8},
        model_cfg={
            "hidden_dims": list(layer_dims),
            "init_temperature": 1.0,
            "shared_temperature": False,
            "learnable_tau": False,
            "use_softmax": True,
            "even_init": {"type": "normal", "mean": 0.5},
            "odd_init": {"type": "normal", "mean": 0.5},
            "bias_init": {"type": "normal", "mean": 1.0},
            "grad_scalar": True,
        },
        training_cfg={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "optimizer_kwargs": {"lr": lr},
            "loss": "MSELoss",
            "lr": lr,
            "start_temperature": start_temperature,
            "end_temperature": end_temperature,
            "variance_weight": variance_weight,
        },
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Boolean benchmark task")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Baseline JSON config with task/model/training")
    parser.add_argument("--task", default=None,
                        help="Task name override (uses task params from --task-params)")
    parser.add_argument("--task-params", default=None,
                        help="JSON object overriding task params")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    task_cfg = cfg["task"]
    task_name = args.task or task_cfg["task"]
    task_params = dict(task_cfg.get("params", {}))
    if args.task_params:
        task_params.update(json.loads(args.task_params))
    training_cfg = dict(cfg.get("training", {}))
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else int(task_cfg.get("seed", 0))

    metrics = run_single(
        task_name=task_name,
        task_params=task_params,
        model_cfg=cfg.get("model", {}),
        training_cfg=training_cfg,
        seed=seed,
        discretization_threshold=float(cfg.get("discretization_threshold", 0.5)),
    )
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
