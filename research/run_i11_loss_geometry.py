"""I11: loss geometry under deterministic repeated-label conflicts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-i11")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.run_initialization_i2 import model_for  # noqa: E402
from research.run_i8_boolean_ste import metrics as old_metrics  # noqa: E402

OUT = ROOT / "research/operator_results"
CKOUT = OUT / "i11_loss_geometry_checkpoints"
FIG = ROOT / "research/figures"
LOSSES = ("MSE", "BCE", "POWER_1_5", "POWER_1_25", "MAE")
ETAS = (0.0, 0.1, 0.2, 0.4)
REPEATS = 10
EVAL_STEPS = [0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]


def make_model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2")


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def scalar_optimum(q, loss, alpha=None):
    if loss in ("MSE", "BCE"):
        return q
    if loss == "MAE":
        return 1.0 if q > .5 else 0.0 if q < .5 else None
    power = 1.0 / (float(alpha) - 1.0)
    a, b = q**power, (1.0 - q)**power
    return a / (a + b)


def scalar_risk(p, q, loss, alpha=None):
    p = torch.as_tensor(p, dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    if loss == "MSE":
        return q * (1 - p) ** 2 + (1 - q) * p**2
    if loss == "BCE":
        return q * (-torch.log(p)) + (1 - q) * (-torch.log1p(-p))
    if loss == "MAE":
        return q * (1 - p) + (1 - q) * p
    return q * (1 - p) ** alpha + (1 - q) * p**alpha


def scalar_optimize(q, loss, alpha=None, steps=4000):
    torch.manual_seed(1000 + int(round(q * 100)) + sum(ord(c) for c in loss))
    z = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([z], lr=.05)
    for _ in range(steps):
        p = torch.sigmoid(z)
        if loss == "BCE":
            value = q * (-torch.log(p)) + (1 - q) * (-torch.log1p(-p))
        elif loss == "MSE":
            value = q * (1 - p) ** 2 + (1 - q) * p**2
        elif loss == "MAE":
            value = q * (1 - p).abs() + (1 - q) * p.abs()
        else:
            value = q * (1 - p).abs() ** alpha + (1 - q) * p.abs() ** alpha
        opt.zero_grad()
        value.backward()
        opt.step()
    return float(torch.sigmoid(z).detach())


def analytic_tables():
    qs = (.50, .55, .60, .70, .80, .90, .95)
    rows = []
    for q in qs:
        rows.append({"q": q, "MSE": scalar_optimum(q, "MSE"), "BCE": scalar_optimum(q, "BCE"),
                     "POWER_1_5": scalar_optimum(q, "POWER_1_5", 1.5),
                     "POWER_1_25": scalar_optimum(q, "POWER_1_25", 1.25),
                     "MAE": scalar_optimum(q, "MAE")})
    return rows


def gradient_geometry():
    ps = torch.linspace(.001, .999, 1000, dtype=torch.float64)
    result = {"p": ps.tolist(), "y1": {}, "y0": {}}
    for name in LOSSES:
        alpha = 1.5 if name == "POWER_1_5" else 1.25 if name == "POWER_1_25" else None
        if name == "MSE":
            g1, g0 = 2 * (ps - 1), 2 * ps
        elif name == "BCE":
            g1, g0 = -1 / ps, 1 / (1 - ps)
        elif name == "MAE":
            g1, g0 = -torch.ones_like(ps), torch.ones_like(ps)
        else:
            g1 = -alpha * (1 - ps) ** (alpha - 1)
            g0 = alpha * ps ** (alpha - 1)
        result["y1"][name] = g1.abs().tolist()
        result["y0"][name] = g0.abs().tolist()
    return result


def step_geometry():
    ps = torch.linspace(.001, .999, 1000, dtype=torch.float64)
    result = {"p": ps.tolist()}
    for tau in (.05, .1):
        z = (torch.abs(ps - 1) - .5) / tau
        value = torch.sigmoid(z)
        magnitude = value * (1 - value) / tau
        result[f"tau_{tau:.2f}"] = {"loss_y1": value.tolist(), "abs_gradient_y1": magnitude.tolist()}
    return result


def noisy_dataset(clean_x, clean_y, eta, seed):
    flips = int(round(REPEATS * eta))
    generator = torch.Generator().manual_seed(100000 + seed * 100 + int(round(eta * 100)))
    x = clean_x.repeat_interleave(REPEATS, dim=0)
    y = clean_y.repeat_interleave(REPEATS, dim=0)
    for row in range(clean_y.shape[0]):
        for bit in range(clean_y.shape[1]):
            if flips:
                selected = torch.randperm(REPEATS, generator=generator)[:flips]
                y[row * REPEATS + selected, bit] = 1.0 - y[row * REPEATS + selected, bit]
    counts_one = y.reshape(clean_y.shape[0], REPEATS, clean_y.shape[1]).sum(dim=1)
    counts_zero = REPEATS - counts_one
    return x, y, counts_one, counts_zero


def weighted_loss(output, clean_y, counts_one, counts_zero, loss_name):
    eps = 1e-7
    if loss_name == "BCE":
        p = output.clamp(eps, 1 - eps)
        values = counts_one * (-torch.log(p)) + counts_zero * (-torch.log1p(-p))
    elif loss_name == "MSE":
        values = counts_one * (1 - output).square() + counts_zero * output.square()
    elif loss_name == "MAE":
        values = counts_one * (1 - output).abs() + counts_zero * output.abs()
    else:
        alpha = 1.5 if loss_name == "POWER_1_5" else 1.25
        values = counts_one * (1 - output).abs().pow(alpha) + counts_zero * output.abs().pow(alpha)
    return values.mean() / REPEATS


def endpoint_stats(output, target):
    distance = torch.minimum(output, 1 - output)
    confidence = torch.where(target > .5, output, 1 - output)
    return {
        "mean_D_endpoint": float(distance.mean()),
        "median_D_endpoint": float(distance.median()),
        "p95_D_endpoint": float(torch.quantile(distance, .95)),
        "fraction_D_endpoint_lt_.01": float((distance < .01).float().mean()),
        "fraction_D_endpoint_lt_.05": float((distance < .05).float().mean()),
        "fraction_abs_p_minus_.5_lt_.05": float((distance > .45).float().mean()),
        "fraction_abs_p_minus_.5_lt_.10": float((distance > .40).float().mean()),
        "mean_clean_confidence": float(confidence.mean()),
        "minimum_clean_confidence": float(confidence.min()),
        "p05_clean_confidence": float(torch.quantile(confidence, .05)),
    }


def clean_metrics(net, x, y):
    with torch.no_grad():
        output = net(x)
        base = old_metrics(net, x, y)
        return {
            **base,
            "endpoint": endpoint_stats(output, y),
            "finite": bool(torch.isfinite(output).all()),
        }


def save_checkpoint(path, state, seed, loss_name, eta, step, net, x, y, initial_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    check = make_model()
    check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    got = clean_metrics(check, x, y)
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reload_verified": True, "reload_metrics": got, "seed": seed,
        "loss": loss_name, "eta": eta, "step": step,
        "initial_state_sha256": initial_hash,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def run_one(seed, loss_name, eta, clean_x, clean_y, steps=3000):
    base = model_for("I2-B", seed, torch.device("cpu"))
    initial_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    initial_hash = state_hash(initial_state)
    net = make_model()
    net.load_state_dict(initial_state)
    train_x, train_y, counts_one, counts_zero = noisy_dataset(clean_x, clean_y, eta, seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.0)
    records, states = [], []
    best_mse = (float("inf"), None, None)
    best_bool = (-1.0, None, None)
    for step in range(steps + 1):
        if step in EVAL_STEPS:
            clean = clean_metrics(net, clean_x, clean_y)
            record = {"step": step, "clean": clean}
            records.append(record)
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if clean["continuous"]["mse"] < best_mse[0]:
                best_mse = (clean["continuous"]["mse"], step, state)
            bool_score = clean["boolean"]["exact_accuracy"]
            if bool_score > best_bool[0]:
                best_bool = (bool_score, step, state)
        if step == steps:
            break
        # The weighted form is exactly the expanded repeated-label objective,
        # evaluated once per unique input row.  Pair outputs with clean_x,
        # while counts_one/counts_zero carry the ten repeated labels.
        output = net(clean_x)
        loss = weighted_loss(output, clean_y, counts_one, counts_zero, loss_name)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    checkpoint_meta = {}
    for kind, item in (("best_clean_continuous", best_mse), ("best_clean_boolean", best_bool), ("final", (0, steps, final_state))):
        checkpoint_meta[kind] = save_checkpoint(CKOUT / f"I11_seed{seed}_{loss_name}_eta{eta:.1f}_{kind}.pt", item[2], seed, loss_name, eta, item[1], net, clean_x, clean_y, initial_hash)
    return {
        "seed": seed, "loss": loss_name, "eta": eta, "steps": steps,
        "optimizer_steps": steps, "training_examples": int(train_x.shape[0]),
        "repeated_label_weighted_equivalent": True,
        "initial_state_sha256": initial_hash,
        "flip_count_per_label": int(round(REPEATS * eta)),
        "trajectory": records, "checkpoints": checkpoint_meta,
    }


def plot_analytic(table, gradients):
    FIG.mkdir(parents=True, exist_ok=True)
    q = torch.linspace(.001, .999, 500).tolist()
    plt.figure(figsize=(7, 5))
    for name, alpha in (("MSE", 2.0), ("BCE", 2.0), ("POWER_1_5", 1.5), ("POWER_1_25", 1.25), ("MAE", 1.0)):
        vals = []
        for value in q:
            if name == "MAE": vals.append(1.0 if value > .5 else 0.0)
            elif name in ("MSE", "BCE"): vals.append(value)
            else: vals.append(scalar_optimum(value, name, alpha))
        plt.plot(q, vals, label=name)
    plt.plot(q, q, "k--", alpha=.4)
    plt.xlabel("q = P(y=1|x)"); plt.ylabel("population optimum p*"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_population_optima.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 5))
    for name, vals in gradients["y1"].items(): plt.plot(gradients["p"], vals, label=name)
    plt.yscale("log"); plt.xlabel("p"); plt.ylabel("|dL/dp| for y=1"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_loss_gradients.png", dpi=150); plt.close()
    step = step_geometry()
    plt.figure(figsize=(7, 5))
    for tau in ("tau_0.05", "tau_0.10"):
        plt.plot(step["p"], step[tau]["loss_y1"], label=f"tau={tau.split('_')[1]} loss")
        plt.plot(step["p"], step[tau]["abs_gradient_y1"], linestyle="--", label=f"tau={tau.split('_')[1]} |gradient|")
    plt.xlabel("p"); plt.ylabel("step-loss / gradient for y=1"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_step_loss_analysis.png", dpi=150); plt.close()


def plot_results(runs):
    if not runs: return
    groups = {}
    for run in runs:
        final = run["trajectory"][-1]["clean"]
        groups.setdefault((run["loss"], run["eta"]), []).append(final)
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        xs, ys = [], []
        for eta in ETAS:
            vals = [groups[(loss, eta)][0]["boolean"]["exact_accuracy"] for _ in [0] if (loss, eta) in groups]
            if vals: xs.append(eta); ys.append(vals[0])
        if xs: plt.plot(xs, ys, marker="o", label=loss)
    plt.xlabel("label-flip fraction"); plt.ylabel("clean Boolean exact accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_noise_boolean_accuracy.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        xs, ys = [], []
        for eta in ETAS:
            vals = [groups[(loss, eta)][0]["endpoint"]["mean_D_endpoint"] for _ in [0] if (loss, eta) in groups]
            if vals: xs.append(eta); ys.append(vals[0])
        if xs: plt.plot(xs, ys, marker="o", label=loss)
    plt.xlabel("label-flip fraction"); plt.ylabel("mean endpoint distance"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_endpoint_distance.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        xs, ys = [], []
        for eta in ETAS:
            vals = [groups[(loss, eta)][0]["endpoint"]["mean_clean_confidence"] for _ in [0] if (loss, eta) in groups]
            if vals: xs.append(eta); ys.append(vals[0])
        if xs: plt.plot(xs, ys, marker="o", label=loss)
    plt.xlabel("label-flip fraction"); plt.ylabel("mean clean confidence"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i11_clean_confidence.png", dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="3", help="comma-separated seeds; stage 1 defaults to seed 3")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--loss", choices=LOSSES)
    parser.add_argument("--eta", type=float, choices=ETAS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default=str(OUT / "i11_loss_geometry_results.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    clean_x, clean_y = task["X"], task["Y"]
    table = analytic_tables()
    gradients = gradient_geometry()
    step = step_geometry()
    plot_analytic(table, gradients)
    scalar = [{"q": q, "loss": loss, "analytic": scalar_optimum(q, loss, 1.5 if loss == "POWER_1_5" else 1.25 if loss == "POWER_1_25" else None), "optimized": scalar_optimize(q, loss, 1.5 if loss == "POWER_1_5" else 1.25 if loss == "POWER_1_25" else None)} for q in (.5, .6, .8, .9) for loss in LOSSES]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    runs = []
    selected_losses = (args.loss,) if args.loss else LOSSES
    selected_etas = (args.eta,) if args.eta is not None else ETAS
    for seed in seeds:
        hashes = set()
        for loss in selected_losses:
            for eta in selected_etas:
                run = run_one(seed, loss, eta, clean_x, clean_y, args.steps)
                hashes.add(run["initial_state_sha256"])
                runs.append(run)
                print(seed, loss, eta, run["trajectory"][-1]["clean"]["boolean"]["exact_accuracy"], flush=True)
        if len(hashes) != 1:
            raise RuntimeError(f"paired initialization mismatch for seed {seed}: {hashes}")
    plot_results(runs)
    bad_label = {}
    p = .99
    for loss in LOSSES:
        alpha = 1.5 if loss == "POWER_1_5" else 1.25 if loss == "POWER_1_25" else None
        if loss == "MSE": good, bad = abs(2 * (p - 1)), abs(2 * p)
        elif loss == "BCE": good, bad = 1 / p, 1 / (1 - p)
        elif loss == "MAE": good = bad = 1.0
        else: good, bad = alpha * (1 - p) ** (alpha - 1), alpha * p ** (alpha - 1)
        bad_label[loss] = {"correct_label_gradient": good, "corrupted_label_gradient": bad, "ratio": bad / good}
    payload = {
        "experiment": "I11-loss-geometry",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "stage": "seed3" if seeds == [3] else "replication",
        "seeds": seeds,
        "architecture": {"input_dim": 8, "width": 64, "residual_blocks": 2, "operator": "lehmer_p2"},
        "initializer": "I2-B meanfield sigma2 + BIAS_ONE",
        "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": args.steps, "repeats": REPEATS,
                      "weighted_repeated_objective": True, "losses": LOSSES, "etas": ETAS},
        "analytic_population_optima": table,
        "scalar_optimization": scalar,
        "gradient_geometry": gradients,
        "step_loss_analysis": step,
        "bad_label_influence_at_p_.99": bad_label,
        "runs": runs,
    }
    path = Path(args.output)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
