"""Stage B2: small compositional Boolean truth-table comparison."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402

OUT = ROOT / "research" / "operator_results"
FIG = ROOT / "research" / "figures"
OPS = ["hardmax", "lehmer_p1", "lehmer_p2", "log_hazard", "probabilistic_or", "softmax_value_a16"]
TASKS = ["xor2", "majority5", "parity4", "full_adder", "multiplexer4"]
MILESTONES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


def metrics(out: torch.Tensor, target: torch.Tensor) -> dict:
    pred = out >= .5
    correct = pred == (target >= .5)
    return {"mse": float((out-target).square().mean()),
            "bit_accuracy": float(correct.float().mean()),
            "exact_accuracy": float(correct.all(dim=-1).float().mean())}


def evaluate(model, x, y, threshold=.5):
    with torch.no_grad():
        soft = model(x)
        hard = model.forward_hard(x)
        boolean = model.to_discrete(threshold)(x.bool()).float()
    return {"continuous": metrics(soft, y), "hard": metrics(hard, y), "boolean": metrics(boolean, y)}


def gate_stats(model):
    rows = []
    for i, layer in enumerate(model.expectation_layers):
        g = layer.effective_gate().detach()
        b = layer.actual_bias().detach()
        rows.append({"layer": i, "gate_mean": float(g.mean()), "gate_d": float(torch.minimum(g,1-g).mean()),
                     "gate_margin_mean": float((g-.5).abs().mean()),
                     "gate_margin_lt_01": float(((g-.5).abs()<.1).float().mean()),
                     "gate_margin_lt_05": float(((g-.5).abs()<.05).float().mean()),
                     "gate_sigmoid_deriv_mean": float((g*(1-g)).mean()),
                     "gate_sat_lt_1e3": float((g*(1-g)<1e-3).float().mean()),
                     "bias_d": float(torch.minimum(b,1-b).mean()),
                     "bias_margin_mean": float((b-.5).abs().mean())})
    return rows


def threshold_stability(model, x, y):
    result = {}
    for t in THRESHOLDS:
        result[str(t)] = evaluate(model, x, y, t)["boolean"]
    return result


def run(epochs=1000, width=16, blocks=1, seeds=(0,1,2)):
    results = []
    for task_name in TASKS:
        task_key, task_params = {"majority5": ("majority", {"n": 5}), "parity4": ("parity", {"n": 4})}.get(task_name, (task_name, {}))
        task = build_task(task_key, task_params)
        x, y = task["X"], task["Y"]
        for op in OPS:
            for seed in seeds:
                torch.manual_seed(seed)
                bias_init = lambda t: nn.init.normal_(t, mean=.5, std=.1)
                model = SigmoidOrModernLogicGateNet(x.shape[1], y.shape[1], width=width,
                    num_residual_blocks=blocks, or_operator=op,
                    bias_initialization=bias_init)
                optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                best = None; milestones = {}; trajectory = []
                for epoch in range(epochs + 1):
                    out = model(x); loss = (out-y).square().mean()
                    optimizer.zero_grad(); loss.backward()
                    grad_norm = float(torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])))
                    with torch.no_grad():
                        ev = evaluate(model, x, y)
                        rec = {"epoch": epoch, **ev, "gradient_norm": grad_norm, "gate_stats": gate_stats(model)}
                        if epoch == 0 or epoch % 25 == 0 or epoch == epochs:
                            trajectory.append(rec)
                        for m in MILESTONES:
                            if str(m) not in milestones and ev["continuous"]["mse"] < m:
                                milestones[str(m)] = rec
                        if best is None or ev["continuous"]["mse"] < best["continuous"]["mse"]:
                            best = {"epoch": epoch, **ev, "gate_stats": gate_stats(model), "state": copy.deepcopy(model.state_dict())}
                    if epoch == epochs: break
                    optimizer.step()
                model.load_state_dict(best["state"])
                best.pop("state")
                result = {"task": task_name, "operator": op, "seed": seed, "width": width,
                          "num_residual_blocks": blocks, "best": best,
                          "milestones": milestones, "threshold_stability": threshold_stability(model, x, y),
                          "trajectory": trajectory}
                ckdir = OUT / "stage_b2_checkpoints"; ckdir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckdir / f"B2_{task_name}_{op}_seed{seed}.pt")
                results.append(result)
                print(task_name, op, seed, best["continuous"]["mse"], best["boolean"]["exact_accuracy"], flush=True)
    return results


def plot(results):
    for task in TASKS:
        plt.figure(figsize=(9,6))
        for op in OPS:
            runs = [r for r in results if r["task"] == task and r["operator"] == op]
            for r in runs:
                pts = [(z["continuous"]["mse"], 1-z["boolean"]["exact_accuracy"]) for z in r["trajectory"] if z["continuous"]["mse"] > 0]
                if pts: plt.plot([p[0] for p in pts], [p[1] for p in pts], alpha=.4, label=op if r["seed"]==0 else None)
        plt.xscale("log"); plt.yscale("log"); plt.xlabel("continuous MSE"); plt.ylabel("Boolean truth-table error"); plt.title(f"B2 consistency: {task}"); plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / f"stage_b2_consistency_{task}.png", dpi=160); plt.close()


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=1000); p.add_argument("--width", type=int, default=16); p.add_argument("--blocks", type=int, default=1)
    args = p.parse_args(); OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    results = run(args.epochs, args.width, args.blocks)
    path = OUT / "stage_b2_results.json"
    with path.open("w") as f: json.dump({"experiment":"B2", "operators":OPS, "tasks":TASKS, "epochs":args.epochs, "width":args.width, "blocks":args.blocks, "results":results}, f, indent=2)
    plot(results); print(json.dumps({"output":str(path), "runs":len(results)}, indent=2))


if __name__ == "__main__": main()
