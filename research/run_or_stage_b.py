"""Stage B0/B1 CPU experiments for OR operator discretization consistency."""

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
from layers import SigmoidOrLogicLayer  # noqa: E402
from discrete_logic_net import DiscreteOrNorGateLayer  # noqa: E402

OUT = ROOT / "research" / "operator_results"
FIG = ROOT / "research" / "figures"
OPS = ["hardmax", "lehmer_p1", "lehmer_p2", "log_hazard", "probabilistic_or", "softmax_value_a16"]
MILESTONES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]


def metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    pred = output >= .5
    exact = (pred == (target >= .5)).all(dim=-1).float().mean()
    return {"mse": float(torch.mean((output - target) ** 2)),
            "bit_accuracy": float((pred == (target >= .5)).float().mean()),
            "exact_accuracy": float(exact)}


def bool_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    return metrics(output.float(), target)


def layer_stats(layer: SigmoidOrLogicLayer) -> dict:
    g = layer.effective_gate().detach()
    b = layer.actual_bias().detach()
    return {"gate_mean": float(g.mean()), "gate_d": float(torch.minimum(g, 1-g).mean()),
            "gate_margin_mean": float((g-.5).abs().mean()),
            "gate_margin_lt_01": float(((g-.5).abs()<.1).float().mean()),
            "gate_margin_lt_005": float(((g-.5).abs()<.05).float().mean()),
            "gate_sigmoid_deriv_mean": float((g*(1-g)).mean()),
            "gate_sat_lt_1e-3": float((g*(1-g)<1e-3).float().mean()),
            "bias_d": float(torch.minimum(b, 1-b).mean()),
            "bias_margin_mean": float((b-.5).abs().mean())}


def threshold_state(layer: SigmoidOrLogicLayer, threshold: float = .5) -> DiscreteOrNorGateLayer:
    d = DiscreteOrNorGateLayer(layer.in_features, layer.out_features)
    with torch.no_grad():
        w, b = layer.to_discrete(threshold)
        d.weight.copy_(w); d.bias.copy_(b)
    return d


def direct_discrete(layer: SigmoidOrLogicLayer, x: torch.Tensor, threshold: float = .5) -> torch.Tensor:
    return threshold_state(layer, threshold)(x).float()


def train_b0(epochs: int = 400, seeds=(0, 1, 2)) -> dict:
    records = []
    for op_name in OPS:
        for target_value in (0.0, 1.0):
            # One active literal is enough to expose edge optimization.
            a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            target = torch.full((1, 1), target_value)
            for seed in seeds:
                torch.manual_seed(seed)
                layer = SigmoidOrLogicLayer(4, 1, op_name, gate_initialization=.5,
                                             bias_initialization=lambda x: nn.init.constant_(x, 0.0))
                opt = torch.optim.Adam(layer.parameters(), lr=.05)
                trajectory = []
                milestones = {}
                for epoch in range(epochs + 1):
                    out = layer(a)
                    loss = ((out-target) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    if epoch == 0 or epoch % 10 == 0 or epoch == epochs:
                        with torch.no_grad():
                            g = layer.effective_gate()
                            trajectory.append({"epoch": epoch, "loss": float(loss), "g_mean": float(g.mean()), "grad_norm": float(layer.raw_edge.grad.norm()) if layer.raw_edge.grad is not None else 0.0})
                    if epoch > 0:
                        for m in MILESTONES:
                            if m not in milestones and float(loss) < m:
                                milestones[str(m)] = {"epoch": epoch, "g": layer.effective_gate().detach().flatten().tolist()}
                    if epoch == epochs: break
                    opt.step()
                with torch.no_grad():
                    hard = layer.hard_forward(a)
                    boolean = direct_discrete(layer, a)
                records.append({"operator": op_name, "target": target_value, "seed": seed,
                                "final_loss": float(loss), "final_g": layer.effective_gate().detach().flatten().tolist(),
                                "hard": float(hard), "boolean": float(boolean), "milestones": milestones,
                                "trajectory": trajectory})
    return records


def task_data(task: str):
    rows = torch.tensor([[float((i >> j) & 1) for j in range(4)] for i in range(16)])
    if task == "or4": return rows, (rows.any(dim=1).float().unsqueeze(1))
    if task == "identity4": return rows, rows.clone()
    if task == "not4": return rows, 1-rows
    raise ValueError(task)


def train_b1(epochs: int = 600, seeds=(0, 1, 2)) -> dict:
    records = []
    for task in ("or4", "identity4", "not4"):
        x, y = task_data(task)
        for op_name in OPS:
            for seed in seeds:
                torch.manual_seed(seed)
                layer = SigmoidOrLogicLayer(4, y.shape[1], op_name, gate_initialization=.5,
                                             bias_initialization=lambda t: nn.init.normal_(t, mean=.5, std=.1))
                opt = torch.optim.Adam(layer.parameters(), lr=.01)
                best = {"loss": float("inf")}; trajectory = []; milestones = {}
                for epoch in range(epochs + 1):
                    out = layer(x); loss = ((out-y)**2).mean()
                    opt.zero_grad(); loss.backward()
                    with torch.no_grad():
                        hard_out = layer.hard_forward(x)
                        bool_out = direct_discrete(layer, x)
                        rec = {"epoch": epoch, "continuous": metrics(out,y),
                               "hard": metrics(hard_out,y), "boolean": bool_metrics(bool_out,y), **layer_stats(layer),
                               "raw_edge_grad_norm": float(layer.raw_edge.grad.norm()),
                               "raw_edge_grad_mean_abs": float(layer.raw_edge.grad.abs().mean())}
                        if epoch == 0 or epoch % 10 == 0 or epoch == epochs: trajectory.append(rec)
                        for m in MILESTONES:
                            if m not in milestones and rec["continuous"]["mse"] < m:
                                milestones[str(m)] = copy.deepcopy(rec)
                        if rec["continuous"]["mse"] < best["loss"]:
                            best = {"loss": rec["continuous"]["mse"], "epoch": epoch, "record": copy.deepcopy(rec), "state": copy.deepcopy(layer.state_dict())}
                    if epoch == epochs: break
                    opt.step()
                records.append({"task": task, "operator": op_name, "seed": seed,
                                "best_continuous": {k:v for k,v in best.items() if k != "state"},
                                "milestones": milestones, "trajectory": trajectory})
                # Keep only compact checkpoints at scientifically defined milestones.
                ckdir = OUT / "stage_b_checkpoints"; ckdir.mkdir(exist_ok=True)
                torch.save(best["state"], ckdir / f"B1_{task}_{op_name}_seed{seed}_best.pt")
    return records


def plot_b1(records: list[dict]) -> None:
    for task in ("or4", "identity4", "not4"):
        plt.figure(figsize=(9, 6))
        for op in OPS:
            runs = [r for r in records if r["task"] == task and r["operator"] == op]
            for r in runs:
                xy = [(z["continuous"]["mse"], 1-z["boolean"]["exact_accuracy"]) for z in r["trajectory"] if z["continuous"]["mse"] > 0]
                if xy: plt.plot([a for a,b in xy], [b for a,b in xy], alpha=.35, label=op if r["seed"] == 0 else None)
        plt.xscale("log"); plt.yscale("log"); plt.xlabel("continuous MSE"); plt.ylabel("Boolean truth-table error"); plt.title(f"B1 discretization consistency: {task}"); plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / f"stage_b1_consistency_{task}.png", dpi=160); plt.close()


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    b0 = train_b0(); b1 = train_b1()
    result = {"experiment": "Stage-B0-B1", "operators": OPS, "mse_milestones": MILESTONES, "b0": b0, "b1": b1}
    with (OUT / "stage_b_results.json").open("w") as f: json.dump(result, f, indent=2)
    plot_b1(b1)
    print(json.dumps({"output": str(OUT / "stage_b_results.json"), "b0_runs": len(b0), "b1_runs": len(b1)}, indent=2))


if __name__ == "__main__": main()
