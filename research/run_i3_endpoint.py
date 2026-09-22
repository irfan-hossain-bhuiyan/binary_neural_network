"""I3 endpoint consistency diagnostics and controlled MSE/BCE continuation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
I3CK = OUT / "i3_endpoint_checkpoints"
EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
PARENT_SEEDS = (2, 3, 4)


def model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2,
                                       or_operator="lehmer_p2")


def load_parent(seed: int) -> tuple[SigmoidOrModernLogicGateNet, Path, str]:
    path = CK / f"I2_I2-B_seed{seed}_best_continuous_mse.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    parent_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = model()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net, path, parent_sha


def common_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    error = (output - target).abs()
    binary = (output >= .5) == (target >= .5)
    return {
        "mse": float(error.square().mean()),
        "bce": float(F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), target)),
        "mae": float(error.mean()),
        "e_inf": float(error.max()),
        "p95_abs_error": float(torch.quantile(error.flatten(), .95)),
        "p99_abs_error": float(torch.quantile(error.flatten(), .99)),
        "bit_accuracy": float(binary.float().mean()),
        "exact_accuracy": float(binary.all(dim=-1).float().mean()),
        "wrong_rows": int((~binary.all(dim=-1)).sum()),
        "wrong_bits": int((~binary).sum()),
    }


@torch.no_grad()
def evaluate(net: SigmoidOrModernLogicGateNet, x: torch.Tensor, y: torch.Tensor,
             threshold: float = .5) -> dict:
    continuous = net(x)
    hard = net.forward_hard(x)
    boolean = net.to_discrete(threshold)(x.bool()).float()
    return {
        "continuous": common_metrics(continuous, y),
        "hard": common_metrics(hard, y),
        "boolean": common_metrics(boolean, y),
    }


def raw_gate_stats(net: SigmoidOrModernLogicGateNet) -> list[dict]:
    rows = []
    for i, layer in enumerate(net.expectation_layers):
        r = layer.raw_edge.detach(); g = torch.sigmoid(r); b = layer.actual_bias().detach()
        absr = r.abs(); deriv = g * (1 - g)
        rows.append({
            "layer": i, "raw_mean": float(r.mean()), "raw_std": float(r.std()),
            "raw_abs_q50": float(torch.quantile(absr, .50)),
            "raw_abs_q75": float(torch.quantile(absr, .75)),
            "raw_abs_q90": float(torch.quantile(absr, .90)),
            "raw_abs_q95": float(torch.quantile(absr, .95)),
            "raw_abs_q99": float(torch.quantile(absr, .99)),
            "raw_positive": float((r > 0).float().mean()), "raw_negative": float((r < 0).float().mean()),
            **{f"raw_abs_gt_{n}": float((absr > n).float().mean()) for n in (2, 4, 6, 8, 10)},
            **{f"raw_pos_gt_{n}": float((r > n).float().mean()) for n in (2, 4, 6, 8, 10)},
            **{f"raw_neg_lt_{n}": float((r < -n).float().mean()) for n in (2, 4, 6, 8, 10)},
            "D_g": float(torch.minimum(g, 1 - g).mean()),
            "g_lt_01": float((g < .01).float().mean()), "g_gt_99": float((g > .99).float().mean()),
            "g_lt_001": float((g < .001).float().mean()), "g_gt_999": float((g > .999).float().mean()),
            "g_middle": float(((g > .45) & (g < .55)).float().mean()),
            "gate_derivative_mean": float(deriv.mean()), "gate_derivative_median": float(deriv.median()),
            "bias_D": float(torch.minimum(b, 1 - b).mean()), "bias_lt_01": float((b < .01).float().mean()),
            "bias_gt_99": float((b > .99).float().mean()), "bias_middle": float(((b > .45) & (b < .55)).float().mean()),
        })
    return rows


@torch.no_grad()
def layer_trace(net: SigmoidOrModernLogicGateNet, x: torch.Tensor) -> dict:
    disc = net.to_discrete(.5)
    h, hb = x, x.bool(); rows = []

    def add(name: str, continuous: torch.Tensor, boolean: torch.Tensor, fan_in: int | None = None,
            contributions: torch.Tensor | None = None) -> None:
        thresholded = continuous >= .5
        mismatch = thresholded != boolean
        item = {
            "name": name, "continuous_mean": float(continuous.mean()), "continuous_std": float(continuous.std(unbiased=False)),
            "continuous_q01": float(torch.quantile(continuous.flatten(), .01)),
            "continuous_q99": float(torch.quantile(continuous.flatten(), .99)),
            "continuous_below_half": float((~thresholded).float().mean()),
            "discrete_zero_fraction": float((~boolean).float().mean()),
            "threshold_discrete_mismatch_fraction": float(mismatch.float().mean()),
            "threshold_discrete_mismatch_count": int(mismatch.sum()),
        }
        if contributions is not None and fan_in is not None:
            f = continuous
            bound = 1.0 / (4.0 * fan_in)
            zero_side = f < .5
            item.update({
                "fan_in": fan_in, "zero_safety_bound": bound,
                "zero_safety_fraction": float((zero_side & (f < bound)).float().mean()),
                "zero_side_fraction": float(zero_side.float().mean()),
                "zero_side_max": float(f[zero_side].max()) if zero_side.any() else 0.0,
                "zero_side_p95": float(torch.quantile(f[zero_side], .95)) if zero_side.any() else 0.0,
                "zero_side_p99": float(torch.quantile(f[zero_side], .99)) if zero_side.any() else 0.0,
            })
        rows.append(item)

    add("input", h, hb)
    h = net.stem(h); hb = disc.stem(hb); add("stem", h, hb, net.stem.in_features, net.stem.contributions(x))
    for i, block in enumerate(net.blocks):
        h1 = block.layer1(h); hb1 = disc.blocks[i].layer1(hb)
        add(f"block{i}.layer1", h1, hb1, block.layer1.in_features, block.layer1.contributions(h))
        h2 = block.layer2(h1); hb2 = disc.blocks[i].layer2(hb1)
        add(f"block{i}.layer2", h2, hb2, block.layer2.in_features, block.layer2.contributions(h1))
        h = h + h2 - 2 * h * h2; hb = hb ^ hb2; add(f"block{i}.residual", h, hb)
    out = net.head(h); outb = disc.head(hb)
    add("head", out, outb, net.head.in_features, net.head.contributions(h))
    certified = all(row.get("threshold_discrete_mismatch_fraction", 1.0) == 0.0 and
                    row.get("zero_safety_fraction", 0.0) >= row.get("zero_side_fraction", 1.0)
                    for row in rows if "zero_safety_fraction" in row)
    return {"stages": rows, "network_certified": certified}


def wrong_rows(net: SigmoidOrModernLogicGateNet, x: torch.Tensor, y: torch.Tensor) -> list[dict]:
    with torch.no_grad():
        cont = net(x); pred = net.to_discrete(.5)(x.bool()).float()
        bad = (pred != y).any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
        return [{"row": i, "input": x[i].int().tolist(), "target": y[i].int().tolist(),
                 "boolean_prediction": pred[i].int().tolist(), "continuous_prediction": cont[i].tolist()} for i in bad]


def checkpoint_save(path: Path, state: dict[str, torch.Tensor]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic(seed: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    net, path, parent_sha = load_parent(seed)
    return {
        "seed": seed, "parent_checkpoint": str(path), "parent_sha256": parent_sha,
        "metrics": evaluate(net, x, y), "raw_gate_stats": raw_gate_stats(net),
        "layer_trace": layer_trace(net, x), "wrong_rows": wrong_rows(net, x, y),
    }


def continuation(seed: int, arm: str, steps: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    net, parent_path, parent_sha = load_parent(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)
    records = []
    best = {"mse": (float("inf"), None, None), "bce": (float("inf"), None, None), "e_inf": (float("inf"), None, None)}
    first_boolean = None

    def record(step: int) -> None:
        nonlocal first_boolean
        ev = evaluate(net, x, y)
        ev["step"] = step
        ev["raw_gate_stats"] = raw_gate_stats(net)
        ev["layer_trace"] = layer_trace(net, x)
        ev["wrong_rows"] = wrong_rows(net, x, y)
        records.append(ev)
        if first_boolean is None and ev["boolean"]["exact_accuracy"] == 1.0:
            first_boolean = {"step": step, "mse": ev["continuous"]["mse"]}
        for metric in ("mse", "bce", "e_inf"):
            value = ev["continuous"][metric]
            if value < best[metric][0]:
                best[metric] = (value, copy.deepcopy(ev),
                                {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    record(0)
    for step in range(1, steps + 1):
        output = net(x)
        if arm == "mse":
            loss = (output - y).square().mean()
        else:
            loss = F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in EVAL_STEPS[1:] or step == steps:
            record(step)
    saved = {}
    for kind, (_, ev, state) in best.items():
        if ev is not None and state is not None:
            path = I3CK / f"I3_seed{seed}_{arm}_best_{kind}.pt"
            saved[f"best_{kind}"] = {"path": str(path), "sha256": checkpoint_save(path, state), "step": ev["step"]}
    final_path = I3CK / f"I3_seed{seed}_{arm}_final.pt"
    saved["final"] = {"path": str(final_path), "sha256": checkpoint_save(final_path,
        {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}), "step": steps}
    return {"seed": seed, "arm": arm, "parent_checkpoint": str(parent_path), "parent_sha256": parent_sha,
            "optimizer_state_policy": "fresh Adam for both arms; checkpoint had no optimizer state",
            "learning_rate": .01, "steps": steps, "first_boolean_exact": first_boolean,
            "trajectory": records, "saved_checkpoints": saved}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", default="3")
    args = p.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    task = build_task("bitwise_xor_truth_table", {"bits": 4}); x = task["X"].float().to(device); y = task["Y"].float().to(device)
    I3CK.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    diagnostics = [diagnostic(s, x, y) for s in PARENT_SEEDS]
    continuations = [continuation(s, arm, args.steps, x, y) for s in seeds for arm in ("mse", "bce")]
    sha = os.environ.get("RESEARCH_GIT_SHA")
    if not sha:
        try: sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError: sha = "unknown"
    payload = {"experiment": "I3-endpoint-consistency", "git_sha": sha, "device": str(device), "steps": args.steps,
               "training": {"optimizer": "Adam", "lr": .01, "loss_arms": ["mse", "binary_cross_entropy_probability_space"],
                            "optimizer_state_policy": "fresh Adam; parent optimizer state unavailable"},
               "diagnostics": diagnostics, "continuations": continuations}
    (OUT / "i3_endpoint_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT / "i3_endpoint_results.json")


if __name__ == "__main__": main()
