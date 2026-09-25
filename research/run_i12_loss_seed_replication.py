"""I12: five-seed clean XOR replication for four output losses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i11_loss_geometry import make_model, state_hash  # noqa: E402
from research.run_i3_endpoint import common_metrics  # noqa: E402
from research.run_initialization_i2 import model_for  # noqa: E402

OUT = ROOT / "research/operator_results"
CKOUT = OUT / "i12_loss_seed_replication_checkpoints"
LOSSES = ("MSE", "BCE", "POWER_1_5", "POWER_1_25")
EVAL_STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000)
THRESHOLDS = (.30, .35, .40, .45, .50, .55, .60, .65, .70)
MILESTONES = (.25, .10, .05, .01, .005, .001)


def loss_value(output: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "MSE":
        return (output - target).square().mean()
    if name == "BCE":
        return F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), target)
    alpha = 1.5 if name == "POWER_1_5" else 1.25
    return (output - target).abs().pow(alpha).mean()


def endpoint_stats(output: torch.Tensor, target: torch.Tensor) -> dict:
    distance = torch.minimum(output, 1 - output)
    confidence = torch.where(target > .5, output, 1 - output)
    return {
        "mean": float(distance.mean()),
        "median": float(distance.median()),
        "p95": float(torch.quantile(distance, .95)),
        "fraction_lt_.01": float((distance < .01).float().mean()),
        "fraction_lt_.05": float((distance < .05).float().mean()),
        "fraction_abs_minus_.5_lt_.05": float((distance > .45).float().mean()),
        "fraction_abs_minus_.5_lt_.10": float((distance > .40).float().mean()),
        "mean_confidence": float(confidence.mean()),
        "minimum_confidence": float(confidence.min()),
        "p05_confidence": float(torch.quantile(confidence, .05)),
    }


def gate_bias_stats(net) -> list[dict]:
    rows = []
    for i, layer in enumerate(net.expectation_layers):
        raw = layer.raw_edge.detach(); gate = torch.sigmoid(raw); bias = layer.actual_bias().detach()
        rows.append({
            "layer": i,
            "gate_D": float(torch.minimum(gate, 1 - gate).mean()),
            "gate_lt_.01": float((gate < .01).float().mean()),
            "gate_gt_.99": float((gate > .99).float().mean()),
            "raw_abs_gt_2": float((raw.abs() > 2).float().mean()),
            "raw_abs_gt_4": float((raw.abs() > 4).float().mean()),
            "raw_abs_lt_.1": float((raw.abs() < .1).float().mean()),
            "raw_abs_lt_.5": float((raw.abs() < .5).float().mean()),
            "bias_D": float(torch.minimum(bias, 1 - bias).mean()),
            "bias_lt_.01": float((bias < .01).float().mean()),
            "bias_gt_.99": float((bias > .99).float().mean()),
            "bias_abs_from_.5_lt_.05": float((bias.sub(.5).abs() < .05).float().mean()),
            "bias_abs_from_.5_lt_.10": float((bias.sub(.5).abs() < .10).float().mean()),
        })
    return rows


@torch.no_grad()
def evaluate(net, x, y, threshold=.5) -> dict:
    output = net(x)
    hard = net.forward_hard(x)
    boolean = net.to_discrete(threshold)(x.bool()).float()
    return {
        "continuous": common_metrics(output, y),
        "hard": common_metrics(hard, y),
        "boolean": common_metrics(boolean, y),
        "endpoint": endpoint_stats(output, y),
        "finite": bool(torch.isfinite(output).all()),
    }


def threshold_robustness(net, x, y) -> dict:
    return {f"{t:.2f}": evaluate(net, x, y, t)["boolean"] for t in THRESHOLDS}


def snapshot(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_verified(path: Path, state, seed, loss_name, kind, step, x, y, initial_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    check = make_model()
    check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    got = evaluate(check, x, y)
    return {
        "path": str(path), "sha256": sha_file(path), "reload_verified": True,
        "reload_metrics": got, "seed": seed, "loss": loss_name, "kind": kind,
        "step": step, "initial_state_sha256": initial_hash,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def run_one(seed: int, loss_name: str, x: torch.Tensor, y: torch.Tensor, steps: int) -> dict:
    # Construct once, then reload the byte-identical state for the selected loss.
    base = model_for("I2-B", seed, torch.device("cpu"))
    initial = snapshot(base)
    initial_hash = state_hash(initial)
    net = make_model(); net.load_state_dict(initial)
    assert state_hash(snapshot(net)) == initial_hash
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.0)
    trajectory = []
    best_einf = (float("inf"), None, None)
    best_boolean = (-1.0, None, None)
    first_cont = first_hard = first_bool = None
    first_bool_state = None
    cont_regressions = hard_regressions = bool_regressions = 0
    last_cont = last_hard = last_bool = False
    milestones = {}
    for step in range(steps + 1):
        if step in EVAL_STEPS or step == steps:
            ev = evaluate(net, x, y)
            rec = {"step": step, **ev}
            trajectory.append(rec)
            c = ev["continuous"]["exact_accuracy"] == 1.0
            h = ev["hard"]["exact_accuracy"] == 1.0
            b = ev["boolean"]["exact_accuracy"] == 1.0
            if first_cont is None and c: first_cont = step
            if first_hard is None and h: first_hard = step
            if first_bool is None and b:
                first_bool = step
                first_bool_state = snapshot(net)
            if last_cont and not c: cont_regressions += 1
            if last_hard and not h: hard_regressions += 1
            if last_bool and not b: bool_regressions += 1
            last_cont, last_hard, last_bool = c, h, b
            if ev["continuous"]["e_inf"] < best_einf[0]:
                best_einf = (ev["continuous"]["e_inf"], step, snapshot(net))
            if ev["boolean"]["exact_accuracy"] > best_boolean[0]:
                best_boolean = (ev["boolean"]["exact_accuracy"], step, snapshot(net))
            for threshold in MILESTONES:
                key = f"{threshold:g}"
                if key not in milestones and ev["continuous"]["e_inf"] < threshold:
                    milestones[key] = {"step": step, "boolean_exact": ev["boolean"]["exact_accuracy"]}
        if step == steps: break
        output = net(x)
        loss = loss_value(output, y, loss_name)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    final_state = snapshot(net)
    checkpoint_dir = CKOUT / f"seed{seed}_{loss_name}"
    checkpoints = {
        "best_continuous_e_inf": save_verified(checkpoint_dir / "best_continuous_e_inf.pt", best_einf[2], seed, loss_name, "best_continuous_e_inf", best_einf[1], x, y, initial_hash),
        "best_boolean": save_verified(checkpoint_dir / "best_boolean.pt", best_boolean[2], seed, loss_name, "best_boolean", best_boolean[1], x, y, initial_hash),
        "final": save_verified(checkpoint_dir / "final.pt", final_state, seed, loss_name, "final", steps, x, y, initial_hash),
    }
    if first_bool_state is not None:
        checkpoints["first_boolean_exact"] = save_verified(
            checkpoint_dir / "first_boolean_exact.pt", first_bool_state, seed, loss_name,
            "first_boolean_exact", first_bool, x, y, initial_hash,
        )
    # Reload final state for topology diagnostics; no selected checkpoint is
    # allowed to overwrite the actual end-of-training state.
    final_net = make_model(); final_net.load_state_dict(final_state)
    final_eval = evaluate(final_net, x, y)
    return {
        "seed": seed, "loss": loss_name, "steps": steps, "optimizer_steps": steps,
        "initial_state_sha256": initial_hash, "trajectory": trajectory,
        "milestones": milestones, "first_continuous_exact": first_cont,
        "first_hard_exact": first_hard, "first_boolean_exact": first_bool,
        "continuous_regressions_after_exact": cont_regressions,
        "hard_regressions_after_exact": hard_regressions,
        "boolean_regressions_after_exact": bool_regressions,
        "final": final_eval, "final_gate_bias_stats": gate_bias_stats(final_net),
        "threshold_robustness": threshold_robustness(final_net, x, y),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss", choices=LOSSES, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    result = run_one(args.seed, args.loss, x, y, args.steps)
    payload = {
        "experiment": "I12-loss-seed-replication", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": args.seed, "loss": args.loss,
        "architecture": {"input_dim": 8, "width": 64, "residual_blocks": 2, "operator": "lehmer_p2"},
        "initializer": "I2-B meanfield sigma2 + BIAS_ONE",
        "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": args.steps, "dataset_rows": 256, "batch": 256},
        "run": result,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(args.seed, args.loss, result["final"]["boolean"]["exact_accuracy"], result["final"]["continuous"]["e_inf"], flush=True)


if __name__ == "__main__":
    main()
