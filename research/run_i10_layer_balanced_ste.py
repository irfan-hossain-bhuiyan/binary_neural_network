"""I10: causal-agnostic, per-layer Boolean STE gradient composition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i8_boolean_ste import (  # noqa: E402
    bool_loss,
    exact_discrete_trace,
    forward_equivalence,
    make_model,
    mask_diff,
    mask_state,
    metrics,
    task_loss,
)
from research.run_i9_ste_replication import redundant_active  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
CKOUT = OUT / "i10_layer_balanced_ste_checkpoints"
EVAL_STEPS = [0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
GRAD_STEPS = {0, 10, 100, 500}
RHO = 0.5
EPS = 1e-12


def parent_path(seed: int) -> Path:
    return CK / f"I2_I2-B_seed{seed}_best_continuous_mse.pt"


def load_parent(seed: int):
    path = parent_path(seed)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = make_model()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, path, sha


def all_grads(net, loss, params):
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(values, params)]


def edge_grads(net, loss):
    edges = [layer.raw_edge for layer in net.expectation_layers]
    values = torch.autograd.grad(loss, edges, retain_graph=True, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(values, edges)]


def flat_norm(grads):
    return torch.sqrt(sum((g.detach() * g.detach()).sum() for g in grads))


def cosine(a, b):
    aa, bb = a.reshape(-1), b.reshape(-1)
    den = aa.norm() * bb.norm()
    return float(torch.dot(aa, bb) / den) if den > 0 else None


def layer_lambdas(task_edges, bool_edges, rho=RHO, epsilon=EPS):
    """Return one frozen multiplier per layer from that layer's two norms."""
    return [
        float(rho * gt.norm() / (gb.norm() + epsilon)) if gb.norm() > epsilon else 0.0
        for gt, gb in zip(task_edges, bool_edges)
    ]


def calibration(net, x, y):
    """Compute both diagnostic STE variants and sigmoid per-layer lambdas."""
    params = list(net.parameters())
    task = task_loss(net, x, y)
    task_all = all_grads(net, task, params)
    edge_params = [layer.raw_edge for layer in net.expectation_layers]
    param_indices = {id(p): i for i, p in enumerate(params)}
    task_edges = [task_all[param_indices[id(p)]] for p in edge_params]
    variants = {}
    for variant in ("STE_SIGMOID", "STE_CONST025"):
        boolean = bool_loss(net, x, y, variant)
        bg = edge_grads(net, boolean)
        rows = []
        for i, (gt, gb) in enumerate(zip(task_edges, bg)):
            tn, bn = float(gt.norm()), float(gb.norm())
            lam = layer_lambdas([gt], [gb])[0]
            active = int((gb.abs() >= 1e-8).sum())
            scaled = lam * float(gb.norm())
            rows.append({
                "layer": i,
                "task_norm": tn,
                "bool_norm": bn,
                "lambda": lam,
                "scaled_bool_norm": scaled,
                "scaled_ratio": scaled / tn if tn > 0 else None,
                "coverage": float((gb.abs() >= 1e-8).float().mean()),
                "active_parameter_count": active,
                "effective_pressure": scaled / max(1, active),
                "mean_abs_scaled_active": float((lam * gb.abs())[gb.abs() >= 1e-8].mean()) if active else 0.0,
                "cosine": cosine(gt, gb),
            })
        variants[variant] = {
            "bool_loss": float(boolean.detach()),
            "task_global_norm": float(flat_norm(task_edges)),
            "bool_global_norm": float(flat_norm(bg)),
            "layers": rows,
            "global_cosine": cosine(torch.cat([g.reshape(-1) for g in task_edges]), torch.cat([g.reshape(-1) for g in bg])),
            "nonzero_parameter_count": int(sum((g.abs() >= 1e-8).sum() for g in bg)),
        }
    sigmoid_rows = variants["STE_SIGMOID"]["layers"]
    return {
        "rho": RHO,
        "epsilon": EPS,
        "task_loss": float(task.detach()),
        "task_global_norm": float(flat_norm(task_edges)),
        "variants": variants,
        "lambdas": [row["lambda"] for row in sigmoid_rows],
        "forward_equivalence": {v: forward_equivalence(net, x, v) for v in ("STE_SIGMOID", "STE_CONST025")},
        "redundant_active": redundant_active(net, x, y, "STE_SIGMOID"),
    }


def compose_gradients(net, task_grads, bool_grads, lambdas):
    """Return task gradients with per-layer Boolean additions on raw edges only."""
    params = list(net.parameters())
    edge_by_id = {id(layer.raw_edge): i for i, layer in enumerate(net.expectation_layers)}
    out = []
    for p, gt in zip(params, task_grads):
        layer_index = edge_by_id.get(id(p))
        if layer_index is None:
            out.append(gt.detach().clone())
        else:
            out.append((gt + lambdas[layer_index] * bool_grads[layer_index]).detach().clone())
    return out


def gradient_snapshot(net, x, y, variant, lambdas):
    params = list(net.parameters())
    task = task_loss(net, x, y)
    boolean = bool_loss(net, x, y, variant)
    tg = all_grads(net, task, params)
    edges = [layer.raw_edge for layer in net.expectation_layers]
    param_indices = {id(p): i for i, p in enumerate(params)}
    edge_indices = [param_indices[id(p)] for p in edges]
    bg = edge_grads(net, boolean)
    fg = compose_gradients(net, tg, bg, lambdas)
    rows = []
    for i, (gt, gb) in enumerate(zip((tg[j] for j in edge_indices), bg)):
        scaled = lambdas[i] * gb
        rows.append({
            "layer": i,
            "task_norm": float(gt.norm()),
            "bool_norm": float(gb.norm()),
            "scaled_bool_norm": float(scaled.norm()),
            "scaled_ratio": float(scaled.norm() / gt.norm()) if gt.norm() > 0 else None,
            "final_norm": float((gt + scaled).norm()),
            "cosine_task_bool": cosine(gt, gb),
            "coverage": float((gb.abs() >= 1e-8).float().mean()),
            "active_parameter_count": int((gb.abs() >= 1e-8).sum()),
            "effective_pressure": float(scaled.norm() / max(1, int((gb.abs() >= 1e-8).sum()))),
        })
    repair_task = tg[edge_indices[4]][62, 10]
    repair_bool = bg[4][62, 10]
    return {
        "task_loss": float(task.detach()),
        "bool_loss": float(boolean.detach()),
        "layers": rows,
        "repair_edge": {
            "task_gradient": float(repair_task),
            "bool_gradient": float(repair_bool),
            "composed_gradient": float(repair_task + lambdas[4] * repair_bool),
        },
        "global_task_norm": float(flat_norm([tg[i] for i in edge_indices])),
        "global_bool_norm": float(flat_norm(bg)),
        "global_composed_norm": float(flat_norm([fg[i] for i in edge_indices])),
        "global_cosine": cosine(torch.cat([tg[i].reshape(-1) for i in edge_indices]), torch.cat([g.reshape(-1) for g in bg])),
    }


def manual_optimizer_step(net, optimizer, x, y, variant, lambdas):
    """One update: compute separate gradients, compose, then call step once."""
    optimizer.zero_grad(set_to_none=True)
    params = list(net.parameters())
    task = task_loss(net, x, y)
    tg = all_grads(net, task, params)
    boolean = bool_loss(net, x, y, variant)
    bg = edge_grads(net, boolean)
    final = compose_gradients(net, tg, bg, lambdas)
    for p, g in zip(params, final):
        p.grad = g
    optimizer.step()
    return float(task.detach()), float(boolean.detach())


def threshold_robustness(net, x, y):
    vals = []
    for threshold in (.30, .35, .40, .45, .50, .55, .60, .65, .70):
        pred = net.to_discrete(threshold)(x.bool()).float()
        vals.append((threshold, float(((pred >= .5) == (y >= .5)).all(dim=-1).float().mean())))
    exact = [t for t, a in vals if a == 1.0]
    runs = []
    if exact:
        current = [exact[0]]
        for t in exact[1:]:
            if abs(t - current[-1] - .05) < 1e-8:
                current.append(t)
            else:
                runs.append(current)
                current = [t]
        runs.append(current)
    containing = [run for run in runs if .50 in run]
    return {"thresholds": vals, "largest_contiguous_containing_half": ([containing[0][0], containing[0][-1]] if containing else None)}


def eval_record(net, x, y, step, variant, lambdas, diagnostics=False):
    result = {
        "step": step,
        "metrics": metrics(net, x, y),
        "L_bool": float(bool_loss(net, x, y, variant).detach()),
        "threshold_robustness": threshold_robustness(net, x, y),
        "mask_hashes": mask_state(net)["hashes"],
        "repair_edge": {
            "raw": float(net.expectation_layers[4].raw_edge[62, 10].detach()),
            "gate": float(torch.sigmoid(net.expectation_layers[4].raw_edge[62, 10]).detach()),
            "boolean_bit": bool(net.expectation_layers[4].raw_edge[62, 10].detach() >= 0),
        },
    }
    if diagnostics:
        result["gradient_diagnostics"] = gradient_snapshot(net, x, y, variant, lambdas)
    return result


def save_checkpoint(path, state, metadata, x, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    check = make_model()
    check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    metadata = dict(metadata)
    metadata.update({
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reload_verified": True,
        "reload_metrics": metrics(check, x, y),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    return metadata


def run_arm(seed, arm, x, y, steps=3000):
    net, parent, parent_sha = load_parent(seed)
    parent_metrics = metrics(net, x, y)
    cal = calibration(net, x, y)
    lambdas = cal["lambdas"]
    variant = "STE_SIGMOID"
    active = arm == "layer_balanced_ste"
    if not active:
        lambdas = [0.0] * len(lambdas)
    initial = mask_state(net)
    previous = None
    records, states, transitions = [], [], []
    exact_history = []
    first = None

    def record(step, diagnostic=False):
        nonlocal previous, first
        ev = eval_record(net, x, y, step, variant, lambdas, diagnostic)
        current = mask_state(net)
        diff = mask_diff(previous, current) if previous is not None else mask_diff(initial, current)
        ev["mask_change_since_previous"] = diff
        ev["mask_change_since_parent"] = mask_diff(initial, current)
        if previous is not None and diff["edge_bits"] + diff["bias_bits"]:
            transitions.append({"step": step, **diff})
        previous = current
        records.append(ev)
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        exact = ev["metrics"]["boolean"]["exact_accuracy"] == 1.0
        exact_history.append(exact)
        if first is None and exact:
            first = {"step": step, "mse": ev["metrics"]["continuous"]["mse"], "e_inf": ev["metrics"]["continuous"]["e_inf"]}

    record(0, True)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.0)
    for step in range(1, steps + 1):
        if active:
            manual_optimizer_step(net, optimizer, x, y, variant, lambdas)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(net, x, y)
            loss.backward()
            optimizer.step()
        if step in EVAL_STEPS[1:] or step == steps:
            record(step, step in GRAD_STEPS)

    regressions = 0
    if first:
        reached = False
        for state in exact_history:
            if state:
                reached = True
            elif reached:
                regressions += 1
    longest = current_streak = 0
    for state in exact_history:
        current_streak = current_streak + 1 if state else 0
        longest = max(longest, current_streak)
    best_mse = min(range(len(records)), key=lambda i: records[i]["metrics"]["continuous"]["mse"] if math.isfinite(records[i]["metrics"]["continuous"]["mse"]) else float("inf"))
    best_boolean = max(range(len(records)), key=lambda i: (records[i]["metrics"]["boolean"]["exact_accuracy"], -records[i]["step"]))
    base = {
        "seed": seed,
        "arm": arm,
        "active_losses": ["top8_row_bce", "boolean_ste_mse"] if active else ["top8_row_bce"],
        "ste_auxiliary_active": active,
        "ste_variant": variant if active else None,
        "rho": RHO if active else 0.0,
        "lambdas": lambdas if active else [0.0] * len(lambdas),
        "calibration": cal,
        "parent_checkpoint": str(parent),
        "parent_sha256": parent_sha,
        "continuous_operator": "lehmer_p2",
        "steps": steps,
    }
    saved = {}
    for kind, index in (("best_continuous", best_mse), ("best_boolean", best_boolean), ("final", len(states) - 1)):
        saved[kind] = save_checkpoint(CKOUT / f"I10_seed{seed}_{arm}_{kind}.pt", states[index], {**base, "step": records[index]["step"]}, x, y)
    if first:
        index = next(i for i, r in enumerate(records) if r["step"] == first["step"])
        saved["first_boolean_exact"] = save_checkpoint(CKOUT / f"I10_seed{seed}_{arm}_first_boolean_exact.pt", states[index], {**base, "step": first["step"]}, x, y)
    return {
        **base,
        "parent_metrics": parent_metrics,
        "first_boolean_recovery": first,
        "boolean_regressions_after_first": regressions,
        "longest_consecutive_exact_evaluations": longest,
        "mask_transitions": transitions,
        "trajectory": records,
        "saved_checkpoints": saved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("all", "control", "layer_balanced_ste"), default="all")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    parent, path, sha = load_parent(args.seed)
    cal = calibration(parent, x, y)
    payload = {
        "experiment": "I10-layer-balanced-boolean-ste",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": args.seed,
        "parent_checkpoint": str(path),
        "parent_sha256": sha,
        "parent_metrics": metrics(parent, x, y),
        "calibration": cal,
        "forward_equivalence": cal["forward_equivalence"],
    }
    if args.seed == 4:
        arms = ("control",) if args.arm in ("all", "control") else tuple()
    else:
        arms = ("control", "layer_balanced_ste") if args.arm == "all" else (args.arm,)
    payload["arms"] = [run_arm(args.seed, arm, x, y, args.steps) for arm in arms]
    path_out = OUT / "i10_layer_balanced_ste_results.json"
    if path_out.exists():
        old = json.loads(path_out.read_text())
        by = {(r["seed"], r["arm"]): r for r in old.get("runs", [])}
        for r in payload["arms"]:
            by[(r["seed"], r["arm"])] = r
        payload["runs"] = list(by.values())
    else:
        payload["runs"] = payload.pop("arms")
    path_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(path_out)


if __name__ == "__main__":
    main()
