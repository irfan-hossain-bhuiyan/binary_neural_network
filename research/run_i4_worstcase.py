"""I4 worst-case BCE continuation from the verified I2-B checkpoints."""
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
from research.run_i3_endpoint import common_metrics, evaluate, raw_gate_stats  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
I4CK = OUT / "i4_worstcase_checkpoints"
EVAL_STEPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
HARD_ROWS = [239, 255]


def make_model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2,
                                       or_operator="lehmer_p2")


def load_parent(seed: int):
    path = CK / f"I2_I2-B_seed{seed}_best_continuous_mse.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = make_model()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, path, sha


def mask_state(net: SigmoidOrModernLogicGateNet) -> dict:
    edges, biases, hashes = [], [], []
    for layer in net.expectation_layers:
        e = (layer.effective_gate().detach() >= .5).cpu()
        b = (layer.actual_bias().detach() >= .5).cpu()
        edges.append(e); biases.append(b)
        hashes.append({"edge": hashlib.sha256(e.numpy().tobytes()).hexdigest(),
                       "bias": hashlib.sha256(b.numpy().tobytes()).hexdigest()})
    return {"edges": edges, "biases": biases, "hashes": hashes}


def mask_diff(a: dict | None, b: dict) -> dict:
    if a is None:
        return {"edge_bits": 0, "bias_bits": 0, "layers": []}
    layers = []
    for ea, eb, ba, bb in zip(a["edges"], b["edges"], a["biases"], b["biases"]):
        layers.append({"edge_bits": int((ea != eb).sum()), "bias_bits": int((ba != bb).sum())})
    return {"edge_bits": sum(x["edge_bits"] for x in layers),
            "bias_bits": sum(x["bias_bits"] for x in layers), "layers": layers}


def metrics(net, x, y) -> dict:
    with torch.no_grad():
        out = net(x)
        element = F.binary_cross_entropy(out.clamp(1e-7, 1 - 1e-7), y, reduction="none")
        row = element.mean(dim=1)
        base = evaluate(net, x, y)
        base.update({
            "max_element_bce": float(element.max()),
            "top16_mean_element_bce": float(torch.topk(element.flatten(), 16).values.mean()),
            "max_row_bce": float(row.max()),
            "top8_mean_row_bce": float(torch.topk(row, 8).values.mean()),
            "rows": {str(i): {"continuous": out[i].tolist(), "target": y[i].tolist(),
                              "hard": net.forward_hard(x[i:i+1])[0].tolist(),
                              "boolean": net.to_discrete(.5)(x[i:i+1].bool())[0].float().tolist()}
                     for i in HARD_ROWS},
        })
        return base


def gradient_groups(net, x, y) -> dict:
    out = net(x)
    element = F.binary_cross_entropy(out.clamp(1e-7, 1 - 1e-7), y, reduction="none")
    groups = {"hard_rows": torch.tensor(HARD_ROWS), "other_rows": torch.tensor([i for i in range(len(x)) if i not in HARD_ROWS])}
    params = [p for p in net.parameters() if p.requires_grad]
    result = {}
    for name, idx in groups.items():
        loss = element[idx].mean()
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        rows = []
        for layer, (edge, bias) in enumerate(zip(grads[0::2], grads[1::2])):
            ge = edge.detach() if edge is not None else torch.zeros_like(net.expectation_layers[layer].raw_edge)
            gb = bias.detach() if bias is not None else torch.zeros_like(net.expectation_layers[layer].bias)
            rows.append({"layer": layer, "edge_norm": float(ge.norm()), "bias_norm": float(gb.norm()),
                         "edge_mean_abs": float(ge.abs().mean()), "bias_mean_abs": float(gb.abs().mean())})
        result[name] = rows
    cosines = []
    h = result["hard_rows"]; o = result["other_rows"]
    # Recompute flattened gradients to preserve signs for cosine diagnostics.
    for layer in range(len(net.expectation_layers)):
        lh = torch.autograd.grad(element[groups["hard_rows"]].mean(), params, retain_graph=True, allow_unused=True)
        lo = torch.autograd.grad(element[groups["other_rows"]].mean(), params, retain_graph=True, allow_unused=True)
        a = torch.cat([lh[2*layer].flatten(), lh[2*layer+1].flatten()]); b = torch.cat([lo[2*layer].flatten(), lo[2*layer+1].flatten()])
        denom = a.norm() * b.norm()
        cosines.append({"layer": layer, "cosine": float(torch.dot(a, b) / denom) if denom > 0 else None})
    result["cosine_by_layer"] = cosines
    net.zero_grad(set_to_none=True)
    return result


def run_arm(seed: int, arm: str, steps: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    net, parent, parent_sha = load_parent(seed)
    parent_eval = metrics(net, x, y)
    if seed == 3:
        if parent_eval["boolean"]["wrong_rows"] != 2 or set(i for i in HARD_ROWS if parent_eval["rows"][str(i)]["boolean"] != parent_eval["rows"][str(i)]["target"]) != set(HARD_ROWS):
            raise RuntimeError("seed3 parent does not match the documented rows 239/255 checkpoint")
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    initial_mask = mask_state(net); previous_mask = None; records = []; states = []; first_recovery = None
    transitions = []

    def record(step: int) -> None:
        nonlocal previous_mask, first_recovery
        ev = metrics(net, x, y); current_mask = mask_state(net)
        ev.update({"step": step, "raw_gate_stats": raw_gate_stats(net),
                   "mask_hashes": current_mask["hashes"],
                   "mask_change_since_previous": mask_diff(previous_mask, current_mask),
                   "mask_change_since_start": mask_diff(initial_mask, current_mask)})
        if step in (0, 100, 500): ev["gradient_groups"] = gradient_groups(net, x, y)
        records.append(ev)
        if first_recovery is None and ev["boolean"]["exact_accuracy"] == 1.0:
            first_recovery = {"step": step, "mse": ev["continuous"]["mse"], "mean_bce": ev["continuous"]["bce"],
                              "e_inf": ev["continuous"]["e_inf"], "top16_bce": ev["top16_mean_element_bce"],
                              "top8_row_bce": ev["top8_mean_row_bce"]}
        if previous_mask is not None and ev["mask_change_since_previous"]["edge_bits"] + ev["mask_change_since_previous"]["bias_bits"]:
            transitions.append({"step": step, **ev["mask_change_since_previous"]})
        previous_mask = current_mask
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    record(0)
    for step in range(1, steps + 1):
        out = net(x); element = F.binary_cross_entropy(out.clamp(1e-7, 1 - 1e-7), y, reduction="none")
        if arm == "mean_bce": loss = element.mean()
        elif arm == "top16_bit_bce": loss = torch.topk(element.reshape(-1), 16).values.mean()
        elif arm == "top8_row_bce": loss = torch.topk(element.mean(dim=1), 8).values.mean()
        else: raise ValueError(arm)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in EVAL_STEPS[1:] or step == steps: record(step)

    saved = {}
    for kind, key in (("best_mse", "mse"), ("best_boolean", None)):
        if key:
            idx = min(range(len(records)), key=lambda i: records[i]["continuous"][key])
        else:
            idx = max(range(len(records)), key=lambda i: (records[i]["boolean"]["exact_accuracy"], -records[i]["step"]))
        path = I4CK / f"I4_seed{seed}_{arm}_{kind}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states[idx], path)
        saved[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "eval_step": records[idx]["step"]}
    final = I4CK / f"I4_seed{seed}_{arm}_final.pt"; torch.save(states[-1], final)
    saved["final"] = {"path": str(final), "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "eval_step": steps}
    return {"seed": seed, "arm": arm, "parent": str(parent), "parent_sha256": parent_sha,
            "optimizer": {"name": "Adam", "lr": .01, "state": "fresh"}, "steps": steps,
            "initial_metrics": parent_eval, "first_boolean_recovery": first_recovery,
            "mask_transitions": transitions, "trajectory": records, "saved_checkpoints": saved}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed3-only", action="store_true")
    p.add_argument("--seed2-only", action="store_true")
    p.add_argument("--arm", choices=("mean_bce", "top16_bit_bce", "top8_row_bce"))
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    task = build_task("bitwise_xor_truth_table", {"bits": 4}); x = task["X"].float().to(device); y = task["Y"].float().to(device)
    if args.arm:
        seed = 2 if args.seed2_only else 3
        arms = [(seed, args.arm)]
    elif args.seed2_only:
        arms = [(2, "mean_bce"), (2, "top8_row_bce")]
    else:
        arms = [(3, a) for a in ("mean_bce", "top16_bit_bce", "top8_row_bce")]
        if not args.seed3_only: arms += [(2, "mean_bce"), (2, "top8_row_bce")]
    results = [run_arm(seed, arm, args.steps, x, y) for seed, arm in arms]
    sha = os.environ.get("RESEARCH_GIT_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {"experiment": "I4-worstcase-endpoint-optimization", "git_sha": sha, "device": str(device), "steps": args.steps,
               "parent_checkpoint": "I2-B best_continuous_mse", "hard_rows": HARD_ROWS, "results": results}
    (OUT / "i4_worstcase_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT / "i4_worstcase_results.json")


if __name__ == "__main__": main()
