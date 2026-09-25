"""A3: whole-word tolerance losses on exhaustive four-bit addition."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.meanfield_initialization import (  # noqa: E402
    bias_one_normal_init_, meanfield_gaussian_edge_init_,
)

OUT = ROOT / "research/operator_results"
LOSSES = ("BASELINE", "BIT_MEAN_RATIONAL", "ROW_MAX_RATIONAL", "ROW_MAX_SOFTPLUS")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 3000
EPSILON = 0.1
Q = 2.0
K = 20.0
EVAL_STEPS = {0, *range(10, 501, 10), *range(525, STEPS + 1, 25), STEPS}
DIAG_STEPS = {0, 100, 500, 1000, 2000, 3000}


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(seed: int, device: torch.device):
    eg = torch.Generator(device="cpu").manual_seed(100_000 + seed)
    bg = torch.Generator(device="cpu").manual_seed(200_000 + seed)

    def edge_init(tensor, fan_in):
        meanfield_gaussian_edge_init_(tensor, fan_in, sigma=2.0, generator=eg)

    def bias_init(tensor):
        bias_one_normal_init_(tensor, std=0.1, generator=bg)

    return SigmoidOrModernLogicGateNet(
        8, 5, width=64, num_residual_blocks=2, or_operator="lehmer_p2",
        gate_initializations=[0.5] * 6, bias_initialization=bias_init,
        edge_initialization=edge_init, residual_enabled=True,
    ).to(device)


def loss_value(output, target, name: str):
    d = (output - target).abs()
    if name == "BASELINE":
        return d.square().mean()
    if name == "BIT_MEAN_RATIONAL":
        z = d.square()
        return (z / (z + EPSILON ** Q)).mean()
    row = d.max(dim=1).values
    if name == "ROW_MAX_RATIONAL":
        z = row.square()
        return (z / (z + EPSILON ** Q)).mean()
    if name == "ROW_MAX_SOFTPLUS":
        raw = F.softplus(K * (row - EPSILON))
        zero = F.softplus(torch.tensor(-K * EPSILON, dtype=row.dtype, device=row.device))
        one = F.softplus(torch.tensor(K * (1.0 - EPSILON), dtype=row.dtype, device=row.device))
        return ((raw - zero) / (one - zero)).mean()
    raise ValueError(name)


def rational_scalar(d):
    z = d.square()
    return z / (z + EPSILON ** Q)


def softplus_scalar(d):
    raw = F.softplus(K * (d - EPSILON))
    zero = F.softplus(torch.tensor(-K * EPSILON, dtype=d.dtype, device=d.device))
    one = F.softplus(torch.tensor(K * (1.0 - EPSILON), dtype=d.dtype, device=d.device))
    return (raw - zero) / (one - zero)


def metric(output, target, chain):
    pred = output >= .5; truth = target >= .5
    rows = (pred == truth).all(dim=1); bits = pred == truth
    err = (output - target).abs(); row_err = err.max(dim=1).values
    carry = {}
    for n in range(5):
        mask = chain == n
        carry[str(n)] = {
            "rows": int(mask.sum()),
            "exact_accuracy": float(rows[mask].float().mean()),
            "bit_accuracy": float(bits[mask].float().mean()),
            "median_row_error": float(row_err[mask].median()),
            "mean_row_error": float(row_err[mask].mean()),
        }
    worst = err.argmax(dim=1)
    return {
        "bit_accuracy": float(bits.float().mean()), "exact_accuracy": float(rows.float().mean()),
        "wrong_rows": int((~rows).sum()), "wrong_bits": int((~bits).sum()),
        "mse": float(err.square().mean()), "mae": float(err.mean()), "e_inf": float(err.max()),
        "p95_abs_error": float(torch.quantile(err.reshape(-1), .95)),
        "p99_abs_error": float(torch.quantile(err.reshape(-1), .99)),
        "per_bit_accuracy": [float(v) for v in bits.float().mean(dim=0)],
        "per_bit_wrong_count": [int(v) for v in (~bits).sum(dim=0)],
        "row_tolerance_fractions": {str(t): float((row_err < t).float().mean()) for t in (.25, .1, .05, .01)},
        "worst_bit_counts": [int((worst == i).sum()) for i in range(5)],
        "carry_chain": carry,
    }


@torch.no_grad()
def evaluate(model, x, y, chain):
    cont = model(x)
    hard = model.forward_hard(x)
    boolean = model.to_discrete(.5).to(x.device)(x.bool()).float()
    return {
        "continuous": metric(cont, y, chain), "hard": metric(hard, y, chain),
        "boolean": metric(boolean, y, chain),
        "bit_disagreement_fraction": float(((cont >= .5) != boolean).float().mean()),
        "row_disagreement_fraction": float(((cont >= .5) != boolean).any(dim=1).float().mean()),
        "mean_hamming_continuous_to_boolean": float(((cont >= .5) != boolean).sum(dim=1).float().mean()),
    }


def xor(a, b):
    return a + b - 2 * a * b


def forward_trace(model, x):
    stem = model.stem(x); current = stem; traces = []
    for i, block in enumerate(model.blocks):
        xin = current; h1 = block.layer1(xin); branch = block.layer2(h1); out = xor(xin, branch)
        traces.append((i, xin, branch, out)); current = out
    return model.head(current), traces


def gradient_diagnostics(model, x, y, loss_name):
    model.zero_grad(set_to_none=True)
    out, traces = forward_trace(model, x)
    loss = loss_value(out, y, loss_name)
    blocks = []
    for i, xin, branch, yout in traces:
        gy = torch.autograd.grad(loss, yout, retain_graph=True)[0]
        gx = torch.autograd.grad(loss, xin, retain_graph=True)[0]
        direct = (1 - 2 * branch) * gy
        gbranch = torch.autograd.grad(branch, xin, grad_outputs=(1 - 2 * xin) * gy, retain_graph=True)[0]
        blocks.append({
            "block": i,
            "grad_input_l2": float(gx.norm()), "grad_output_l2": float(gy.norm()),
            "grad_input_output_ratio": float(gx.norm() / gy.norm().clamp_min(1e-20)),
            "mean_abs_input_output_ratio": float(gx.abs().mean() / gy.abs().mean().clamp_min(1e-20)),
            "cosine_input_output": float(torch.nn.functional.cosine_similarity(gx.reshape(1, -1), gy.reshape(1, -1)).item()),
            "mean_direct_gain": float((1 - 2 * branch).detach().mean()),
            "mean_abs_direct_gain": float((1 - 2 * branch).detach().abs().mean()),
            "direct_l2": float(direct.norm()), "branch_l2": float(gbranch.norm()),
            "direct_over_total": float(direct.norm() / gx.norm().clamp_min(1e-20)),
            "branch_over_total": float(gbranch.norm() / gx.norm().clamp_min(1e-20)),
            "direct_branch_cosine": float(torch.nn.functional.cosine_similarity(direct.reshape(1, -1), gbranch.reshape(1, -1)).item()),
            "decomposition_relative_error": float((gx - direct - gbranch).norm() / gx.norm().clamp_min(1e-20)),
            "branch_mean_min": float(torch.minimum(branch, 1 - branch).detach().mean()),
            "branch_fraction_lt_0_1": float((branch.detach() < .1).float().mean()),
            "branch_fraction_gt_0_9": float((branch.detach() > .9).float().mean()),
            "branch_fraction_mid_0_4_0_6": float(((branch.detach() >= .4) & (branch.detach() <= .6)).float().mean()),
        })
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    by_id = {id(p): (g if g is not None else torch.zeros_like(p)) for p, g in zip(params, grads)}
    layers = [("stem", model.stem)]
    for i, block in enumerate(model.blocks): layers.extend([(f"block{i}.layer1", block.layer1), (f"block{i}.layer2", block.layer2)])
    layers.append(("head", model.head))
    layer_records = []
    for name, layer in layers:
        g = torch.cat([by_id[id(layer.raw_edge)].reshape(-1), by_id[id(layer.bias)].reshape(-1)])
        v = torch.cat([layer.raw_edge.detach().reshape(-1), layer.bias.detach().reshape(-1)])
        layer_records.append({"layer": name, "gradient_l2": float(g.norm()), "mean_abs_gradient": float(g.abs().mean()), "max_abs_gradient": float(g.abs().max()), "parameter_l2": float(v.norm()), "gradient_parameter_ratio": float(g.norm() / v.norm().clamp_min(1e-20))})
    return {"loss": float(loss.detach()), "blocks": blocks, "layers": layer_records, "first_logic_to_last_logic_gradient_ratio": float(layer_records[0]["gradient_l2"] / max(layer_records[-1]["gradient_l2"], 1e-20))}


def output_gradient_sparsity(model, x, y, loss_name):
    out = model(x); out.retain_grad(); loss = loss_value(out, y, loss_name)
    go = torch.autograd.grad(loss, out)[0]
    nonzero = go.abs() >= 1e-12
    row_bit = nonzero.float()
    return {
        "loss": float(loss.detach()), "fraction_nonzero_output_gradient": float(nonzero.float().mean()),
        "per_bit_gradient_l2": [float(go[:, i].norm()) for i in range(go.shape[1])],
        "per_bit_nonzero_fraction": [float(nonzero[:, i].float().mean()) for i in range(go.shape[1])],
        "max_credit_bit_fraction": [float((go.abs().argmax(dim=1) == i).float().mean()) for i in range(go.shape[1])],
        "gradient_l2": float(go.norm()),
    }


def train_one(seed, loss_name, x, y, chain, device):
    model = make_model(seed, device); init_hash = state_hash(snapshot(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0)
    start = time.perf_counter(); core_seconds = 0.0; step_times = []
    trajectory = []; recovery = {k: None for k in ("continuous_exact", "boolean_exact", "stable_boolean_exact", "e_inf_lt_0_25", "e_inf_lt_0_10", "e_inf_lt_0_05", "e_inf_lt_0_01")}; regressions = {k: 0 for k in recovery}; prior = {k: False for k in recovery}; diag_steps = set(DIAG_STEPS)
    def record(step, native=None):
        if step not in EVAL_STEPS: return
        rec = evaluate(model, x, y, chain); rec.update({"step": step, "native_loss": native, "core_seconds": core_seconds})
        conditions = {"continuous_exact": rec["continuous"]["exact_accuracy"] >= 1.0, "boolean_exact": rec["boolean"]["exact_accuracy"] >= 1.0, "e_inf_lt_0_25": rec["continuous"]["e_inf"] < .25, "e_inf_lt_0_10": rec["continuous"]["e_inf"] < .1, "e_inf_lt_0_05": rec["continuous"]["e_inf"] < .05, "e_inf_lt_0_01": rec["continuous"]["e_inf"] < .01}
        needs = step in diag_steps or any(now and recovery[k] is None for k, now in conditions.items())
        if needs:
            rec["gradient_diagnostics"] = gradient_diagnostics(model, x, y, loss_name)
            rec["output_gradient_sparsity"] = output_gradient_sparsity(model, x, y, loss_name)
        trajectory.append(rec)
        for k, now in conditions.items():
            if now and recovery[k] is None: recovery[k] = {"step": step, "core_seconds": core_seconds}
            if prior[k] and not now: regressions[k] += 1
            prior[k] = now
    initial = evaluate(model, x, y, chain); initial.update({"step": 0, "native_loss": float(loss_value(model(x), y, loss_name)), "core_seconds": 0.0, "gradient_diagnostics": gradient_diagnostics(model, x, y, loss_name), "output_gradient_sparsity": output_gradient_sparsity(model, x, y, loss_name)})
    trajectory.append(initial)
    for step in range(1, STEPS + 1):
        t0 = time.perf_counter(); optimizer.zero_grad(set_to_none=True); native = loss_value(model(x), y, loss_name); native.backward(); optimizer.step(); elapsed = time.perf_counter() - t0; core_seconds += elapsed; step_times.append(elapsed); record(step, float(native.detach()))
    # stable exact means exact at a scheduled point with no later regression.
    if recovery["boolean_exact"] is not None and regressions["boolean_exact"] == 0:
        recovery["stable_boolean_exact"] = recovery["boolean_exact"]
    warm = step_times[10:]
    return {"seed": seed, "loss": loss_name, "blocks": 2, "residual_enabled": True, "initial_state_sha256": init_hash, "optimizer": {"name": "Adam", "lr": .01, "weight_decay": 0.0, "steps": STEPS, "loss": loss_name}, "trajectory": trajectory, "recovery": recovery, "regressions": regressions, "stable_recovery": {k: recovery[k] is not None and regressions.get(k, 0) == 0 for k in recovery}, "final": trajectory[-1], "timing": {"total_wall_seconds": time.perf_counter() - start, "optimizer_core_seconds": core_seconds, "optimizer_steps": STEPS, "examples_processed": STEPS * len(x), "mean_ms_per_step": 1000 * statistics.mean(step_times), "median_ms_per_step_after_warmup": 1000 * statistics.median(warm)}, "checkpoint_policy": "disabled"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--output", default=str(OUT / "a3_tolerance_loss_results.json")); args = ap.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    task = build_task("binary_addition", {"bits": 4}); x = task["X"].float().to(device); y = task["Y"].float().to(device); chain = task["carry_chain_length"].to(device)
    results = []; hashes = {}
    for seed in SEEDS:
        hashes[seed] = state_hash(snapshot(make_model(seed, device)))
        for loss_name in LOSSES:
            run = train_one(seed, loss_name, x, y, chain, device)
            if run["initial_state_sha256"] != hashes[seed]: raise RuntimeError("paired initialization mismatch")
            results.append(run); print(seed, loss_name, run["recovery"], flush=True)
    payload = {"experiment": "A3-whole-word-tolerance-loss", "git_sha": None, "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}, "a2_audit": {"valid": True, "selected_blocks": 2, "rule": "deepest XOR-residual A2 depth with >=2/3 continuous-exact runs; fallback to two blocks", "evidence": "A2 XOR-residual depths 1,2,4,8 each had 0/3 continuous-exact runs; fallback applied", "a2_result": "research/operator_results/a2_depth_gradient_results.json"}, "baseline_selection": {"loss": "MSE", "reason": "A1 recovery rates tied at zero; MSE had strongest task-independent continuous result; A2 uses MSE only"}, "task": {"name": "binary_addition", "bits": 4, "rows": 256, "target_generator_agreement": bool(torch.equal(task["target_integer"], task["target_ripple"])), "target_bit_frequency": [float(v) for v in y.mean(dim=0)], "carry_chain_counts": {str(i): int((chain == i).sum()) for i in range(5)}}, "architecture": {"input_dim": 8, "width": 64, "output_dim": 5, "blocks": 2, "residual_enabled": True, "operator": "lehmer_p2", "initializer": "I2-B meanfield sigma2 + BIAS_ONE"}, "losses": {"epsilon": EPSILON, "q": Q, "softplus_k": K, "arms": list(LOSSES), "definitions": {"BASELINE": "mean |p-y|^2", "BIT_MEAN_RATIONAL": "mean T(|p-y|), T=d^2/(d^2+epsilon^2)", "ROW_MAX_RATIONAL": "mean T(max_j |p-y|)", "ROW_MAX_SOFTPLUS": "mean normalized softplus(k(max_j|p-y|-epsilon))"}}, "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": STEPS, "batch_size": 256, "evaluation_schedule": "every 10 through 500, then every 25", "checkpoint_policy": "disabled"}, "paired_initial_state_hashes": {str(k): v for k, v in hashes.items()}, "results": results}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n"); print(out)


if __name__ == "__main__": main()
