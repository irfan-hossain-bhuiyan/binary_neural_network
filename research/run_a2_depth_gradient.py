"""A2: depth and XOR-residual gradient propagation on four-bit addition.

The runner uses the A1-selected MSE task loss, full-batch Adam, and the same
I2-B initializer. It records activation-gradient decomposition diagnostics but
never saves parameter checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.meanfield_initialization import (  # noqa: E402
    bias_one_normal_init_,
    meanfield_gaussian_edge_init_,
)

OUT = ROOT / "research/operator_results"
LOSSES = ("MSE",)
DEPTHS = (0, 1, 2, 4, 8)
SEEDS = (0, 1, 2)
STEPS = 3000
EVAL_STEPS = {0, *range(10, 501, 10), *range(525, STEPS + 1, 25), STEPS}
DIAG_STEPS = {0, 100, 500, 1000, 2000, 3000}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(seed: int, blocks: int, residual: bool, device: torch.device):
    edge_generator = torch.Generator(device="cpu").manual_seed(100_000 + seed)
    bias_generator = torch.Generator(device="cpu").manual_seed(200_000 + seed)

    def edge_init(tensor: torch.Tensor, fan_in: int) -> None:
        meanfield_gaussian_edge_init_(tensor, fan_in, sigma=2.0, generator=edge_generator)

    def bias_init(tensor: torch.Tensor) -> None:
        bias_one_normal_init_(tensor, std=0.1, generator=bias_generator)

    return SigmoidOrModernLogicGateNet(
        8, 5, width=64, num_residual_blocks=blocks,
        or_operator="lehmer_p2", gate_initializations=[0.5] * (2 + 2 * blocks),
        bias_initialization=bias_init, edge_initialization=edge_init,
        residual_enabled=residual,
    ).to(device)


def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - 2 * a * b


def forward_trace(model, x: torch.Tensor, retain: bool = False):
    """Return output and block input/branch/output tensors for diagnostics."""
    stem = model.stem(x)
    current = stem
    traces = []
    for index, block in enumerate(model.blocks):
        block_input = current
        if retain:
            block_input.retain_grad()
        h1 = block.layer1(block_input)
        branch = block.layer2(h1)
        if retain:
            branch.retain_grad()
        output = xor(block_input, branch) if block.residual_enabled else branch
        if retain:
            output.retain_grad()
        traces.append({"index": index, "input": block_input, "branch": branch, "output": output})
        current = output
    head = model.head(current)
    return head, traces


def loss_value(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output - target).square().mean()


def state_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    pred = output >= 0.5
    truth = target >= 0.5
    rows = (pred == truth).all(dim=1)
    bits = pred == truth
    error = (output - target).abs()
    return {
        "bit_accuracy": float(bits.float().mean()),
        "exact_accuracy": float(rows.float().mean()),
        "wrong_rows": int((~rows).sum()),
        "wrong_bits": int((~bits).sum()),
        "mse": float(error.square().mean()),
        "mae": float(error.mean()),
        "e_inf": float(error.max()),
        "p95_abs_error": float(torch.quantile(error.reshape(-1), 0.95)),
        "p99_abs_error": float(torch.quantile(error.reshape(-1), 0.99)),
        "per_bit_accuracy": [float(v) for v in bits.float().mean(dim=0)],
        "per_bit_wrong_count": [int(v) for v in (~bits).sum(dim=0)],
    }


def carry_metrics(output: torch.Tensor, target: torch.Tensor, chain: torch.Tensor) -> dict:
    pred = output >= 0.5
    truth = target >= 0.5
    rows = (pred == truth).all(dim=1)
    out = {}
    for n in range(5):
        mask = chain == n
        out[str(n)] = {
            "rows": int(mask.sum()),
            "exact_accuracy": float(rows[mask].float().mean()),
            "bit_accuracy": float((pred[mask] == truth[mask]).float().mean()),
        }
    return out


@torch.no_grad()
def evaluate(model, x, y, chain) -> dict:
    continuous = model(x)
    hard = model.forward_hard(x)
    boolean = model.to_discrete(0.5)(x.bool()).float()
    return {
        "continuous": state_metrics(continuous, y),
        "hard": state_metrics(hard, y),
        "boolean": state_metrics(boolean, y),
        "continuous_carry_chain": carry_metrics(continuous, y, chain),
        "hard_carry_chain": carry_metrics(hard, y, chain),
        "boolean_carry_chain": carry_metrics(boolean, y, chain),
        "bit_disagreement_fraction": float(((continuous >= .5) != boolean).float().mean()),
        "row_disagreement_fraction": float(((continuous >= .5) != boolean).any(dim=1).float().mean()),
        "mean_hamming_continuous_to_boolean": float(((continuous >= .5) != boolean).sum(dim=1).float().mean()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa, bb = a.reshape(-1), b.reshape(-1)
    den = aa.norm() * bb.norm()
    if float(den) <= 1e-20:
        return 0.0
    return float(torch.dot(aa, bb) / den)


def layer_parameters(model):
    layers = [("stem", model.stem)]
    for i, block in enumerate(model.blocks):
        layers.extend([(f"block{i}.layer1", block.layer1), (f"block{i}.layer2", block.layer2)])
    layers.append(("head", model.head))
    return layers


def gradient_diagnostics(model, x, y, residual_enabled: bool) -> dict:
    """Measure activation and parameter gradients at the current state."""
    model.zero_grad(set_to_none=True)
    output, traces = forward_trace(model, x, retain=True)
    loss = loss_value(output, y)
    # Keep one graph while obtaining blockwise decomposition and parameters.
    block_records = []
    for trace in traces:
        xin, branch, yout = trace["input"], trace["branch"], trace["output"]
        gy = torch.autograd.grad(loss, yout, retain_graph=True, allow_unused=True)[0]
        gx = torch.autograd.grad(loss, xin, retain_graph=True, allow_unused=True)[0]
        if gy is None or gx is None:
            continue
        transfer = {
            "grad_input_l2": float(gx.norm()),
            "grad_output_l2": float(gy.norm()),
            "grad_input_output_ratio": float(gx.norm() / gy.norm().clamp_min(1e-20)),
            "mean_abs_input_output_ratio": float(gx.abs().mean() / gy.abs().mean().clamp_min(1e-20)),
            "cosine_input_output": cosine(gx, gy),
        }
        if residual_enabled:
            direct = (1.0 - 2.0 * branch) * gy
            branch_upstream = (1.0 - 2.0 * xin) * gy
            gbranch = torch.autograd.grad(branch, xin, grad_outputs=branch_upstream, retain_graph=True, allow_unused=True)[0]
            if gbranch is None:
                gbranch = torch.zeros_like(gx)
            recomposed = direct + gbranch
            transfer.update({
                "mean_direct_gain": float((1.0 - 2.0 * branch).detach().mean()),
                "mean_abs_direct_gain": float((1.0 - 2.0 * branch).detach().abs().mean()),
                "median_abs_direct_gain": float(torch.median((1.0 - 2.0 * branch).detach().abs())),
                "fraction_abs_direct_gain_lt_0_1": float(((1.0 - 2.0 * branch).abs() < .1).float().mean()),
                "fraction_abs_direct_gain_gt_0_9": float(((1.0 - 2.0 * branch).abs() > .9).float().mean()),
                "direct_l2": float(direct.norm()),
                "branch_l2": float(gbranch.norm()),
                "direct_over_total": float(direct.norm() / gx.norm().clamp_min(1e-20)),
                "branch_over_total": float(gbranch.norm() / gx.norm().clamp_min(1e-20)),
                "direct_branch_cosine": cosine(direct, gbranch),
                "decomposition_relative_error": float((gx - recomposed).norm() / gx.norm().clamp_min(1e-20)),
                "branch_mean_min": float(torch.minimum(branch, 1.0 - branch).detach().mean()),
                "branch_fraction_lt_0_1": float((branch.detach() < .1).float().mean()),
                "branch_fraction_gt_0_9": float((branch.detach() > .9).float().mean()),
                "branch_fraction_mid_0_4_0_6": float(((branch.detach() >= .4) & (branch.detach() <= .6)).float().mean()),
            })
        else:
            transfer.update({
                "mean_direct_gain": None, "mean_abs_direct_gain": None,
                "median_abs_direct_gain": None, "fraction_abs_direct_gain_lt_0_1": None,
                "fraction_abs_direct_gain_gt_0_9": None, "direct_l2": None,
                "branch_l2": float(gx.norm()), "direct_over_total": None,
                "branch_over_total": 1.0, "direct_branch_cosine": None,
                "decomposition_relative_error": None,
                "branch_mean_min": float(torch.minimum(branch, 1.0 - branch).detach().mean()),
                "branch_fraction_lt_0_1": float((branch.detach() < .1).float().mean()),
                "branch_fraction_gt_0_9": float((branch.detach() > .9).float().mean()),
                "branch_fraction_mid_0_4_0_6": float(((branch.detach() >= .4) & (branch.detach() <= .6)).float().mean()),
            })
        transfer["block"] = trace["index"]
        block_records.append(transfer)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    grad_by_id = {id(p): (g if g is not None else torch.zeros_like(p)) for p, g in zip(params, grads)}
    layer_records = []
    for name, layer in layer_parameters(model):
        selected = [layer.raw_edge, layer.bias]
        g = torch.cat([grad_by_id[id(p)].reshape(-1) for p in selected])
        v = torch.cat([p.detach().reshape(-1) for p in selected])
        layer_records.append({
            "layer": name, "gradient_l2": float(g.norm()),
            "mean_abs_gradient": float(g.abs().mean()), "max_abs_gradient": float(g.abs().max()),
            "parameter_l2": float(v.norm()),
            "gradient_parameter_ratio": float(g.norm() / v.norm().clamp_min(1e-20)),
            "raw_edge_gradient_l2": float(grad_by_id[id(layer.raw_edge)].norm()),
        })
    first = layer_records[0]["gradient_l2"]
    last = layer_records[-1]["gradient_l2"]
    return {
        "loss": float(loss.detach()),
        "blocks": block_records,
        "layers": layer_records,
        "first_logic_to_last_logic_gradient_ratio": float(first / max(last, 1e-20)),
    }


def mismatch_trace(model, x):
    disc = model.to_discrete(0.5)
    current = x
    exact = x.bool()
    rows = []
    def record(name, cont, boolean):
        mismatch = (cont >= .5) != boolean.bool()
        rows.append({"name": name, "bit_mismatch_fraction": float(mismatch.float().mean()), "sample_mismatch_fraction": float(mismatch.any(dim=1).float().mean())})
    record("input", current, exact)
    current, _ = model.stem(current), None
    exact = disc.stem(exact)
    record("stem", current, exact)
    for i, block in enumerate(model.blocks):
        h1 = block.layer1(current); e1 = disc.blocks[i].layer1(exact); record(f"block{i}.layer1", h1, e1)
        h2 = block.layer2(h1); e2 = disc.blocks[i].layer2(e1); record(f"block{i}.layer2", h2, e2)
        current = xor(current, h2) if block.residual_enabled else h2
        exact = exact ^ e2 if block.residual_enabled else e2
        record(f"block{i}.residual", current, exact)
    record("head", model.head(current), disc.head(exact))
    return rows


def evaluate_with_trace(model, x, y, chain, residual):
    rec = evaluate(model, x, y, chain)
    rec["layer_mismatch"] = mismatch_trace(model, x)
    rec["gradient_diagnostics"] = gradient_diagnostics(model, x, y, residual)
    return rec


def train_one(seed, blocks, residual, x, y, chain, device):
    model = make_model(seed, blocks, residual, device)
    init_hash = state_hash(snapshot(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0)
    start = time.perf_counter(); core_seconds = 0.0; step_times = []
    recovery = {"continuous_exact": None, "boolean_exact": None, "e_inf_lt_0_1": None}
    regressions = {k: 0 for k in recovery}; prior = {k: False for k in recovery}
    trajectory = []
    diag_steps = set(DIAG_STEPS)
    def record(step, native=None):
        nonlocal diag_steps
        if step in EVAL_STEPS:
            rec = evaluate(model, x, y, chain)
            rec.update({"step": step, "native_loss": native, "core_seconds": core_seconds})
            conditions = {"continuous_exact": rec["continuous"]["exact_accuracy"] >= 1.0,
                          "boolean_exact": rec["boolean"]["exact_accuracy"] >= 1.0,
                          "e_inf_lt_0_1": rec["continuous"]["e_inf"] < .1}
            needs_diag = step in diag_steps or any(
                now and recovery[key] is None for key, now in conditions.items()
            )
            if needs_diag:
                rec.update({"layer_mismatch": mismatch_trace(model, x), "gradient_diagnostics": gradient_diagnostics(model, x, y, residual)})
            trajectory.append(rec)
            for key, now in conditions.items():
                if now and recovery[key] is None: recovery[key] = {"step": step, "core_seconds": core_seconds}
                if prior[key] and not now: regressions[key] += 1
                prior[key] = now
                if now and step not in DIAG_STEPS: diag_steps.add(step)
    initial = evaluate_with_trace(model, x, y, chain, residual)
    initial.update({"step": 0, "native_loss": float(loss_value(model(x), y)), "core_seconds": 0.0})
    trajectory.append(initial)
    for key, now in {"continuous_exact": initial["continuous"]["exact_accuracy"] >= 1.0, "boolean_exact": initial["boolean"]["exact_accuracy"] >= 1.0, "e_inf_lt_0_1": initial["continuous"]["e_inf"] < .1}.items(): prior[key] = now
    for step in range(1, STEPS + 1):
        started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        native = loss_value(model(x), y); native.backward(); optimizer.step()
        elapsed = time.perf_counter() - started; core_seconds += elapsed; step_times.append(elapsed)
        record(step, float(native.detach()))
    final = trajectory[-1]
    warm = step_times[10:]
    return {
        "seed": seed, "blocks": blocks, "residual_enabled": residual,
        "initial_state_sha256": init_hash,
        "optimizer": {"name": "Adam", "lr": .01, "weight_decay": 0.0, "steps": STEPS, "loss": "MSE"},
        "trajectory": trajectory, "recovery": recovery, "regressions": regressions,
        "stable_recovery": {k: recovery[k] is not None and regressions[k] == 0 for k in recovery},
        "final": final,
        "timing": {"total_wall_seconds": time.perf_counter() - start, "optimizer_core_seconds": core_seconds, "optimizer_steps": STEPS, "examples_processed": STEPS * len(x), "mean_ms_per_step": 1000 * sum(step_times) / len(step_times), "median_ms_per_step_after_warmup": 1000 * float(torch.tensor(warm).median())},
        "checkpoint_policy": "disabled",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(OUT / "a2_depth_gradient_results.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    task = build_task("binary_addition", {"bits": 4}); x = task["X"].float().to(device); y = task["Y"].float().to(device); chain = task["carry_chain_length"].to(device)
    results = []
    for blocks in DEPTHS:
        modes = (True,) if blocks == 0 else (True, False)
        for residual in modes:
            for seed in SEEDS:
                # Verify the only paired variable is the residual equation.
                a = make_model(seed, blocks, residual, device); b = make_model(seed, blocks, residual, device)
                if state_hash(snapshot(a)) != state_hash(snapshot(b)): raise RuntimeError("initialization is not deterministic")
                result = train_one(seed, blocks, residual, x, y, chain, device)
                results.append(result); print(blocks, residual, seed, result["recovery"], flush=True)
    payload = {
        "experiment": "A2-depth-xor-residual-gradient-propagation", "git_sha": None,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "baseline_selection": {"a1_loss": "MSE", "reason": "all A1 recovery rates tied at zero; MSE had best final continuous exact rate and E_inf without comparing native loss scale"},
        "task": {"name": "binary_addition", "bits": 4, "rows": len(x), "target_generator_agreement": bool(torch.equal(task["target_integer"], task["target_ripple"])), "target_bit_frequency": [float(v) for v in y.mean(dim=0)], "carry_chain_counts": {str(i): int((chain == i).sum()) for i in range(5)}},
        "architecture": {"input_dim": 8, "width": 64, "output_dim": 5, "operator": "lehmer_p2", "blocks": list(DEPTHS), "residual_modes": ["XOR_RESIDUAL", "NO_RESIDUAL"], "initializer": "I2-B meanfield sigma2 + BIAS_ONE"},
        "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": STEPS, "batch_size": len(x), "loss": "MSE", "evaluation_schedule": "every 10 through 500, then every 25", "checkpoint_policy": "disabled"},
        "results": results,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n"); print(out)


if __name__ == "__main__": main()
