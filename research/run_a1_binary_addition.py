"""A1: exhaustive four-bit binary addition benchmark.

The runner keeps the XOR architecture, Lehmer-p2 operator, I2-B initializer,
and full-batch Adam training fixed while measuring both Boolean quality and
time to recovery.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
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
    bias_one_normal_init_,
    meanfield_gaussian_edge_init_,
)

OUT = ROOT / "research/operator_results"
CKOUT = OUT / "a1_binary_addition_checkpoints"
LOSSES = ("MSE", "BCE", "POWER_1_5", "POWER_1_25")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 3000
BITS = 4
THRESHOLDS = (0.25, 0.10, 0.05, 0.01)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_model(seed: int, device: torch.device) -> SigmoidOrModernLogicGateNet:
    """Construct the I2-B model generalized only to a five-bit head."""
    edge_generator = torch.Generator(device="cpu").manual_seed(100_000 + seed)
    bias_generator = torch.Generator(device="cpu").manual_seed(200_000 + seed)

    def edge_init(tensor: torch.Tensor, fan_in: int) -> None:
        meanfield_gaussian_edge_init_(tensor, fan_in, sigma=2.0, generator=edge_generator)

    def bias_init(tensor: torch.Tensor) -> None:
        bias_one_normal_init_(tensor, std=0.1, generator=bias_generator)

    return SigmoidOrModernLogicGateNet(
        8,
        5,
        width=64,
        num_residual_blocks=2,
        or_operator="lehmer_p2",
        gate_initializations=[0.5] * 6,
        bias_initialization=bias_init,
        edge_initialization=edge_init,
    ).to(device)


def loss_value(output: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "MSE":
        return (output - target).square().mean()
    if loss_name == "BCE":
        return F.binary_cross_entropy(output.clamp(1e-7, 1.0 - 1e-7), target)
    if loss_name == "POWER_1_5":
        return (output - target).abs().pow(1.5).mean()
    if loss_name == "POWER_1_25":
        return (output - target).abs().pow(1.25).mean()
    raise ValueError(f"unknown loss: {loss_name}")


def metric_dict(output: torch.Tensor, target: torch.Tensor) -> dict:
    prediction = output >= 0.5
    expected = target >= 0.5
    row_correct = (prediction == expected).all(dim=1)
    bit_correct = prediction == expected
    error = (output - target).abs()
    return {
        "bit_accuracy": float(bit_correct.float().mean()),
        "exact_accuracy": float(row_correct.float().mean()),
        "wrong_rows": int((~row_correct).sum()),
        "wrong_bits": int((~bit_correct).sum()),
        "mse": float((output - target).square().mean()),
        "mae": float(error.mean()),
        "e_inf": float(error.max()),
        "p95_abs_error": float(torch.quantile(error.reshape(-1), 0.95)),
        "p99_abs_error": float(torch.quantile(error.reshape(-1), 0.99)),
        "per_bit_accuracy": [float(value) for value in bit_correct.float().mean(dim=0)],
        "per_bit_wrong_count": [int(value) for value in (~bit_correct).sum(dim=0)],
    }


def carry_group_metrics(output: torch.Tensor, target: torch.Tensor, chain: torch.Tensor) -> dict:
    prediction = output >= 0.5
    expected = target >= 0.5
    row_correct = (prediction == expected).all(dim=1)
    result = {}
    for length in range(BITS + 1):
        mask = chain == length
        result[str(length)] = {
            "rows": int(mask.sum()),
            "exact_accuracy": float(row_correct[mask].float().mean()) if mask.any() else None,
            "bit_accuracy": float((prediction[mask] == expected[mask]).float().mean()) if mask.any() else None,
        }
    return result


def mismatch_trace(model: SigmoidOrModernLogicGateNet, x: torch.Tensor) -> list[dict]:
    discrete = model.to_discrete(0.5).to(x.device)
    continuous = x
    boolean = x.bool()
    rows = []

    def record(name: str, cont: torch.Tensor, exact: torch.Tensor) -> None:
        mismatch = (cont >= 0.5) != exact.bool()
        rows.append({
            "name": name,
            "bit_mismatch_fraction": float(mismatch.float().mean()),
            "sample_mismatch_fraction": float(mismatch.any(dim=1).float().mean()),
        })

    record("input", continuous, boolean)
    continuous = model.stem(continuous)
    boolean = discrete.stem(boolean)
    record("stem", continuous, boolean)
    for index, block in enumerate(model.blocks):
        continuous_1 = block.layer1(continuous)
        boolean_1 = discrete.blocks[index].layer1(boolean)
        record(f"block{index}.layer1", continuous_1, boolean_1)
        continuous_2 = block.layer2(continuous_1)
        boolean_2 = discrete.blocks[index].layer2(boolean_1)
        record(f"block{index}.layer2", continuous_2, boolean_2)
        continuous = continuous + continuous_2 - 2.0 * continuous * continuous_2
        boolean = boolean ^ boolean_2
        record(f"block{index}.residual", continuous, boolean)
    record("head", model.head(continuous), discrete.head(boolean))
    return rows


@torch.no_grad()
def evaluate(model: SigmoidOrModernLogicGateNet, x: torch.Tensor, y: torch.Tensor, chain: torch.Tensor) -> dict:
    continuous = model(x)
    hard = model.forward_hard(x)
    boolean = model.to_discrete(0.5).to(x.device)(x.bool()).float()
    return {
        "continuous": metric_dict(continuous, y),
        "hard": metric_dict(hard, y),
        "boolean": metric_dict(boolean, y),
        "continuous_carry_chain": carry_group_metrics(continuous, y, chain),
        "boolean_carry_chain": carry_group_metrics(boolean, y, chain),
        "hard_carry_chain": carry_group_metrics(hard, y, chain),
        "bit_disagreement_fraction": float(((continuous >= 0.5) != boolean).float().mean()),
        "row_disagreement_fraction": float(((continuous >= 0.5) != boolean).any(dim=1).float().mean()),
        "mean_hamming_continuous_to_boolean": float(((continuous >= 0.5) != boolean).sum(dim=1).float().mean()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def scale_diagnostic(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict:
    base = make_model(0, device)
    initial = snapshot(base)
    records = {}
    updates = {}
    for scale in (1.0, 10.0):
        model = make_model(0, device)
        model.load_state_dict({key: value.to(device) for key, value in initial.items()})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = snapshot(model)
        output = model(x)
        loss = scale * loss_value(output, y, "MSE")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        synchronize(device)
        optimizer.step()
        after = snapshot(model)
        per_tensor = {}
        total_sq = 0.0
        for key in before:
            delta = (after[key] - before[key]).float()
            norm = float(delta.norm())
            per_tensor[key] = norm
            total_sq += norm * norm
        updates[str(scale)] = {"per_tensor_l2": per_tensor, "global_l2": total_sq**0.5}
    updates["ratio_10x_to_1x"] = {
        key: updates["10.0"]["per_tensor_l2"][key] / max(updates["1.0"]["per_tensor_l2"][key], 1e-30)
        for key in updates["1.0"]["per_tensor_l2"]
    }
    updates["global_ratio"] = updates["10.0"]["global_l2"] / max(updates["1.0"]["global_l2"], 1e-30)
    return updates


def full_adder_smoke() -> dict:
    task = build_task("full_adder", {})
    expected = []
    for a in (0, 1):
        for b in (0, 1):
            for carry in (0, 1):
                expected.append((a ^ b ^ carry, int(a + b + carry >= 2)))
    observed = [tuple(int(value) for value in row) for row in task["Y"].tolist()]
    return {"rows": len(observed), "passed": observed == expected}


def save_checkpoint(
    path: Path,
    model: SigmoidOrModernLogicGateNet,
    record: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    chain: torch.Tensor,
    condition: str,
    seed: int,
    kind: str,
) -> dict:
    state = snapshot(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    check = make_model(seed, x.device)
    check.load_state_dict({key: value.to(x.device) for key, value in state.items()})
    checked = evaluate(check, x, y, chain)
    for mode in ("continuous", "hard", "boolean"):
        for key in ("bit_accuracy", "exact_accuracy", "e_inf"):
            if abs(checked[mode][key] - record[mode][key]) > 1e-6:
                raise RuntimeError(f"checkpoint verification failed for {path}: {mode}.{key}")
    try:
        stored_path = str(path.relative_to(OUT))
    except ValueError:
        stored_path = str(path)
    return {
        "path": stored_path,
        "sha256": file_hash(path),
        "kind": kind,
        "seed": seed,
        "condition": condition,
        "step": record["step"],
        "metrics": checked,
        "reload_verified": True,
    }


def train_one(
    seed: int,
    loss_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    chain: torch.Tensor,
    device: torch.device,
    save_checkpoints: bool = False,
) -> dict:
    model = make_model(seed, device)
    initial_state = snapshot(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    run_start = time.perf_counter()
    core_seconds = 0.0
    step_times: list[float] = []
    eval_steps = {0, *range(10, 501, 10), *range(525, STEPS + 1, 25), STEPS}
    trajectory = []
    recovery = {
        "continuous_bit_exact": None,
        "continuous_exact": None,
        "hard_exact": None,
        "boolean_exact": None,
        **{f"e_inf_lt_{threshold:g}": None for threshold in THRESHOLDS},
    }
    regressions = {key: 0 for key in recovery}
    previously_true = {key: False for key in recovery}
    saved: dict[str, dict] = {}
    best_e_inf = float("inf")
    first_continuous_step = None
    first_boolean_step = None

    def maybe_save(kind: str, path: Path, metrics: dict) -> None:
        if save_checkpoints:
            saved[kind] = save_checkpoint(
                path, model, metrics, x, y, chain, loss_name, seed, kind
            )

    def maybe_recover(record: dict) -> None:
        nonlocal first_continuous_step, first_boolean_step, best_e_inf
        metrics = record
        conditions = {
            "continuous_bit_exact": metrics["continuous"]["bit_accuracy"] >= 1.0,
            "continuous_exact": metrics["continuous"]["exact_accuracy"] >= 1.0,
            "hard_exact": metrics["hard"]["exact_accuracy"] >= 1.0,
            "boolean_exact": metrics["boolean"]["exact_accuracy"] >= 1.0,
            **{
                f"e_inf_lt_{threshold:g}": metrics["continuous"]["e_inf"] < threshold
                for threshold in THRESHOLDS
            },
        }
        for key, true_now in conditions.items():
            if true_now and recovery[key] is None:
                recovery[key] = {"step": metrics["step"], "core_seconds": metrics["core_seconds"]}
            if previously_true[key] and not true_now:
                regressions[key] += 1
            previously_true[key] = true_now
        if first_continuous_step is None and conditions["continuous_exact"]:
            first_continuous_step = metrics["step"]
            maybe_save("first_continuous_exact",
                CKOUT / f"seed{seed}_{loss_name.lower()}" / "first_continuous_exact.pt",
                metrics,
            )
        if first_boolean_step is None and conditions["boolean_exact"]:
            first_boolean_step = metrics["step"]
            maybe_save("first_boolean_exact",
                CKOUT / f"seed{seed}_{loss_name.lower()}" / "first_boolean_exact.pt",
                metrics,
            )
        if metrics["continuous"]["e_inf"] < best_e_inf:
            best_e_inf = metrics["continuous"]["e_inf"]
            maybe_save("best_e_inf",
                CKOUT / f"seed{seed}_{loss_name.lower()}" / "best_e_inf.pt",
                metrics,
            )

    def evaluate_and_record(step: int, native_loss: float | None) -> None:
        record = evaluate(model, x, y, chain)
        record.update({
            "step": step,
            "native_loss": native_loss,
            "core_seconds": core_seconds,
            "layer_mismatch": mismatch_trace(model, x),
        })
        trajectory.append(record)
        maybe_recover(record)

    initial_eval = evaluate(model, x, y, chain)
    initial_eval.update({
        "step": 0,
        "native_loss": float(loss_value(model(x), y, loss_name).detach()),
        "core_seconds": 0.0,
        "layer_mismatch": mismatch_trace(model, x),
    })
    trajectory.append(initial_eval)
    maybe_recover(initial_eval)
    maybe_save("initial",
        CKOUT / f"seed{seed}_{loss_name.lower()}" / "initial.pt",
        initial_eval,
    )

    for step in range(1, STEPS + 1):
        synchronize(device)
        started = time.perf_counter()
        output = model(x)
        native = loss_value(output, y, loss_name)
        optimizer.zero_grad(set_to_none=True)
        native.backward()
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        step_times.append(elapsed)
        core_seconds += elapsed
        if step in eval_steps:
            evaluate_and_record(step, float(native.detach()))

    maybe_save("final",
        CKOUT / f"seed{seed}_{loss_name.lower()}" / "final.pt",
        trajectory[-1],
    )
    total_wall_seconds = time.perf_counter() - run_start
    stable = {
        key: recovery[key] is not None and regressions[key] == 0
        for key in recovery
    }
    final = trajectory[-1]
    wrong_rows = []
    boolean_output = model.to_discrete(0.5).to(device)(x.bool()).float()
    wrong = ((boolean_output >= 0.5) != (y >= 0.5)).any(dim=1)
    for index in torch.where(wrong)[0].tolist():
        row = x[index].int().tolist()
        a = sum(row[i] << i for i in range(BITS))
        b = sum(row[BITS + i] << i for i in range(BITS))
        wrong_rows.append({
            "row": index,
            "a": a,
            "b": b,
            "sum": a + b,
            "carry_chain_length": int(chain[index]),
            "continuous_output": model(x[index:index + 1]).detach().cpu().tolist()[0],
            "hard_output": model.forward_hard(x[index:index + 1]).detach().cpu().tolist()[0],
            "boolean_output": boolean_output[index].cpu().tolist(),
            "target": y[index].cpu().tolist(),
        })
    warmup = step_times[10:] if len(step_times) > 10 else step_times
    return {
        "seed": seed,
        "loss": loss_name,
        "initial_state_sha256": state_hash(initial_state),
        "optimizer": {"name": "Adam", "lr": 0.01, "weight_decay": 0.0, "steps": STEPS},
        "trajectory": trajectory,
        "recovery": recovery,
        "regressions": regressions,
        "stable_recovery": stable,
        "checkpoints": saved,
        "checkpoint_policy": "disabled by default; pass --save-checkpoints for opt-in diagnostics",
        "final": final,
        "final_wrong_rows": wrong_rows,
        "timing": {
            "total_wall_seconds": total_wall_seconds,
            "optimizer_core_seconds": core_seconds,
            "optimizer_steps": STEPS,
            "examples_processed": STEPS * len(x),
            "mean_ms_per_step": 1000.0 * sum(step_times) / len(step_times),
            "median_ms_per_step_after_warmup": 1000.0 * float(torch.tensor(warmup).median()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(OUT / "a1_binary_addition_results.json"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Opt in to heavyweight .pt checkpoints; disabled by default.",
    )
    args = parser.parse_args()
    if args.steps != STEPS:
        raise ValueError("A1 uses exactly 3000 optimizer steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = build_task("binary_addition", {"bits": BITS})
    x = task["X"].float().to(device)
    y = task["Y"].float().to(device)
    chain = task["carry_chain_length"].to(device)
    full_adder = full_adder_smoke()
    if not full_adder["passed"]:
        raise RuntimeError("full-adder smoke test failed")
    target_agreement = bool(torch.equal(task["target_integer"], task["target_ripple"]))
    if not target_agreement:
        raise RuntimeError("integer/ripple target mismatch")
    scale = scale_diagnostic(x, y, device)
    results = []
    for seed in SEEDS:
        initial = make_model(seed, device)
        initial_state = snapshot(initial)
        paired_hash = state_hash(initial_state)
        for loss_name in LOSSES:
            check = make_model(seed, device)
            if state_hash(snapshot(check)) != paired_hash:
                raise RuntimeError(f"paired initialization mismatch for seed {seed}")
            result = train_one(
                seed, loss_name, x, y, chain, device,
                save_checkpoints=args.save_checkpoints,
            )
            if result["initial_state_sha256"] != paired_hash:
                raise RuntimeError(f"training initialization mismatch for seed {seed}/{loss_name}")
            results.append(result)
            print(seed, loss_name, result["recovery"], flush=True)
    payload = {
        "experiment": "A1-exact-binary-addition",
        "git_sha": None,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else None,
        },
        "task": {
            "name": "binary_addition",
            "bits": BITS,
            "rows": len(x),
            "input_order": "a0,a1,a2,a3,b0,b1,b2,b3",
            "output_order": "s0,s1,s2,s3,s4",
            "target_generator_agreement": target_agreement,
            "target_integer_sha256": hashlib.sha256(task["target_integer"].cpu().numpy().tobytes()).hexdigest(),
            "target_ripple_sha256": hashlib.sha256(task["target_ripple"].cpu().numpy().tobytes()).hexdigest(),
            "target_bit_frequency": [float(value) for value in y.mean(dim=0).cpu()],
            "carry_chain_counts": {str(i): int((chain == i).sum()) for i in range(BITS + 1)},
            "full_adder_smoke": full_adder,
        },
        "architecture": {
            "input_dim": 8,
            "width": 64,
            "residual_blocks": 2,
            "output_dim": 5,
            "operator": "lehmer_p2",
            "initializer": "I2-B meanfield sigma2 + BIAS_ONE",
        },
        "training": {
            "optimizer": "Adam",
            "lr": 0.01,
            "weight_decay": 0.0,
            "steps": STEPS,
            "batch_size": len(x),
            "losses": list(LOSSES),
            "evaluation_schedule": "every 10 steps through 500, then every 25",
            "scheduler": None,
            "regularizer": None,
            "ste": False,
            "checkpoint_policy": "disabled by default; use --save-checkpoints only for temporary local diagnostics",
        },
        "loss_scale_diagnostic": scale,
        "seeds": list(SEEDS),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(output)


if __name__ == "__main__":
    main()
