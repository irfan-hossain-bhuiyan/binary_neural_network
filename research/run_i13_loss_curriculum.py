"""I13: POWER_1_5 to POWER_1_25 clean-XOR loss curriculum."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i11_loss_geometry import make_model, state_hash  # noqa: E402
from research.run_i12_loss_seed_replication import (  # noqa: E402
    evaluate, gate_bias_stats, threshold_robustness, snapshot,
)
from research.run_initialization_i2 import model_for  # noqa: E402

OUT = ROOT / "research/operator_results"
CKOUT = OUT / "i13_loss_curriculum_checkpoints"
SEEDS = tuple(range(5))
STEPS = 3000
EVAL_INTERVAL = 25
EVAL_STEPS = tuple(range(0, STEPS + 1, EVAL_INTERVAL))
EXPECTED_INITIAL_HASHES = {
    0: "d8fad20680124095e77b8921161097c7a4a6f49033840fba05d37b3243d5ef6a",
    1: "e9dda60f4dc96cfbe1a83f7cc4b89d77baed27ea78617440138d8c8053f531cd",
    2: "c21d073cc2c86a097786e45f730910bb970dab1d2400254f2a7489577bed59f4",
    3: "9fe12ddd7c1e003c794d990533a7ecc44d5d729c7f6b3f5018b95c8243bad031",
    4: "7ebc3a89d2db1815cd5525e9e34ce908ecdebbb98770f9b603ae4926d7b4ef03",
}


def alpha_loss(output, target, alpha):
    return (output - target).abs().pow(alpha).mean()


def curriculum_alpha(step, trigger):
    if trigger is None or step <= trigger:
        return 1.5, "POWER_1_5"
    progress = min(1.0, (step - trigger) / 500.0)
    alpha = 1.5 - .25 * progress
    return alpha, "ANNEAL" if progress < 1.0 else "POWER_1_25"


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path, state, seed, kind, step, alpha, phase, initial_hash, x, y, trigger):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    check = make_model()
    check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    metrics = evaluate(check, x, y)
    return {
        "path": str(path), "sha256": sha_file(path), "reload_verified": True,
        "reload_metrics": metrics, "seed": seed, "kind": kind, "step": step,
        "alpha": alpha, "curriculum_phase": phase, "trigger_step": trigger,
        "initial_state_sha256": initial_hash,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def run_one(seed, x, y, steps=STEPS):
    base = model_for("I2-B", seed, torch.device("cpu"))
    initial = snapshot(base)
    initial_hash = state_hash(initial)
    if initial_hash != EXPECTED_INITIAL_HASHES[seed]:
        raise RuntimeError(f"I12 initial hash mismatch seed {seed}: {initial_hash}")
    net = make_model(); net.load_state_dict(initial)
    if state_hash(snapshot(net)) != initial_hash:
        raise RuntimeError(f"initial reload mismatch seed {seed}")
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.0)
    trajectory = []
    trigger = None
    exact_streak = 0
    first_cont = first_hard = first_bool = None
    bool_regressions = 0
    last_bool = False
    states = {}
    best_einf = (float("inf"), None, None, None, None)
    for step in range(steps + 1):
        alpha, phase = curriculum_alpha(step, trigger)
        if step % EVAL_INTERVAL == 0 or step == steps:
            ev = evaluate(net, x, y)
            c_exact = ev["continuous"]["exact_accuracy"] == 1.0
            h_exact = ev["hard"]["exact_accuracy"] == 1.0
            b_exact = ev["boolean"]["exact_accuracy"] == 1.0
            if first_cont is None and c_exact: first_cont = step
            if first_hard is None and h_exact: first_hard = step
            if first_bool is None and b_exact:
                first_bool = step
                states["first_boolean_exact"] = snapshot(net)
            if last_bool and not b_exact: bool_regressions += 1
            last_bool = b_exact
            if c_exact: exact_streak += 1
            else: exact_streak = 0
            if trigger is None and step <= 2000 and exact_streak >= 3:
                trigger = step
                states["before_trigger"] = snapshot(net)
                alpha, phase = curriculum_alpha(step, trigger)
            rec = {
                "step": step, "alpha": alpha, "curriculum_phase": phase,
                **ev,
                "current_l_alpha": float(alpha_loss(net(x), y, alpha).detach()),
                "edge_mask_changes_since_previous": None,
                "bias_mask_changes_since_previous": None,
                "edge_hamming_from_initial": None,
                "bias_hamming_from_initial": None,
            }
            edge_masks = []; bias_masks = []
            for layer in net.expectation_layers:
                edge_masks.append((layer.effective_gate().detach() >= .5).cpu())
                bias_masks.append((layer.actual_bias().detach() >= .5).cpu())
            if trajectory:
                rec["edge_mask_changes_since_previous"] = int(sum((a != b).sum() for a, b in zip(edge_masks, previous_edges)))
                rec["bias_mask_changes_since_previous"] = int(sum((a != b).sum() for a, b in zip(bias_masks, previous_biases)))
            rec["edge_hamming_from_initial"] = int(sum((a != b).sum() for a, b in zip(edge_masks, initial_edges))) if 'initial_edges' in locals() else 0
            rec["bias_hamming_from_initial"] = int(sum((a != b).sum() for a, b in zip(bias_masks, initial_biases))) if 'initial_biases' in locals() else 0
            if not trajectory:
                initial_edges, initial_biases = [m.clone() for m in edge_masks], [m.clone() for m in bias_masks]
                rec["edge_hamming_from_initial"] = rec["bias_hamming_from_initial"] = 0
            previous_edges, previous_biases = edge_masks, bias_masks
            trajectory.append(rec)
            if ev["continuous"]["e_inf"] < best_einf[0]:
                best_einf = (ev["continuous"]["e_inf"], step, snapshot(net), alpha, phase)
            if trigger is not None and step == trigger + 500:
                states["end_anneal"] = snapshot(net)
        if step == steps: break
        output = net(x)
        loss = alpha_loss(output, y, alpha)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    if trigger is None:
        states.pop("before_trigger", None); states.pop("end_anneal", None)
    final_state = snapshot(net)
    checkpoint_dir = CKOUT / f"seed{seed}"
    checkpoints = {
        "best_e_inf": save_checkpoint(checkpoint_dir / "best_e_inf.pt", best_einf[2], seed, "best_e_inf", best_einf[1], best_einf[3], best_einf[4], initial_hash, x, y, trigger),
        "final": save_checkpoint(checkpoint_dir / "final.pt", final_state, seed, "final", steps, trajectory[-1]["alpha"], trajectory[-1]["curriculum_phase"], initial_hash, x, y, trigger),
    }
    for kind, state in states.items():
        step = trigger if kind == "before_trigger" else trigger + 500 if kind == "end_anneal" else first_bool
        alpha, phase = curriculum_alpha(step, trigger)
        checkpoints[kind] = save_checkpoint(checkpoint_dir / f"{kind}.pt", state, seed, kind, step, alpha, phase, initial_hash, x, y, trigger)
    final_net = make_model(); final_net.load_state_dict(final_state)
    final_eval = evaluate(final_net, x, y)
    stable = first_bool is not None and all(r["boolean"]["exact_accuracy"] == 1.0 for r in trajectory if r["step"] >= first_bool)
    return {
        "seed": seed, "initial_state_sha256": initial_hash, "optimizer_steps": steps,
        "trigger_step": trigger, "curriculum_not_triggered": trigger is None,
        "first_continuous_exact": first_cont, "first_hard_exact": first_hard,
        "first_boolean_exact": first_bool, "alpha_at_first_boolean_exact": curriculum_alpha(first_bool, trigger)[0] if first_bool is not None else None,
        "boolean_regressions_after_first_exact": bool_regressions,
        "stable_boolean_exact": stable, "trajectory": trajectory,
        "final": final_eval, "final_gate_bias_stats": gate_bias_stats(final_net),
        "threshold_robustness": threshold_robustness(final_net, x, y),
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--steps", type=int, default=STEPS); parser.add_argument("--threads", type=int, default=1); parser.add_argument("--output", required=True); args = parser.parse_args()
    torch.set_num_threads(args.threads)
    task = build_task("bitwise_xor_truth_table", {"bits": 4}); x, y = task["X"].float(), task["Y"].float()
    result = run_one(args.seed, x, y, args.steps)
    payload = {
        "experiment": "I13-loss-curriculum", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "architecture": {"input_dim": 8, "width": 64, "residual_blocks": 2, "operator": "lehmer_p2"},
        "initializer": "I2-B meanfield sigma2 + BIAS_ONE", "dataset_rows": 256,
        "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": args.steps, "initial_alpha": 1.5, "final_alpha": 1.25, "anneal_steps": 500, "trigger": "continuous exact for 3 consecutive evaluations every 25 steps; eligibility through step 2000"},
        "run": result,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(args.seed, result["trigger_step"], result["final"]["boolean"]["exact_accuracy"], flush=True)


if __name__ == "__main__": main()
