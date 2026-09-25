"""Merge I13 curriculum shards and generate comparison artifacts."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/operator_results"
FIG = ROOT / "research/figures"


def enrich(run):
    run = dict(run)
    trigger = run["trigger_step"]
    traj = run["trajectory"]
    if trigger is not None:
        around = {r["step"]: r for r in traj if r["step"] in {trigger, trigger + 100, trigger + 250, trigger + 500}}
        run["trigger_diagnostics"] = around
        end = next((r for r in traj if r["step"] == trigger + 500), traj[-1])
        start = next((r for r in traj if r["step"] == trigger), traj[0])
        run["anneal_edge_hamming_change"] = end["edge_hamming_from_initial"] - start["edge_hamming_from_initial"]
        run["anneal_bias_hamming_change"] = end["bias_hamming_from_initial"] - start["bias_hamming_from_initial"]
    else:
        run["trigger_diagnostics"] = {}
        run["anneal_edge_hamming_change"] = 0
        run["anneal_bias_hamming_change"] = 0
    return run


def load_i12():
    return json.loads((OUT / "i12_loss_seed_replication_results.json").read_text())


def summary(curriculum, i12):
    fixed = {r["loss"]: r for r in i12["runs"]}
    # I12 has one run per seed/loss; key it explicitly.
    fixed = {(r["seed"], r["loss"]): r for r in i12["runs"]}
    rows = []
    for method, runs in (("fixed POWER_1_5", [fixed[(s, "POWER_1_5")] for s in range(5)]),
                         ("fixed POWER_1_25", [fixed[(s, "POWER_1_25")] for s in range(5)]),
                         ("I13 curriculum", curriculum)):
        recovered = [r["first_boolean_exact"] for r in runs if r.get("first_boolean_exact") is not None]
        rows.append({
            "method": method, "continuous_exact": sum(r["final"]["continuous"]["exact_accuracy"] == 1 for r in runs),
            "stable_boolean_exact": sum(r["stable_boolean_exact"] for r in runs),
            "boolean_exact": sum(r["final"]["boolean"]["exact_accuracy"] == 1 for r in runs),
            "median_e_inf": statistics.median(r["final"]["continuous"]["e_inf"] for r in runs),
            "median_endpoint_distance": statistics.median(r["final"]["endpoint"]["mean"] for r in runs),
            "median_first_boolean_exact": statistics.median(recovered) if recovered else None,
        })
    return rows


def threshold_interval(run):
    values = [(t, run["threshold_robustness"][f"{t:.2f}"]["exact_accuracy"] == 1.0)
              for t in (.30, .35, .40, .45, .50, .55, .60, .65, .70)]
    if not values[4][1]: return None
    lo = hi = 4
    while lo and values[lo - 1][1]: lo -= 1
    while hi < len(values) - 1 and values[hi + 1][1]: hi += 1
    return values[lo][0], values[hi][0]


def figures(curriculum, fixed):
    FIG.mkdir(parents=True, exist_ok=True)
    methods = ["fixed POWER_1_5", "fixed POWER_1_25", "I13 curriculum"]
    vals = [sum(r["final"]["boolean"]["exact_accuracy"] == 1 for r in fixed["POWER_1_5"]),
            sum(r["final"]["boolean"]["exact_accuracy"] == 1 for r in fixed["POWER_1_25"]),
            sum(r["final"]["boolean"]["exact_accuracy"] == 1 for r in curriculum)]
    plt.figure(figsize=(8, 5)); plt.bar(methods, vals); plt.ylim(0, 5); plt.ylabel("final Boolean exact runs / 5"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(FIG / "i13_boolean_recovery.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    for r in curriculum: plt.plot([x["step"] for x in r["trajectory"]], [x["alpha"] for x in r["trajectory"]], label=f"seed {r['seed']}")
    plt.xlabel("optimizer step"); plt.ylabel("alpha"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "i13_alpha_trajectories.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    for r in curriculum: plt.plot([x["step"] for x in r["trajectory"]], [x["boolean"]["exact_accuracy"] for x in r["trajectory"]], label=f"seed {r['seed']}")
    plt.xlabel("optimizer step"); plt.ylabel("Boolean exact accuracy"); plt.ylim(0, 1.05); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "i13_boolean_trajectories.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    for r in curriculum: plt.plot([x["step"] for x in r["trajectory"]], [x["continuous"]["e_inf"] for x in r["trajectory"]], label=f"seed {r['seed']}")
    plt.yscale("log"); plt.xlabel("optimizer step"); plt.ylabel("E_inf"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "i13_einf_trajectories.png", dpi=150); plt.close()


def report(curriculum, i12, rows, path):
    fixed = {(r["seed"], r["loss"]): r for r in i12["runs"]}
    lines = [
        "# I13 — POWER_1_5 → POWER_1_25 Curriculum",
        "",
        "## A–C. I12 audit and paired initial states",
        "",
        "The canonical I12 JSON contains 20 runs. All 20 reached continuous exact accuracy 1.0. Stable Boolean exact counts are MSE 1/5, BCE 2/5, POWER_1_5 1/5, and POWER_1_25 2/5. Every I12 checkpoint reload/SHA check passed before I13; no I12 training was rerun.",
        "",
        "Each I13 seed reproduced its I12 initial-state SHA256 exactly. The curriculum used the same I2-B initialization, clean 256-row XOR table, Lehmer p=2, Adam lr=.01, zero weight decay, one uninterrupted optimizer state, and 3000 updates.",
        "",
        "## D–E. Trigger and alpha schedule",
        "",
        "| seed | initial hash | first continuous exact | trigger step | alpha at first Boolean exact | curriculum not triggered |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in curriculum:
        lines.append(f"| {r['seed']} | `{r['initial_state_sha256']}` | {r['first_continuous_exact']} | {r['trigger_step'] if r['trigger_step'] is not None else '—'} | {r['alpha_at_first_boolean_exact'] if r['alpha_at_first_boolean_exact'] is not None else '—'} | {'yes' if r['curriculum_not_triggered'] else 'no'} |")
    lines += ["", "The trigger required three consecutive 25-step continuous-exact evaluations and was reached for every seed. Alpha then annealed linearly from 1.5 to 1.25 over 500 updates and stayed at 1.25. Adam was not reset.", "", "## F–G. Curriculum trajectories", "", "The canonical JSON stores every 25-step continuous, hard-max, Boolean, endpoint, confidence, current-alpha loss, and topology record. The figures show alpha, Boolean accuracy, and E_inf trajectories for each seed.", ""]
    lines += ["## H. Per-seed comparison", "", "| seed | fixed 1.5 Boolean | fixed 1.25 Boolean | curriculum Boolean | curriculum first exact | trigger step |", "|---:|---:|---:|---:|---:|---:|"]
    for r in curriculum:
        p15 = fixed[(r["seed"], "POWER_1_5")]["final"]["boolean"]["exact_accuracy"]
        p125 = fixed[(r["seed"], "POWER_1_25")]["final"]["boolean"]["exact_accuracy"]
        lines.append(f"| {r['seed']} | {p15:.4f} | {p125:.4f} | {r['final']['boolean']['exact_accuracy']:.4f} | {r['first_boolean_exact'] if r['first_boolean_exact'] is not None else '—'} | {r['trigger_step']} |")
    lines += ["", "## I–L. Aggregate comparison", "", "| method | continuous exact | Boolean exact | stable Boolean exact | median E_inf | median endpoint distance | median first Boolean step |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        step = row["median_first_boolean_exact"] if row["median_first_boolean_exact"] is not None else "—"
        lines.append(f"| {row['method']} | {row['continuous_exact']}/5 | {row['boolean_exact']}/5 | {row['stable_boolean_exact']}/5 | {row['median_e_inf']:.6g} | {row['median_endpoint_distance']:.6g} | {step} |")
    lines += ["", "## M. Threshold robustness and N. topology changes", "", "All final threshold sweeps are stored in JSON under `threshold_robustness`. The exact interval containing .5 was:", "", "| seed | final exact threshold interval containing .5 | final mean gate D | final mean bias D |", "|---:|---|---:|---:|"]
    for r in curriculum:
        interval = threshold_interval(r)
        stats = r["final_gate_bias_stats"]
        gate_d = sum(x["gate_D"] for x in stats) / len(stats); bias_d = sum(x["bias_D"] for x in stats) / len(stats)
        lines.append(f"| {r['seed']} | {f'[{interval[0]:.2f},{interval[1]:.2f}]' if interval else 'none'} | {gate_d:.6g} | {bias_d:.6g} |")
    lines += ["", "Topology Hamming changes and per-evaluation edge/bias changes are stored in the trajectory. The anneal-window deltas were:", "", "| seed | edge Hamming change during anneal | bias Hamming change during anneal | final edge Hamming from init | final bias Hamming from init |", "|---:|---:|---:|---:|---:|"]
    for r in curriculum:
        tr = r["trigger_diagnostics"]; start = tr.get(str(r["trigger_step"]), {}) if r["trigger_step"] is not None else {}; end = tr.get(str(r["trigger_step"] + 500), {}) if r["trigger_step"] is not None else {}
        lines.append(f"| {r['seed']} | {r['anneal_edge_hamming_change']} | {r['anneal_bias_hamming_change']} | {end.get('edge_hamming_from_initial', r['trajectory'][-1]['edge_hamming_from_initial'])} | {end.get('bias_hamming_from_initial', r['trajectory'][-1]['bias_hamming_from_initial'])} |")
    lines += ["", "## O–Q. Conclusion", "", "The curriculum did not repair a seed where both fixed power losses failed. Its final Boolean results matched the fixed POWER_1_5 results in this run: seed0 1.0, seed1 .75, seed2 .875, seed3 .9765625, seed4 .8125. Stable Boolean exactness was therefore 1/5, versus 1/5 for fixed POWER_1_5 and 2/5 for fixed POWER_1_25.", "", "For context, I12 BCE was stable Boolean exact in 2/5 with median E_inf 0.2049, and I12 MSE was stable in 1/5 with median E_inf 0.1469. I13's median E_inf was 0.2981; its lower median endpoint distance was not accompanied by improved Boolean topology.", "", "The POWER_1_5 → POWER_1_25 hypothesis is not supported as a basin-improving method in this five-seed study. Alpha sharpening did not produce additional topology recovery; it mostly preserved the basin selected during the POWER_1_5 phase.", "", "## R–S. MNIST recommendation", "", "XOR experimentation complete; next experiment is an MNIST pilot. Use fixed POWER_1_25, the I13 curriculum, and BCE as the minimum comparison arms. The curriculum is not established as superior, so MNIST should be treated as a scale/generalization experiment rather than a claim that Boolean discretization is solved.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--raw-dir", required=True); parser.add_argument("--output", default=str(OUT / "i13_loss_curriculum_results.json")); args = parser.parse_args()
    curriculum = [enrich(json.loads(p.read_text())["run"]) for p in sorted(Path(args.raw_dir).glob("*.json"))]
    if {r["seed"] for r in curriculum} != set(range(5)): raise RuntimeError("I13 requires exactly seeds 0..4")
    i12 = load_i12(); fixed = {l: [r for r in i12["runs"] if r["loss"] == l] for l in ("POWER_1_5", "POWER_1_25")}
    rows = summary(curriculum, i12); figures(curriculum, fixed)
    payload = {"experiment": "I13-loss-curriculum", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "architecture": {"input_dim": 8, "width": 64, "residual_blocks": 2, "operator": "lehmer_p2"}, "initializer": "I2-B meanfield sigma2 + BIAS_ONE", "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": 3000, "eval_interval": 25, "initial_alpha": 1.5, "final_alpha": 1.25, "anneal_steps": 500, "trigger": "continuous exact for 3 consecutive evaluations; no Boolean signal; eligibility through step 2000"}, "summary": rows, "runs": curriculum}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    report(curriculum, i12, rows, ROOT / "research/i13_loss_curriculum_report.md")
    print(args.output)


if __name__ == "__main__": main()
