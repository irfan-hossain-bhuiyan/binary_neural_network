"""Merge I12 run shards and generate the reproducible report/figures."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "research/figures"
OUT = ROOT / "research/operator_results"
LOSSES = ("MSE", "BCE", "POWER_1_5", "POWER_1_25")


def longest_exact(run):
    best = cur = 0
    for row in run["trajectory"]:
        if row["boolean"]["exact_accuracy"] == 1.0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def enrich(run):
    run = dict(run)
    run["longest_boolean_exact_evaluation_streak"] = longest_exact(run)
    run["stable_boolean_exact"] = bool(
        run["final"]["boolean"]["exact_accuracy"] == 1.0
        and run["boolean_regressions_after_exact"] == 0
    )
    return run


def aggregate(runs):
    rows = []
    for loss in LOSSES:
        group = [r for r in runs if r["loss"] == loss]
        recovered = [r["first_boolean_exact"] for r in group if r["first_boolean_exact"] is not None]
        rows.append({
            "loss": loss,
            "seeds": len(group),
            "continuous_exact_runs": sum(r["final"]["continuous"]["exact_accuracy"] == 1.0 for r in group),
            "hard_exact_runs": sum(r["final"]["hard"]["exact_accuracy"] == 1.0 for r in group),
            "boolean_exact_runs": sum(r["final"]["boolean"]["exact_accuracy"] == 1.0 for r in group),
            "stable_boolean_exact_runs": sum(r["stable_boolean_exact"] for r in group),
            "median_final_e_inf": statistics.median(r["final"]["continuous"]["e_inf"] for r in group),
            "median_endpoint_distance": statistics.median(r["final"]["endpoint"]["mean"] for r in group),
            "median_first_boolean_exact_step": statistics.median(recovered) if recovered else None,
            "recovered_steps": sorted(recovered),
        })
    return rows


def exact_threshold_interval(run):
    values = [(t, run["threshold_robustness"][f"{t:.2f}"]["exact_accuracy"] == 1.0)
              for t in (.30, .35, .40, .45, .50, .55, .60, .65, .70)]
    center = 4
    if not values[center][1]:
        return None
    lo = hi = center
    while lo > 0 and values[lo - 1][1]: lo -= 1
    while hi + 1 < len(values) and values[hi + 1][1]: hi += 1
    return values[lo][0], values[hi][0]


def make_figures(runs, summary):
    FIG.mkdir(parents=True, exist_ok=True)
    # Recovery counts.
    plt.figure(figsize=(8, 5))
    vals = [next(x for x in summary if x["loss"] == l)["stable_boolean_exact_runs"] for l in LOSSES]
    plt.bar(LOSSES, vals); plt.ylim(0, 5); plt.ylabel("stable Boolean exact runs / 5")
    plt.tight_layout(); plt.savefig(FIG / "i12_boolean_recovery_rate.png", dpi=150); plt.close()
    # Final E_inf by seed.
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        g = sorted((r for r in runs if r["loss"] == loss), key=lambda r: r["seed"])
        plt.plot([r["seed"] for r in g], [r["final"]["continuous"]["e_inf"] for r in g], marker="o", label=loss)
    plt.yscale("log"); plt.xlabel("seed"); plt.ylabel("final E_inf"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i12_einf_by_loss.png", dpi=150); plt.close()
    # Endpoint distance by seed.
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        g = sorted((r for r in runs if r["loss"] == loss), key=lambda r: r["seed"])
        plt.plot([r["seed"] for r in g], [r["final"]["endpoint"]["mean"] for r in g], marker="o", label=loss)
    plt.xlabel("seed"); plt.ylabel("mean endpoint distance"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i12_endpoint_distance.png", dpi=150); plt.close()
    # Per-seed Boolean exact heatmap-like grouped lines.
    plt.figure(figsize=(8, 5))
    for loss in LOSSES:
        g = sorted((r for r in runs if r["loss"] == loss), key=lambda r: r["seed"])
        plt.plot([r["seed"] for r in g], [r["final"]["boolean"]["exact_accuracy"] for r in g], marker="o", label=loss)
    plt.ylim(0, 1.05); plt.xlabel("seed"); plt.ylabel("final Boolean exact accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "i12_seed_outcomes.png", dpi=150); plt.close()


def report(runs, summary, out_path):
    hashes = {seed: sorted({r["initial_state_sha256"] for r in runs if r["seed"] == seed}) for seed in range(5)}
    lines = [
        "# I12 — Five-Seed Clean Loss Replication",
        "",
        "## A. I11 audit and correction",
        "",
        "The canonical I11 JSON and checkpoints were audited before I12. Four clean eta=0 arms agreed with their checkpoints. The MSE seed3 eta=0 JSON trajectory was correct, but its recorded final checkpoint had been overwritten and its SHA did not match; that single arm was rerun with the original configuration and the canonical JSON/checkpoint metadata were repaired. No other I11 arm was retrained.",
        "",
        "The I11 report was corrected: POWER_1_5 is better at eta=.1, while POWER_1_25 is better at eta=.2 and .4. This is a tradeoff, not monotonic dominance by smaller alpha.",
        "",
        "Kaggle execution was attempted previously but the API was unavailable from this environment (DNS/network access). I12 therefore ran locally with one CPU thread per concurrent arm; no remote result is claimed.",
        "",
        "## B. Configuration and paired initialization",
        "",
        "All 20 runs used the exact 256-row XOR truth table, I2-B (mean-field sigma=2 + BIAS_ONE), Lehmer p=2, Adam lr=.01, zero weight decay, full batch, and 3000 optimizer steps. Each seed's four losses reloaded one byte-identical initial state.",
        "",
        "| seed | initial state SHA256 | paired arms |\n|---:|---|:---:|",
    ]
    for seed in range(5):
        lines.append(f"| {seed} | `{hashes[seed][0] if len(hashes[seed]) == 1 else hashes[seed]}` | {'yes' if len(hashes[seed]) == 1 else 'NO'} |")
    lines += ["", "## C. Aggregate results (denominator 5 seeds)", "", "| loss | continuous exact | hard exact | Boolean exact | stable Boolean exact | median final E_inf | median endpoint distance | median first Boolean step |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        step = "—" if s["median_first_boolean_exact_step"] is None else f"{s['median_first_boolean_exact_step']:.0f}"
        lines.append(f"| {s['loss']} | {s['continuous_exact_runs']}/5 | {s['hard_exact_runs']}/5 | {s['boolean_exact_runs']}/5 | {s['stable_boolean_exact_runs']}/5 | {s['median_final_e_inf']:.6g} | {s['median_endpoint_distance']:.6g} | {step} |")
    lines += ["", "## D. Per-seed final results", "", "| seed | loss | continuous exact | hard exact | Boolean exact | wrong rows | final E_inf | first Boolean exact |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(runs, key=lambda x: (x["seed"], LOSSES.index(x["loss"]))):
        lines.append(f"| {r['seed']} | {r['loss']} | {r['final']['continuous']['exact_accuracy']:.4f} | {r['final']['hard']['exact_accuracy']:.4f} | {r['final']['boolean']['exact_accuracy']:.4f} | {r['final']['boolean']['wrong_rows']} | {r['final']['continuous']['e_inf']:.6g} | {r['first_boolean_exact'] if r['first_boolean_exact'] is not None else '—'} |")
    lines += ["", "## E. Endpoint, threshold, and parameter diagnostics", "", "| loss | median confidence | median endpoint distance | final exact threshold intervals containing .5 | median gate D | median bias D |", "|---|---:|---:|---|---:|---:|"]
    for loss in LOSSES:
        group = [r for r in runs if r["loss"] == loss]
        intervals = [exact_threshold_interval(r) for r in group if exact_threshold_interval(r) is not None]
        gate_d = [sum(v["gate_D"] for v in r["final_gate_bias_stats"]) / len(r["final_gate_bias_stats"]) for r in group]
        bias_d = [sum(v["bias_D"] for v in r["final_gate_bias_stats"]) / len(r["final_gate_bias_stats"]) for r in group]
        conf = [r["final"]["endpoint"]["mean_confidence"] for r in group]
        end = [r["final"]["endpoint"]["mean"] for r in group]
        interval_text = ", ".join(f"[{a:.2f},{b:.2f}]" for a, b in intervals) if intervals else "none"
        lines.append(f"| {loss} | {statistics.median(conf):.6g} | {statistics.median(end):.6g} | {len(intervals)}/5 ({interval_text}) | {statistics.median(gate_d):.6g} | {statistics.median(bias_d):.6g} |")
    lines += ["", "The clean seed3 POWER_1_25 result was reproducible: Boolean exact first appeared at step 750 in I11 and I12, with final Boolean exact 1.0 and E_inf about 0.00376. Among successful Boolean runs, POWER_1_25 had median E_inf 0.00343 versus 0.00307 for BCE and 0.00801 for POWER_1_5; across all seeds its median is worse because three seeds failed to reach the Boolean basin. Lower-alpha endpoint sharpening is therefore beneficial conditional on success, but it does not guarantee the topology transition.", "", "## F. Interpretation and MNIST decision", "", "Every run reached continuous exact accuracy in this clean function-recovery setting, but Boolean recovery was much more seed-sensitive. POWER_1_25 reached stable Boolean exactness in 2/5 seeds (0 and 3), POWER_1_5 in 1/5, MSE in 1/5, and BCE in 2/5. POWER_1_25 therefore reproduced the strong seed3 result but did not generalize to the required 3/5 adequate or 4/5 strong threshold. POWER_1_5 was better than POWER_1_25 on the final Boolean score in three seeds, but it was not clearly more stable for continuous learning because all 20 runs reached continuous exactness.", "", "The predefined MNIST decision rule does not pass for POWER_1_25: stable Boolean exact is 2/5. No I13 run was started. If one final XOR experiment is required before MNIST, the smallest justified study is a three-seed paired POWER_1_5→POWER_1_25 curriculum using the same clean setup; otherwise the evidence is insufficient to choose a curriculum. MNIST should wait for that decision.", "", "The results do not support claiming POWER_1_25 is universally superior. In this five-seed clean-XOR study it produces the most endpoint-like outputs when it succeeds, but its optimization basin is not yet reliable enough for the MNIST go decision.", ""]
    out_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--raw-dir", required=True); p.add_argument("--output", default=str(OUT / "i12_loss_seed_replication_results.json")); args = p.parse_args()
    runs = []
    for path in sorted(Path(args.raw_dir).glob("*.json")):
        payload = json.loads(path.read_text()); runs.append(enrich(payload["run"]))
    expected = {(s, l) for s in range(5) for l in LOSSES}
    actual = {(r["seed"], r["loss"]) for r in runs}
    if actual != expected: raise RuntimeError(f"missing/duplicate runs: expected {len(expected)}, got {len(actual)}")
    summary = aggregate(runs); make_figures(runs, summary)
    payload = {"experiment": "I12-loss-seed-replication", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "architecture": {"input_dim": 8, "width": 64, "residual_blocks": 2, "operator": "lehmer_p2"}, "initializer": "I2-B meanfield sigma2 + BIAS_ONE", "training": {"optimizer": "Adam", "lr": .01, "weight_decay": 0.0, "steps": 3000, "dataset_rows": 256, "losses": LOSSES}, "kaggle_status": {"available": False, "note": "Kaggle API unavailable from this environment; runs executed locally."}, "summary": summary, "runs": runs}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    report(runs, summary, ROOT / "research/i12_loss_seed_replication_report.md")
    print(args.output)


if __name__ == "__main__": main()
