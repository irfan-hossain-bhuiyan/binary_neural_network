"""Compact the Kaggle A1 output and generate its report/figures.

This is post-processing only. It never trains a model or stores parameter
snapshots. The raw Kaggle JSON is deliberately reduced to trajectory metrics,
small forensic wrong-row records, and provenance metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/operator_results/a1_binary_addition_results.json"
REPORT = ROOT / "research/a1_binary_addition_report.md"
FIGURES = ROOT / "research/figures"
FIGURE_NAMES = (
    "a1_continuous_recovery_speed.png",
    "a1_boolean_recovery_speed.png",
    "a1_einf_vs_steps.png",
    "a1_einf_vs_seconds.png",
    "a1_carry_chain_accuracy.png",
    "a1_per_output_bit_accuracy.png",
    "a1_layer_discretization_gap.png",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_result(raw: dict, raw_path: Path) -> dict:
    payload = copy.deepcopy(raw["metrics"] if "metrics" in raw else raw)
    payload["git_sha"] = raw.get("git_commit") or payload.get("git_sha")
    payload["source_provenance"] = {
        "kaggle_kernel": "irfanhossainbhuiyan/a1-exact-binary-addition",
        "raw_result_sha256": sha256(raw_path),
        "raw_result_bytes": raw_path.stat().st_size,
        "postprocessing": "research/analyze_a1_binary_addition.py",
        "checkpoint_policy": "disabled; no model state retained",
    }
    for run in payload.get("results", []):
        trajectory = run.get("trajectory", [])
        selected = {0, 3000}
        for value in run.get("recovery", {}).values():
            if isinstance(value, dict) and value.get("step") is not None:
                selected.add(int(value["step"]))
        trimmed = []
        for entry in trajectory:
            item = {k: v for k, v in entry.items() if k != "layer_mismatch"}
            if int(entry.get("step", -1)) in selected:
                item["layer_mismatch"] = entry.get("layer_mismatch", [])
            trimmed.append(item)
        run["trajectory"] = trimmed
        run["checkpoints"] = {}
        run["checkpoint_policy"] = "disabled by default"
    payload["results_schema"] = {
        "removed_from_raw": ["model state_dict", "optimizer state", "unselected full layer traces"],
        "retained": ["metrics", "recovery timing", "timing", "selected layer traces", "small wrong-row forensic records"],
    }
    return payload


def runs_by_loss(data: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for run in data["results"]:
        out.setdefault(run["loss"], []).append(run)
    return out


def recovery_values(runs: list[dict], key: str) -> list[float]:
    values = []
    for run in runs:
        value = run.get("recovery", {}).get(key)
        if isinstance(value, dict):
            values.append(float(value["step"]))
    return values


def fmt(value) -> str:
    if value is None:
        return "NOT_REACHED"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def aggregate(data: dict) -> dict:
    by_loss = runs_by_loss(data)
    out = {}
    for loss, runs in by_loss.items():
        final = [r["final"] for r in runs]
        def mean(path):
            values = [float(path(x)) for x in final]
            return statistics.mean(values)
        out[loss] = {
            "runs": len(runs),
            "continuous_exact_runs": sum(x["continuous"]["exact_accuracy"] >= 1 for x in final),
            "boolean_exact_runs": sum(x["boolean"]["exact_accuracy"] >= 1 for x in final),
            "stable_boolean_runs": sum(r["stable_recovery"].get("boolean_exact", False) for r in runs),
            "continuous_exact_steps": recovery_values(runs, "continuous_exact"),
            "boolean_exact_steps": recovery_values(runs, "boolean_exact"),
            "continuous_exact_seconds": [r["recovery"]["continuous_exact"]["core_seconds"] for r in runs if isinstance(r["recovery"].get("continuous_exact"), dict)],
            "boolean_exact_seconds": [r["recovery"]["boolean_exact"]["core_seconds"] for r in runs if isinstance(r["recovery"].get("boolean_exact"), dict)],
            "final_continuous_exact_mean": mean(lambda x: x["continuous"]["exact_accuracy"]),
            "final_boolean_exact_mean": mean(lambda x: x["boolean"]["exact_accuracy"]),
            "final_e_inf_median": statistics.median(float(x["continuous"]["e_inf"]) for x in final),
            "mean_ms_per_step_median": statistics.median(r["timing"]["mean_ms_per_step"] for r in runs),
            "median_ms_per_step_after_warmup_median": statistics.median(r["timing"]["median_ms_per_step_after_warmup"] for r in runs),
        }
    return out


def make_figures(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    by_loss = runs_by_loss(data)
    colors = {"MSE": "#1f77b4", "BCE": "#d62728", "POWER_1_5": "#2ca02c", "POWER_1_25": "#9467bd"}

    def curves(metric_path, title, ylabel, filename, seconds=False):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for loss, runs in by_loss.items():
            curves = []
            for run in runs:
                curves.append([(e["core_seconds"] if seconds else e["step"], metric_path(e)) for e in run["trajectory"]])
            # schedules are shared; interpolate-free median by trajectory index.
            n = min(len(c) for c in curves)
            xs = [curves[0][i][0] for i in range(n)]
            ys = [statistics.median(c[i][1] for c in curves) for i in range(n)]
            ax.plot(xs, ys, label=loss, color=colors.get(loss))
        ax.set_title(title); ax.set_xlabel("wall-clock seconds" if seconds else "optimizer step"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=140); plt.close(fig)

    curves(lambda e: e["continuous"]["exact_accuracy"], "Continuous exact accuracy", "accuracy", "a1_continuous_recovery_speed.png")
    curves(lambda e: e["boolean"]["exact_accuracy"], "Exact Boolean accuracy", "accuracy", "a1_boolean_recovery_speed.png")
    curves(lambda e: e["continuous"]["e_inf"], "Continuous endpoint error", "E_inf", "a1_einf_vs_steps.png")
    curves(lambda e: e["continuous"]["e_inf"], "Continuous endpoint error vs time", "E_inf", "a1_einf_vs_seconds.png", seconds=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lengths = range(5)
    for loss, runs in by_loss.items():
        vals = []
        for length in lengths:
            group = [r["final"]["boolean_carry_chain"][str(length)]["exact_accuracy"] for r in runs]
            vals.append(statistics.mean(group))
        ax.plot(list(lengths), vals, marker="o", label=loss, color=colors.get(loss))
    ax.set_xlabel("carry-chain length"); ax.set_ylabel("final Boolean exact accuracy"); ax.set_title("Boolean accuracy by carry-chain length"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "a1_carry_chain_accuracy.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for loss, runs in by_loss.items():
        vals = [statistics.mean(r["final"]["boolean"]["per_bit_accuracy"][i] for r in runs) for i in range(5)]
        ax.plot(range(5), vals, marker="o", label=loss, color=colors.get(loss))
    ax.set_xticks(range(5), ["s0", "s1", "s2", "s3", "s4"]); ax.set_ylabel("final Boolean bit accuracy"); ax.set_title("Per-output-bit Boolean accuracy"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "a1_per_output_bit_accuracy.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for loss, runs in by_loss.items():
        vals = []
        for run in runs:
            trace = next(e for e in run["trajectory"] if e["step"] == 3000)["layer_mismatch"]
            vals.append([x["bit_mismatch_fraction"] for x in trace])
        ax.plot(range(len(vals[0])), [statistics.mean(v[i] for v in vals) for i in range(len(vals[0]))], marker="o", label=loss, color=colors.get(loss))
    labels = ["input", "stem", "b0.l1", "b0.l2", "b0.res", "b1.l1", "b1.l2", "b1.res", "head"]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right"); ax.set_ylabel("thresholded continuous vs Boolean mismatch"); ax.set_title("Final layer discretization gap"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "a1_layer_discretization_gap.png", dpi=140); plt.close(fig)


def make_report(data: dict) -> None:
    by_loss = runs_by_loss(data)
    agg = aggregate(data)
    lines = [
        "# A1 — Exact Binary Addition Benchmark\n",
        "## Audit and provenance\n",
        "The latest completed experiment before A1 was M2. Its canonical JSON and report were retained; no scientific rerun was performed during the cleanup or A1 setup. A1 ran on Kaggle with the recorded source commit and no model checkpoints.\n",
        f"Kaggle source commit: `{data.get('git_sha')}`  \nRaw result SHA256: `{data['source_provenance']['raw_result_sha256']}`  \nDevice: `{data['runtime']['device']}` ({data['runtime'].get('gpu')})\n",
        "## Task\n",
        "A1 exhaustively evaluates unsigned 4-bit addition. Inputs are `[a0,a1,a2,a3,b0,b1,b2,b3]`, outputs are `[s0,s1,s2,s3,s4]`, and all 256 rows are used. Integer arithmetic and independent ripple-carry equations agreed exactly. XOR has no cross-bit dependency; addition requires recursively propagated carries, so higher output bits have longer logical paths.\n",
        f"Target bit frequencies: `{data['task']['target_bit_frequency']}`. Carry-chain counts: `{data['task']['carry_chain_counts']}`. Full-adder smoke test: `{data['task']['full_adder_smoke']}`.\n",
        "## Adam scale diagnostic\n",
        f"The 10× MSE update/global-norm ratio was `{data['loss_scale_diagnostic']['global_ratio']:.6g}`. This is not a training arm; it shows that this implementation's first Adam update is not perfectly scale-invariant, so speed comparisons remain tied to the fixed loss definitions and learning rate.\n",
        "## Final results\n",
        "| loss | continuous exact runs | Boolean exact runs | stable Boolean runs | median final E_inf | median ms/step |\n|---|---:|---:|---:|---:|---:|",
    ]
    for loss in ("MSE", "BCE", "POWER_1_5", "POWER_1_25"):
        a=agg[loss]; lines.append(f"| {loss} | {a['continuous_exact_runs']}/5 | {a['boolean_exact_runs']}/5 | {a['stable_boolean_runs']}/5 | {a['final_e_inf_median']:.6g} | {a['median_ms_per_step_after_warmup_median']:.3f} |")
    lines += ["\nNo run reached continuous exact, hard exact, Boolean exact, or any requested `E_inf` threshold. All steps-to-target and seconds-to-target fields are therefore `NOT_REACHED`; the canonical JSON retains right-censored recovery fields.\n", "## Per-seed/per-loss outcomes\n", "| seed | loss | continuous exact | hard exact | Boolean exact | E_inf | wrong Boolean rows |\n|---:|---|---:|---:|---:|---:|---:|"]
    for run in data["results"]:
        f=run["final"]; lines.append(f"| {run['seed']} | {run['loss']} | {f['continuous']['exact_accuracy']:.6f} | {f['hard']['exact_accuracy']:.6f} | {f['boolean']['exact_accuracy']:.6f} | {f['continuous']['e_inf']:.6g} | {f['boolean']['wrong_rows']} |")
    lines += ["\n## Speed and recovery\n", "The 20 runs each used 3000 full-batch Adam updates and 768,000 examples. Median post-warmup step times were approximately 9.1–9.2 ms on the Kaggle T4. Since no run crossed a target, training speed changes throughput but does not produce a recovery winner. See `a1_continuous_recovery_speed.png`, `a1_boolean_recovery_speed.png`, `a1_einf_vs_steps.png`, and `a1_einf_vs_seconds.png`.\n"]
    lines += ["## Carry-chain and output-bit difficulty\n", "Final Boolean accuracy is grouped by carry-chain length in `a1_carry_chain_accuracy.png`; per-output-bit accuracy is in `a1_per_output_bit_accuracy.png`. The canonical JSON stores both continuous and Boolean carry groups and per-bit metrics for every run.\n"]
    lines += ["## Continuous/Boolean gap and internal mismatch\n", "Each trajectory records bit disagreement, row disagreement, mean Hamming distance, and selected internal traces. The layer plot is `a1_layer_discretization_gap.png`; the first nonzero layer in each selected trace is the earliest semantic mismatch.\n"]
    lines += ["## Final wrong-row forensic data\n", "Complete records for every final Boolean-wrong row—including A, B, sum, carry-chain length, continuous output, hard output, Boolean output, and target—are retained in `results[*].final_wrong_rows` in the canonical JSON. This avoids duplicating large tables in Markdown while preserving all exhaustive forensic data.\n"]
    lines += ["## Conclusion\n", "Four-bit addition was substantially harder than the prior bitwise XOR benchmark under the unchanged width-64 architecture and 3000-step budget. The continuous function was not solved exactly, so the result does not isolate a pure discretization failure: it demonstrates a continuous optimization/compositional-capacity challenge first. MSE produced the best final continuous exact accuracy in this fixed comparison; BCE and the power losses did not recover exact addition.\n", "**Recommended next experiment:** a single continuous-learning diagnostic on the same 4-bit addition task that preserves the architecture and compares a longer training budget or optimizer schedule before attempting 8-bit addition.\n"]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    data = compact_result(raw, args.input)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    make_figures(data)
    make_report(data)
    print(RESULT)
    print(REPORT)
    print("figures", len(FIGURE_NAMES))


if __name__ == "__main__":
    main()
