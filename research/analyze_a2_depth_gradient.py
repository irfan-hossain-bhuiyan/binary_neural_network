"""Compact A2 output and generate the depth/residual report and figures."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/operator_results/a2_depth_gradient_results.json"
REPORT = ROOT / "research/a2_depth_gradient_report.md"
FIGURES = ROOT / "research/figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(raw: dict, raw_path: Path) -> dict:
    data = copy.deepcopy(raw.get("metrics", raw))
    data["git_sha"] = raw.get("git_commit") or data.get("git_sha")
    data["source_provenance"] = {
        "kaggle_kernel": "irfanhossainbhuiyan/a2-depth-gradient-propagation",
        "raw_result_sha256": sha256(raw_path),
        "raw_result_bytes": raw_path.stat().st_size,
        "postprocessing": "research/analyze_a2_depth_gradient.py",
        "checkpoint_policy": "disabled; no model state retained",
    }
    for run in data.get("results", []):
        # Keep all task metrics and the selected diagnostics, but never retain
        # tensors or unselected activation arrays.
        run["checkpoint_policy"] = "disabled"
    return data


def key(run: dict) -> tuple[int, bool]:
    return int(run["blocks"]), bool(run["residual_enabled"])


def grouped(data: dict) -> dict[tuple[int, bool], list[dict]]:
    out: dict[tuple[int, bool], list[dict]] = {}
    for run in data["results"]:
        out.setdefault(key(run), []).append(run)
    return out


def median_recovery(runs: list[dict], name: str):
    vals = [r["recovery"][name]["step"] for r in runs if isinstance(r["recovery"].get(name), dict)]
    return statistics.median(vals) if vals else None


def fmt(x):
    if x is None:
        return "NOT_REACHED"
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def figure_data(data: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    groups = grouped(data)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 4: "#d62728", 8: "#9467bd"}
    labels = {True: "XOR_RESIDUAL", False: "NO_RESIDUAL"}

    def curve(metric, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for (blocks, residual), runs in groups.items():
            n = min(len(r["trajectory"]) for r in runs)
            xs = [runs[0]["trajectory"][i]["step"] for i in range(n)]
            ys = [statistics.median(metric(r["trajectory"][i]) for r in runs) for i in range(n)]
            ax.plot(xs, ys, label=f"{blocks} blocks {labels[residual]}", color=colors[blocks], linestyle="-" if residual else "--")
        ax.set_title(title); ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=140); plt.close(fig)

    curve(lambda e: e["continuous"]["exact_accuracy"], "Continuous exact recovery by depth", "exact-row accuracy", "a2_recovery_vs_depth.png")

    # Parameter gradient and activation transfer by layer/depth.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (blocks, residual), runs in groups.items():
        vals = []
        for run in runs:
            diag = next(e for e in run["trajectory"] if e["step"] == 0)["gradient_diagnostics"]["layers"]
            vals.append([x["gradient_l2"] for x in diag])
        n = len(vals[0]); means = [statistics.mean(v[i] for v in vals) for i in range(n)]
        ax.plot(range(n), means, marker="o", label=f"{blocks} {'res' if residual else 'plain'}", color=colors[blocks], linestyle="-" if residual else "--")
    ax.set_yscale("log"); ax.set_xlabel("logic-layer index"); ax.set_ylabel("initial parameter gradient L2"); ax.set_title("Initial logic-layer gradient norms"); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIGURES / "a2_gradient_by_layer.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for residual, style in ((True, "-"), (False, "--")):
        xs, ys = [], []
        for blocks in (1, 2, 4, 8):
            rs = groups[(blocks, residual)]
            vals = [b["mean_abs_direct_gain"] for r in rs for b in next(e for e in r["trajectory"] if e["step"] == 0)["gradient_diagnostics"]["blocks"]]
            xs.append(blocks); ys.append(statistics.mean(vals))
        ax.plot(xs, ys, marker="o", linestyle=style, label=labels[residual])
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("blocks"); ax.set_ylabel("mean |1-2F|"); ax.set_title("Initial XOR skip gain"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "a2_direct_skip_gain.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for residual, style in ((True, "-"), (False, "--")):
        xs, ys = [], []
        for blocks in (1, 2, 4, 8):
            rs = groups[(blocks, residual)]
            vals = [b["branch_mean_min"] for r in rs for b in next(e for e in r["trajectory"] if e["step"] == 3000)["gradient_diagnostics"]["blocks"]]
            xs.append(blocks); ys.append(statistics.mean(vals))
        ax.plot(xs, ys, marker="o", linestyle=style, label=labels[residual])
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("blocks"); ax.set_ylabel("mean min(F, 1-F)"); ax.set_title("Final branch polarization"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "a2_branch_polarization.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for (blocks, residual), runs in groups.items():
        vals = []
        for length in range(5):
            vals.append(statistics.mean(r["final"]["boolean_carry_chain"][str(length)]["exact_accuracy"] for r in runs))
        ax.plot(range(5), vals, marker="o", label=f"{blocks} {'res' if residual else 'plain'}", color=colors[blocks], linestyle="-" if residual else "--")
    ax.set_xticks(range(5)); ax.set_xlabel("carry-chain length"); ax.set_ylabel("final Boolean exact accuracy"); ax.set_title("Boolean accuracy by carry-chain difficulty"); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIGURES / "a2_carry_chain_accuracy.png", dpi=140); plt.close(fig)


def report(data: dict):
    groups = grouped(data)
    lines = [
        "# A2 — XOR-Residual Depth / Gradient Propagation\n",
        "## A. A1 baseline and audit\n",
        "A1 is the latest completed benchmark before A2. Its canonical JSON, report, and task tests agreed; no A1 rerun was required. A1 selected **MSE** for A2 because all requested recovery rates were zero, while MSE had the strongest task-independent final continuous result (highest continuous exact rate and lowest E_inf). Native objective magnitudes were not used.\n",
        f"A2 Kaggle commit: `{data.get('git_sha')}`; device `{data.get('runtime', {}).get('device')}` ({data.get('runtime', {}).get('gpu')}). Raw result SHA256: `{data.get('source_provenance', {}).get('raw_result_sha256')}`.\n",
        "## B. Architecture matrix\n",
        "All runs use input 8, width 64, output 5, Lehmer-p2, I2-B mean-field sigma=2 plus BIAS_ONE, full-batch Adam (lr=.01, 3000 updates), and seeds 0–2. Depths are 0, 1, 2, 4, and 8 two-layer width-preserving blocks. Depth 0 has no residual factorial; positive depths have matched XOR_RESIDUAL and NO_RESIDUAL arms. No checkpoints were retained.\n",
        "## C. XOR residual Jacobian\n",
        r"For (y=x+F(x)-2xF(x)), (J_y=\operatorname{diag}(1-2F)+\operatorname{diag}(1-2x)J_F). The direct skip gain is (D_{skip}=\operatorname{diag}(1-2F)): it is approximately +1 when (F\approx0), −1 when (F\approx1), and vanishes near (F=.5). Thus this is a signed, state-dependent residual path rather than an additive identity. The runner verifies (g_x=g_{direct}+g_{branch}) numerically at initialization and diagnostic checkpoints.\n",
        "## D–F. Gradient propagation and cancellation\n",
        "For each block the JSON records input/output activation-gradient norms, transfer ratios, cosine, direct and branch norms, direct/branch cosine, and relative reconstruction error. It also records per-layer parameter gradient norms and gradient/parameter ratios. The direct/branch cosine diagnoses reinforcement versus cancellation; values near −1 indicate cancellation.\n",
        "## G–H. Depth and polarization\n",
        "Initial and final branch statistics include mean min(F,1−F), fractions below .1/above .9, and fraction near .5. These can be compared with mean |1−2F| to test whether polarization strengthens the direct XOR path.\n",
        "## I–M. Recovery and carry structure\n",
        "The tables below use task-independent continuous, hard-max, and exact Boolean metrics. Carry-chain groups and per-output-bit metrics remain in the canonical JSON for every evaluation.\n",
        "| blocks | residual | cont exact runs | Boolean exact runs | stable Boolean runs | median cont step | median Bool step | min initial transfer |\n|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for blocks in (0, 1, 2, 4, 8):
        modes = (True,) if blocks == 0 else (True, False)
        for residual in modes:
            rs = groups[(blocks, residual)]
            c = sum(r["final"]["continuous"]["exact_accuracy"] >= 1 for r in rs)
            b = sum(r["final"]["boolean"]["exact_accuracy"] >= 1 for r in rs)
            stable = sum(r["stable_recovery"].get("boolean_exact", False) for r in rs)
            transfers = []
            for r in rs:
                initial_diag = next(e for e in r["trajectory"] if e["step"] == 0)["gradient_diagnostics"]["blocks"]
                if initial_diag:
                    transfers.append(min(x["grad_input_output_ratio"] for x in initial_diag))
            lines.append(f"| {blocks} | {'XOR_RESIDUAL' if residual else 'NO_RESIDUAL'} | {c}/3 | {b}/3 | {stable}/3 | {fmt(median_recovery(rs, 'continuous_exact'))} | {fmt(median_recovery(rs, 'boolean_exact'))} | {statistics.median(transfers) if transfers else 'NA'} |")
    lines += [
        "\nRecovery timestamps are `NOT_REACHED` when a condition did not occur; they are never replaced by step 3000. The canonical trajectories include per-evaluation MSE/MAE/E_inf, continuous/hard/Boolean accuracy, wrong rows/bits, carry groups, per-bit metrics, functional disagreement, internal mismatch traces, timing, and diagnostics at steps 0, 100, 500, 1000, 2000, 3000 (plus any first recovery checkpoint).\n",
        "## N. Interpretation\n",
        "Use `a2_recovery_vs_depth.png`, `a2_gradient_by_layer.png`, `a2_direct_skip_gain.png`, `a2_branch_polarization.png`, and `a2_carry_chain_accuracy.png` for the depth/residual comparison. A residual benefit is supported only if it improves task-independent recovery and/or gradient transfer relative to the matched no-residual arm. If both modes fail together, the limiting mechanism is more likely depth/operator optimization than the skip. If direct gain is weak early and increases as F polarizes, the data support a weak-early/strong-late XOR path.\n",
        "**Recommended subsequent experiment:** introduce the planned threshold-aware loss on the unchanged A1-selected MSE baseline, using the A2 diagnostics to determine whether its pressure should target the first failing semantic layer. Do not change depth, operator, initialization, or residual structure in that follow-up.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.input.read_text())
    data = compact(raw, args.input)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    figure_data(data)
    report(data)
    print(RESULT); print(REPORT)


if __name__ == "__main__":
    main()
