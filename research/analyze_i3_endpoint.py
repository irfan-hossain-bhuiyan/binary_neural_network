"""Summarize I3 continuation trajectories and make the requested plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/operator_results/i3_endpoint_results.json"
FIG = ROOT / "research/figures"


def main() -> None:
    data = json.loads(RESULT.read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in data["continuations"]:
        for point in arm["trajectory"]:
            rows.append((arm["seed"], arm["arm"], point))

    def plot(xkey: str, ykey: str, name: str, xlabel: str, ylabel: str, logx=False) -> None:
        plt.figure(figsize=(7, 5))
        for seed, arm, p in rows:
            x = p["continuous"][xkey]
            y = p["boolean"][ykey]
            plt.scatter(x, y, s=12, alpha=.7, label=f"seed {seed} {arm}")
        if logx: plt.xscale("log")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(FIG / name, dpi=160); plt.close()

    plot("mse", "wrong_rows", "i3_mse_vs_boolean_error.png", "continuous MSE", "Boolean wrong rows", True)
    plot("bce", "wrong_rows", "i3_bce_vs_boolean_error.png", "continuous BCE", "Boolean wrong rows", True)
    plot("e_inf", "wrong_rows", "i3_einf_vs_boolean_error.png", "continuous E_inf", "Boolean wrong rows", True)

    plt.figure(figsize=(7, 5))
    for seed, arm, p in rows:
        plt.plot(p["step"], p["boolean"]["wrong_rows"], "o-", label=f"seed {seed} {arm}")
    plt.xlabel("continuation step"); plt.ylabel("Boolean wrong rows"); plt.grid(alpha=.25); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "i3_boolean_error_vs_step.png", dpi=160); plt.close()

    # Aggregate diagnostics by step; one curve per arm/seed is intentionally
    # retained so seed-dependent basins are visible.
    plt.figure(figsize=(7, 5))
    for arm in data["continuations"]:
        for point in arm["trajectory"]:
            value = sum(layer["raw_abs_gt_6"] for layer in point["raw_gate_stats"]) / len(point["raw_gate_stats"])
            plt.scatter(point["step"], value, s=14, label=f"seed {arm['seed']} {arm['arm']}" if point["step"] == 0 else None)
    plt.xlabel("continuation step"); plt.ylabel("mean fraction |raw edge| > 6"); plt.grid(alpha=.25); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "i3_raw_logit_growth.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    for arm in data["continuations"]:
        for point in arm["trajectory"]:
            vals = [s["threshold_discrete_mismatch_fraction"] for s in point["layer_trace"]["stages"]]
            plt.plot(range(len(vals)), vals, alpha=.25, color="tab:blue" if arm["arm"] == "mse" else "tab:orange")
    plt.xlabel("stage index"); plt.ylabel("threshold/Boolean mismatch fraction"); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIG / "i3_layer_mismatch.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    for arm in data["continuations"]:
        for point in arm["trajectory"]:
            vals = [s.get("zero_safety_fraction", 0.0) for s in point["layer_trace"]["stages"] if "zero_safety_fraction" in s]
            plt.plot(range(len(vals)), vals, alpha=.25, color="tab:blue" if arm["arm"] == "mse" else "tab:orange")
    plt.xlabel("logic-stage index"); plt.ylabel("zero-side safety fraction"); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIG / "i3_zero_safety.png", dpi=160); plt.close()

    lines = ["# I3 Endpoint Continuation Analysis", "", "The continuation starts from verified I2-B best-continuous checkpoints.", ""]
    lines += ["| seed | arm | final MSE | final BCE | final E_inf | final Boolean wrong rows |", "|---:|---|---:|---:|---:|---:|"]
    for arm in data["continuations"]:
        p = arm["trajectory"][-1]
        lines.append(f"| {arm['seed']} | {arm['arm']} | {p['continuous']['mse']:.6g} | {p['continuous']['bce']:.6g} | {p['continuous']['e_inf']:.6g} | {p['boolean']['wrong_rows']} |")
    lines += ["", "The same checkpoint and learning rate were used for both arms; Adam state was reset identically because the parent checkpoints did not contain optimizer state.", "", "A zero-loss endpoint argument is conditional: it applies to exact endpoint values in the closure of the sigmoid parameterization. Finite low loss alone does not force every internal gate or output to an endpoint."]
    (ROOT / "research/i3_endpoint_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
