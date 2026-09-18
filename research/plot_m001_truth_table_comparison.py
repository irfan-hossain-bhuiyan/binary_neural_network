"""Plot archived M001-R and matched M001-NR truth-table trajectories."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "kaggle" / "results"
FIGURES = ROOT / "research" / "figures"

runs = {
    "M001-R (XOR residual)": RESULTS / "6a33da8_M001-R_seed0.json",
    "M001-NR (no residual)": RESULTS / "feb9db1_M001-NR_seed0.json",
    "M002 (fixed temperature)": RESULTS / "f8e998f_M002-fixed-temperature_seed0.json",
    "M003 (no explicit regularizer)": RESULTS / "62f1705_M003-no-explicit-regularizer_seed0.json",
}
colors = {"soft": "#1f77b4", "hard-max": "#ff7f0e", "Boolean": "#2ca02c"}
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
axes = axes.ravel()
for ax, (label, path) in zip(axes, runs.items()):
    metrics = json.loads(path.read_text())["metrics"]
    trajectory = metrics["trajectory"]
    epochs = [entry["epoch"] for entry in trajectory]
    for key, pretty in [
        ("continuous_exact_accuracy", "soft"),
        ("continuous_hardmax_exact_accuracy", "hard-max"),
        ("discrete_exact_accuracy", "Boolean"),
    ]:
        valid = [(entry["epoch"], entry[key]) for entry in trajectory
                 if entry.get(key) is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, label=pretty, color=colors[pretty], linewidth=1.2,
                    marker="o", markersize=2.2)
    ready = metrics.get("parameter_binarized_first_epoch")
    if ready is not None:
        ax.axvline(ready, color="black", linestyle="--", alpha=0.6,
                   label=f"first binarized: {ready}")
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_xlim(0, max(epochs))
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Full truth-table exact accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Modern 4-bit XOR truth-table experiments (seed 0)")
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "m001_truth_table_residual_control.png", dpi=180)
