"""Plot the committed M002b temperature-free XOR truth-table run."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "kaggle/results/c555781_M002b-temperature-free-self-sharpening_seed0.json"
OUTPUT = ROOT / "research/figures/m002b_temperature_free_trajectory.png"
metrics = json.loads(RESULT.read_text())["metrics"]
trajectory = metrics["trajectory"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
for key, label, color in [
    ("continuous_exact_accuracy", "continuous soft", "#1f77b4"),
    ("continuous_hardmax_exact_accuracy", "continuous hard-max", "#ff7f0e"),
    ("discrete_exact_accuracy", "thresholded Boolean", "#2ca02c"),
]:
    pts = [(e["epoch"], e[key]) for e in trajectory if e.get(key) is not None]
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2,
                label=label, color=color)
ax.axvline(525, linestyle="--", color="black", alpha=0.6,
           label="first perfect soft truth table: 525")
ax.set_title("Full truth-table exact accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Exact accuracy")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[0, 1]
for key, label, color, style in [
    ("D_g", "D_g", "#1f77b4", "-"),
    ("D_b", "D_b", "#ff7f0e", "-"),
    ("g_corner_05", "g corner fraction (0.05/0.95)", "#1f77b4", "--"),
    ("b_corner_05", "bias corner fraction (0.05/0.95)", "#ff7f0e", "--"),
]:
    pts = [(e["epoch"], e["polarization"].get(key)) for e in trajectory
           if e.get("polarization", {}).get(key) is not None]
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, marker=".", markersize=2.5, linewidth=1.0,
                label=label, color=color, linestyle=style)
ax.set_title("Gate/bias polarization")
ax.set_xlabel("Epoch")
ax.set_ylabel("Distance / fraction")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1, 0]
layer_names = ["stem", "block_0.layer1", "block_0.layer2",
               "block_1.layer1", "block_1.layer2", "head"]
layer_ids = [f"layer_{i}" for i in range(6)]
for layer, lid in zip(layer_names, layer_ids):
    pts = [(e["epoch"], e["temperature_free_parameters"][lid]["g"]["mean"])
           for e in trajectory if lid in e.get("temperature_free_parameters", {})]
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, marker=".", markersize=2.5, linewidth=1.0, label=layer)
ax.set_title("Mean effective gate by layer")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean g = tanh(softplus(theta))")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25)

ax = axes[1, 1]
for layer in layer_names:
    pts = [(e["epoch"], e["theta_mean_abs_gradient_by_layer"].get(layer))
           for e in trajectory if e.get("theta_mean_abs_gradient_by_layer", {}).get(layer) is not None]
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, linewidth=1.0, label=layer)
ax.set_yscale("log")
ax.set_title("Mean absolute gradient with respect to theta")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean |dL/dtheta| (log scale)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25, which="both")

fig.suptitle("M002b — temperature-free self-sharpening, seed 0")
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180)
