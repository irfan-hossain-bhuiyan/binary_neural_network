"""Rebuild report figures from the archived preliminary M001 Kaggle JSON."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle/results/091faa0_M001_modern_4bit_preliminary.json"
OUT = ROOT / "research/figures"
OUT.mkdir(parents=True, exist_ok=True)
metrics = json.loads(DATA.read_text())["metrics"]
traj = metrics["trajectory"]
epoch = np.array([r["epoch"] for r in traj])

def series(key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in traj], dtype=float)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Training objective: task and regularization components have different scales.
fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.plot(epoch, series("task_loss"), color="#1768ac", lw=1.1, label="Task loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Task loss (MSE)", color="#1768ac")
ax.tick_params(axis="y", labelcolor="#1768ac")
ax.grid(axis="y", alpha=.22)
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(epoch, series("regularization_loss"), color="#d97706", lw=.8, alpha=.78,
         label="Regularization loss")
ax2.set_ylabel("Regularization loss", color="#d97706")
ax2.tick_params(axis="y", labelcolor="#d97706")
ax.set_title("M001 preliminary 4-bit training losses (seed 0)")
fig.tight_layout()
fig.savefig(OUT / "m001_training_losses.png", bbox_inches="tight")
plt.close(fig)

# Performance and thresholded diagnostic series.
fig, ax = plt.subplots(figsize=(9.2, 4.6))
for key, label, color, style in [
    ("continuous_bit_accuracy", "Continuous bit accuracy", "#1768ac", "-"),
    ("continuous_exact_accuracy", "Continuous exact accuracy", "#0f766e", "-"),
]:
    ax.plot(epoch, series(key), color=color, lw=1.7, marker="o", ms=2.5, label=label)
disc_ep=[]; disc_exact=[]; ready=[]
for r in traj:
    if r.get("discrete_exact_accuracy") is not None:
        disc_ep.append(r["epoch"]); disc_exact.append(r["discrete_exact_accuracy"])
        ready.append(bool(r.get("discretization_ready")))
ax.scatter(np.array(disc_ep)[np.array(ready)], np.array(disc_exact)[np.array(ready)],
           s=18, color="#7c3aed", label="Thresholded exact accuracy (ready checkpoint)", zorder=3)
ax.scatter(np.array(disc_ep)[~np.array(ready)], np.array(disc_exact)[~np.array(ready)],
           s=22, facecolors="none", edgecolors="#9ca3af", label="Thresholded exact accuracy (diagnostic only)", zorder=3)
ax.set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(-.02, 1.02),
       title="Learning and thresholded-model performance")
ax.grid(alpha=.22)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "m001_accuracy_trajectory.png", bbox_inches="tight")
plt.close(fig)

# Readiness measures and operational threshold.
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.plot(epoch, series("D_w"), color="#1768ac", label="$D_w$")
ax.plot(epoch, series("D_b"), color="#d97706", label="$D_b$")
ax.axhline(.01, color="#111827", lw=1, ls="--", label="Distance threshold 0.01")
ax.set(xlabel="Epoch", ylabel="Mean distance to nearest Boolean corner", ylim=(-.001, .09),
       title="Parameter polarization and readiness (first ready: epoch 5)")
ax.grid(alpha=.22)
ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(epoch, np.array([r["polarization"]["w_corner_05"] for r in traj]),
         color="#16a34a", lw=1.25, label="$w$ corner fraction")
ax2.plot(epoch, np.array([r["polarization"]["b_corner_05"] for r in traj]),
         color="#9333ea", lw=1.25, label="$b$ corner fraction")
ax2.axhline(.95, color="#4b5563", lw=1, ls=":", label="Corner threshold 0.95")
ax2.set_ylabel("Fraction within 0.05 of 0 or 1")
ax2.set_ylim(.65, 1.01)
handles, labels=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(handles+h2, labels+l2, loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "m001_polarization_readiness.png", bbox_inches="tight")
plt.close(fig)

# Direct skip gain and actual activation-gradient measurements.
blocks=metrics["residual_diagnostics"]
labels=["Block 1", "Block 2"]
fig, axs=plt.subplots(1,2,figsize=(9.2,4.2))
qnames=["p10","p25","median","p75","p90"]
x=np.arange(len(qnames)); width=.34
for j,key in enumerate(("block_0","block_1")):
    g=blocks[key]["direct_gain"]
    vals=[g["p10"],g["p25"],g["median"],g["p75"],g["p90"]]
    axs[0].plot(x, vals, marker="o", lw=1.7, label=labels[j])
axs[0].set_xticks(x,qnames); axs[0].set_ylim(0,1.04)
axs[0].set_ylabel("$|1-2F(x)|$")
axs[0].set_title("Direct XOR skip gain")
axs[0].grid(axis="y",alpha=.2); axs[0].legend()
for j,key in enumerate(("block_0","block_1")):
    g=blocks[key]["actual_activation_gradients"]
    vals=[g["input"]["mean_abs_grad"],g["output"]["mean_abs_grad"]]
    axs[1].bar(np.array([0,1])+j*width, vals, width=width, label=labels[j])
axs[1].set_xticks([width/2,1+width/2],["Block input", "Block output"])
axs[1].set_ylabel("Mean absolute activation gradient")
axs[1].set_title("Actual task-loss gradient")
axs[1].set_yscale("log"); axs[1].grid(axis="y",alpha=.2); axs[1].legend()
fig.suptitle("Residual diagnostics at the epoch-2000 checkpoint", y=1.03)
fig.tight_layout()
fig.savefig(OUT / "m001_residual_gradients.png", bbox_inches="tight")
plt.close(fig)

# Activation distribution summaries for x, h1, branch F(x), and y.
fig, axs=plt.subplots(1,2,figsize=(9.2,4.2))
series_names=["input","h1","h2","output"]
series_labels=["Block input $x$","First layer $h_1$","Branch $F(x)$","XOR output $y$"]
colors=["#1768ac","#0f766e","#d97706","#7c3aed"]
for i,key in enumerate(("block_0","block_1")):
    for name,label,color in zip(series_names,series_labels,colors):
        s=blocks[key]["activations"][name]
        axs[i].bar(label, s["mean"], color=color, alpha=.88)
    axs[i].set_ylim(0,1)
    axs[i].tick_params(axis="x",labelrotation=25)
    axs[i].set_title(labels[i])
    axs[i].set_ylabel("Mean activation")
    axs[i].grid(axis="y",alpha=.2)
fig.suptitle("Residual-block activation means at epoch 2000",y=1.02)
fig.tight_layout()
fig.savefig(OUT / "m001_activation_means.png",bbox_inches="tight")
plt.close(fig)
