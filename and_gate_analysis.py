import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── grid ──────────────────────────────────────────────────────────────────────
N = 200
a = np.linspace(1e-4, 1 - 1e-4, N)
A, B = np.meshgrid(a, a)          # shape (N, N), rows=b axis, cols=a axis

EPS = 1e-7
T   = 4.0   # softmin temperature

# ── function definitions ───────────────────────────────────────────────────────
def product(a, b):
    return a * b

def grad_product(a, b):
    return b, a          # ∂/∂a, ∂/∂b

def softmin(a, b, t=T):
    ea = np.exp(-t * a);  eb = np.exp(-t * b)
    s  = ea + eb
    return (a * eb + b * ea) / s

def grad_softmin(a, b, t=T):
    ea = np.exp(-t * a);  eb = np.exp(-t * b)
    s  = ea + eb
    f  = (a * eb + b * ea) / s
    # ∂f/∂a = p_a * [1 + (a - f)/t]   where p_a = ea/s
    pa = ea / s;  pb = eb / s
    ga = pa * (1 + (a - f) / t)
    gb = pb * (1 + (b - f) / t)
    return ga, gb

def harmonic(a, b):
    return 2 * a * b / np.clip(a + b, EPS, None)

def grad_harmonic(a, b):
    s  = np.clip(a + b, EPS, None)
    ga = 2 * b**2 / s**2
    gb = 2 * a**2 / s**2
    return ga, gb

def geometric(a, b):
    return np.sqrt(np.clip(a * b, EPS, None))

def grad_geometric(a, b):
    f  = geometric(a, b)
    ga = f / (2 * np.clip(a, EPS, None))
    gb = f / (2 * np.clip(b, EPS, None))
    return ga, gb

def yager2(a, b):
    return np.maximum(1 - np.sqrt((1-a)**2 + (1-b)**2), 0)

def grad_yager2(a, b):
    d  = np.sqrt(np.clip((1-a)**2 + (1-b)**2, EPS, None))
    ga = (1 - a) / d
    gb = (1 - b) / d
    mask = (yager2(a, b) > 0).astype(float)
    return ga * mask, gb * mask

def logspace(a, b):
    logit_a = np.log(a / np.clip(1 - a, EPS, None))
    logit_b = np.log(b / np.clip(1 - b, EPS, None))
    return 1 / (1 + np.exp(-(logit_a + logit_b)))

def grad_logspace(a, b):
    f  = logspace(a, b)
    da = f * (1 - f) / np.clip(a * (1 - a), EPS, None)
    db = f * (1 - f) / np.clip(b * (1 - b), EPS, None)
    return da, db

def avg_prod_softmin(a, b):
    return (product(a, b) + softmin(a, b)) / 2

def grad_avg_prod_softmin(a, b):
    gpa, gpb = grad_product(a, b)
    gsa, gsb = grad_softmin(a, b)
    return (gpa + gsa) / 2, (gpb + gsb) / 2

# ── registry ──────────────────────────────────────────────────────────────────
functions = [
    ("Product\na·b",           product,          grad_product),
    ("Softmin\nsoftmin(a,b)",  softmin,           grad_softmin),
    ("Harmonic\n2ab/(a+b)",    harmonic,          grad_harmonic),
    ("Geometric\n√(ab)",       geometric,         grad_geometric),
    ("Yager p=2\nmax(1-‖…‖,0)",yager2,           grad_yager2),
    ("Log-space\nσ(logit_a+logit_b)", logspace,  grad_logspace),
    ("Avg(Prod,Softmin)\n(·+sm)/2",avg_prod_softmin, grad_avg_prod_softmin),
]

# ── figure layout ─────────────────────────────────────────────────────────────
n_fns = len(functions)
fig_w = 5 * 3          # 3 columns per function: f, ∂/∂a, |∇|
fig_h = 3.6 * n_fns

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0e0e0e")
fig.suptitle(
    "Differentiable AND-gate candidates  ·  f(a,b)  |  ∂f/∂a  |  gradient magnitude  |∇f|",
    fontsize=13, color="white", y=1.002, fontweight="bold"
)

outer = gridspec.GridSpec(n_fns, 1, figure=fig, hspace=0.55)

CMAP_F  = "plasma"
CMAP_G  = "viridis"
CMAP_MG = "inferno"

def make_axes(row_gs):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=row_gs, wspace=0.35)
    return [fig.add_subplot(inner[0]), fig.add_subplot(inner[1]), fig.add_subplot(inner[2])]

def style_ax(ax, title, cmap, data, xlabel="a →", ylabel="b ↑"):
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    im = ax.imshow(
        data, origin="lower", extent=[0, 1, 0, 1],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear"
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="white")
    cb.outline.set_edgecolor("#444")
    ax.set_title(title, fontsize=8.5, color="white", pad=4)
    ax.set_xlabel(xlabel, fontsize=7, color="#aaa")
    ax.set_ylabel(ylabel, fontsize=7, color="#aaa")
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_facecolor("#0e0e0e")

for i, (name, fn, gfn) in enumerate(functions):
    F          = fn(A, B)
    GA, GB     = gfn(A, B)
    GMAG       = np.sqrt(GA**2 + GB**2)

    axs = make_axes(outer[i])

    # row label
    fig.text(
        0.01, 1 - (i + 0.5) / n_fns,
        name, ha="left", va="center",
        fontsize=9, color="white", fontweight="bold",
        transform=fig.transFigure
    )

    style_ax(axs[0], "f(a, b)  — output",          CMAP_F,  F)
    style_ax(axs[1], "∂f/∂a  — gradient w.r.t. a", CMAP_G,  GA)
    style_ax(axs[2], "|∇f|  — gradient magnitude",  CMAP_MG, GMAG)

    # annotate the 4 truth-table corners on the output plot
    for ca, cb, label in [(0.02, 0.02, "00"), (0.97, 0.02, "10"),
                           (0.02, 0.97, "01"), (0.97, 0.97, "11")]:
        fv = fn(np.array([ca]), np.array([cb]))[0]
        axs[0].text(ca, cb, f"{fv:.2f}", ha="center", va="center",
                    fontsize=6, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1, lw=0))

plt.subplots_adjust(left=0.10, right=0.98, top=0.995, bottom=0.01)
out_path = "/mnt/user-data/outputs/and_gate_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
