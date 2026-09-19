"""CPU stress tests for differentiable OR candidates.

This is deliberately independent of network training.  It measures semantic
properties, gradients, width scaling, duplicate accumulation, and searches
for fractional endpoint counterexamples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from or_surrogates import candidate_factories, get_operator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "operator_results"
FIG = ROOT / "research" / "figures"
WIDTHS = [2, 4, 8, 16, 64, 256, 784]


def quantiles(x: torch.Tensor) -> dict:
    x = x.detach().flatten().double()
    if not x.numel():
        return {k: float("nan") for k in ("min", "max", "mean", "median", "p01", "p05", "p25", "p75", "p95", "p99")}
    q = torch.quantile(x, torch.tensor([.01, .05, .25, .75, .95, .99], dtype=x.dtype))
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "median": float(x.median()),
            "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5])}


def make_regime(name: str, n: int, batch: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + n + len(name))
    if name == "all_zero": return torch.zeros(batch, n, dtype=torch.float64)
    if name == "one_epsilon":
        x = torch.zeros(batch, n, dtype=torch.float64); x[:, 0] = 1e-6; return x
    if name == "one_near_one":
        x = torch.zeros(batch, n, dtype=torch.float64); x[:, 0] = 1 - 1e-6; return x
    if name == "one_exact_one":
        x = torch.zeros(batch, n, dtype=torch.float64); x[:, 0] = 1; return x
    if name == "winner_weak":
        x = torch.rand(batch, n, generator=g, dtype=torch.float64) * .01; x[:, 0] = .8; return x
    if name == "equal_active":
        x = torch.zeros(batch, n, dtype=torch.float64); x[:, :max(1, n // 4)] = .25; return x
    if name == "all_equal": return torch.full((batch, n), .5, dtype=torch.float64)
    if name == "boolean_random": return torch.randint(0, 2, (batch, n), generator=g, dtype=torch.float64)
    if name == "uniform": return torch.rand(batch, n, generator=g, dtype=torch.float64)
    if name == "beta02":
        torch.manual_seed(1000 + n + len(name)); return torch.distributions.Beta(.2, .2).sample((batch, n)).double()
    if name == "beta22":
        torch.manual_seed(1000 + n + len(name)); return torch.distributions.Beta(2, 2).sample((batch, n)).double()
    if name == "near_zero": return torch.rand(batch, n, generator=g, dtype=torch.float64) * .01
    if name == "near_one": return 1 - torch.rand(batch, n, generator=g, dtype=torch.float64) * .01
    raise ValueError(name)


REGIMES = ["all_zero", "one_epsilon", "one_near_one", "one_exact_one", "winner_weak", "equal_active", "all_equal", "boolean_random", "uniform", "beta02", "beta22", "near_zero", "near_one"]


def gradient_record(op, v: torch.Tensor) -> dict:
    v = v.clone().detach().requires_grad_(True)
    try:
        y = op(v)
        grad = torch.autograd.grad(y.sum(), v, allow_unused=True)[0]
        if grad is None: grad = torch.zeros_like(v)
        finite = torch.isfinite(grad)
        abs_g = grad.abs()
        q = quantiles(abs_g[finite]) if finite.any() else quantiles(abs_g.flatten()[:0])
        flat = abs_g[finite]
        top_share = float(flat.max() / flat.sum()) if flat.numel() and flat.sum() else 0.0
        topk = min(5, flat.numel())
        top5 = float(torch.topk(flat, topk).values.sum() / flat.sum()) if topk and flat.sum() else 0.0
        return {"output": quantiles(y), "gradient_abs": q, "l1": float(abs_g[finite].sum()) if finite.any() else float("nan"),
                "l2": float(torch.linalg.vector_norm(grad[finite])) if finite.any() else float("nan"),
                "linf": float(abs_g[finite].max()) if finite.any() else float("nan"),
                "fraction_lt_1e-12": float((abs_g < 1e-12).double().mean()),
                "fraction_lt_1e-9": float((abs_g < 1e-9).double().mean()),
                "fraction_lt_1e-6": float((abs_g < 1e-6).double().mean()),
                "fraction_negative": float((grad < 0).double().mean()),
                "fraction_positive": float((grad > 0).double().mean()),
                "largest_gradient_share": top_share, "top5_gradient_share": top5,
                "nan_count": int(torch.isnan(grad).sum()), "inf_count": int(torch.isinf(grad).sum())}
    except Exception as exc:
        return {"error": repr(exc)}


def duplicate_data() -> dict:
    values = [.01, .1, .25, .5, .75, .9, .99]
    reps = [1, 2, 4, 8, 16, 64, 256, 784]
    result = {}
    for name in candidate_factories():
        op = get_operator(name); result[name] = {}
        for c in values:
            result[name][str(c)] = [float(op(torch.full((1, n), c, dtype=torch.float64))[0]) for n in reps]
    return {"values": values, "repetitions": reps, "results": result}


def endpoint_search(steps: int = 250, restarts: int = 8) -> dict:
    # Search for a low-error endpoint reached with fractional coordinates.
    out = {}
    for name in candidate_factories():
        op = get_operator(name); out[name] = {}
        for target in (0.0, 1.0):
            best = None
            for seed in range(restarts):
                torch.manual_seed(20000 + seed)
                logits = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
                opt = torch.optim.Adam([logits], lr=.08)
                for _ in range(steps):
                    v = torch.sigmoid(logits)
                    loss = (op(v) - target).square().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    v = torch.sigmoid(logits); err = float((op(v) - target).abs())
                    frac = float((v * (1 - v)).mean())
                    score = frac if err < 1e-5 else -err
                    rec = {"v": v[0].tolist(), "error": err, "fractionality": frac, "min": float(v.min()), "max": float(v.max()), "threshold_or": float((v >= .5).any()), "target": target}
                    if best is None or score > best["score"]: best = {"score": score, **rec}
            out[name][str(target)] = best
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true"); args = parser.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    records = {"widths": WIDTHS, "regimes": REGIMES, "operators": {}}
    widths = [2, 8, 64, 784] if args.quick else WIDTHS
    regimes = ["all_zero", "one_exact_one", "winner_weak", "all_equal", "uniform", "beta22", "near_one"] if args.quick else REGIMES
    for name in candidate_factories():
        op = get_operator(name); records["operators"][name] = {}
        for n in widths:
            records["operators"][name][str(n)] = {}
            for regime in regimes:
                records["operators"][name][str(n)][regime] = gradient_record(op, make_regime(regime, n))
    records["duplicates"] = duplicate_data()
    records["endpoint_search"] = endpoint_search(100 if args.quick else 250, 4 if args.quick else 8)
    with (OUT / "operator_stress_results.json").open("w") as f: json.dump(records, f, indent=2)

    # Plots use the same regime across candidates and widths.
    for regime in ("winner_weak", "all_equal", "near_one"):
        plt.figure(figsize=(10, 6))
        for name in candidate_factories():
            ys = []
            for n in widths:
                rec = records["operators"][name][str(n)][regime]
                ys.append(rec.get("gradient_abs", {}).get("mean", float("nan")))
            plt.plot(widths, ys, marker="o", label=name)
        plt.xscale("log"); plt.yscale("log"); plt.xlabel("fan-in"); plt.ylabel("mean |gradient|"); plt.title(f"OR gradient scaling: {regime}"); plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / f"or_gradient_width_{regime}.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 6))
    reps = records["duplicates"]["repetitions"]
    for name in candidate_factories(): plt.plot(reps, records["duplicates"]["results"][name]["0.1"], marker="o", label=name)
    plt.xscale("log"); plt.xlabel("duplicate count"); plt.ylabel("F([0.1,...,0.1])"); plt.title("Duplicate accumulation"); plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG / "or_duplicate_accumulation.png", dpi=160); plt.close()
    print(json.dumps({"operators": list(candidate_factories()), "output": str(OUT / "operator_stress_results.json"), "figures": str(FIG)}, indent=2))


if __name__ == "__main__": main()
