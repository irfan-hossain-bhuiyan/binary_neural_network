"""Forensic, no-retraining analysis of the Stage B3 XOR circuits.

This script deliberately loads saved B3 checkpoints.  It does not optimize a
model or alter any experiment configuration.  Its outputs are JSON summaries
and four diagnostic figures used by ``b3_forensic_analysis.md``.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402

OUT = ROOT / "research" / "operator_results"
CK = ROOT / "artifacts" / "checkpoints" / "B3R"
RESULTS_PATH = ROOT / "research" / "operator_results" / "stage_b3r_results.json"
FIG = ROOT / "research" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OPS = ("lehmer_p2", "probabilistic_or")
SEEDS = (0, 1, 2)
LAYERS = ("stem", "block0.layer1", "block0.layer2", "block1.layer1", "block1.layer2", "head")
THRESHOLDS = [round(0.20 + 0.025 * i, 3) for i in range(25)]


def model_for(op: str) -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(
        8, 4, width=64, num_residual_blocks=2, or_operator=op,
        bias_initialization=lambda t: nn.init.normal_(t, mean=.5, std=.1),
    )


def load_model(op: str, seed: int, suffix: str = "best_continuous", checkpoint_dir: Path | None = None):
    model = model_for(op)
    suffix = {"best_continuous": "best_continuous_mse", "best_hard": "best_hard_exact", "best_boolean": "best_boolean_exact"}.get(suffix, suffix)
    path = (checkpoint_dir or CK) / f"B3R_{op}_seed{seed}_{suffix}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, path


def task_data():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    return task["X"].float(), task["Y"].float()


def layer_list(model):
    return list(model.expectation_layers)


def continuous_trace(model, x):
    """Return layer inputs, contributions and outputs, including residual outputs."""
    h = x
    rows = []
    residual_inputs = {}
    for name, layer in zip(LAYERS, layer_list(model)):
        if name.endswith(".layer1"):
            residual_inputs[name.split(".")[0]] = h
        v = layer.contributions(h)
        y = layer.or_operator(v)
        rows.append({"name": name, "input": h, "v": v, "output": y})
        h = y
        if name == "block0.layer2":
            start = residual_inputs["block0"]
            h = start + y - 2 * start * y
            rows.append({"name": "block0.residual", "input": start, "v": None, "output": h})
        elif name == "block1.layer2":
            start = residual_inputs["block1"]
            h = start + y - 2 * start * y
            rows.append({"name": "block1.residual", "input": start, "v": None, "output": h})
    return rows


def discrete_trace(model, x, threshold):
    """Trace Boolean layer outputs with exactly the converted model topology."""
    d = model.to_discrete(threshold)
    h = x.bool()
    rows = []
    residual_inputs = {}
    layer_iter = iter(layer_list(d))
    for name in LAYERS:
        if name.endswith(".layer1"):
            residual_inputs[name.split(".")[0]] = h
        layer = next(layer_iter)
        y = layer(h)
        rows.append((name, h, y))
        h = y
        if name == "block0.layer2":
            start = residual_inputs["block0"]
            h = start ^ y
            rows.append(("block0.residual", start, h))
        elif name == "block1.layer2":
            start = residual_inputs["block1"]
            h = start ^ y
            rows.append(("block1.residual", start, h))
    return rows, d


def basic_stats(t):
    q = torch.quantile(t.flatten().float(), torch.tensor([.01, .05, .25, .5, .75, .95, .99]))
    return {
        "mean": float(t.mean()), "std": float(t.std()),
        "q01": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
        "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]), "q99": float(q[6]),
    }


def contribution_report(model, x):
    reports = {}
    for row in continuous_trace(model, x):
        if row["v"] is None:
            continue
        v = row["v"].detach()
        flat = v.flatten()
        top = torch.sort(v, dim=-1, descending=True).values
        f = torch.zeros_like(v)
        # For each output neuron, condition on its Boolean-style output.
        ybit = (row["output"].detach() >= .5).unsqueeze(-1).expand_as(v)
        cond = {}
        for label, mask in (("output0", ~ybit), ("output1", ybit)):
            z = v[mask]
            cond[label] = basic_stats(z) if z.numel() else {}
        reports[row["name"]] = {
            "all": basic_stats(v),
            "conditioned": cond,
            "max_per_neuron": basic_stats(top[..., 0]),
            "runner_up": basic_stats(top[..., 1]),
            "winner_margin": basic_stats(top[..., 0] - top[..., 1]),
            "counts_gt": {str(t): float((flat > t).float().mean()) for t in (.1, .25, .5, .75, .9)},
        }
    return reports


def lehmer_report(model, x):
    if model.or_operator != "lehmer_p2":
        return {}
    reports = {}
    for row in continuous_trace(model, x):
        if row["v"] is None:
            continue
        v = row["v"].detach()
        den = v.square().sum(-1, keepdim=True)
        f = v.pow(3).sum(-1, keepdim=True) / den.clamp_min(1e-30)
        sign_term = 3 * v - 2 * f
        grad = v / den.clamp_min(1e-30) * sign_term
        reports[row["name"]] = {
            "fraction_positive": float((grad > 0).float().mean()),
            "fraction_negative": float((grad < 0).float().mean()),
            "fraction_near_zero": float((grad.abs() < 1e-8).float().mean()),
            "mean_abs_negative": float(grad[grad < 0].abs().mean()) if (grad < 0).any() else 0.0,
            "mean_abs_positive": float(grad[grad > 0].abs().mean()) if (grad > 0).any() else 0.0,
            "mean_v_positive": float(v[grad > 0].mean()) if (grad > 0).any() else 0.0,
            "mean_v_negative": float(v[grad < 0].mean()) if (grad < 0).any() else 0.0,
        }
    return reports


def circuit_report(model, threshold=.5):
    result = {}
    for name, layer in zip(LAYERS, layer_list(model)):
        g = layer.effective_gate().detach() >= threshold
        b = layer.actual_bias().detach() >= threshold
        fan = g.sum(1)
        result[name] = {
            "selected_edges": int(g.sum()), "selected_fraction": float(g.float().mean()),
            "negated_selected_edges": int((g & b).sum()),
            "fan_in_mean": float(fan.float().mean()), "fan_in_min": int(fan.min()),
            "fan_in_max": int(fan.max()), "zero_fan_in_neurons": int((fan == 0).sum()),
            "single_input_neurons": int((fan == 1).sum()), "multi_input_neurons": int((fan > 1).sum()),
            "bias_selected_fraction": float(b.float().mean()),
        }
    return result


def metrics(out, y):
    pred = out >= .5
    ok = pred == (y >= .5)
    return {"mse": float((out - y).square().mean()), "bit": float(ok.float().mean()), "exact": float(ok.all(-1).float().mean())}


def output_report(model, x, y):
    with torch.no_grad():
        c, h = model(x), model.forward_hard(x)
    with torch.no_grad():
        b = model.to_discrete(.5)(x.bool()).float()
    result = {"continuous": metrics(c, y), "hard": metrics(h, y), "boolean": metrics(b, y), "bits": {}}
    for i in range(4):
        result["bits"][str(i)] = {
            "continuous": float(((c[:, i] >= .5) == (y[:, i] >= .5)).float().mean()),
            "hard": float(((h[:, i] >= .5) == (y[:, i] >= .5)).float().mean()),
            "boolean": float(((b[:, i] >= .5) == (y[:, i] >= .5)).float().mean()),
        }
    for t in THRESHOLDS:
        with torch.no_grad():
            result.setdefault("thresholds", {})[str(t)] = metrics(model.to_discrete(t)(x.bool()).float(), y)
    exact_ts = [t for t in THRESHOLDS if result["thresholds"][str(t)]["exact"] == 1.0]
    runs=[]; current=[]
    for t in THRESHOLDS:
        if t in exact_ts:
            current.append(t)
        elif current:
            runs.append(current); current=[]
    if current: runs.append(current)
    result["functional_threshold_intervals"] = [[run[0], run[-1]] for run in runs]
    result["functional_threshold_interval"] = max(result["functional_threshold_intervals"], key=lambda z: z[1]-z[0], default=None)
    return result


def hardening_report(model, x, y):
    """Replace one continuous logic layer at a time by hard max."""
    def forward_modes(modes):
        h = x
        layers = layer_list(model)
        i = 0
        h = layers[i].hard_forward(h) if modes[i] else layers[i](h); i += 1
        for block in model.blocks:
            h1 = block.layer1.hard_forward(h) if modes[i] else block.layer1(h); i += 1
            h2 = block.layer2.hard_forward(h1) if modes[i] else block.layer2(h1); i += 1
            h = h + h2 - 2 * h * h2
        h = layers[i].hard_forward(h) if modes[i] else layers[i](h)
        return h
    base = metrics(model(x), y)
    result = {"continuous": base}
    for i, name in enumerate(LAYERS):
        modes = [False] * len(LAYERS); modes[i] = True
        result[name] = metrics(forward_modes(modes), y)
    return result


def functional_layers(model, x):
    rows, _ = discrete_trace(model, x, .5)
    result = {}
    for name, inp, out in rows:
        a = out.float()
        unique = torch.unique(a, dim=0).shape[0]
        ones = a.mean(0)
        result[name] = {
            "unique_activation_vectors": int(unique),
            "activation_one_fraction": float(a.mean()),
            "dead_neurons": int((ones == 0).sum()), "constant_one_neurons": int((ones == 1).sum()),
            "balanced_neurons": int(((ones > .25) & (ones < .75)).sum()),
            "duplicate_neuron_functions": int(a.shape[1] - torch.unique(a.T, dim=0).shape[0]),
        }
    return result


def error_report(model, x, y):
    with torch.no_grad():
        pred = model.to_discrete(.5)(x.bool()).float()
    wrong = (pred != y).any(-1)
    return {
        "wrong_rows": [int(i) for i in torch.nonzero(wrong, as_tuple=False).flatten()],
        "wrong_row_count": int(wrong.sum()),
        "bit_exact_accuracy": [float(((pred[:, i] >= .5) == (y[:, i] >= .5)).float().mean()) for i in range(y.shape[1])],
        "errors_when_operands_equal": int(((pred != y).any(-1) & (x[:, :4] == x[:, 4:]).all(-1)).sum()),
        "errors_when_a_zero": int(((pred != y).any(-1) & (x[:, :4] == 0).all(-1)).sum()),
        "errors_when_b_zero": int(((pred != y).any(-1) & (x[:, 4:] == 0).all(-1)).sum()),
    }


def accumulation_report(model, x):
    if model.or_operator != "probabilistic_or":
        return {}
    result = {}
    for row in continuous_trace(model, x):
        if row["v"] is None:
            continue
        v = row["v"].detach()
        prob = row["output"].detach()
        hard = v.max(-1).values
        gap = prob - hard
        mask = (prob > .99) & (hard < .75)
        examples = []
        for idx in torch.nonzero(mask.flatten(), as_tuple=False).flatten()[:5]:
            sample, out = divmod(int(idx), v.shape[1])
            vals = torch.sort(v[sample, out], descending=True).values[:8]
            examples.append({"sample": sample, "output": out, "prob_or": float(prob[sample, out]), "hardmax": float(hard[sample, out]), "sum_v": float(v[sample, out].sum()), "top_v": [float(q) for q in vals]})
        result[row["name"]] = {"prob_minus_hard_stats": basic_stats(gap), "fraction_prob_gt_.99_hard_lt_.75": float(mask.float().mean()), "examples": examples}
    return result


def analyze_one(op, seed, x, y, checkpoint_dir):
    model, path = load_model(op, seed, checkpoint_dir=checkpoint_dir)
    return {
        "checkpoint": str(path), "output": output_report(model, x, y),
        "contributions": contribution_report(model, x), "lehmer_gradients": lehmer_report(model, x),
        "circuit": circuit_report(model), "functional_layers": functional_layers(model, x),
        "hardening": hardening_report(model, x, y), "accumulation": accumulation_report(model, x),
        "errors": error_report(model, x, y),
    }


def verify_checkpoints(results_path, checkpoint_dir, x, y):
    """Verify hashes and metrics against the supplied B3R result JSON only."""
    archive = json.loads(results_path.read_text())
    results = archive.get("metrics", archive)["results"]
    entries = [entry for result in results for entry in result.get("checkpoint_manifest", [])]
    if not entries:
        raise RuntimeError("B3R result JSON has no checkpoint manifest")
    for entry in entries:
        path = checkpoint_dir / Path(entry["checkpoint"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        model = model_for(entry["operator"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model.eval()
        observed = output_report(model, x, y)
        expected = entry["metrics"]
        for mode in ("continuous", "hard", "boolean"):
            for metric in ("mse", "bit_accuracy", "exact_accuracy"):
                actual = observed[mode][{"bit_accuracy": "bit", "exact_accuracy": "exact"}.get(metric, "mse")]
                if abs(actual - expected[mode][metric]) > 1e-6:
                    raise RuntimeError(f"checkpoint metric mismatch: {path} {mode}.{metric}")


def make_figures(all_data):
    # 1. Seed comparison: continuous MSE and Boolean exact at best checkpoints.
    fig, ax = plt.subplots(figsize=(8, 5))
    for op in OPS:
        vals = [all_data[op][str(s)]["output"]["continuous"]["mse"] for s in SEEDS]
        ax.plot(SEEDS, vals, "o-", label=op)
    ax.set_yscale("log"); ax.set_xlabel("seed"); ax.set_ylabel("minimum continuous MSE"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "b3_lehmer_seed_comparison.png", dpi=160); plt.close(fig)

    # 2. Hardening sensitivity.
    fig, ax = plt.subplots(figsize=(9, 5)); xs = range(len(LAYERS))
    for op in OPS:
        vals=[]
        for s in SEEDS:
            d=all_data[op][str(s)]["hardening"]; vals.append([d[n]["exact"] for n in LAYERS])
        ax.plot(xs, torch.tensor(vals).mean(0), "o-", label=op)
    ax.set_xticks(list(xs), LAYERS, rotation=35, ha="right"); ax.set_ylabel("exact accuracy after one hard layer"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "b3_layer_hardening.png", dpi=160); plt.close(fig)

    # 3. Functional threshold interval endpoints.
    fig, ax = plt.subplots(figsize=(8, 5))
    for op in OPS:
        for s in SEEDS:
            interval=all_data[op][str(s)]["output"]["functional_threshold_interval"]
            if interval: ax.plot([interval[0], interval[1]], [f"{op}:s{s}"]*2, "|-", label=f"{op} s{s}")
    ax.set_xlim(.18,.82); ax.set_xlabel("threshold"); ax.set_title("Functional threshold intervals"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG / "b3_functional_threshold_margin.png", dpi=160); plt.close(fig)

    # 4. Probabilistic accumulation gap by layer.
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEEDS:
        d=all_data["probabilistic_or"][str(s)]["accumulation"]
        names=[n for n in LAYERS if n in d]
        vals=[d[n]["fraction_prob_gt_.99_hard_lt_.75"] for n in names]
        ax.plot(names, vals, "o-", label=f"seed {s}")
    ax.set_ylabel("fraction prob-OR>.99 while max<.75"); ax.tick_params(axis="x", rotation=35); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "b3_prob_or_accumulation.png", dpi=160); plt.close(fig)


def main():
    global CK
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--checkpoints", type=Path, default=CK)
    parser.add_argument("--output", type=Path, default=OUT / "b3r_forensic_analysis.json")
    args = parser.parse_args()
    CK = args.checkpoints
    x, y = task_data()
    verify_checkpoints(args.results, args.checkpoints, x, y)
    data = {op: {str(s): analyze_one(op, s, x, y, args.checkpoints) for s in SEEDS} for op in OPS}
    # Parameter-level distance is useful context, but functional-layer
    # diagnostics in the report remain the primary comparison.
    for op in OPS:
        models = {s: load_model(op, s, checkpoint_dir=args.checkpoints)[0] for s in SEEDS}
        pairwise = {}
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                distance = 0
                for la, lb in zip(layer_list(models[a]), layer_list(models[b])):
                    distance += int(torch.count_nonzero((la.effective_gate() >= .5) ^ (lb.effective_gate() >= .5)))
                    distance += int(torch.count_nonzero((la.actual_bias() >= .5) ^ (lb.actual_bias() >= .5)))
                pairwise[f"seed{a}_vs_seed{b}"] = distance
        data[op]["pairwise_parameter_hamming"] = pairwise
    out = args.output
    out.write_text(json.dumps(data, indent=2))
    make_figures(data)
    print(json.dumps({"output": str(out), "figures": [str(FIG / n) for n in ("b3_lehmer_seed_comparison.png", "b3_layer_hardening.png", "b3_functional_threshold_margin.png", "b3_prob_or_accumulation.png")]}, indent=2))


if __name__ == "__main__":
    main()
