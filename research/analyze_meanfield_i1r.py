"""I1R replicated mean-field propagation diagnostics (no training)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from discrete_logic_net import DiscreteModernLogicGateNet, DiscreteOrNorGateLayer  # noqa: E402
from layers import SigmoidOrLogicLayer  # noqa: E402
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.meanfield_initialization import gaussian_bias_init_, meanfield_gaussian_edge_init_, target_selected_probability  # noqa: E402

P0S = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
BIAS_SPECS = {"CURRENT": (0.5, 0.1), "ONE": (1.0, 0.1), "BALANCED_POLARIZED": (0.5, 2.0)}


def generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def controlled_binary(n: int, d: int, p_zero: float, seed: int, device: torch.device) -> torch.Tensor:
    cpu = torch.ones(n, d, dtype=torch.float32)
    g = generator(seed, torch.device("cpu"))
    for j in range(d):
        cpu[torch.randperm(n, generator=g)[:round(n * p_zero)], j] = 0
    return cpu.to(device)


def activation_stats(h: torch.Tensor) -> dict[str, float]:
    h = h.detach().float()
    q = torch.quantile(h, torch.tensor([.01, .05, .25, .5, .75, .95, .99], device=h.device))
    return {"mean": float(h.mean().cpu()), "std": float(h.std(unbiased=False).cpu()),
            "q01": float(q[0].cpu()), "q05": float(q[1].cpu()), "q25": float(q[2].cpu()),
            "median": float(q[3].cpu()), "q75": float(q[4].cpu()), "q95": float(q[5].cpu()),
            "q99": float(q[6].cpu()), "lt01": float((h < .1).float().mean().cpu()),
            "lt04": float((h < .4).float().mean().cpu()), "mid": float(((h >= .4) & (h <= .6)).float().mean().cpu()),
            "gt06": float((h > .6).float().mean().cpu()), "gt09": float((h > .9).float().mean().cpu()),
            "lt05": float((h < .5).float().mean().cpu())}


def meanfield_curve(p0: float, q: float, s: float | None = None, depth: int = 12) -> list[float]:
    s = target_selected_probability(64) if s is None else s
    values = [float(p0)]
    for _ in range(depth):
        values.append(float((1 - s * (1 - q + (2 * q - 1) * values[-1])) ** 64))
    return values


def build_chain(sigma: float, bias_name: str, seed: int, device: torch.device, depth: int = 12):
    edge_g = generator(100_000 + seed, device)
    bias_g = generator(200_000 + seed, device)
    mean, std = BIAS_SPECS[bias_name]
    layers, discrete = [], []
    for _ in range(depth):
        layer = SigmoidOrLogicLayer(
            64, 64, "lehmer_p2", .5,
            lambda t, mean=mean, std=std, g=bias_g: gaussian_bias_init_(t, mean, std, g),
            lambda t, fan_in, sigma=sigma, g=edge_g: meanfield_gaussian_edge_init_(t, fan_in, sigma, g),
        ).to(device)
        d = DiscreteOrNorGateLayer(64, 64).to(device)
        with torch.no_grad():
            d.weight.copy_(layer.effective_gate() >= .5)
            d.bias.copy_(layer.actual_bias() >= .5)
        layers.append(layer); discrete.append(d)
    return layers, discrete


def meanfield_curve(p0: float, q: float, s: float | None = None, depth: int = 12) -> list[float]:
    s = target_selected_probability(64) if s is None else s
    values = [float(p0)]
    for _ in range(depth):
        values.append(float((1 - s * (1 - q + (2 * q - 1) * values[-1])) ** 64))
    return values


def one_chain(layers, discrete, p0: float, seed: int, n_bool: int, n_cont: int, device: torch.device):
    xb = controlled_binary(n_bool, 64, p0, seed + 10_000, device).bool()
    xc = controlled_binary(n_cont, 64, p0, seed + 20_000, device)
    rows = []
    for depth in range(len(layers) + 1):
        rows.append({"depth": depth, "p_zero_bool": float((~xb).float().mean().cpu()), "continuous": activation_stats(xc)})
        if depth == len(layers): break
        xc = layers[depth](xc); xb = discrete[depth](xb)
    return rows


def summarize_runs(runs):
    result = []
    for depth in range(len(runs[0])):
        vals = [run[depth] for run in runs]; p = torch.tensor([v["p_zero_bool"] for v in vals]); cs = [v["continuous"] for v in vals]
        entry = {"depth": depth, "p_zero_bool_mean": float(p.mean()), "p_zero_bool_std": float(p.std(unbiased=False)),
                 "p_zero_bool_q05": float(torch.quantile(p, .05)), "p_zero_bool_q25": float(torch.quantile(p, .25)),
                 "p_zero_bool_median": float(torch.quantile(p, .5)), "p_zero_bool_q75": float(torch.quantile(p, .75)),
                 "p_zero_bool_q95": float(torch.quantile(p, .95)),
                 "continuous_mean": {k: sum(c[k] for c in cs) / len(cs) for k in cs[0]}}
        if depth < len(runs[0]) - 1:
            rho = []
            for run in runs:
                d0 = run[depth]["p_zero_bool"] - .5; d1 = run[depth + 1]["p_zero_bool"] - .5
                if abs(d0) > 1e-6: rho.append(d1 / d0)
            if rho:
                r = torch.tensor(rho)
                entry.update({"rho_mean": float(r.mean()), "rho_median": float(r.median()), "rho_q05": float(torch.quantile(r, .05)),
                              "rho_q95": float(torch.quantile(r, .95)), "rho_abs_mean": float(r.abs().mean()), "rho_count": len(rho)})
        result.append(entry)
    return result


def fanin_stats(sigma: float, bias_name: str, seeds: int, device: torch.device):
    rows, s_values, q_values = [], [], []
    for seed in range(seeds):
        layers, _ = build_chain(sigma, bias_name, seed, device, depth=1); layer = layers[0]
        k = (layer.effective_gate() >= .5).sum(1)
        rows.append({"zero": float((k == 0).float().mean().cpu()), "one": float((k == 1).float().mean().cpu()),
                     "two": float((k == 2).float().mean().cpu()), "three": float((k == 3).float().mean().cpu()),
                     "four_plus": float((k >= 4).float().mean().cpu()), "mean": float(k.float().mean().cpu()),
                     "median": float(k.float().median().cpu()), "max": int(k.max().cpu())})
        s_values.append(float((layer.effective_gate() >= .5).float().mean().cpu()))
        q_values.append(float((layer.actual_bias() >= .5).float().mean().cpu()))
    result = {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}
    result["s_observed"] = sum(s_values) / len(s_values); result["q_observed"] = sum(q_values) / len(q_values)
    return result


def discrete_residual_trace(discrete, x):
    hb = x.bool(); out = [("input", hb)]; hb = discrete.stem(hb); out.append(("stem", hb))
    for bi, block in enumerate(discrete.blocks):
        h1 = block.layer1(hb); out.append((f"block{bi}.layer1", h1))
        h2 = block.layer2(h1); out.append((f"block{bi}.layer2", h2)); hb = hb ^ h2; out.append((f"block{bi}.residual", hb))
    out.append(("head", discrete.head(hb))); return out


@torch.no_grad()
def residual_stats(sigma: float, bias_name: str, seeds: int, device: torch.device):
    x = build_task("bitwise_xor_truth_table", {"bits": 4})["X"].float().to(device); all_rows = []
    for seed in range(seeds):
        edge_g = generator(300_000 + seed, device); bias_g = generator(400_000 + seed, device); mean, std = BIAS_SPECS[bias_name]
        model = SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2", gate_initializations=[.5] * 6,
            bias_initialization=lambda t, mean=mean, std=std, g=bias_g: gaussian_bias_init_(t, mean, std, g),
            edge_initialization=lambda t, fan_in, sigma=sigma, g=edge_g: meanfield_gaussian_edge_init_(t, fan_in, sigma, g)).to(device)
        discrete = model.to_discrete(.5).to(device); continuous = [("input", x)]; h = model.stem(x); continuous.append(("stem", h))
        for bi, block in enumerate(model.blocks):
            h1 = block.layer1(h); continuous.append((f"block{bi}.layer1", h1)); h2 = block.layer2(h1); continuous.append((f"block{bi}.layer2", h2)); h = h + h2 - 2 * h * h2; continuous.append((f"block{bi}.residual", h))
        continuous.append(("head", model.head(h))); exact = dict(discrete_residual_trace(discrete, x.bool()))
        all_rows.append({name: {"continuous": activation_stats(value), "continuous_below_half": float((value < .5).float().mean().cpu()), "discrete_zero": float((~exact[name]).float().mean().cpu())} for name, value in continuous})
    names = list(all_rows[0]); return [{"name": name, "continuous_mean": {k: sum(row[name]["continuous"][k] for row in all_rows) / len(all_rows) for k in all_rows[0][name]["continuous"]},
        "continuous_below_half_mean": sum(row[name]["continuous_below_half"] for row in all_rows) / len(all_rows), "discrete_zero_mean": sum(row[name]["discrete_zero"] for row in all_rows) / len(all_rows)} for name in names]


def bias_signal_stats(name: str, device: torch.device):
    mean, std = BIAS_SPECS[name]; g = generator(91_000 + list(BIAS_SPECS).index(name), device)
    probe = SigmoidOrLogicLayer(64, 64, "lehmer_p2", .5,
                                lambda t, mean=mean, std=std, g=g: gaussian_bias_init_(t, mean, std, g)).to(device)
    b = probe.actual_bias().detach(); gain = (1 - 2 * b).abs()
    x = controlled_binary(8192, 64, .5, 70_000, device); literal = x.unsqueeze(1) + b.unsqueeze(0) - 2 * x.unsqueeze(1) * b.unsqueeze(0)
    return {"effective_mean": float(b.mean().cpu()), "effective_std": float(b.std().cpu()), "threshold_one": float((b >= .5).float().mean().cpu()),
            "le01": float((b <= .01).float().mean().cpu()), "ge99": float((b >= .99).float().mean().cpu()), "middle": float(((b > .4) & (b < .6)).float().mean().cpu()),
            "gain_mean": float(gain.mean().cpu()), "gain_median": float(gain.median().cpu()), "gain_lt01": float((gain < .1).float().mean().cpu()), "gain_gt09": float((gain > .9).float().mean().cpu()),
            "literal_variance": float(literal.var(unbiased=False).cpu()), "input_variance": float(x.var(unbiased=False).cpu()), "literal_to_input_variance_ratio": float((literal.var(unbiased=False) / x.var(unbiased=False)).cpu())}


def cuda_boolean_smoke(device: torch.device):
    if device.type != "cuda": return {"performed": False}
    torch.manual_seed(1234); cpu = DiscreteModernLogicGateNet(8, 4, width=64, num_residual_blocks=2)
    for layer in cpu.expectation_layers:
        layer.weight.copy_(torch.rand_like(layer.weight.float()) > .5); layer.bias.copy_(torch.rand_like(layer.bias.float()) > .5)
    gpu = DiscreteModernLogicGateNet(8, 4, width=64, num_residual_blocks=2).to(device); gpu.load_state_dict(cpu.state_dict())
    x = controlled_binary(8192, 8, .5, 1235, torch.device("cpu")).bool()
    return {"performed": True, "equal": bool(torch.equal(cpu(x), gpu(x.to(device)).cpu()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); ap.add_argument("--network-seeds", type=int, default=64); ap.add_argument("--residual-seeds", type=int, default=64); ap.add_argument("--bool-batch", type=int, default=8192); ap.add_argument("--continuous-batch", type=int, default=512); ap.add_argument("--output", default=str(ROOT / "research/operator_results/initialization_i1r_full.json")); args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cpu": torch.set_num_threads(4)
    result = {"config": {"device": str(device), "network_seeds": args.network_seeds, "residual_seeds": args.residual_seeds, "bool_batch": args.bool_batch, "continuous_batch": args.continuous_batch, "p0": P0S, "depth": 12}, "cuda_boolean_smoke": cuda_boolean_smoke(device), "theory": {"s_target_64": target_selected_probability(64), "ideal_theory": {}, "observed_parameter_theory": {}}, "plain_chain": {}, "fanin": {}, "bias_signal": {}, "residual_network": {}}
    for name in BIAS_SPECS:
        result["bias_signal"][name] = bias_signal_stats(name, device); q = 1.0 if name == "ONE" else .5
        result["theory"]["ideal_theory"][name] = [{"p0": p, "trajectory": meanfield_curve(p, q)} for p in P0S]
        for sigma in (2.0, 4.0, 6.0):
            key = f"{name}_sigma{int(sigma)}"; result["plain_chain"][key] = {}; result["fanin"][key] = fanin_stats(sigma, name, args.network_seeds, device); obs = result["fanin"][key]
            result["theory"]["observed_parameter_theory"][key] = [{"p0": p, "trajectory": meanfield_curve(p, obs["q_observed"], obs["s_observed"])} for p in P0S]
            if sigma in (2.0, 4.0): result["residual_network"][key] = residual_stats(sigma, name, args.residual_seeds, device)
            for seed in range(args.network_seeds):
                layers, discrete = build_chain(sigma, name, seed, device)
                for p0 in P0S:
                    result["plain_chain"].setdefault(key, {}).setdefault(str(p0), []).append(one_chain(layers, discrete, p0, seed, args.bool_batch, args.continuous_batch, device))
            for p0 in P0S: result["plain_chain"][key][str(p0)] = summarize_runs(result["plain_chain"][key][str(p0)])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)); print(output)


if __name__ == "__main__": main()
