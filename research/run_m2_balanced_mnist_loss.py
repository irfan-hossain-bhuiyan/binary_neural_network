"""M2 balanced Boolean-seeking MNIST loss study.

This runner deliberately keeps the experiment small and explicit: binary
MNIST inputs, the existing sigmoid/Lehmer-p2 XOR-residual network, and paired
initial states/minibatch orders for BCE, POWER_1_25, and the I13 curriculum.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.meanfield_initialization import (  # noqa: E402
    bias_one_normal_init_, meanfield_gaussian_edge_init_,
)

OUT = ROOT / "research/operator_results"
CKOUT = OUT / "m2_balanced_mnist_loss_checkpoints"
FIG = ROOT / "research/figures"
DATA_PATH = ROOT / "artifacts/mnist_binary.pt"
LOSSES = ("BCE", "BALANCED_BCE", "BALANCED_POWER_1_25")
SEEDS = (0, 1, 2)
SPLIT_SEED = 20260924
EPOCHS = 60
DIAG_N = 1024
EVAL_EPOCHS = (0, 5, 10, 20, 30, 40, 50, 60)
M1_RESULT = OUT / "m1_mnist_pilot_results.json"
M1_INITIAL_HASHES = {}
if M1_RESULT.exists():
    try:
        _m1 = json.loads(M1_RESULT.read_text())
        M1_INITIAL_HASHES = {str(r["seed"]): r["initial_state_sha256"] for r in _m1.get("runs", []) if r["loss"] == "BCE"}
    except Exception:
        M1_INITIAL_HASHES = {}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(seed: int) -> SigmoidOrModernLogicGateNet:
    edge_gen = torch.Generator(device="cpu").manual_seed(100_000 + seed)
    bias_gen = torch.Generator(device="cpu").manual_seed(200_000 + seed)

    def edge_init(tensor, fan_in):
        meanfield_gaussian_edge_init_(tensor, fan_in, sigma=2.0, generator=edge_gen)

    def bias_init(tensor):
        bias_one_normal_init_(tensor, std=0.1, generator=bias_gen)

    return SigmoidOrModernLogicGateNet(
        784, 10, width=256, num_residual_blocks=2,
        or_operator="lehmer_p2", gate_initializations=[0.5] * 6,
        bias_initialization=bias_init, edge_initialization=edge_init,
    )


def load_data() -> dict[str, torch.Tensor | str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"MNIST artifact not found: {DATA_PATH}")
    blob = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    raw_x, labels = blob["X"], blob["Y"]
    if tuple(raw_x.shape) != (70_000, 28, 28) or tuple(labels.shape) != (70_000,):
        raise ValueError(f"unexpected MNIST artifact shapes: {raw_x.shape}, {labels.shape}")
    # The artifact is the canonical uint8-valued MNIST train followed by test.
    x = (raw_x.reshape(70_000, 784) >= 128).float()
    y = F.one_hot(labels.long(), num_classes=10).float()
    gen = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    perm = torch.randperm(60_000, generator=gen)
    train_idx, val_idx = perm[:55_000], perm[55_000:]
    return {
        "train_x": x[train_idx], "train_y": y[train_idx],
        "val_x": x[val_idx], "val_y": y[val_idx],
        "test_x": x[60_000:], "test_y": y[60_000:],
        "train_idx": train_idx, "val_idx": val_idx,
        "artifact_sha256": file_hash(DATA_PATH),
    }


def alpha_for_step(loss_name: str, step: int, total_steps: int) -> tuple[float | None, str]:
    if loss_name in ("BCE", "BALANCED_BCE"):
        return None, loss_name
    return 1.25, "BALANCED_POWER_1_25"


def balanced_loss_value(output, target, loss_name):
    eps = 1e-7
    positive = target.bool()
    negative = ~positive
    if loss_name == "BCE":
        return F.binary_cross_entropy(output.clamp(eps, 1 - eps), target)
    if loss_name == "BALANCED_BCE":
        pos = -torch.log(output.clamp(eps, 1 - eps).masked_select(positive)).mean()
        neg = -torch.log1p(-output.clamp(eps, 1 - eps).masked_select(negative)).reshape(-1, 9).mean(dim=1).mean()
        return pos + neg
    if loss_name == "BALANCED_POWER_1_25":
        pos = (1 - output.masked_select(positive)).abs().pow(1.25).mean()
        neg = output.masked_select(negative).abs().reshape(-1, 9).pow(1.25).mean(dim=1).mean()
        return pos + neg
    raise ValueError(loss_name)


def loss_value(output, target, loss_name, step, total_steps):
    alpha, phase = alpha_for_step(loss_name, step, total_steps)
    return balanced_loss_value(output, target, loss_name), alpha, phase

def output_summary(output, target) -> dict:
    threshold = (output >= 0.5).float()
    strict = (threshold == target).all(dim=1)
    valid = threshold.sum(dim=1) == 1
    zero = threshold.sum(dim=1) == 0
    multi = threshold.sum(dim=1) > 1
    pred = output.argmax(dim=1)
    true = target.argmax(dim=1)
    distance = torch.minimum(output, 1 - output)
    confidence = target * output + (1 - target) * (1 - output)
    non_target = output.masked_fill(target.bool(), -1)
    margin = output.masked_select(target.bool()).reshape(-1) - non_target.max(dim=1).values
    return {
        "mse": float((output - target).square().mean()),
        "bce": float(F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), target)),
        "mae": float((output - target).abs().mean()),
        "e_inf": float((output - target).abs().max()),
        "p95_abs_error": float(torch.quantile((output - target).abs().reshape(-1), .95)),
        "p99_abs_error": float(torch.quantile((output - target).abs().reshape(-1), .99)),
        "argmax_accuracy": float((pred == true).float().mean()),
        "threshold_strict_accuracy": float(strict.float().mean()),
        "valid_onehot_rate": float(valid.float().mean()),
        "zero_hot_rate": float(zero.float().mean()),
        "multi_hot_rate": float(multi.float().mean()),
        "target_bit_active_rate": float(threshold.masked_select(target.bool()).mean()),
        "average_active_bits": float(threshold.sum(dim=1).float().mean()),
        "mean_endpoint_distance": float(distance.mean()),
        "median_endpoint_distance": float(distance.median()),
        "p90_endpoint_distance": float(torch.quantile(distance.reshape(-1), .90)),
        "p95_endpoint_distance": float(torch.quantile(distance.reshape(-1), .95)),
        "fraction_endpoint_lt_.01": float((distance < .01).float().mean()),
        "fraction_endpoint_lt_.05": float((distance < .05).float().mean()),
        "fraction_abs_minus_.5_lt_.05": float((distance > .45).float().mean()),
        "fraction_abs_minus_.5_lt_.10": float((distance > .40).float().mean()),
        "mean_confidence": float(confidence.mean()),
        "minimum_confidence": float(confidence.min()),
        "p05_confidence": float(torch.quantile(confidence, .05)),
        "mean_class_margin": float(margin.mean()),
        "median_class_margin": float(margin.median()),
        "p05_class_margin": float(torch.quantile(margin, .05)),
        "fraction_margin_gt_0": float((margin > 0).float().mean()),
        "fraction_margin_gt_.25": float((margin > .25).float().mean()),
        "fraction_margin_gt_.5": float((margin > .5).float().mean()),
    }


@torch.no_grad()
def semantics(model, x_float, x_bool, target, batch_size=256) -> dict:
    outputs, hards, bools = [], [], []
    disc = model.to_discrete(.5).to(x_float.device)
    for start in range(0, len(x_float), batch_size):
        xf, xb = x_float[start:start+batch_size], x_bool[start:start+batch_size]
        outputs.append(model(xf).cpu())
        hards.append(model.forward_hard(xf).cpu())
        bools.append(disc(xb).float().cpu())
    out, hard, boolean = torch.cat(outputs), torch.cat(hards), torch.cat(bools)
    cont = output_summary(out, target.cpu())
    hard_m = output_summary(hard, target.cpu())
    bool_m = output_summary(boolean, target.cpu())
    ct = (out >= .5).float()
    disagreement = (ct != boolean).float()
    ct_ok, b_ok = (ct == target.cpu()).all(1), (boolean == target.cpu()).all(1)
    bool_m.update({
        "bit_disagreement_fraction": float(disagreement.mean()),
        "sample_disagreement_fraction": float(disagreement.any(1).float().mean()),
        "mean_hamming_continuous_to_boolean": float(disagreement.sum(1).mean()),
        "both_correct": int((ct_ok & b_ok).sum()),
        "continuous_correct_boolean_wrong": int((ct_ok & ~b_ok).sum()),
        "continuous_wrong_boolean_correct": int((~ct_ok & b_ok).sum()),
        "both_wrong": int((~ct_ok & ~b_ok).sum()),
    })
    return {"continuous": cont, "hard": hard_m, "boolean": bool_m}


def parameter_stats(model) -> list[dict]:
    rows = []
    for i, layer in enumerate(model.expectation_layers):
        g, b, r = layer.effective_gate().detach(), layer.actual_bias().detach(), layer.raw_edge.detach()
        rows.append({"layer": i, "in_features": layer.in_features, "out_features": layer.out_features,
                     "gate_mean": float(g.mean()), "gate_mean_distance": float(torch.minimum(g, 1-g).mean()),
                     "gate_lt_.01": float((g < .01).float().mean()), "gate_gt_.99": float((g > .99).float().mean()),
                     "raw_lt_.1": float((r.abs() < .1).float().mean()), "raw_lt_.5": float((r.abs() < .5).float().mean()),
                     "raw_gt_2": float((r.abs() > 2).float().mean()), "raw_gt_4": float((r.abs() > 4).float().mean()),
                     "bias_mean_distance": float(torch.minimum(b, 1-b).mean()),
                     "bias_near_.5": float((b.sub(.5).abs() < .05).float().mean()),
                     "bias_near_endpoints": float((torch.minimum(b, 1-b) < .05).float().mean())})
    return rows


def trace_stages(model, x_float, x_bool) -> list[dict]:
    disc = model.to_discrete(.5).to(x_float.device)
    h, hb, rows = x_float, x_bool, []
    def add(name, c, b):
        rows.append({"name": name, "bit_mismatch_fraction": float(((c >= .5) != b).float().mean()),
                     "sample_mismatch_fraction": float((((c >= .5) != b).any(1)).float().mean())})
    add("input", h, hb)
    h, hb = model.stem(h), disc.stem(hb); add("stem", h, hb)
    for i, block in enumerate(model.blocks):
        h1, hb1 = block.layer1(h), disc.blocks[i].layer1(hb); add(f"block{i}.layer1", h1, hb1)
        h2, hb2 = block.layer2(h1), disc.blocks[i].layer2(hb1); add(f"block{i}.layer2", h2, hb2)
        h, hb = h + h2 - 2*h*h2, hb ^ hb2; add(f"block{i}.residual", h, hb)
    h, hb = model.head(h), disc.head(hb); add("head", h, hb)
    return rows


def mask_hamming(model, initial_state):
    names = ["stem", "blocks.0.layer1", "blocks.0.layer2", "blocks.1.layer1", "blocks.1.layer2", "head"]
    edge = bias = 0
    for name, layer in zip(names, model.expectation_layers):
        edge += int(((torch.sigmoid(initial_state[f"{name}.raw_edge"]) >= .5) != (layer.effective_gate() >= .5).cpu()).sum())
        bias += int(((initial_state[f"{name}.bias"].clamp(0, 1) >= .5) != (layer.actual_bias() >= .5).cpu()).sum())
    return edge, bias


def initial_diagnostics(model, x):
    stats = parameter_stats(model)
    with torch.no_grad():
        h = x[:256]
        acts = []
        for layer in model.expectation_layers:
            h = layer(h); acts.append({"mean": float(h.mean()), "std": float(h.std(unbiased=False)),
                                      "lt_.1": float((h < .1).float().mean()), "gt_.9": float((h > .9).float().mean()),
                                      "near_.5": float((h.sub(.5).abs() < .05).float().mean())})
    return {"parameter_stats": stats, "activation_stats": acts}


def smoke_test(device, batch_size):
    model = make_model(0).to(device)
    x = torch.randint(0, 2, (batch_size, 784), device=device).float()
    y = F.one_hot(torch.arange(batch_size, device=device) % 10, 10).float()
    model.zero_grad(set_to_none=True)
    out = model(x); loss = out.square().mean(); loss.backward()
    return {"batch_size": batch_size, "device": str(device), "output_shape": list(out.shape),
            "loss": float(loss.detach()), "finite": bool(torch.isfinite(loss)),
            "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}


def evaluate_checkpoint(model, state, data, device):
    model.load_state_dict(state); model.to(device); model.eval()
    return semantics(model, data["val_x"].to(device), data["val_x"].to(device).bool(), data["val_y"], 256)


def distribution_stats(model, x, target, batch_size=256):
    values=[]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(model(x[start:start+batch_size]).detach())
    out=torch.cat(values)
    pos=out.masked_select(target.bool()); neg=out.masked_select(~target.bool())
    def stats(v):
        return {"mean":float(v.mean()),"median":float(v.median()),"p05":float(torch.quantile(v,.05)),"p95":float(torch.quantile(v,.95)),"lt_.1":float((v<.1).float().mean()),"gt_.9":float((v>.9).float().mean())}
    return {"target":stats(pos),"non_target":stats(neg)}


def gradient_sanity() -> dict:
    rows=[]
    for p in (0.01,0.1,0.5,0.9):
        pred=torch.full((10,10),p,requires_grad=True)
        target=F.one_hot(torch.arange(10),10).float()
        vals={}
        for name in LOSSES:
            pred.grad=None
            loss=balanced_loss_value(pred,target,name)
            grad=torch.autograd.grad(loss,pred,retain_graph=True)[0]
            vals[name]={"loss":float(loss.detach()),"target_grad_mean":float(grad.diagonal().mean()),"non_target_grad_mean":float((grad.sum()-grad.diagonal().sum())/90),"target_abs_grad_mean":float(grad.diagonal().abs().mean()),"non_target_abs_grad_mean":float((grad.abs().sum()-grad.diagonal().abs().sum())/90),"target_abs_grad_aggregate":float(grad.diagonal().abs().mean()),"non_target_abs_grad_aggregate":float((grad.abs().sum()-grad.diagonal().abs().sum())/10)}
        rows.append({"p":p,"losses":vals})
    return {"description":"synthetic uniform-prediction gradient sanity; target is one-hot across ten classes","rows":rows}


def run_one(seed, loss_name, data, device, batch_size, epochs=EPOCHS, smoke=False):
    base = make_model(seed)
    initial = snapshot(base); initial_hash = state_hash(initial)
    m1_expected = M1_INITIAL_HASHES.get(str(seed))
    if m1_expected is not None and initial_hash != m1_expected:
        raise RuntimeError(f"M2 seed {seed} initial hash {initial_hash} does not match M1 {m1_expected}")
    model = make_model(seed); model.load_state_dict(initial); model.to(device)
    if state_hash(snapshot(model)) != initial_hash:
        raise RuntimeError(f"paired initial reload mismatch seed {seed}")
    tx, ty = data["train_x"], data["train_y"]
    vx, vy = data["val_x"], data["val_y"]
    # Keep the fixed binary split resident on the accelerator for the pilot;
    # this changes no numerical semantics and avoids host-to-device copies.
    xb, yb = tx.to(device), ty.to(device)
    vx_dev, vy_dev = vx.to(device), vy.to(device)
    xbool = vx_dev.bool()
    steps_per_epoch = (len(tx) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    # One deterministic order per epoch, reused by every loss arm of this seed.
    orders = []
    gen = torch.Generator(device="cpu").manual_seed(700_000 + seed)
    for _ in range(epochs): orders.append(torch.randperm(len(tx), generator=gen))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    records, saved = [], {}
    init_diag = initial_diagnostics(model, xb[:256])
    best = {"continuous": (-1.0, None, None), "continuous_strict": (-1.0, None, None), "boolean": (-1.0, None, None)}
    step = 0
    epoch_records = {}
    for epoch in range(epochs + 1):
        model.eval()
        val = semantics(model, vx_dev, xbool, vy_dev, 256)
        ps = parameter_stats(model)
        edge_h, bias_h = mask_hamming(model, initial)
        rec = {"epoch": epoch, "step": step, **val, "parameter_stats": ps, "edge_hamming_from_initial": edge_h, "bias_hamming_from_initial": bias_h}
        if epoch in EVAL_EPOCHS:
            rec["output_distributions"] = distribution_stats(model, vx_dev, vy_dev, 256)
        if epoch in EVAL_EPOCHS:
            rec["layer_trace_1024"] = trace_stages(model, vx_dev[:DIAG_N], xbool[:DIAG_N])
        records.append(rec); epoch_records[epoch] = rec
        keys = (("continuous", val["continuous"]["argmax_accuracy"]), ("continuous_strict", val["continuous"]["threshold_strict_accuracy"]), ("boolean", val["boolean"]["threshold_strict_accuracy"]))
        for key, score in keys:
            if score > best[key][0]: best[key] = (score, epoch, snapshot(model))
        if epoch == epochs: break
        model.train(); total_obj = 0.0; finite = True
        order = orders[epoch]
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            out = model(xb[ids])
            loss, alpha, phase = loss_value(out, yb[ids], loss_name, step, total_steps)
            if not torch.isfinite(loss): finite = False; break
            opt.zero_grad(set_to_none=True); loss.backward();
            grad_norms = [float(p.grad.detach().norm()) for p in model.parameters() if p.grad is not None]
            opt.step(); total_obj += float(loss.detach()); step += 1
        records[-1]["training_objective_mean"] = total_obj / max(1, len(orders[epoch]))
        records[-1]["alpha_last_step"] = alpha; records[-1]["curriculum_phase_last_step"] = phase
        records[-1]["gradient_norm_mean_last_epoch"] = sum(grad_norms) / max(1, len(grad_norms)) if grad_norms else 0.0
        records[-1]["finite"] = finite
        if not finite: break
    final_state = snapshot(model)
    # Ensure epoch-20 validation exists if a non-finite run stopped early.
    selected = {"best_validation_continuous_argmax": best["continuous"],
                "best_validation_continuous_strict": best["continuous_strict"],
                "best_validation_boolean_strict": best["boolean"]}
    ckmeta = {}
    ckdir = CKOUT / f"seed{seed}_{loss_name.lower()}"; ckdir.mkdir(parents=True, exist_ok=True)
    for kind, (score, ep, state) in selected.items():
        if state is None: continue
        path = ckdir / f"{kind}.pt"; torch.save(state, path)
        check = make_model(seed); check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        got = evaluate_checkpoint(check, snapshot(check), data, torch.device("cpu"))
        ckmeta[kind] = {"path": str(path), "sha256": file_hash(path), "epoch": ep, "metric": score, "reload_verified": True,
                        "reload_validation": got, "initial_state_sha256": initial_hash}
    path = ckdir / "final.pt"; torch.save(final_state, path)
    check = make_model(seed); check.load_state_dict(final_state)
    ckmeta["final"] = {"path": str(path), "sha256": file_hash(path), "epoch": len(records)-1, "reload_verified": True,
                        "reload_validation": evaluate_checkpoint(check, final_state, data, torch.device("cpu")), "initial_state_sha256": initial_hash}
    # Evaluate every validation-selected checkpoint on the untouched official
    # test split.  Selection happened using validation only.
    test_metrics = {}
    for kind, meta in ckmeta.items():
        check = make_model(seed)
        check.load_state_dict(torch.load(meta["path"], map_location="cpu", weights_only=True))
        test_metrics[kind] = semantics(check, data["test_x"], data["test_x"].bool(), data["test_y"], 256)
    return {"seed": seed, "loss": loss_name, "initial_state_sha256": initial_hash, "batch_size": batch_size,
            "epochs": epochs, "steps": step, "steps_per_epoch": steps_per_epoch, "trajectory": records,
            "checkpoints": ckmeta, "test_metrics": test_metrics, "initial_diagnostics": init_diag,
            "finite": bool(records[-1].get("finite", True)), "final": records[-1]}


def run_smoke_and_matrix(args):
    data = load_data(); device = torch.device(args.device)
    batch = args.batch_size
    smoke = None
    for b in (batch, 32, 16):
        try:
            smoke = smoke_test(device, b); batch = b; break
        except (RuntimeError, MemoryError) as exc:
            if "out of memory" not in str(exc).lower() and b == 16: raise
            if device.type == "cuda": torch.cuda.empty_cache()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = os.environ.get("RESEARCH_GIT_SHA", "uncommitted")
    payload = {"experiment": "M2-balanced-mnist-loss", "git_sha": git_sha,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                           "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
               "data": {"source": str(DATA_PATH), "artifact_sha256": data["artifact_sha256"], "split_seed": SPLIT_SEED,
                        "train": 55_000, "validation": 5_000, "test": 10_000, "input_threshold_uint8": 128},
               "architecture": {"input_dim": 784, "width": 256, "residual_blocks": 2, "output_dim": 10, "operator": "lehmer_p2", "initializer": "I2-B meanfield sigma2 + BIAS_ONE"},
               "training": {"optimizer": "Adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": args.epochs, "batch_size": batch, "losses": list(LOSSES), "no_scheduler": True},
               "smoke_test": smoke, "gradient_sanity": gradient_sanity(), "split_indices_sha256": state_hash({"train": data["train_idx"], "val": data["val_idx"]}),
               "runs": []}
    if args.preflight:
        for loss in LOSSES:
            payload["runs"].append(run_one(0, loss, data, device, batch, epochs=2, smoke=True))
        return payload
    for seed in SEEDS:
        for loss in LOSSES:
            print(f"M2 running seed={seed} loss={loss} batch={batch}", flush=True)
            payload["runs"].append(run_one(seed, loss, data, device, batch, epochs=args.epochs))
    return payload


def main():
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--epochs", type=int, default=60); p.add_argument("--preflight", action="store_true"); p.add_argument("--output", default=str(OUT / "m2_balanced_mnist_loss_results.json")); args = p.parse_args()
    result = run_smoke_and_matrix(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__": main()
