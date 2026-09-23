"""I8 exact Boolean forward with straight-through gate gradients."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i3_endpoint import common_metrics  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
I8CK = OUT / "i8_boolean_ste_checkpoints"
PARENT = CK / "I2_I2-B_seed3_best_continuous_mse.pt"
EVAL_STEPS = [0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
GRAD_STEPS = {0, 10, 100, 500}


class _SigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw):
        ctx.save_for_backward(raw)
        return (raw >= 0).to(raw.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (raw,) = ctx.saved_tensors
        soft = torch.sigmoid(raw)
        return grad_output * soft * (1 - soft)


class _ConstantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw):
        return (raw >= 0).to(raw.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0.25


def make_model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2")


def load_parent() -> tuple[SigmoidOrModernLogicGateNet, Path, str]:
    sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    net = make_model()
    net.load_state_dict(torch.load(PARENT, map_location="cpu", weights_only=True))
    return net, PARENT, sha


def ste_gate(raw: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "STE_SIGMOID":
        return _SigmoidSTE.apply(raw)
    if variant == "STE_CONST025":
        return _ConstantSTE.apply(raw)
    raise ValueError(variant)


def _logic_ste(layer, x: torch.Tensor, variant: str) -> torch.Tensor:
    bias = (layer.actual_bias() >= 0.5).to(x.dtype).detach()
    literal = x.unsqueeze(1) + bias.unsqueeze(0) - 2 * x.unsqueeze(1) * bias.unsqueeze(0)
    contribution = ste_gate(layer.raw_edge, variant).unsqueeze(0) * literal
    return 1 - torch.prod(1 - contribution, dim=-1)


def boolean_ste_trace(net, x: torch.Tensor, variant: str) -> dict[str, torch.Tensor]:
    """Exact Boolean forward values with surrogate gate derivatives."""
    stages: dict[str, torch.Tensor] = {"input": x}
    h = x
    h = _logic_ste(net.stem, h, variant); stages["stem"] = h
    for i, block in enumerate(net.blocks):
        h1 = _logic_ste(block.layer1, h, variant); stages[f"block{i}.layer1"] = h1
        h2 = _logic_ste(block.layer2, h1, variant); stages[f"block{i}.layer2"] = h2
        h = h + h2 - 2 * h * h2; stages[f"block{i}.residual"] = h
    stages["head"] = _logic_ste(net.head, h, variant)
    return stages


def exact_discrete_trace(net, x: torch.Tensor) -> dict[str, torch.Tensor]:
    disc = net.to_discrete(0.5)
    stages: dict[str, torch.Tensor] = {"input": x.bool()}
    hb = x.bool()
    hb = disc.stem(hb); stages["stem"] = hb
    for i, block in enumerate(disc.blocks):
        h1 = block.layer1(hb); stages[f"block{i}.layer1"] = h1
        h2 = block.layer2(h1); stages[f"block{i}.layer2"] = h2
        hb = hb ^ h2; stages[f"block{i}.residual"] = hb
    stages["head"] = disc.head(hb)
    return stages


def forward_equivalence(net, x: torch.Tensor, variant: str) -> dict:
    ste = boolean_ste_trace(net, x, variant)
    exact = exact_discrete_trace(net, x)
    mismatches = {name: int((ste[name].detach() != exact[name].float()).sum()) for name in ste}
    return {"variant": variant, "mismatches": mismatches, "total_mismatch": sum(mismatches.values()),
            "all_equal": sum(mismatches.values()) == 0}


def task_loss(net, x, y):
    out = net(x)
    if not torch.isfinite(out).all():
        return out.sum() * 0 + torch.tensor(float("nan"), device=out.device)
    row_bce = F.binary_cross_entropy(out.clamp(1e-7, 1 - 1e-7), y, reduction="none").mean(dim=1)
    return torch.topk(row_bce, 8).values.mean()


def bool_loss(net, x, y, variant: str):
    out = boolean_ste_trace(net, x, variant)["head"]
    return (out - y).square().mean()


def _nan_metrics():
    return {"finite": False, "mse": float("nan"), "bce": float("nan"), "mae": float("nan"),
            "e_inf": float("nan"), "p95_abs_error": float("nan"), "p99_abs_error": float("nan"),
            "bit_accuracy": float("nan"), "exact_accuracy": float("nan"), "wrong_rows": None, "wrong_bits": None}


def metrics(net, x, y):
    with torch.no_grad():
        out = net(x); hard = net.forward_hard(x); boolean = net.to_discrete(0.5)(x.bool()).float()
        return {"continuous": common_metrics(out, y) if torch.isfinite(out).all() else _nan_metrics(),
                "hard": common_metrics(hard, y) if torch.isfinite(hard).all() else _nan_metrics(),
                "boolean": common_metrics(boolean, y) if torch.isfinite(boolean).all() else _nan_metrics()}


def mask_state(net):
    edges, biases, hashes = [], [], []
    for layer in net.expectation_layers:
        edge = (layer.raw_edge.detach() >= 0).cpu(); bias = (layer.actual_bias().detach() >= .5).cpu()
        edges.append(edge); biases.append(bias)
        hashes.append({"edge": hashlib.sha256(edge.numpy().tobytes()).hexdigest(),
                       "bias": hashlib.sha256(bias.numpy().tobytes()).hexdigest()})
    return {"edges": edges, "biases": biases, "hashes": hashes}


def mask_diff(a, b):
    if a is None: return {"edge_bits": 0, "bias_bits": 0, "layers": []}
    layers = [{"edge_bits": int((ea != eb).sum()), "bias_bits": int((ba != bb).sum())}
              for ea, eb, ba, bb in zip(a["edges"], b["edges"], a["biases"], b["biases"])]
    return {"edge_bits": sum(v["edge_bits"] for v in layers), "bias_bits": sum(v["bias_bits"] for v in layers), "layers": layers}


def grad_vector(net, loss):
    params = [layer.raw_edge for layer in net.expectation_layers]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


def gradient_summary(net, task_grads, bool_grads, task_loss_value, bool_loss_value, variant):
    rows = []
    for i, (layer, gt, gb) in enumerate(zip(net.expectation_layers, task_grads, bool_grads)):
        raw = layer.raw_edge.detach()
        total = gt + current_lambda * gb
        toward = ((raw < 0) & (gt < 0)).float().mean() if variant == "TASK" else (((raw < 0) & (gb < 0)) | ((raw >= 0) & (gb > 0))).float().mean()
        a = gt.flatten(); b = gb.flatten(); denom = a.norm() * b.norm()
        rows.append({"layer": i, "task_l2": float(gt.norm()), "bool_l2": float(gb.norm()),
                     "combined_l2": float(total.norm()), "task_mean_abs": float(gt.abs().mean()),
                     "bool_mean_abs": float(gb.abs().mean()), "bool_median_abs": float(gb.abs().median()),
                     "bool_fraction_lt_1e-8": float((gb.abs() < 1e-8).float().mean()),
                     "bool_cosine_task": float(torch.dot(a, b) / denom) if denom > 0 else None,
                     "bool_fraction_toward_threshold": float(toward)})
    return rows


def initial_diagnostics(net, x, y, variant):
    task = task_loss(net, x, y); bl = bool_loss(net, x, y, variant)
    tg = grad_vector(net, task); bg = grad_vector(net, bl)
    layer = 4; gt = tg[layer][62, 10]; gb = bg[layer][62, 10]
    gt_norm = tg[layer].norm(); gb_norm = bg[layer].norm()
    rho_initial = .5
    global current_lambda
    current_lambda = float(rho_initial * gt_norm / (gb_norm + 1e-12))
    combined = gt + current_lambda * gb
    if combined >= 0:
        for rho in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
            candidate = float(rho * gt_norm / (gb_norm + 1e-12))
            if gt + candidate * gb < 0:
                current_lambda = candidate; break
        else:
            current_lambda = float(current_lambda)
    causal = {"block1.layer2_62_10": float(bg[4][62, 10]), "block1.layer2_62_15": float(bg[4][62, 15]),
              "block1.layer1_output10_l2": float(bg[3][10].norm()), "block1.layer1_output15_l2": float(bg[3][15].norm()),
              "head_output2_l2": float(bg[5][2].norm())}
    return {"variant": variant, "task_loss": float(task.detach()), "bool_loss": float(bl.detach()),
            "G_task_layer4": float(gt_norm), "G_bool_layer4": float(gb_norm),
            "rho_initial": rho_initial, "rho_final": float(current_lambda * (gb_norm + 1e-12) / (gt_norm + 1e-12)),
            "lambda": current_lambda, "repair_edge": {"raw": float(net.expectation_layers[4].raw_edge[62,10].detach()),
              "gate": float(torch.sigmoid(net.expectation_layers[4].raw_edge[62,10]).detach()),
              "task_gradient": float(gt), "bool_gradient": float(gb), "combined_gradient": float(gt + current_lambda * gb)},
            "causal_bool_gradients": causal, "layers": gradient_summary(net, tg, bg, task, bl, variant)}


def eval_record(net, x, y, step, variant, diagnostics=False):
    m = metrics(net, x, y); raw = net.expectation_layers[4].raw_edge[62, 10].detach()
    ste_out = boolean_ste_trace(net, x, variant)["head"].detach()
    boolean = net.to_discrete(.5)(x.bool()).float()
    focus = {str(i): {"ste_output": ste_out[i].tolist(), "boolean_output": boolean[i].tolist(), "target": y[i].tolist()}
             for i in (239, 255)}
    ev = {"step": step, "metrics": m, "L_bool": float(bool_loss(net, x, y, variant).detach()),
          "repair_edge": {"raw": float(raw), "gate": float(torch.sigmoid(raw)), "boolean_bit": bool(raw >= 0)},
          "focus_rows": focus, "mask_hashes": mask_state(net)["hashes"]}
    if diagnostics:
        task = task_loss(net, x, y); bl = bool_loss(net, x, y, variant)
        tg = grad_vector(net, task); bg = grad_vector(net, bl)
        ev["gradient_diagnostics"] = {"task_repair": float(tg[4][62,10]), "bool_repair": float(bg[4][62,10]),
                                       "combined_repair": float(tg[4][62,10] + current_lambda * bg[4][62,10]),
                                       "layers": gradient_summary(net, tg, bg, task, bl, variant)}
    return ev


def save_checkpoint(path, net, metadata, x, y, operator="p2"):
    path.parent.mkdir(parents=True, exist_ok=True); torch.save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, path)
    check = make_model(); check.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    check_eval = metrics(check, x, y)
    metadata = dict(metadata); metadata.update({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                "reload_verified": True, "reload_metrics": check_eval,
                                                "operator": operator, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    return metadata


def run_arm(name, variant, x, y, steps=3000):
    net, parent, parent_sha = load_parent(); initial = mask_state(net); previous = None; records=[]; states=[]; transitions=[]; first=None; crossed=None
    global current_lambda
    if variant is None:
        current_lambda = 0.0
        diag = initial_diagnostics(net, x, y, "STE_SIGMOID")
    else:
        diag = initial_diagnostics(net, x, y, variant)
    if variant is None:
        chosen_variant = "STE_SIGMOID"
    else:
        chosen_variant = variant
    def record(step):
        nonlocal previous, first, crossed
        ev=eval_record(net,x,y,step,chosen_variant,step in GRAD_STEPS); records.append(ev); states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        cur=mask_state(net); diff=mask_diff(previous,cur) if previous is not None else mask_diff(initial,cur)
        ev["mask_change_since_previous"]=diff; ev["mask_change_since_parent"]=mask_diff(initial,cur)
        if previous is not None and diff["edge_bits"]+diff["bias_bits"]: transitions.append({"step":step,**diff})
        old=previous; previous=cur
        if first is None and ev["metrics"]["boolean"]["exact_accuracy"]==1.0: first={"step":step,"mse":ev["metrics"]["continuous"]["mse"]}
        if crossed is None and old is not None:
            oldbit=bool((old["edges"][4][62,10]).item()); newbit=bool((cur["edges"][4][62,10]).item())
            if oldbit != newbit: crossed={"step":step,"old_bit":oldbit,"new_bit":newbit,"wrong_rows_before":records[-2]["metrics"]["boolean"]["wrong_rows"],"wrong_rows_after":ev["metrics"]["boolean"]["wrong_rows"]}
    record(0); opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=0.)
    for step in range(1,steps+1):
        lt=task_loss(net,x,y); loss=lt if variant is None else lt + current_lambda * bool_loss(net,x,y,chosen_variant)
        if not torch.isfinite(loss): record(step); break
        opt.zero_grad(); loss.backward(); opt.step()
        if step in EVAL_STEPS[1:] or step==steps: record(step)
    saved={}; by_mse=min(range(len(records)),key=lambda i: records[i]["metrics"]["continuous"]["mse"] if math.isfinite(records[i]["metrics"]["continuous"]["mse"]) else float('inf')); by_bool=max(range(len(records)),key=lambda i:(records[i]["metrics"]["boolean"]["exact_accuracy"],-records[i]["step"]))
    for kind,idx in (("best_continuous",by_mse),("best_boolean",by_bool),("final",len(states)-1)):
        path=I8CK/f"I8_{name}_{kind}.pt"; saved[kind]=save_checkpoint(path, type("State",(),{"state_dict":lambda self:states[idx]})(), {"step":records[idx]["step"],"arm":name,"variant":chosen_variant,"lambda":current_lambda,"rho":diag["rho_final"],"parent":str(parent),"parent_sha256":parent_sha}, x,y)
    return {"arm":name,"variant":chosen_variant,"lambda":current_lambda,"diagnostics":diag,"trajectory":records,"first_boolean_recovery":first,"repair_edge_crossing":crossed,"mask_transitions":transitions,"saved_checkpoints":saved,"parent":str(parent),"parent_sha256":parent_sha}


def main():
    global current_lambda
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=3000); p.add_argument("--diagnostics-only",action="store_true"); p.add_argument("--arm",choices=("all","control","ste"),default="all"); args=p.parse_args(); torch.set_num_threads(1)
    task=build_task("bitwise_xor_truth_table",{"bits":4}); x=task["X"].float(); y=task["Y"].float(); parent, path, sha=load_parent()
    eq={v:forward_equivalence(parent,x,v) for v in ("STE_SIGMOID","STE_CONST025")}
    diags={v:initial_diagnostics(parent,x,y,v) for v in ("STE_SIGMOID","STE_CONST025")}
    chosen=None
    for v in ("STE_SIGMOID","STE_CONST025"):
        if diags[v]["repair_edge"]["combined_gradient"]<0: chosen=v; break
    payload={"experiment":"I8-boolean-ste","git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"parent":str(path),"parent_sha256":sha,"forward_equivalence":eq,"initial_diagnostics":diags,"chosen_variant":chosen}
    if not args.diagnostics_only and chosen is not None:
        current_lambda=diags[chosen]["lambda"]
        selected = ("control", "ste") if args.arm == "all" else (args.arm,)
        payload["arms"] = []
        for arm in selected:
            item = run_arm(arm, None if arm == "control" else chosen, x, y, args.steps)
            payload["arms"].append(item)
        output_path = OUT / "i8_boolean_ste_results.json"
        if output_path.exists():
            old = json.loads(output_path.read_text())
            if old.get("experiment") == payload["experiment"]:
                by_arm = {item["arm"]: item for item in old.get("arms", [])}
                by_arm.update({item["arm"]: item for item in payload["arms"]})
                payload["arms"] = [by_arm[k] for k in sorted(by_arm)]
    (OUT/"i8_boolean_ste_results.json").write_text(json.dumps(payload,indent=2)+"\n"); print(OUT/"i8_boolean_ste_results.json")


if __name__=="__main__":
    import math
    main()
