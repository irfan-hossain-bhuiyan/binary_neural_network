"""I7 Lehmer-to-hardmax continuation from the verified I2-B seed-3 parent."""
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
from research.or_surrogates import HardMax, LehmerMean  # noqa: E402
from research.run_i4_worstcase import metrics, mask_state  # noqa: E402
from research.run_i3_endpoint import common_metrics  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
I7CK = OUT / "i7_operator_homotopy_checkpoints"
EVAL_STEPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2300, 2400, 2500, 2750, 3000]
DIAG_STEPS = {0, 10, 100, 500, 1000, 2000}


def make_model():
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2")


def load_parent():
    path = CK / "I2_I2-B_seed3_best_continuous_mse.pt"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = make_model(); net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, path, sha


def set_operator(net, operator: str, p: float | None = None) -> dict:
    if operator == "hardmax":
        for layer in net.expectation_layers:
            layer.or_operator = HardMax()
        return {"operator_type": "hardmax", "p": None}
    if operator != "lehmer" or p is None:
        raise ValueError((operator, p))
    for layer in net.expectation_layers:
        layer.or_operator = LehmerMean(float(p))
    return {"operator_type": "lehmer", "p": float(p)}


def current_spec(net) -> dict:
    op = net.expectation_layers[0].or_operator
    return {"operator_type": "hardmax", "p": None} if isinstance(op, HardMax) else {"operator_type": "lehmer", "p": float(op.p)}


def active_p(net) -> float | None:
    op = net.expectation_layers[0].or_operator
    return None if isinstance(op, HardMax) else float(op.p)


def task_loss(net, x, y):
    output = net(x)
    if not torch.isfinite(output).all():
        return output.sum() * 0 + torch.tensor(float("nan"), device=output.device)
    row = F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), y, reduction="none").mean(dim=1)
    return torch.topk(row, 8).values.mean()


def _safe_metrics(output, target):
    return {"finite": False, "mse": float("nan"), "bce": float("nan"), "mae": float("nan"),
            "e_inf": float("nan"), "p95_abs_error": float("nan"), "p99_abs_error": float("nan"),
            "bit_accuracy": float("nan"), "exact_accuracy": float("nan"),
            "wrong_rows": None, "wrong_bits": None}


def safe_metrics(net, x, y):
    with torch.no_grad():
        output = net(x)
        hard = net.forward_hard(x)
        boolean = net.to_discrete(.5)(x.bool()).float()
        return {"continuous": common_metrics(output, y) if torch.isfinite(output).all() else _safe_metrics(output, y),
                "hard": common_metrics(hard, y) if torch.isfinite(hard).all() else _safe_metrics(hard, y),
                "boolean": common_metrics(boolean, y) if torch.isfinite(boolean).all() else _safe_metrics(boolean, y)}


@torch.no_grad()
def path_values(net, x):
    h = x
    h = net.stem(h)
    for bi, block in enumerate(net.blocks):
        h1 = block.layer1(h); h2 = block.layer2(h1)
        h = h + h2 - 2 * h * h2
        if bi == 1:
            residual62 = h[:, 62]
    out = net.head(h)
    return {"residual_source62": residual62, "head_bit2": out[:, 2]}


@torch.no_grad()
def causal_trace(net, x, rows=(239, 255)):
    """Small p-sweep trace for the I6 causal source and its predecessors."""
    selected = {}
    h = net.stem(x)
    selected["stem"] = h
    for bi, block in enumerate(net.blocks):
        h1 = block.layer1(h); selected[f"block{bi}.layer1"] = h1
        h2 = block.layer2(h1); selected[f"block{bi}.layer2"] = h2
        h = h + h2 - 2 * h * h2
        selected[f"block{bi}.residual"] = h
    selected["head"] = net.head(h)
    return {str(row): {name: {"all": [float(v) for v in tensor[row].flatten()],
                              "indices": {"source10": float(tensor[row, 10]) if tensor.shape[-1] > 10 else None,
                                          "source15": float(tensor[row, 15]) if tensor.shape[-1] > 15 else None,
                                          "source62": float(tensor[row, 62]) if tensor.shape[-1] > 62 else None,
                                          "bit2": float(tensor[row, 2]) if tensor.shape[-1] > 2 else None}}
                             for name, tensor in selected.items()} for row in rows}


@torch.no_grad()
def concentration(net, x) -> list[dict]:
    # Capture each layer's actual input under the currently active operator.
    inputs = [x]
    h = net.stem(x)
    inputs.append(h)
    for block in net.blocks:
        h1 = block.layer1(h); inputs.append(h1)
        h2 = block.layer2(h1); h = h + h2 - 2 * h * h2
        inputs.append(h)
    p = active_p(net)
    rows = []
    for i, (layer, inp) in enumerate(zip(net.expectation_layers, inputs)):
        v = layer.contributions(inp)
        if p is None:
            alpha = torch.nn.functional.one_hot(v.argmax(dim=-1), v.shape[-1]).to(v.dtype)
            valid = torch.ones_like(v[..., 0], dtype=torch.bool)
        else:
            w = v.pow(p); den = w.sum(dim=-1, keepdim=True)
            valid = den.squeeze(-1) > 1e-12
            alpha = torch.where(valid.unsqueeze(-1), w / den.clamp_min(1e-12), torch.zeros_like(w))
        top = v.topk(2, dim=-1).values
        neff = torch.where(valid, 1 / alpha.square().sum(dim=-1).clamp_min(1e-12), torch.zeros_like(valid, dtype=v.dtype))
        rows.append({"layer": i, "active_p": p, "valid_fraction": float(valid.float().mean()),
                     "max_alpha_mean": float(alpha.max(dim=-1).values[valid].mean()) if valid.any() else 0.0,
                     "N_eff_mean": float(neff[valid].mean()) if valid.any() else 0.0,
                     "top_gap_mean": float((top[..., 0] - top[..., 1]).mean()),
                     "top_gap_p05": float(torch.quantile((top[..., 0] - top[..., 1]).flatten(), .05)),
                     "top_gap_p50": float(torch.quantile((top[..., 0] - top[..., 1]).flatten(), .50))})
    return rows


@torch.no_grad()
def semantic_eval(net, x, y) -> dict:
    old = current_spec(net); result = {}
    for label, spec in (("p2", ("lehmer", 2.0)), ("p4", ("lehmer", 4.0)), ("p8", ("lehmer", 8.0)), ("hardmax", ("hardmax", None))):
        set_operator(net, spec[0], spec[1]); result[label] = safe_metrics(net, x, y)["continuous"]
    set_operator(net, old["operator_type"], old["p"])
    result["boolean"] = safe_metrics(net, x, y)["boolean"]
    return result


def gradient_diagnostics(net, x, y):
    loss = task_loss(net, x, y); params = [l.raw_edge for l in net.expectation_layers]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    layer_rows=[]
    for i,g in enumerate(grads):
        g = g if g is not None else torch.zeros_like(params[i])
        layer_rows.append({"layer": i, "mean_abs": float(g.abs().mean()), "median_abs": float(g.abs().median()), "fraction_lt_1e-8": float((g.abs()<1e-8).float().mean())})
    target_grad = grads[4][62, 10] if grads[4] is not None else torch.zeros((), device=loss.device)
    return {"repair_edge_gradient": float(target_grad.detach()), "layers": layer_rows}


def evaluate_point(net, x, y, step):
    spec=current_spec(net); base=safe_metrics(net,x,y); sem=semantic_eval(net,x,y); paths=path_values(net,x)
    return {"step":step, **spec, "current":base, "semantic_eval":sem,
            "focus": {str(row): {"residual_source62": float(paths["residual_source62"][row]), "head_bit2": float(paths["head_bit2"][row]),
                                  "residual_source62_threshold": bool(paths["residual_source62"][row]>=.5),
                                  "boolean_output": bool(sem["boolean"]["exact_accuracy"] >= 0 and net.to_discrete(.5)(x.bool())[row,2])} for row in (239,255)},
            "repair_edge_raw": float(net.expectation_layers[4].raw_edge[62,10].detach()),
            "repair_edge_gate": float(torch.sigmoid(net.expectation_layers[4].raw_edge[62,10]).detach()),
            "repair_edge_bit": bool(torch.sigmoid(net.expectation_layers[4].raw_edge[62,10]).detach()>=.5),
            "mask_hashes": mask_state(net)["hashes"],
            "concentration": concentration(net,x),
            "gradient_diagnostics": gradient_diagnostics(net,x,y) if step in DIAG_STEPS else None}


def schedule(step):
    if step < 750: return ("lehmer", 2 + 2*step/749)
    if step < 1500: return ("lehmer", 4 + 4*(step-750)/749)
    if step < 2250: return ("lehmer", 8.0)
    return ("hardmax", None)


def run_arm(name, steps, x, y):
    net,parent,parent_sha=load_parent(); records=[]; states=[]; first=None; previous=None; transitions=[]
    if name=="p2": fixed=("lehmer",2.0)
    elif name=="p4": fixed=("lehmer",4.0)
    elif name=="hardmax": fixed=("hardmax",None)
    else: fixed=None
    opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=0.)
    def record(step):
        nonlocal previous, first
        spec=schedule(step) if fixed is None else fixed; set_operator(net,spec[0],spec[1])
        ev=evaluate_point(net,x,y,step); records.append(ev); states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        if first is None and ev["current"]["boolean"]["exact_accuracy"]==1.0: first={"step":step,"mse":ev["current"]["continuous"]["mse"]}
        current=mask_state(net)
        if previous is not None:
            changed=sum(int((a!=b).sum()) for a,b in zip(previous["edges"],current["edges"]))+sum(int((a!=b).sum()) for a,b in zip(previous["biases"],current["biases"]))
            if changed: transitions.append({"step":step,"changed_bits":changed})
        previous=current
    record(0)
    for step in range(1,steps+1):
        spec=schedule(step) if fixed is None else fixed; set_operator(net,spec[0],spec[1])
        loss=task_loss(net,x,y)
        if not torch.isfinite(loss):
            record(step)
            break
        opt.zero_grad(); loss.backward(); opt.step()
        if step in EVAL_STEPS[1:] or step==steps: record(step)
    saved={}
    for kind,index in (("final",len(states)-1),("best_boolean",max(range(len(states)),key=lambda i:(records[i]["current"]["boolean"]["exact_accuracy"],-records[i]["step"])))):
        spec=records[index]; path=I7CK/f"I7_{name}_{kind}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(states[index],path)
        saved[kind]={"path":str(path),"step":spec["step"],"operator_type":spec["operator_type"],"p":spec["p"],"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"arm":name,"parent":str(parent),"parent_sha256":parent_sha,"first_boolean_recovery":first,"trajectory":records,"mask_transitions":transitions,"saved_checkpoints":saved}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=3000); p.add_argument("--arm",choices=("all","p2","p4","hardmax","homotopy"),default="all"); p.add_argument("--sweep-only",action="store_true"); args=p.parse_args(); torch.set_num_threads(1)
    task=build_task("bitwise_xor_truth_table",{"bits":4}); x=task["X"].float(); y=task["Y"].float(); parent,parent_path,parent_sha=load_parent()
    sweep=[]
    for label,spec in [("p2",("lehmer",2.0)),("p2.5",("lehmer",2.5)),("p3",("lehmer",3.0)),("p4",("lehmer",4.0)),("p6",("lehmer",6.0)),("p8",("lehmer",8.0)),("hardmax",("hardmax",None))]:
        set_operator(parent,spec[0],spec[1]); paths = path_values(parent,x)
        sweep.append({"operator":label,"spec":current_spec(parent),"metrics":safe_metrics(parent,x,y),
                      "paths": {str(i): {"residual_source62": float(paths["residual_source62"][i]),
                                         "head_bit2": float(paths["head_bit2"][i])} for i in (239, 255)},
                      "causal_trace": causal_trace(parent, x)})
    payload={"experiment":"I7-operator-homotopy","git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"parent":str(parent_path),"parent_sha256":parent_sha,"p_sweep":sweep}
    if not args.sweep_only:
        arms=("p2","p4","hardmax","homotopy") if args.arm=="all" else (args.arm,)
        payload["continuations"]=[run_arm(a,args.steps,x,y) for a in arms]
    output_path = OUT / "i7_operator_homotopy_results.json"
    if output_path.exists() and not args.sweep_only:
        existing = json.loads(output_path.read_text())
        if existing.get("experiment") == payload["experiment"]:
            old = {item["arm"]: item for item in existing.get("continuations", [])}
            for item in payload.get("continuations", []):
                old[item["arm"]] = item
            payload["continuations"] = [old[k] for k in sorted(old)]
            payload["p_sweep"] = existing.get("p_sweep", payload["p_sweep"])
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(output_path)


if __name__=="__main__": main()
