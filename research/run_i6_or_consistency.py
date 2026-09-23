"""I6 forensic analysis and threshold-consistency continuation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i4_worstcase import mask_state, metrics  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
I4CK = ROOT / "research/operator_results/i4_worstcase_checkpoints"
I5CK = ROOT / "research/operator_results/i5_gate_regularization_checkpoints"
OUT = ROOT / "research/operator_results"
I6CK = OUT / "i6_or_consistency_checkpoints"
EVAL_STEPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
FOCUS_ROWS = (239, 255)


def make_model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2,
                                       or_operator="lehmer_p2")


def load_state(path: Path) -> tuple[SigmoidOrModernLogicGateNet, str]:
    net = make_model()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, hashlib.sha256(path.read_bytes()).hexdigest()


def lehmer_from_contributions(v: torch.Tensor) -> torch.Tensor:
    den = v.square().sum(dim=-1)
    safe = torch.where(den > 0, den, torch.ones_like(den))
    out = v.pow(3).sum(dim=-1) / safe
    return torch.where(den > 0, out, torch.zeros_like(out))


def threshold_consistency_penalty(f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Penalize only the relaxed-zero / hard-one threshold mismatch."""
    zero_side = (f.detach() < .5).to(f.dtype)
    return (zero_side * F.relu(m - .5).square()).mean()


def contribution_trace(layer, x: torch.Tensor) -> dict[str, torch.Tensor]:
    v = layer.contributions(x)
    f = lehmer_from_contributions(v)
    actual = layer(x)
    if not torch.allclose(f, actual, atol=1e-7, rtol=1e-6):
        raise RuntimeError("reconstructed Lehmer output does not match layer output")
    return {"v": v, "F": f, "M": v.max(dim=-1).values}


def network_trace(net, x: torch.Tensor) -> dict:
    disc = net.to_discrete(.5)
    h, hb, hh = x, x.bool(), x
    stages = [{"name": "input", "continuous": h, "boolean": hb, "hard": hh}]
    stem = contribution_trace(net.stem, h); hard_v = net.stem.contributions(hh)
    h = stem["F"]; hh = hard_v.max(dim=-1).values; hb = disc.stem(hb)
    stages.append({"name": "stem", "continuous": h, "boolean": hb, "hard": hh, "hard_M": hard_v.max(dim=-1).values, **stem})
    for bi, block in enumerate(net.blocks):
        h1t = contribution_trace(block.layer1, h); h1 = h1t["F"]
        hard_v1 = block.layer1.contributions(hh); hh1 = hard_v1.max(dim=-1).values
        hb1 = disc.blocks[bi].layer1(hb)
        stages.append({"name": f"block{bi}.layer1", "continuous": h1, "boolean": hb1, "hard": hh1,
                       "hard_M": hard_v1.max(dim=-1).values, **h1t})
        h2t = contribution_trace(block.layer2, h1); h2 = h2t["F"]
        hard_v2 = block.layer2.contributions(hh1); hh2 = hard_v2.max(dim=-1).values
        hb2 = disc.blocks[bi].layer2(hb1)
        stages.append({"name": f"block{bi}.layer2", "continuous": h2, "boolean": hb2, "hard": hh2,
                       "hard_M": hard_v2.max(dim=-1).values, **h2t})
        h = h + h2 - 2 * h * h2; hh = hh + hh2 - 2 * hh * hh2; hb = hb ^ hb2
        stages.append({"name": f"block{bi}.residual", "continuous": h, "boolean": hb, "hard": hh})
    ht = contribution_trace(net.head, h); hout = ht["F"]
    hard_vh = net.head.contributions(hh); hhard = hard_vh.max(dim=-1).values; hbout = disc.head(hb)
    stages.append({"name": "head", "continuous": hout, "boolean": hbout, "hard": hhard,
                   "hard_input": hh, "hard_v": hard_vh,
                   "hard_M": hard_vh.max(dim=-1).values, **ht})
    return {"stages": stages, "discrete": disc}


def violation_stats(trace: dict) -> list[dict]:
    out = []
    for stage in trace["stages"]:
        if "F" not in stage:
            continue
        f, m = stage["F"], stage["M"]
        mask = (f < .5) & (m >= .5)
        gaps = (m - f)[mask]
        out.append({"layer": stage["name"], "count": int(mask.sum()),
                    "fraction": float(mask.float().mean()),
                    "samples_with_violation": int(mask.any(dim=-1).sum()),
                    "mean_gap": float(gaps.detach().mean()) if gaps.numel() else 0.0,
                    "mean_M": float(m[mask].detach().mean()) if gaps.numel() else 0.0,
                    "max_M": float(m[mask].detach().max()) if gaps.numel() else 0.0,
                    "mean_F": float(f[mask].detach().mean()) if gaps.numel() else 0.0})
    return out


def row_focus(trace: dict, row: int, output_index: int = 2) -> dict:
    result = {}
    for stage in trace["stages"]:
        if "F" in stage:
            f, m = stage["F"][row, output_index], stage["M"][row, output_index]
            result[stage["name"]] = {"F": float(f), "M": float(m),
                                      "hard": float(stage["hard"][row, output_index]),
                                      "hard_M": float(stage.get("hard_M", stage["hard"])[row, output_index]),
                                      "threshold_violation": bool(f < .5 and m >= .5),
                                      "threshold_F": bool(f >= .5),
                                      "threshold_M": bool(m >= .5),
                                      "discrete": bool(stage["boolean"][row, output_index])}
        elif stage["name"] != "input":
            value = stage["continuous"][row, output_index]
            result[stage["name"]] = {"continuous": float(value),
                                      "threshold_continuous": bool(value >= .5),
                                      "discrete": bool(stage["boolean"][row, output_index]),
                                      "mismatch": bool((value >= .5) != stage["boolean"][row, output_index])}
    return result


def head_table(trace: dict, row: int, output_index: int = 2) -> list[dict]:
    stage = next(s for s in trace["stages"] if s["name"] == "head")
    layer = stage["layer"] if "layer" in stage else None
    # The explicit trace omits the module to keep tensors compact; use the
    # caller-provided head layer in enrich_head_table instead.
    return []


def enrich_head_table(net, trace: dict, row: int, output_index: int = 2) -> list[dict]:
    stage = next(s for s in trace["stages"] if s["name"] == "head")
    layer = net.head; x = stage["input"][row]
    # stage input is retained by the explicit trace below.
    b, g = layer.actual_bias()[output_index], layer.effective_gate()[output_index]
    a = layer.contributions(x.unsqueeze(0))[0, output_index] / g.clamp_min(1e-12)
    v = stage["v"][row, output_index]
    hv = stage["hard_v"][row, output_index]
    alpha = v.square() / v.square().sum().clamp_min(1e-12)
    disc = trace["discrete"]
    db = disc.head.bias[output_index]
    dw = disc.head.weight[output_index]
    rows = []
    for j in torch.argsort(v, descending=True).tolist():
        rows.append({"source": j, "x_continuous": float(x[j]),
                     "x_boolean": bool(stage["boolean_input"][row, j]),
                     "bias": float(b[j]), "bias_bit": bool(db[j]),
                     "raw_gate": float(layer.raw_edge[output_index, j]),
                     "gate": float(g[j]), "gate_bit": bool(dw[j]),
                     "literal": float(a[j]), "contribution": float(v[j]),
                     "hard_source": float(stage["hard_input"][row, j]),
                     "hard_contribution": float(hv[j]),
                     "lehmer_weight": float(alpha[j]),
                     "weighted_contribution": float(alpha[j] * v[j]),
                     "v_ge_half": bool(v[j] >= .5),
                     "continuous_literal_true": bool((a[j] >= .5)),
                     "exact_boolean_literal_true": bool(stage["boolean_input"][row, j] ^ db[j]),
                     "exact_boolean_edge_active": bool((stage["boolean_input"][row, j] ^ db[j]) & dw[j])})
    return rows


def error_count(disc, x, y) -> tuple[int, int, list[int]]:
    pred = disc(x.bool()).float(); wrong = (pred != y).any(dim=-1)
    return int(wrong.sum()), int((pred != y).sum()), torch.where(wrong)[0].tolist()


def local_repairs(net, x, y) -> dict:
    base = net.to_discrete(.5)
    base_count = error_count(base, x, y)
    candidates = []
    head = base.head
    for kind in ("edge", "bias"):
        for j in range(64):
            d = net.to_discrete(.5)
            tensor = d.head.weight if kind == "edge" else d.head.bias
            old = bool(tensor[2, j]); tensor[2, j] = not old
            count = error_count(d, x, y)
            candidates.append({"kind": kind, "index": j, "old": old, "new": not old,
                               "wrong_rows": count[0], "wrong_bits": count[1], "rows": count[2]})
    ranked = sorted(candidates, key=lambda z: (z["wrong_rows"], z["wrong_bits"]))
    top = ranked[:24]
    pairs = []
    for a, b in itertools.combinations(top, 2):
        d = net.to_discrete(.5)
        for c in (a, b):
            tensor = d.head.weight if c["kind"] == "edge" else d.head.bias
            tensor[2, c["index"]] = not bool(tensor[2, c["index"]])
        count = error_count(d, x, y)
        pairs.append({"first": a, "second": b, "wrong_rows": count[0],
                      "wrong_bits": count[1], "rows": count[2]})
    return {"baseline": {"wrong_rows": base_count[0], "wrong_bits": base_count[1], "rows": base_count[2]},
            "single_best": ranked[:16], "single_exact": [r for r in ranked if r["wrong_rows"] == 0],
            "pair_best": sorted(pairs, key=lambda z: (z["wrong_rows"], z["wrong_bits"]))[:16],
            "pair_exact": [r for r in pairs if r["wrong_rows"] == 0]}


def causal_path(net, trace: dict, row: int, start_index: int = 2) -> list[dict]:
    """Follow the dominant continuous contribution backward from head bit 2."""
    names = ["head", "block1.residual", "block1.layer2", "block1.layer1",
             "block0.residual", "block0.layer2", "block0.layer1", "stem"]
    current = start_index; path = []
    for name in names:
        stage = next(s for s in trace["stages"] if s["name"] == name)
        item = {"layer": name, "output_index": current,
                "continuous": float(stage["continuous"][row, current]),
                "hard": float(stage["hard"][row, current]),
                "boolean": bool(stage["boolean"][row, current])}
        if "v" in stage:
            values = stage["v"][row, current]
            top = torch.argsort(values, descending=True)[:3].tolist()
            item["top_predecessors"] = [{"index": int(j), "v": float(values[j])} for j in top]
            current = int(top[0])
        path.append(item)
    return path


def upstream_repairs(net, trace: dict, x, y, row: int = 239, start_index: int = 2) -> dict:
    """Test single flips on the small dominant causal subgraph."""
    layer_index = {"stem": 0, "block0.layer1": 1, "block0.layer2": 2,
                   "block1.layer1": 3, "block1.layer2": 4, "head": 5}
    path = causal_path(net, trace, row, start_index)
    candidates = []
    for item in path:
        if item["layer"] not in layer_index or "top_predecessors" not in item:
            continue
        li = layer_index[item["layer"]]
        for pred in item["top_predecessors"]:
            for kind in ("edge", "bias"):
                candidates.append({"layer": li, "layer_name": item["layer"], "kind": kind,
                                   "out_index": item["output_index"], "in_index": pred["index"]})
    results=[]
    for c in candidates[:32]:
        d=net.to_discrete(.5); layer=d.expectation_layers[c["layer"]]
        tensor=layer.weight if c["kind"]=="edge" else layer.bias
        old=bool(tensor[c["out_index"],c["in_index"]]); tensor[c["out_index"],c["in_index"]]=not old
        e=error_count(d,x,y); results.append({**c,"old":old,"new":not old,"wrong_rows":e[0],"wrong_bits":e[1],"rows":e[2]})
    return {"path":path,"tested":len(results),"best":sorted(results,key=lambda z:(z["wrong_rows"],z["wrong_bits"]))}


def forward_inputs(net, x):
    h = x; inputs = []
    inputs.append(h); h = net.stem(h)
    for block in net.blocks:
        inputs.append(h); h1 = block.layer1(h)
        inputs.append(h1); h2 = block.layer2(h1)
        h = h + h2 - 2*h*h2
    inputs.append(h)
    return inputs


def tc_regularizer(net, x, selected=(5,)):
    inputs = forward_inputs(net, x)
    vals = []
    for idx in selected:
        layer = net.expectation_layers[idx]
        v = layer.contributions(inputs[idx]); f = lehmer_from_contributions(v); m = v.max(dim=-1).values
        vals.append(threshold_consistency_penalty(f, m))
    return torch.stack(vals).mean()


def edge_norm(net, loss, selected):
    params = [net.expectation_layers[i].raw_edge for i in selected]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None)))), grads


def continuation(net, x, y, selected, lambda_tc, steps, arm):
    opt = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.)
    records = []; previous = None; first = None; states = []
    def record(step):
        nonlocal previous, first
        ev = metrics(net, x, y); tr = network_trace(net, x)
        ev.update({"step": step, "violations": violation_stats(tr),
                   "focus": {str(r): row_focus(tr, r) for r in FOCUS_ROWS},
                   "mask": mask_state(net)["hashes"]})
        records.append(ev); states.append({k: v.detach().cpu().clone() for k,v in net.state_dict().items()})
        if first is None and ev["boolean"]["exact_accuracy"] == 1.0:
            first = {"step": step, "mse": ev["continuous"]["mse"], "e_inf": ev["continuous"]["e_inf"]}
        previous = mask_state(net)
    record(0)
    for step in range(1, steps + 1):
        output = net(x); row = F.binary_cross_entropy(output.clamp(1e-7,1-1e-7),y,reduction="none").mean(dim=1)
        task = torch.topk(row, 8).values.mean(); reg = tc_regularizer(net, x, selected) if arm == "tc" else torch.zeros_like(task)
        loss = task + lambda_tc * reg; opt.zero_grad(); loss.backward(); opt.step()
        if step in EVAL_STEPS[1:] or step == steps: record(step)
    return {"arm": arm, "lambda": lambda_tc, "selected_layers": list(selected), "first_boolean_recovery": first,
            "trajectory": records, "_states": states}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=3000); p.add_argument("--forensic-only",action="store_true"); p.add_argument("--device",default="cpu"); args=p.parse_args()
    torch.set_num_threads(1); device=torch.device(args.device)
    task=build_task("bitwise_xor_truth_table",{"bits":4}); x=task["X"].float().to(device); y=task["Y"].float().to(device)
    parent=CK/"I2_I2-B_seed3_best_continuous_mse.pt"; net,parent_sha=load_state(parent)
    checkpoints={"parent":parent,"i4_top8_final":I4CK/"I4_seed3_top8_row_bce_final.pt",
                 "i4_top8_best":I4CK/"I4_seed3_top8_row_bce_best_mse.pt",
                 "i5_margin_best":I5CK/"I5_margin4_rho0.10_best_E_inf.pt",
                 "i5_margin_final":I5CK/"I5_margin4_rho0.10_final.pt",
                 "i5_polar_final":I5CK/"I5_polar_rho0.10_final.pt"}
    forensic=[]
    for name,path in checkpoints.items():
        m,sha=load_state(path); tr=network_trace(m,x)
        head_stage=next(s for s in tr["stages"] if s["name"]=="head")
        # retain head input/boolean input for the contribution table
        head_stage["input"] = forward_inputs(m,x)[-1]; head_stage["boolean_input"] = tr["stages"][-2]["boolean"]
        focus = {str(r):row_focus(tr,r) for r in FOCUS_ROWS}
        culprit = {}
        head_input = forward_inputs(m, x)[-1]
        head_boolean_input = tr["stages"][-2]["boolean"]
        for r in FOCUS_ROWS:
            table = enrich_head_table(m,tr,r)
            top_hard = max(table, key=lambda z: z["hard_contribution"])
            source = top_hard["source"]
            source_mismatch = bool((head_input[r, source] >= .5) != head_boolean_input[r, source])
            culprit[str(r)] = {"continuous_argmax": max(table, key=lambda z: z["contribution"]),
                               "hard_argmax": top_hard, "head_source_type": "TYPE_U" if source_mismatch else "TYPE_H",
                               "head_source_index_for_hard_argmax": source}
        forensic.append({"name":name,"path":str(path),"sha256":sha,"metrics":metrics(m,x,y),
                         "violations":violation_stats(tr),"focus":{str(r):row_focus(tr,r) for r in FOCUS_ROWS},
                         "head_rows":{"239":enrich_head_table(m,tr,239),"255":enrich_head_table(m,tr,255)},
                         "culprit":culprit,
                         "causal_paths":{"239":causal_path(m,tr,239),"255":causal_path(m,tr,255)},
                         "repairs":local_repairs(m,x,y),
                         "upstream_repairs":upstream_repairs(m,tr,x,y),
                         "mask_hashes":mask_state(m)["hashes"]})
    payload={"experiment":"I6-or-consistency","git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"parent_sha256":parent_sha,"forensic":forensic}
    if not args.forensic_only:
        # The forensic phase identifies the second layer of residual block 1
        # as the first upstream source of the wrong head input.  Apply the
        # consistency penalty there only; the head itself has no F/M violation
        # on the two target rows.
        selected=(4,)
        base, _ = load_state(parent); out=base(x); row=F.binary_cross_entropy(out.clamp(1e-7,1-1e-7),y,reduction="none").mean(dim=1); task_loss=torch.topk(row,8).values.mean(); reg=tc_regularizer(base,x,selected); gt,_=edge_norm(base,task_loss,selected); gr,_=edge_norm(base,reg,selected); lam=.10*gt/(gr+1e-12)
        payload["calibration"]={"task_edge_norm":gt,"tc_edge_norm":gr,"lambda":lam,"initial_ratio":lam*gr/(gt+1e-12),"selected_layers":list(selected)}
        continuations=[]
        for arm, arm_lambda in (("control", 0.0), ("tc", lam)):
            result=continuation(load_state(parent)[0],x,y,selected,arm_lambda,args.steps,arm)
            states=result.pop("_states")
            choices={"final":len(states)-1,
                     "best_E_inf":min(range(len(states)), key=lambda i: result["trajectory"][i]["continuous"]["e_inf"]),
                     "best_boolean":max(range(len(states)), key=lambda i: (result["trajectory"][i]["boolean"]["exact_accuracy"],-result["trajectory"][i]["step"]))}
            saved={}
            for kind,index in choices.items():
                path=I6CK/f"I6_{arm}_{kind}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(states[index],path)
                check,_=load_state(path)
                check_metrics=metrics(check,x,y)
                assert abs(check_metrics["continuous"]["mse"]-result["trajectory"][index]["continuous"]["mse"]) < 1e-9
                saved[kind]={"path":str(path),"step":result["trajectory"][index]["step"],"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            result["saved_checkpoints"]=saved; continuations.append(result)
        payload["continuations"]=continuations
    OUT.joinpath("i6_or_consistency_results.json").write_text(json.dumps(payload,indent=2,default=lambda z: z.tolist() if torch.is_tensor(z) else str(z))+"\n")
    print(OUT/"i6_or_consistency_results.json")


if __name__ == "__main__": main()
