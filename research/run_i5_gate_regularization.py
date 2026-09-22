"""I5 gate-only regularization from the verified I2-B seed-3 checkpoint."""
from __future__ import annotations

import copy
import hashlib
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
from research.run_i4_worstcase import mask_diff, mask_state, metrics, raw_gate_stats  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
I5CK = OUT / "i5_gate_regularization_checkpoints"
EVAL_STEPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]


def make_model() -> SigmoidOrModernLogicGateNet:
    return SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2")


def load_parent():
    path = CK / "I2_I2-B_seed3_best_continuous_mse.pt"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = make_model(); net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, path, sha


def regularizer(net, kind: str) -> torch.Tensor:
    values = []
    for layer in net.expectation_layers:
        if kind == "polar":
            g = layer.effective_gate()
            values.append((4 * g * (1 - g)).mean())
        elif kind == "margin4":
            distance = torch.relu(torch.tensor(4.0, dtype=layer.raw_edge.dtype) - layer.raw_edge.abs()) / 4.0
            values.append(distance.square().mean())
        else:
            raise ValueError(kind)
    return torch.stack(values).mean()


def edge_grad_norm(net, loss: torch.Tensor) -> tuple[float, list[torch.Tensor]]:
    params = [layer.raw_edge for layer in net.expectation_layers]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    actual = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    return float(torch.sqrt(sum((g.square().sum() for g in actual))),), actual


def calibration(net, x, y, kind: str, rho: float) -> dict:
    output = net(x)
    row_loss = F.binary_cross_entropy(output.clamp(1e-7, 1 - 1e-7), y, reduction="none").mean(dim=1)
    task = torch.topk(row_loss, 8).values.mean()
    task_norm, task_grads = edge_grad_norm(net, task)
    reg = regularizer(net, kind)
    reg_norm, reg_grads = edge_grad_norm(net, reg)
    lam = rho * task_norm / (reg_norm + 1e-12)
    return {"regularizer": kind, "rho": rho, "task_edge_grad_norm": task_norm,
            "regularizer_edge_grad_norm": reg_norm, "lambda": lam,
            "initial_ratio": float(lam * reg_norm / (task_norm + 1e-12))}


def gate_metrics(net, kind: str) -> dict:
    layers = []
    for i, layer in enumerate(net.expectation_layers):
        r = layer.raw_edge.detach(); g = torch.sigmoid(r); d = r.abs()
        closest = []
        for flat_index in torch.argsort(d.flatten())[:10].tolist():
            index = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_index), d.shape))
            closest.append({"index": list(index), "value": float(r[index]), "gate": float(g[index])})
        layers.append({"layer": i, "D_g": float(torch.minimum(g, 1-g).mean()),
                       "polar": float((4*g*(1-g)).mean()), "margin4": float(torch.relu(4-d).square().mean()/16),
                       "g_lt_01": float((g < .01).float().mean()), "g_gt_99": float((g > .99).float().mean()),
                       "g_lt_001": float((g < .001).float().mean()), "g_gt_999": float((g > .999).float().mean()),
                       "g_middle": float(((g > .45)&(g < .55)).float().mean()),
                       "raw_abs_q50": float(torch.quantile(d,.50)), "raw_abs_q75": float(torch.quantile(d,.75)),
                       "raw_abs_q90": float(torch.quantile(d,.90)), "raw_abs_q95": float(torch.quantile(d,.95)),
                       "raw_abs_q99": float(torch.quantile(d,.99)),
                       **{f"raw_abs_gt_{n}": float((d > n).float().mean()) for n in (2,4,6,8)},
                       **{f"raw_abs_lt_{n}": float((d < n).float().mean()) for n in (.1,.25,.5,1.)},
                       "closest_raw": closest})
    return {"layers": layers, "R_polar": float(regularizer(net, "polar").detach()), "R_margin4": float(regularizer(net, "margin4").detach())}


def gradient_diagnostic(net, x, y, kind: str, lam: float) -> dict:
    output = net(x); row_loss = F.binary_cross_entropy(output.clamp(1e-7,1-1e-7), y, reduction="none").mean(dim=1)
    task = torch.topk(row_loss, 8).values.mean(); reg = regularizer(net, kind)
    task_norm, tg = edge_grad_norm(net, task); reg_norm, rg = edge_grad_norm(net, reg)
    cosine = []
    for a,b in zip(tg,rg):
        den = a.norm()*b.norm(); cosine.append(float(torch.dot(a.flatten(),b.flatten())/den) if den > 0 else None)
    return {"task_norm": task_norm, "regularizer_norm": reg_norm, "lambda_regularizer_norm": lam*reg_norm,
            "ratio": lam*reg_norm/(task_norm+1e-12), "cosine_by_layer": cosine}


def run_arm(kind: str, rho: float, steps: int, x, y) -> dict:
    net, parent, parent_sha = load_parent(); initial = metrics(net,x,y)
    with torch.no_grad():
        parent_boolean = net.to_discrete(.5)(x.bool()).float()
        wrong = [i for i in range(len(x)) if not torch.equal(parent_boolean[i], y[i])]
    if wrong != [239,255]: raise RuntimeError(f"unexpected parent wrong rows: {wrong}")
    cal = calibration(net,x,y,kind,rho); lam = cal["lambda"]
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.)
    records=[]; states=[]; first=None; previous=None; transitions=[]; initial_mask=mask_state(net)
    def record(step):
        nonlocal previous, first
        ev=metrics(net,x,y); current=mask_state(net)
        ev.update({"step":step,"raw_gate_stats":raw_gate_stats(net),"gate_metrics":gate_metrics(net,kind),
                   "mask_hashes":current["hashes"],"mask_change_since_previous":mask_diff(previous,current),
                   "mask_change_since_parent":mask_diff(initial_mask,current),
                   "gradient_diagnostic": gradient_diagnostic(net,x,y,kind,lam) if step in (0,100,500) else None,
                   "top8_worst_rows": [int(i) for i in torch.topk(F.binary_cross_entropy(net(x).clamp(1e-7,1-1e-7),y,reduction='none').mean(dim=1),8).indices.tolist()]})
        records.append(ev); states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        if first is None and ev["boolean"]["exact_accuracy"] == 1.0: first={"step":step,"mse":ev["continuous"]["mse"],"e_inf":ev["continuous"]["e_inf"],"rows":ev["rows"]}
        diff=ev["mask_change_since_previous"]
        if previous is not None and diff["edge_bits"]+diff["bias_bits"]: transitions.append({"step":step,**diff})
        previous=current
    record(0)
    for step in range(1,steps+1):
        out=net(x); row=F.binary_cross_entropy(out.clamp(1e-7,1-1e-7),y,reduction='none').mean(dim=1); task=torch.topk(row,8).values.mean(); reg=regularizer(net,kind)
        loss=task+lam*reg; optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in EVAL_STEPS[1:] or step==steps: record(step)
    saved={}
    choices={"best_E_inf":min(range(len(records)),key=lambda i:records[i]["continuous"]["e_inf"]),"best_task":min(range(len(records)),key=lambda i:records[i]["top8_mean_row_bce"]),"best_boolean":max(range(len(records)),key=lambda i:(records[i]["boolean"]["exact_accuracy"],-records[i]["step"])),"final":len(records)-1}
    for name,idx in choices.items():
        path=I5CK/f"I5_{kind}_rho{rho:.2f}_{name}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(states[idx],path)
        reloaded=make_model(); reloaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); check=metrics(reloaded,x,y)
        assert abs(check["continuous"]["mse"]-records[idx]["continuous"]["mse"])<1e-9
        saved[name]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"step":records[idx]["step"]}
    return {"arm":f"{kind}_rho{rho:.2f}","regularizer":kind,"rho":rho,"lambda":lam,"calibration":cal,
            "parent":str(parent),"parent_sha256":parent_sha,"initial":initial,"first_boolean_recovery":first,
            "mask_transitions":transitions,"trajectory":records,"saved_checkpoints":saved}


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=3000); p.add_argument('--device',default='cpu'); p.add_argument('--arm', choices=['all','polar_rho0.10','polar_rho0.50','margin4_rho0.10','margin4_rho0.50'], default='all'); p.add_argument('--output', type=Path, default=OUT/'i5_gate_regularization_results.json'); args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    task=build_task('bitwise_xor_truth_table',{'bits':4}); x=task['X'].float().to(device); y=task['Y'].float().to(device)
    requested = [('polar',.10),('polar',.50),('margin4',.10),('margin4',.50)]
    if args.arm != 'all':
        requested = [(k,float(r)) for k,r in requested if f'{k}_rho{r:.2f}' == args.arm]
    results=[run_arm(k,r,args.steps,x,y) for k,r in requested]
    sha=os.environ.get('RESEARCH_GIT_SHA') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    payload={'experiment':'I5-gate-regularization','git_sha':sha,'device':str(device),'steps':args.steps,'loss':'top8-row BCE + lambda R','results':results}
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(args.output)

if __name__=='__main__': main()
