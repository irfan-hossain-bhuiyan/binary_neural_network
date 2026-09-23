"""I9 replication of exact-Boolean STE with global, causal-agnostic calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.run_i8_boolean_ste import (boolean_ste_trace, bool_loss, exact_discrete_trace,
                                         forward_equivalence, make_model, mask_diff, mask_state,
                                         metrics, ste_gate, task_loss)  # noqa: E402

CK = ROOT / "research/operator_results/initialization_i2_checkpoints"
OUT = ROOT / "research/operator_results"
CKOUT = OUT / "i9_ste_replication_checkpoints"
EVAL_STEPS = [0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
GRAD_STEPS = {0, 10, 100, 500}
VARIANTS = ("STE_SIGMOID", "STE_CONST025")


def parent_path(seed: int) -> Path:
    return CK / f"I2_I2-B_seed{seed}_best_continuous_mse.pt"


def load_parent(seed: int):
    path = parent_path(seed); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    net = make_model(); net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net, path, sha


def grads_for(net, loss):
    params = [layer.raw_edge for layer in net.expectation_layers]
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(gs, params)]


def flatten(gs): return torch.cat([g.reshape(-1) for g in gs])


def cosine(a, b):
    den = a.norm() * b.norm()
    return float(torch.dot(a, b) / den) if den > 0 else None


def generic_diagnostics(net, x, y):
    lt = task_loss(net, x, y); gt = grads_for(net, lt); ft = flatten(gt); Gt = float(ft.norm())
    details = {"task_global_norm": Gt, "variants": {}, "task_layers": [float(g.norm()) for g in gt]}
    chosen = None
    for variant in VARIANTS:
        lb = bool_loss(net, x, y, variant); gb = grads_for(net, lb); fb = flatten(gb); Gb = float(fb.norm())
        lam = 0.5 * Gt / (Gb + 1e-12) if Gb > 1e-12 else 0.0
        rows=[]
        for i,(a,b) in enumerate(zip(gt,gb)):
            rows.append({"layer":i,"task_norm":float(a.norm()),"bool_norm":float(b.norm()),
                         "cosine":cosine(a.flatten(),b.flatten()),"coverage":float((b.abs()>=1e-8).float().mean()),
                         "fraction_lt_1e-8":float((b.abs()<1e-8).float().mean())})
        details["variants"][variant]={"bool_loss":float(lb.detach()),"bool_global_norm":Gb,"lambda":lam,
          "lambda_bool_norm":lam*Gb,"ratio":lam*Gb/Gt if Gt>0 else None,
          "repair_edge_diagnostic_only":float(gb[4][62,10]),"layers":rows,
          "global_cosine":cosine(ft,fb),"nonzero_parameter_count":int((fb.abs()>=1e-8).sum())}
        if chosen is None and Gb > 1e-12: chosen=variant
    details["chosen_variant"] = chosen
    details["forward_equivalence"] = {v:forward_equivalence(net,x,v) for v in VARIANTS}
    details["redundant_active"] = redundant_active(net, x, y, chosen or "STE_SIGMOID")
    return details


def redundant_active(net, x, y, variant):
    """Exact Boolean active-edge counts, separated from gradient calibration."""
    disc = net.to_discrete(.5); hb=x.bool(); rows=[]; i=0
    def one(layer, inputs, index):
        w,b=disc.expectation_layers[index].weight,disc.expectation_layers[index].bias
        literal = inputs.unsqueeze(1) ^ b.unsqueeze(0)
        active = literal & w.unsqueeze(0); k=active.sum(-1)
        rows.append({"layer":index,"k0_fraction":float((k==0).float().mean()),"k1_fraction":float((k==1).float().mean()),
                     "k_ge2_fraction":float((k>=2).float().mean()),"max_k":int(k.max())})
        return layer(inputs)
    hb=one(disc.stem,hb,0); i=1
    for block in disc.blocks:
        skip=hb
        hb1=one(block.layer1,hb,i); i+=1
        branch=one(block.layer2,hb1,i); i+=1
        hb=skip ^ branch
    one(disc.head,hb,i)
    # Boolean STE global gradient zero is reported separately; this table is the
    # structural redundant-cause diagnostic requested for the exact forward.
    return {"layers":rows,"variant":variant}


def threshold_robustness(net, x, y):
    vals=[]
    for t in (.30,.35,.40,.45,.50,.55,.60,.65,.70):
        pred=net.to_discrete(t)(x.bool()).float(); vals.append((t, float(((pred>=.5)==(y>=.5)).all(dim=-1).float().mean())))
    exact=[t for t,a in vals if a==1.0]; runs=[]
    if exact:
        cur=[exact[0]]
        for t in exact[1:]:
            if abs(t-cur[-1]-.05)<1e-8: cur.append(t)
            else: runs.append(cur); cur=[t]
        runs.append(cur)
    containing=[r for r in runs if .50 in r]
    return {"thresholds":vals,"exact_runs":runs,"largest_contiguous_containing_half":([containing[0][0],containing[0][-1]] if containing else None)}


def safe_finite(v): return isinstance(v,(int,float)) and math.isfinite(v)


def eval_record(net,x,y,step,variant,lam,diagnostic=False):
    m=metrics(net,x,y); raw=net.expectation_layers[4].raw_edge[62,10].detach(); bool_pred=net.to_discrete(.5)(x.bool()).float()
    rec={"step":step,"metrics":m,"L_bool":float(bool_loss(net,x,y,variant).detach()),
         "repair_edge":{"raw":float(raw),"gate":float(torch.sigmoid(raw)),"boolean_bit":bool(raw>=0)},
         "mask_hashes":mask_state(net)["hashes"],"threshold_robustness":threshold_robustness(net,x,y),
         "focus_rows":{str(i):{"boolean_output":bool_pred[i].tolist(),"target":y[i].tolist()} for i in (239,255)}}
    if diagnostic:
        lt=task_loss(net,x,y); lb=bool_loss(net,x,y,variant); gt=grads_for(net,lt); gb=grads_for(net,lb)
        rec["gradient_diagnostics"]={"task_global_norm":float(flatten(gt).norm()),"bool_global_norm":float(flatten(gb).norm()),
          "task_repair":float(gt[4][62,10]),"bool_repair":float(gb[4][62,10]),"combined_repair":float(gt[4][62,10]+lam*gb[4][62,10]),
          "per_layer_bool_norm":[float(g.norm()) for g in gb],"global_cosine":cosine(flatten(gt),flatten(gb)),
          "coverage":[float((g.abs()>=1e-8).float().mean()) for g in gb]}
    return rec


def save_ckpt(path, state, meta, x, y):
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(state,path)
    chk=make_model(); chk.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); ev=metrics(chk,x,y)
    meta=dict(meta); meta.update({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"reload_verified":True,"reload_metrics":ev,"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()})
    return meta


def run_arm(seed, arm, variant, x, y, steps):
    net,parent,sha=load_parent(seed); parent_m=metrics(net,x,y); initial=mask_state(net); previous=None; records=[]; states=[]; transitions=[]; first=None; exact_history=[]
    diag=generic_diagnostics(net,x,y); lam=0.0 if arm=='control' else float(diag['variants'][variant]['lambda'])
    active = arm=='ste'; used_variant=variant if variant else 'STE_SIGMOID'
    def record(step):
        nonlocal previous,first
        ev=eval_record(net,x,y,step,used_variant,lam,step in GRAD_STEPS); records.append(ev); states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        cur=mask_state(net); diff=mask_diff(previous,cur) if previous is not None else mask_diff(initial,cur); ev['mask_change_since_previous']=diff; ev['mask_change_since_parent']=mask_diff(initial,cur)
        if previous is not None and diff['edge_bits']+diff['bias_bits']: transitions.append({'step':step,**diff})
        previous=cur; exact=ev['metrics']['boolean']['exact_accuracy']==1.0; exact_history.append(exact)
        if first is None and exact: first={'step':step,'mse':ev['metrics']['continuous']['mse'],'e_inf':ev['metrics']['continuous']['e_inf']}
    record(0); opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=0.)
    for step in range(1,steps+1):
        lt=task_loss(net,x,y); loss=lt if not active else lt+lam*bool_loss(net,x,y,used_variant)
        if not torch.isfinite(loss): record(step); break
        opt.zero_grad(); loss.backward(); opt.step()
        if step in EVAL_STEPS[1:] or step==steps: record(step)
    regressions=0
    if first:
        seen=False
        for q in exact_history:
            if q and seen is False: seen=True
            elif seen and not q: regressions+=1
    longest=0; cur=0
    for q in exact_history:
        cur=cur+1 if q else 0; longest=max(longest,cur)
    def finite_mse(i):
        v=records[i]['metrics']['continuous']['mse']; return v if safe_finite(v) else float('inf')
    best_mse=min(range(len(records)),key=finite_mse); best_bool=max(range(len(records)),key=lambda i:(records[i]['metrics']['boolean']['exact_accuracy'],-records[i]['step']))
    saved={}
    base_meta={"seed":seed,"arm":arm,"active_losses":["top8_row_bce","boolean_ste_mse"] if active else ["top8_row_bce"],"ste_auxiliary_active":active,"ste_variant":variant if active else None,"lambda":lam,"rho":.5 if active else 0.0,"parent_checkpoint":str(parent),"parent_sha256":sha,"continuous_operator":"lehmer_p2","steps":steps}
    for kind,idx in [('best_continuous',best_mse),('best_boolean',best_bool),('final',len(states)-1)]:
        saved[kind]=save_ckpt(CKOUT/f"I9_seed{seed}_{arm}_{kind}.pt",states[idx],dict(base_meta,step=records[idx]['step']),x,y)
    if first:
        idx=next(i for i,r in enumerate(records) if r['step']==first['step']); saved['first_boolean_exact']=save_ckpt(CKOUT/f"I9_seed{seed}_{arm}_first_boolean_exact.pt",states[idx],dict(base_meta,step=first['step']),x,y)
    return {"seed":seed,"arm":arm,"active_losses":base_meta['active_losses'],"ste_auxiliary_active":active,"ste_variant":variant if active else None,"lambda":lam,"rho":.5 if active else 0.0,"parent_checkpoint":str(parent),"parent_sha256":sha,"parent_metrics":parent_m,"generic_diagnostics":diag,"first_boolean_recovery":first,"boolean_regressions_after_first":regressions,"longest_consecutive_exact_evaluations":longest,"mask_transitions":transitions,"trajectory":records,"saved_checkpoints":saved}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--arm',choices=('all','control','ste'),default='all'); ap.add_argument('--steps',type=int,default=3000); args=ap.parse_args(); torch.set_num_threads(1)
    task=build_task('bitwise_xor_truth_table',{'bits':4}); x=task['X'].float(); y=task['Y'].float(); parent,pp,psha=load_parent(args.seed); diag=generic_diagnostics(parent,x,y)
    chosen=diag['chosen_variant']; payload={"experiment":"I9-ste-replication","git_sha":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"seed":args.seed,"parent_checkpoint":str(pp),"parent_sha256":psha,"parent_metrics":metrics(parent,x,y),"generic_diagnostics":diag,"chosen_variant":chosen}
    if args.seed==4: arms=('control',) if args.arm=='all' else ((args.arm,) if args.arm=='control' else tuple())
    else: arms=('control','ste') if args.arm=='all' else (args.arm,)
    if chosen is None and 'ste' in arms: arms=tuple(a for a in arms if a!='ste')
    payload['arms']=[run_arm(args.seed,a,chosen,x,y,args.steps) for a in arms]
    path=OUT/'i9_ste_replication_results.json'
    if path.exists():
        old=json.loads(path.read_text()); by={(r['seed'],r['arm']):r for r in old.get('runs',[])}
        for r in payload['arms']: by[(r['seed'],r['arm'])]=r
        payload['runs']=list(by.values()); payload['history']=old.get('history',[])
    else: payload['runs']=payload.pop('arms')
    path.write_text(json.dumps(payload,indent=2)+'\n'); print(path)

if __name__=='__main__': main()
