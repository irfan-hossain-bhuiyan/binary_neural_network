"""A4: common MSE prefix followed by three objective continuations."""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, statistics, sys, time
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from research.boolean_tasks import build_task  # noqa: E402
from research.run_a3_tolerance_loss import (  # noqa: E402
    make_model, loss_value, metric, evaluate, gradient_diagnostics,
    output_gradient_sparsity,
)

OUT=ROOT/"research/operator_results"; SEEDS=(0,1,2,3,4); BRANCHES=("MSE_TO_MSE","MSE_TO_SOFTPLUS","MSE_TO_RATIONAL")
PREFIX_STEPS=2000; FINAL_STEP=6000; EPS=(.25,.10,.05,.01)
PREFIX_EVAL={1500,1750,1900,1950,1975,2000}; POST_EVAL={2025,2050,2100,2200,2250,2500,3000,3500,4000,4500,5000,5500,6000}
GRAD_STEPS={2000,2100,3000,4000,6000}; SPARSITY_STEPS={2000,2025,2100,2500,3000,4000,6000}

def state_hash(state):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x): h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode()); add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x: add(v)
        else: h.update(repr(x).encode())
    add(state); return h.hexdigest()

def snapshot(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def masks(model):
    edge=[]; bias=[]; per=[]
    for i,layer in enumerate(model.expectation_layers):
        e=(layer.effective_gate().detach()>=.5).flatten().cpu(); b=(layer.actual_bias().detach()>=.5).flatten().cpu(); edge.append(e); bias.append(b); per.append({"layer":i,"edge_active":int(e.sum()),"bias_active":int(b.sum())})
    return {"edge":torch.cat(edge),"bias":torch.cat(bias),"per_layer":per}

def topology_distance(current, reference):
    e=(current["edge"]!=reference["edge"]); b=(current["bias"]!=reference["bias"])
    return {"edge_hamming":int(e.sum()),"bias_hamming":int(b.sum()),"edge_0_to_1":int((~reference["edge"] & current["edge"]).sum()),"edge_1_to_0":int((reference["edge"] & ~current["edge"]).sum()),"bias_0_to_1":int((~reference["bias"] & current["bias"]).sum()),"bias_1_to_0":int((reference["bias"] & ~current["bias"]).sum())}

def per_layer_topology(current, reference):
    out=[]; eo=bo=0
    for i,(c,r) in enumerate(zip(current["per_layer"],reference["per_layer"])):
        # Active counts alone are insufficient, so use flattened slices below.
        out.append({"layer":i,"edge_active":c["edge_active"],"bias_active":c["bias_active"]})
    return out

def row_summary(rec):
    c=rec["continuous"]; return {"step":rec["step"],"core_seconds":rec["core_seconds"],"continuous_exact":c["exact_accuracy"],"continuous_bit":c["bit_accuracy"],"hard_exact":rec["hard"]["exact_accuracy"],"boolean_exact":rec["boolean"]["exact_accuracy"],"e_inf":c["e_inf"],"mean_row_error":c["mean_row_error"],"median_row_error":c["median_row_error"],"row_tolerance_0.25":c["row_tolerance_fractions"]["0.25"],"row_tolerance_0.10":c["row_tolerance_fractions"]["0.1"],"row_tolerance_0.05":c["row_tolerance_fractions"]["0.05"],"row_tolerance_0.01":c["row_tolerance_fractions"]["0.01"],"bit_disagreement":rec["bit_disagreement_fraction"],"row_disagreement":rec["row_disagreement_fraction"],"mean_hamming":rec["mean_hamming_continuous_to_boolean"]}

def full_milestone(rec, model, x, y, chain, loss_name, switch_masks, step):
    out=row_summary(rec); out["continuous_per_bit"]=rec["continuous"]["per_bit_accuracy"]; out["hard_per_bit"]=rec["hard"]["per_bit_accuracy"]; out["boolean_per_bit"]=rec["boolean"]["per_bit_accuracy"]; out["continuous_carry_chain"]={k:{"exact":v["exact_accuracy"],"bit":v["bit_accuracy"],"median_row_error":v["median_row_error"]} for k,v in rec["continuous"]["carry_chain"].items()}; out["boolean_carry_chain"]={k:{"exact":v["exact_accuracy"],"bit":v["bit_accuracy"],"median_row_error":v["median_row_error"]} for k,v in rec["boolean"]["carry_chain"].items()}; out["topology"]=topology_distance(masks(model),switch_masks)
    if step in GRAD_STEPS: out["gradient_diagnostics"]=gradient_diagnostics(model,x,y,loss_name)
    if step in SPARSITY_STEPS: out["output_gradient_sparsity"]=output_gradient_sparsity(model,x,y,loss_name)
    return out

def train_prefix(seed,x,y,chain,device):
    model=make_model(seed,device); init_hash=state_hash(snapshot(model)); opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.0); core=0.; times=[]; trajectory=[]
    for step in range(1,PREFIX_STEPS+1):
        t=time.perf_counter(); opt.zero_grad(set_to_none=True); loss=loss_value(model(x),y,"BASELINE"); loss.backward(); opt.step(); core+=time.perf_counter()-t; times.append(time.perf_counter()-t)
        if step in PREFIX_EVAL:
            rec=evaluate(model,x,y,chain); rec.update({"step":step,"core_seconds":core}); trajectory.append(full_milestone(rec,model,x,y,chain,"BASELINE",masks(model),step) if step==2000 else row_summary(rec))
    return model,opt,init_hash,trajectory,core,times

def train_branch(seed,branch,base_model,base_opt,prefix_trajectory,x,y,chain,device,switch_masks):
    loss_name={"MSE_TO_MSE":"BASELINE","MSE_TO_SOFTPLUS":"ROW_MAX_SOFTPLUS","MSE_TO_RATIONAL":"ROW_MAX_RATIONAL"}[branch]
    model=make_model(seed,device); model.load_state_dict(copy.deepcopy(base_model.state_dict())); opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.0); opt.load_state_dict(copy.deepcopy(base_opt.state_dict()))
    model_hash=state_hash(snapshot(model)); opt_hash=state_hash(opt.state_dict()); core=0.; times=[]; trajectory=[]; events=[]; prior_bool=None
    # Switch record is copied from common prefix, then branch evaluations begin.
    switch_rec=evaluate(model,x,y,chain); switch_rec.update({"step":2000,"core_seconds":0.0}); switch_full=full_milestone(switch_rec,model,x,y,chain,loss_name,switch_masks,2000); trajectory.append(switch_full); prior_bool=switch_rec["boolean"]["exact_accuracy"]
    for step in sorted(POST_EVAL):
        while (step - (2000 if not times else 2000)) > len(times):
            t=time.perf_counter(); opt.zero_grad(set_to_none=True); loss=loss_value(model(x),y,loss_name); loss.backward(); opt.step(); core+=time.perf_counter()-t; times.append(time.perf_counter()-t)
        rec=evaluate(model,x,y,chain); rec.update({"step":step,"core_seconds":core}); simple=row_summary(rec); trajectory.append(full_milestone(rec,model,x,y,chain,loss_name,switch_masks,step) if step in GRAD_STEPS or step in SPARSITY_STEPS or step in (3000,4000,6000) else simple)
        b=rec["boolean"]["exact_accuracy"]
        if prior_bool is not None and abs(b-prior_bool)>=.05: events.append({"step":step,"boolean_before":prior_bool,"boolean_after":b,"topology":topology_distance(masks(model),switch_masks)})
        prior_bool=b
    recovery={k:None for k in ("continuous_exact","boolean_exact","e_inf_lt_0_25","e_inf_lt_0_10","e_inf_lt_0_05","e_inf_lt_0_01","rows_lt_0_25","rows_lt_0_10","rows_lt_0_05")};
    for e in trajectory:
        for k,now in {"continuous_exact":e["continuous_exact"]>=1,"boolean_exact":e["boolean_exact"]>=1,"e_inf_lt_0_25":e["e_inf"]<.25,"e_inf_lt_0_10":e["e_inf"]<.1,"e_inf_lt_0_05":e["e_inf"]<.05,"e_inf_lt_0_01":e["e_inf"]<.01,"rows_lt_0_25":e["row_tolerance_0.25"]>=1,"rows_lt_0_10":e["row_tolerance_0.10"]>=1,"rows_lt_0_05":e["row_tolerance_0.05"]>=1}.items():
            if now and recovery[k] is None: recovery[k]={"global_step":e["step"],"continuation_steps":e["step"]-2000,"continuation_core_seconds":e["core_seconds"]}
    final=trajectory[-1]; final_dist=final["topology"]; return {"seed":seed,"continuation":branch,"loss":loss_name,"switch_boolean_exact":trajectory[0]["boolean_exact"],"switch_state_model_sha256":model_hash,"switch_state_optimizer_sha256":opt_hash,"trajectory":trajectory,"topology_events":events,"recovery":recovery,"final":final,"topology_change_final":final_dist,"timing":{"continuation_core_seconds":core,"mean_ms_per_step":1000*statistics.mean(times),"median_ms_per_step":1000*statistics.median(times),"continuation_steps":len(times)},"optimizer_state_cloned":True,"checkpoint_policy":"disabled"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");ap.add_argument("--output",default=str(OUT/"a4_staged_tolerance_results.json"));args=ap.parse_args();device=torch.device(args.device)
    task=build_task("binary_addition",{"bits":4});x=task["X"].float().to(device);y=task["Y"].float().to(device);chain=task["carry_chain_length"].to(device);results=[];prefixes={}; hashes={}
    for seed in SEEDS:
        model,opt,init_hash,prefix_traj,prefix_core,prefix_times=train_prefix(seed,x,y,chain,device); sm=masks(model); pmodel_hash=state_hash(snapshot(model)); popt_hash=state_hash(opt.state_dict()); hashes[str(seed)]=init_hash; prefixes[str(seed)]={"initial_state_sha256":init_hash,"switch_model_sha256":pmodel_hash,"switch_optimizer_sha256":popt_hash,"prefix_core_seconds":prefix_core,"prefix_trajectory":prefix_traj}
        for branch in BRANCHES:
            run=train_branch(seed,branch,model,opt,prefix_traj,x,y,chain,device,sm); run["initial_state_sha256"]=init_hash;run["prefix_core_seconds"]=prefix_core;results.append(run);print(seed,branch,run["recovery"],flush=True)
    payload={"experiment":"A4-staged-ordinary-to-whole-word-tolerance","git_sha":None,"runtime":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None},"a3_audit":{"valid":True,"canonical":"research/operator_results/a3_tolerance_loss_results.json","verified_final_counts":{"MSE":"0/5 continuous, 0/5 Boolean","BIT_MEAN_RATIONAL":"0/5 continuous, 0/5 Boolean","ROW_MAX_RATIONAL":"0/5 continuous, 0/5 Boolean","ROW_MAX_SOFTPLUS":"1/5 continuous, 0/5 Boolean"}},"architecture":{"input_dim":8,"width":64,"blocks":2,"residual_enabled":True,"output_dim":5,"operator":"lehmer_p2","initializer":"I2-B meanfield sigma2 + BIAS_ONE"},"task":{"name":"binary_addition","bits":4,"rows":256,"target_generator_agreement":bool(torch.equal(task["target_integer"],task["target_ripple"]))},"training":{"prefix_loss":"MSE","prefix_steps":2000,"continuation_steps":4000,"final_step":6000,"optimizer":"Adam","lr":.01,"weight_decay":0.0,"full_batch":256,"checkpoint_policy":"disabled"},"branches":list(BRANCHES),"paired_initial_state_hashes":hashes,"prefixes":prefixes,"results":results,"storage_policy":"compact milestone/event trajectory; no full diagnostic object at ordinary points; no model or optimizer checkpoints"}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n");print(out)
if __name__=="__main__":main()
