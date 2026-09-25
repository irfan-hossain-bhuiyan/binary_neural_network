"""A5: target-aware MSE plus unnormalized Gini output polarization."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, statistics, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); torch.set_num_threads(1)
from research.boolean_tasks import build_task
from research.run_a3_tolerance_loss import make_model, metric, evaluate, gradient_diagnostics

SEEDS=(0,1,2,3,4); STEPS=6000; LAMBDA=1.5
EVAL_STEPS=set([0,*range(25,501,25),*range(550,6001,50),1000,2000,3000,4000,5000,6000])
DIAG_STEPS={0,2000,3000,4000,6000}; HIST_BINS=20
ARMS=("MSE_CONTROL","MSE_GINI","MSE_WARMUP_GINI")

def state_hash(state):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v): h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str): h.update(str(k).encode()); add(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v: add(x)
  else: h.update(repr(v).encode())
 add(state); return h.hexdigest()

def snapshot(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def masks(m):
 e=[]; b=[]
 for layer in m.expectation_layers:
  e.append((layer.effective_gate().detach()>=.5).flatten().cpu()); b.append((layer.actual_bias().detach()>=.5).flatten().cpu())
 return {"edge":torch.cat(e),"bias":torch.cat(b)}

def topo(cur,ref):
 e=cur["edge"]!=ref["edge"]; b=cur["bias"]!=ref["bias"]
 return {"edge_hamming":int(e.sum()),"bias_hamming":int(b.sum()),"edge_0_to_1":int((~ref["edge"]&cur["edge"]).sum()),"edge_1_to_0":int((ref["edge"]&~cur["edge"]).sum()),"bias_0_to_1":int((~ref["bias"]&cur["bias"]).sum()),"bias_1_to_0":int((ref["bias"]&~cur["bias"]).sum())}

def loss_parts(out,y):
 mse=(out-y).square().mean(); gini=(out*(1-out)).mean(); return mse,gini

def loss_value(out,y,kind):
 mse,gini=loss_parts(out,y)
 if kind=="MSE": return mse
 if kind=="GINI": return mse+LAMBDA*gini
 raise ValueError(kind)

def component_gradients(model,x,y):
 out=model(x); mse,gini=loss_parts(out,y); params=[p for p in model.parameters() if p.requires_grad]
 gm=torch.autograd.grad(mse,params,retain_graph=True,allow_unused=True); gg=torch.autograd.grad(gini,params,allow_unused=True)
 raw=[]; names=[]
 for i,l in enumerate(model.expectation_layers): raw += [l.raw_edge,l.bias]; names += [f"layer{i}.raw_edge",f"layer{i}.bias"]
 def get(gs,p):
  for q,g in zip(params,gs):
   if q is p: return torch.zeros_like(p) if g is None else g.detach()
  return torch.zeros_like(p)
 gm_raw=[get(gm,l.raw_edge) for l in model.expectation_layers]; gg_raw=[get(gg,l.raw_edge) for l in model.expectation_layers]
 def flat(z): return torch.cat([v.reshape(-1) for v in z])
 a=flat(gm_raw); b=flat(gg_raw)
 per=[]
 for i,(u,v) in enumerate(zip(gm_raw,gg_raw)):
  uu=u.reshape(-1); vv=v.reshape(-1); per.append({"layer":i,"mse_l2":float(uu.norm()),"gini_l2":float(vv.norm()),"cosine":float(F.cosine_similarity(uu.reshape(1,-1),vv.reshape(1,-1)).item()) if uu.norm()>0 and vv.norm()>0 else 0.0})
 return {"mse_loss":float(mse.detach()),"gini_loss":float(gini.detach()),"mse_raw_edge_l2":float(a.norm()),"gini_raw_edge_l2":float(b.norm()),"cosine":float(F.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)).item()) if a.norm()>0 and b.norm()>0 else 0.0,"layers":per}

def error_stats(out,y):
 e=(out-y).square().mean(dim=1); hist=torch.histc(e.detach(),bins=HIST_BINS,min=0,max=1)
 qs=torch.quantile(e.detach(),torch.tensor([.5,.75,.9,.95,.99],device=e.device))
 return {"mean":float(e.mean()),"variance":float(e.var(unbiased=False)),"median":float(qs[0]),"p75":float(qs[1]),"p90":float(qs[2]),"p95":float(qs[3]),"p99":float(qs[4]),"max":float(e.max()),"fractions":{"lt_1e-4":float((e<1e-4).float().mean()),"lt_1e-3":float((e<1e-3).float().mean()),"lt_0.01":float((e<.01).float().mean()),"lt_0.05":float((e<.05).float().mean()),"gt_0.25":float((e>.25).float().mean()),"gt_0.50":float((e>.5).float().mean()),"gt_0.90":float((e>.9).float().mean())},"histogram":{"bins":HIST_BINS,"min":0.0,"max":1.0,"counts":[int(v) for v in hist.cpu()]}}

def polarization(out):
 u=out*(1-out); return {"mean_u":float(u.mean()),"median_u":float(u.median()),"fraction_endpoint_0.01":float(((out<.01)|(out>.99)).float().mean()),"fraction_endpoint_0.05":float(((out<.05)|(out>.95)).float().mean()),"fraction_mid_0.4_0.6":float(((out>.4)&(out<.6)).float().mean())}

def record(model,x,y,chain,step,core,ref_masks,loss_kind,full=False):
 with torch.no_grad(): out=model(x)
 ev=evaluate(model,x,y,chain); c=ev["continuous"]; rec={"step":step,"core_seconds":core,"native_loss":float(loss_value(out,y,loss_kind).detach()),"mse_loss":float(loss_parts(out,y)[0].detach()),"gini_loss":float(loss_parts(out,y)[1].detach()),"continuous_exact":c["exact_accuracy"],"continuous_bit":c["bit_accuracy"],"hard_exact":ev["hard"]["exact_accuracy"],"boolean_exact":ev["boolean"]["exact_accuracy"],"e_inf":c["e_inf"],"bit_disagreement":ev["bit_disagreement_fraction"],"row_disagreement":ev["row_disagreement_fraction"],"mean_hamming":ev["mean_hamming_continuous_to_boolean"],"error_distribution":error_stats(out,y),"polarization":polarization(out),"topology":topo(masks(model),ref_masks),"continuous_per_bit":c["per_bit_accuracy"],"boolean_per_bit":ev["boolean"]["per_bit_accuracy"]}
 if full:
  rec["continuous_carry_chain"]={k:{"exact":v["exact_accuracy"],"bit":v["bit_accuracy"]} for k,v in c["carry_chain"].items()}; rec["boolean_carry_chain"]={k:{"exact":v["exact_accuracy"],"bit":v["bit_accuracy"]} for k,v in ev["boolean"]["carry_chain"].items()}; rec["gradient_diagnostics"]=gradient_diagnostics(model,x,y,loss_kind); rec["component_gradients"]=component_gradients(model,x,y)
 return rec

def train(seed,initial,x,y,chain,device,arm):
 model=make_model(seed,device); model.load_state_dict(copy.deepcopy(initial)); opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.0); init_masks=masks(model); switch_masks=None; core=0.; times=[]; traj=[]; recovery={k:None for k in ("continuous_exact","boolean_exact","stable_boolean_exact","e_inf_lt_0_25","e_inf_lt_0_10","e_inf_lt_0_05","e_inf_lt_0_01")}; prior={k:False for k in recovery}; regress={k:0 for k in recovery}
 def maybe(step):
  nonlocal switch_masks
  if step not in EVAL_STEPS:return
  full=step in DIAG_STEPS or step in (1000,2000,3000,4000,5000,6000)
  if step==2000 and switch_masks is None: switch_masks=masks(model)
  ref=switch_masks if arm=="MSE_WARMUP_GINI" and switch_masks is not None else init_masks
  kind="MSE" if arm=="MSE_CONTROL" or (arm=="MSE_WARMUP_GINI" and step<2000) else "GINI"
  rec=record(model,x,y,chain,step,core,ref,kind,full); rec["phase"]="MSE" if kind=="MSE" else "MSE_PLUS_GINI"; traj.append(rec)
  cond={"continuous_exact":rec["continuous_exact"]>=1,"boolean_exact":rec["boolean_exact"]>=1,"e_inf_lt_0_25":rec["e_inf"]<.25,"e_inf_lt_0_10":rec["e_inf"]<.1,"e_inf_lt_0_05":rec["e_inf"]<.05,"e_inf_lt_0_01":rec["e_inf"]<.01}
  for k,v in cond.items():
   if v and recovery[k] is None: recovery[k]={"step":step,"core_seconds":core}
   if prior[k] and not v: regress[k]+=1
   prior[k]=v
 maybe(0)
 for step in range(1,STEPS+1):
  t=time.perf_counter(); opt.zero_grad(set_to_none=True); out=model(x); kind="MSE" if arm=="MSE_CONTROL" or (arm=="MSE_WARMUP_GINI" and step<=2000) else "GINI"; loss=loss_value(out,y,kind); loss.backward(); opt.step(); dt=time.perf_counter()-t; core+=dt; times.append(dt); maybe(step)
 if recovery["boolean_exact"] is not None and regress["boolean_exact"]==0: recovery["stable_boolean_exact"]=recovery["boolean_exact"]
 return {"seed":seed,"arm":arm,"loss":"MSE" if arm=="MSE_CONTROL" else ("MSE_PLUS_GINI" if arm=="MSE_GINI" else "MSE_WARMUP_MSE_PLUS_GINI"),"initial_state_sha256":state_hash(initial),"switch_step":2000 if arm=="MSE_WARMUP_GINI" else None,"lambda":LAMBDA,"trajectory":traj,"recovery":recovery,"regressions":regress,"final":traj[-1],"timing":{"optimizer_core_seconds":core,"mean_ms_per_step":1000*statistics.mean(times),"median_ms_per_step_after_warmup":1000*statistics.median(times[10:]),"optimizer_steps":STEPS},"checkpoint_policy":"disabled"}

def scalar_checks():
 p=torch.linspace(0,1,10001); out={}
 for y in (0.,1.):
  mse=(p-y).square(); g=mse+LAMBDA*p*(1-p); out[str(int(y))]={"mse_argmin":float(p[mse.argmin()]),"gini_argmin":float(p[g.argmin()]),"gini_wrong_endpoint_derivative":float((2*(1-y)+LAMBDA*(1-2*1)) if y==0 else (-2+LAMBDA*(1-0)))}
 q=.5; rr=q*(1-p).square()+(1-q)*p.square()+LAMBDA*p*(1-p); out["q_0.5"]={"endpoint_risk":float(rr[0]),"mid_risk":float(rr[len(p)//2]),"endpoint_argmin":float(p[rr.argmin()]),"mid_is_max":bool(rr[len(p)//2]>=rr[0])}
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--output',default='research/operator_results/a5_target_gini_results.json'); args=ap.parse_args(); device=torch.device(args.device); task=build_task('binary_addition',{'bits':4}); x=task['X'].float().to(device); y=task['Y'].float().to(device); chain=task['carry_chain_length'].to(device); results=[]; hashes={}
 for seed in SEEDS:
  base=make_model(seed,device); initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}; hashes[str(seed)]=state_hash(initial)
  for arm in ARMS: results.append(train(seed,initial,x,y,chain,device,arm)); print(seed,arm,results[-1]['recovery'],flush=True)
 payload={'experiment':'A5-target-aware-mse-gini','git_sha':None,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None},'a4_audit':{'valid':True,'canonical':'research/operator_results/a4_staged_tolerance_results.json','verified':'A4 completed; no branch reached exact Boolean addition; staged rational improved agreement but hardened wrong topology'},'architecture':{'input_dim':8,'width':64,'blocks':2,'residual_enabled':True,'output_dim':5,'operator':'lehmer_p2','initializer':'I2-B meanfield sigma2 + BIAS_ONE'},'task':{'name':'binary_addition','bits':4,'rows':256,'target_generator_agreement':bool(torch.equal(task['target_integer'],task['target_ripple']))},'objective':{'lambda':LAMBDA,'mse':'mean((p-y)^2)','gini':'mean(p*(1-p))','staged_switch_step':2000},'training':{'optimizer':'Adam','lr':.01,'weight_decay':0.0,'steps':STEPS,'full_batch':256,'checkpoint_policy':'disabled'},'scalar_checks':scalar_checks(),'arms':list(ARMS),'paired_initial_state_hashes':hashes,'results':results,'storage_policy':'compact trajectories; milestone distributions and gradients; no model checkpoints'}
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
