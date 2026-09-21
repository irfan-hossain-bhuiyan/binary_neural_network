"""I1R replicated mean-field propagation diagnostics (no training)."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from layers import SigmoidOrLogicLayer
from models import SigmoidOrModernLogicGateNet
from discrete_logic_net import DiscreteOrNorGateLayer
from research.meanfield_initialization import meanfield_gaussian_edge_init_, target_selected_probability
from research.boolean_tasks import build_task

P0S=(.05,.20,.35,.50,.65,.80,.95)
BIAS={"CURRENT":lambda t: nn.init.normal_(t,mean=.5,std=.1),"ONE":lambda t: nn.init.normal_(t,mean=1.,std=.1),"BALANCED_POLARIZED":lambda t: nn.init.normal_(t,mean=.5,std=2.)}

def controlled_binary(n,d,p,seed):
    x=torch.ones(n,d); g=torch.Generator().manual_seed(seed); z=round(n*p)
    for j in range(d): x[torch.randperm(n,generator=g)[:z],j]=0
    return x

def stats(h):
    h=h.detach().float(); q=torch.quantile(h,torch.tensor([.01,.05,.25,.5,.75,.95,.99]))
    return {"mean":float(h.mean()),"std":float(h.std(unbiased=False)),"q01":float(q[0]),"q05":float(q[1]),"q25":float(q[2]),"median":float(q[3]),"q75":float(q[4]),"q95":float(q[5]),"q99":float(q[6]),"lt01":float((h<.1).float().mean()),"lt04":float((h<.4).float().mean()),"mid":float(((h>=.4)&(h<=.6)).float().mean()),"gt06":float((h>.6).float().mean()),"gt09":float((h>.9).float().mean()),"lt05":float((h<.5).float().mean())}

def layer_pair(m,sigma,bias_fn,seed):
    torch.manual_seed(seed)
    l=SigmoidOrLogicLayer(m,m,"lehmer_p2",.5,bias_fn,lambda t,fan_in: meanfield_gaussian_edge_init_(t,fan_in,sigma))
    d=DiscreteOrNorGateLayer(m,m)
    with torch.no_grad(): d.weight.copy_(l.effective_gate()>=.5); d.bias.copy_(l.actual_bias()>=.5)
    return l,d

def theory(p,m,q):
    s=target_selected_probability(m)
    return (1-s*(1-q+(2*q-1)*p))**m

def one_chain(p0, sigma, bias_name, seed, n_bool, n_cont, depth=12):
    layers=[layer_pair(64,sigma,BIAS[bias_name],seed+i+1) for i in range(depth)]
    xb=controlled_binary(n_bool,64,p0,seed+10000).bool(); xc=xb[:n_cont].float(); rows=[]
    for dep in range(depth+1):
        row={"depth":dep,"p_zero_bool":float((~xb).float().mean()),"continuous":stats(xc)}
        if dep==0: row["selected_fan_in"] = None
        rows.append(row)
        if dep==depth: break
        l,d=layers[dep]; xc=l(xc); xb=d(xb)
    return rows, layers

def summarize_runs(runs):
    out=[]
    for dep in range(len(runs[0])):
        vals=[r[dep] for r in runs]; p=torch.tensor([v['p_zero_bool'] for v in vals]); cs=[v['continuous'] for v in vals]
        out.append({"depth":dep,"p_zero_bool_mean":float(p.mean()),"p_zero_bool_std":float(p.std(unbiased=False)),"p_zero_bool_q05":float(torch.quantile(p,.05)),"p_zero_bool_q25":float(torch.quantile(p,.25)),"p_zero_bool_median":float(torch.quantile(p,.5)),"p_zero_bool_q75":float(torch.quantile(p,.75)),"p_zero_bool_q95":float(torch.quantile(p,.95)),"continuous_mean":{k:sum(c[k] for c in cs)/len(cs) for k in cs[0]}})
    return out

def fanin_stats(sigma,bias_name,seeds):
    rows=[]
    for seed in range(seeds):
        l,_=layer_pair(64,sigma,BIAS[bias_name],seed)
        k=(l.effective_gate()>=.5).sum(1)
        rows.append({"zero":float((k==0).float().mean()),"one":float((k==1).float().mean()),"two":float((k==2).float().mean()),"three":float((k==3).float().mean()),"four_plus":float((k>=4).float().mean()),"mean":float(k.float().mean()),"median":float(k.float().median()),"max":int(k.max())})
    return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}

@torch.no_grad()
def residual_stats(sigma, bias_name, seeds):
    task=build_task('bitwise_xor_truth_table', {'bits':4}); x=task['X'].float()
    out=[]
    for seed in range(seeds):
        torch.manual_seed(seed)
        init=lambda t,fan_in: meanfield_gaussian_edge_init_(t,fan_in,sigma)
        model=SigmoidOrModernLogicGateNet(8,4,width=64,num_residual_blocks=2,or_operator='lehmer_p2',
            gate_initializations=[.5]*6,bias_initialization=BIAS[bias_name],edge_initialization=init)
        h=x; rows=[{'name':'input','continuous':stats(h),'boolean_zero':float((h<.5).float().mean())}]
        h=model.stem(h); rows.append({'name':'stem','continuous':stats(h),'boolean_zero':float((h<.5).float().mean())})
        for bi,block in enumerate(model.blocks):
            h1=block.layer1(h); rows.append({'name':f'block{bi}.layer1','continuous':stats(h1),'boolean_zero':float((h1<.5).float().mean())})
            h2=block.layer2(h1); rows.append({'name':f'block{bi}.layer2','continuous':stats(h2),'boolean_zero':float((h2<.5).float().mean())})
            h=h+h2-2*h*h2; rows.append({'name':f'block{bi}.residual','continuous':stats(h),'boolean_zero':float((h<.5).float().mean())})
        h=model.head(h); rows.append({'name':'head','continuous':stats(h),'boolean_zero':float((h<.5).float().mean())})
        out.append(rows)
    result=[]
    for i,name in enumerate([r['name'] for r in out[0]]):
        cs=[r[i]['continuous'] for r in out]
        result.append({'name':name,'continuous_mean':{k:sum(c[k] for c in cs)/len(cs) for k in cs[0]},'boolean_zero_mean':sum(r[i]['boolean_zero'] for r in out)/len(out)})
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--network-seeds',type=int,default=64); ap.add_argument('--bool-batch',type=int,default=8192); ap.add_argument('--continuous-batch',type=int,default=512); ap.add_argument('--output',default=str(ROOT/'research/operator_results/initialization_i1r.json')); a=ap.parse_args()
    result={"config":{"network_seeds":a.network_seeds,"bool_batch":a.bool_batch,"continuous_batch":a.continuous_batch,"p0":P0S,"depth":12},"theory":{"s_target_64":target_selected_probability(64),"bias_one":[],"balanced":[]},"plain_chain":{},"fanin":{},"bias_signal":{},"residual_network":{}}
    for name,fn in BIAS.items():
        if name == 'ONE':
            t = torch.empty(100000).normal_(1.0, .1)
        elif name == 'BALANCED_POLARIZED':
            t = torch.empty(100000).normal_(.5, 2.0)
        else:
            t = torch.empty(100000).normal_(.5, .1)
        b=torch.clamp(t,0,1); gain=(1-2*b).abs()
        result['bias_signal'][name]={"raw_mean":float(t.mean()),"effective_mean":float(b.mean()),"effective_std":float(b.std()),"threshold_one":float((b>=.5).float().mean()),"le01":float((b<=.01).float().mean()),"ge99":float((b>=.99).float().mean()),"middle":float(((b>.4)&(b<.6)).float().mean()),"gain_mean":float(gain.mean()),"gain_median":float(gain.median()),"gain_lt01":float((gain<.1).float().mean()),"gain_gt09":float((gain>.9).float().mean())}
        q=1.0 if name=='ONE' else (.5 if name=='BALANCED_POLARIZED' else None)
        result['theory']['bias_one' if name=='ONE' else 'balanced' if q==.5 else 'current'] = [{"p0":p,"trajectory":[(lambda x:[x:=theory(x,64,q) for _ in range(12)])(p)]} for p in P0S] if q is not None else []
        for sigma in (2.,4.,6.):
            key=f"{name}_sigma{int(sigma)}"; result['plain_chain'][key]={}; result['fanin'][key]=fanin_stats(sigma,name,a.network_seeds)
            if sigma in (2.,4.): result['residual_network'][key]=residual_stats(sigma,name,min(a.network_seeds,8))
            for p0 in P0S:
                runs=[one_chain(p0,sigma,name,s,a.bool_batch,a.continuous_batch) [0] for s in range(a.network_seeds)]
                result['plain_chain'][key][str(p0)]=summarize_runs(runs)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(out)
if __name__=='__main__': main()
