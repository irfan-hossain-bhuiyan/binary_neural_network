"""I0/I1 diagnostics for the mean-field Gaussian initializer.

This script is intentionally diagnostic-only: it performs no optimization.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from layers import SigmoidOrLogicLayer, xor
from models import SigmoidOrModernLogicGateNet
from research.meanfield_initialization import meanfield_gaussian_edge_init_, meanfield_gaussian_metadata
from discrete_logic_net import DiscreteOrNorGateLayer

def bias_current(t): nn.init.normal_(t, mean=.5, std=.1)
def bias_one(t): nn.init.normal_(t, mean=1.0, std=.1)

def make_layer(m, sigma, bias_fn, seed):
    torch.manual_seed(seed)
    return SigmoidOrLogicLayer(m, m, "lehmer_p2", .5, bias_fn,
                               lambda t, fan_in: meanfield_gaussian_edge_init_(t, fan_in, sigma))

def i0(seeds=32, sigma_values=(2.,4.,6.), fanins=(8,64)):
    rows=[]
    for m in fanins:
        for sigma in sigma_values:
            meta=meanfield_gaussian_metadata(m,sigma); vals=[]
            for seed in range(seeds):
                layer=make_layer(m,sigma,bias_one,seed)
                g=layer.effective_gate().detach(); d=g*(1-g)
                vals.append({"selected":float((g>=.5).float().mean()),"fan_in":float((g>=.5).sum(1).float().mean()),"g_mean":float(g.mean()),"d_mean":float(d.mean()),"d_lt_1e2":float((d<1e-2).float().mean()),"d_lt_1e3":float((d<1e-3).float().mean()),"d_lt_1e4":float((d<1e-4).float().mean())})
            rows.append({"fan_in":m,"sigma":sigma,"metadata":meta,"observed_mean":{k:sum(x[k] for x in vals)/len(vals) for k in vals[0]},"observed_std":{"selected":torch.tensor([x["selected"] for x in vals]).std().item(),"fan_in":torch.tensor([x["fan_in"] for x in vals]).std().item()}})
    return rows

def controlled_binary(batch, features, p_zero, seed):
    n0=round(batch*p_zero); x=torch.ones(batch,features,dtype=torch.float32)
    g=torch.Generator().manual_seed(seed)
    for j in range(features):
        perm=torch.randperm(batch,generator=g); x[perm[:n0],j]=0
    return x

@torch.no_grad()
def plain_chain(p0, bias_fn, sigma=4., depth=12, width=64, seed=0, batch=8192):
    x=controlled_binary(batch,width,p0,seed); layers=[]; states=[]; bool_layers=[]
    for i in range(depth):
        layer=make_layer(width,sigma,bias_fn,seed+i+1); layers.append(layer)
        b=DiscreteOrNorGateLayer(width,width)
        b.weight.copy_(layer.effective_gate()>=.5); b.bias.copy_(layer.actual_bias()>=.5); bool_layers.append(b)
    h=x; hb=x.bool(); states.append({"depth":0,"p_zero_cont":float((h<.5).float().mean()),"p_zero_bool":float((~hb).float().mean())})
    for i,(layer,b) in enumerate(zip(layers,bool_layers),1):
        h=layer(h); hb=b(hb); states.append({"depth":i,"p_zero_cont":float((h<.5).float().mean()),"p_zero_bool":float((~hb).float().mean())})
    return states

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default=str(ROOT/'research/operator_results/initialization_i0_i1.json')); args=ap.parse_args()
    result={"i0":i0(),"i1":{}}
    for name,fn in (("BIAS_CURRENT",bias_current),("BIAS_ONE",bias_one)):
        result["i1"][name]=[{"p0":p,"states":plain_chain(p,fn)} for p in (.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95)]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(args.output)

if __name__=='__main__': main()
