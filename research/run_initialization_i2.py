"""I2: initialization-only comparison on exact 4-bit XOR."""
from __future__ import annotations

import copy, hashlib, json, platform, random, shutil, sys
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.meanfield_initialization import gaussian_bias_init_, meanfield_gaussian_edge_init_  # noqa: E402

OUT=ROOT/'research'/'operator_results'; CK=Path('research/operator_results/initialization_i2_checkpoints')
MILESTONES=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-8]
THRESHOLDS=[.30,.35,.40,.45,.50,.55,.60,.65,.70]
CONDITIONS={
 'I2-A':{'name':'historical_current','edge':'historical','bias':'CURRENT'},
 'I2-B':{'name':'mf_sigma2_one','edge':'meanfield','sigma':2.,'bias':'ONE'},
 'I2-C':{'name':'mf_sigma4_one','edge':'meanfield','sigma':4.,'bias':'ONE'},
 'I2-D':{'name':'mf_sigma2_balanced','edge':'meanfield','sigma':2.,'bias':'BALANCED_POLARIZED'},
 'I2-E':{'name':'mf_sigma4_balanced','edge':'meanfield','sigma':4.,'bias':'BALANCED_POLARIZED'},
}
BIAS={'CURRENT':(.5,.1),'ONE':(1.,.1),'BALANCED_POLARIZED':(.5,2.)}

def gen(seed): return torch.Generator(device='cpu').manual_seed(int(seed))
def bias_init(name, seed):
    mean,std=BIAS[name]; g=gen(seed)
    return lambda t: gaussian_bias_init_(t,mean,std,g)
def edge_init(sigma, seed):
    g=gen(seed)
    return lambda t,m: meanfield_gaussian_edge_init_(t,m,sigma,g)
def historical_model(device):
    # Exact B3 constructor behavior: constant alternating gates and global RNG bias draws.
    return SigmoidOrModernLogicGateNet(8,4,width=64,num_residual_blocks=2,or_operator='lehmer_p2',
        bias_initialization=lambda t: nn.init.normal_(t,mean=.5,std=.1)).to(device)
def model_for(condition, seed, device):
    spec=CONDITIONS[condition]
    if spec['edge']=='historical':
        torch.manual_seed(seed)
        return historical_model(device)
    # Separate paired streams: same seed gives identical masks across sigma2/4.
    edge_seed=100000+seed; bias_seed=200000+seed
    return SigmoidOrModernLogicGateNet(8,4,width=64,num_residual_blocks=2,or_operator='lehmer_p2',
        gate_initializations=[.5]*6,bias_initialization=bias_init(spec['bias'],bias_seed),
        edge_initialization=edge_init(spec['sigma'],edge_seed)).to(device)

def metrics(out,y):
    c=(out>=.5)==(y>=.5)
    return {'mse':float((out-y).square().mean().detach().cpu()),'bit_accuracy':float(c.float().mean().cpu()),'exact_accuracy':float(c.all(dim=-1).float().mean().cpu())}
@torch.no_grad()
def evaluate(model,x,y):
    cont=model(x); hard=model.forward_hard(x); disc=model.to_discrete(.5).to(x.device); boolean=disc(x.bool()).float()
    return {'continuous':metrics(cont,y),'hard':metrics(hard,y),'boolean':metrics(boolean,y)}
def layer_stats(model):
    rows=[]
    for i,l in enumerate(model.expectation_layers):
        g=l.effective_gate().detach(); b=l.actual_bias().detach(); d=g*(1-g); k=(g>=.5).sum(1)
        rows.append({'layer':i,'fan_in':l.in_features,'selected_fraction':float((g>=.5).float().mean().cpu()),'selected_fan_in_mean':float(k.float().mean().cpu()),'zero_fan_in_fraction':float((k==0).float().mean().cpu()),'one_fan_in_fraction':float((k==1).float().mean().cpu()),'mean_g':float(g.mean().cpu()),'std_g':float(g.std().cpu()),'median_g':float(g.median().cpu()),'sigmoid_derivative_mean':float(d.mean().cpu()),'sigmoid_derivative_lt_1e2':float((d<1e-2).float().mean().cpu()),'sigmoid_derivative_lt_1e3':float((d<1e-3).float().mean().cpu()),'sigmoid_derivative_lt_1e4':float((d<1e-4).float().mean().cpu()),'bias_mean':float(b.mean().cpu()),'bias_std':float(b.std().cpu()),'literal_gain_mean':float((1-2*b).abs().mean().cpu())})
    return rows
@torch.no_grad()
def activation_trace(model,x):
    disc=model.to_discrete(.5).to(x.device); h=x; hb=x.bool(); rows=[]
    def add(name,h,hb): rows.append({'name':name,'continuous_mean':float(h.mean().cpu()),'continuous_std':float(h.std(unbiased=False).cpu()),'discrete_zero_fraction':float((~hb).float().mean().cpu())})
    add('input',h,hb); h=model.stem(h); hb=disc.stem(hb); add('stem',h,hb)
    for i,b in enumerate(model.blocks):
        h1=b.layer1(h); hb1=disc.blocks[i].layer1(hb); add(f'block{i}.layer1',h1,hb1)
        h2=b.layer2(h1); hb2=disc.blocks[i].layer2(hb1); add(f'block{i}.layer2',h2,hb2)
        h=h+h2-2*h*h2; hb=hb^hb2; add(f'block{i}.residual',h,hb)
    add('head',model.head(h),disc.head(hb)); return rows
def grad_summary(model):
    out=[]
    for i,l in enumerate(model.expectation_layers):
        g=l.raw_edge.grad; b=l.bias.grad
        out.append({'layer':i,'raw_edge_mean_abs':float(g.abs().mean().cpu()) if g is not None else 0.,'raw_edge_median_abs':float(g.abs().median().cpu()) if g is not None else 0.,'bias_mean_abs':float(b.abs().mean().cpu()) if b is not None else 0.,'bias_median_abs':float(b.abs().median().cpu()) if b is not None else 0.})
    return out
def lehmer_sign(model,x):
    h=x.detach(); result=[]; layers=[]
    layers.append((model.stem,h)); h=model.stem(h)
    for b in model.blocks:
        layers.append((b.layer1,h)); h1=b.layer1(h); layers.append((b.layer2,h1)); h2=b.layer2(h1); h=h+h2-2*h*h2
    layers.append((model.head,h))
    for i,(l,inp) in enumerate(layers):
        v=l.contributions(inp).detach().requires_grad_(True); grad=torch.autograd.grad(l.or_operator(v).sum(),v)[0]
        result.append({'layer':i,'fraction_negative':float((grad<0).float().mean().cpu()),'fraction_positive':float((grad>0).float().mean().cpu()),'fraction_near_zero':float((grad.abs()<1e-8).float().mean().cpu())})
    return result
def mask_hash(model, bias=False):
    h=hashlib.sha256()
    for l in model.expectation_layers:
        t=(l.actual_bias()>=.5) if bias else (l.effective_gate()>=.5)
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def mask_hamming(model, initial_state, bias=False):
    total=0
    names=['stem','blocks.0.layer1','blocks.0.layer2','blocks.1.layer1','blocks.1.layer2','head']
    for i,l in enumerate(model.expectation_layers):
        raw=initial_state[f'{names[i]}.bias' if bias else f'{names[i]}.raw_edge']
        initial=(torch.sigmoid(raw) >= .5) if not bias else (torch.clamp(raw,0,1)>=.5)
        current=(l.effective_gate()>=.5) if not bias else (l.actual_bias()>=.5)
        total += int((initial.to(current.device) != current).sum().item())
    return total
def paired_assertions(seed,device):
    a=model_for('I2-B',seed,device); b=model_for('I2-C',seed,device)
    for la,lb in zip(a.expectation_layers,b.expectation_layers):
        assert torch.equal(la.effective_gate()>=.5,lb.effective_gate()>=.5)
    d=model_for('I2-D',seed,device); e=model_for('I2-E',seed,device)
    for ld,le in zip(d.expectation_layers,e.expectation_layers):
        assert torch.equal(ld.effective_gate()>=.5,le.effective_gate()>=.5)
    return True
def threshold_report(model,x,y):
    return {str(t):evaluate(model,x,y)['boolean'] for t in THRESHOLDS}
def save_verified(path,state,model,x,y,record,condition,seed,epoch,kind,manifest):
    torch.save(state,path); check=model_for(condition,seed,x.device); check.load_state_dict(torch.load(path,map_location=x.device,weights_only=True)); check.eval(); got=evaluate(check,x,y)
    for mode in ('continuous','hard','boolean'):
        for key in ('mse','bit_accuracy','exact_accuracy'):
            if abs(got[mode][key]-record[mode][key])>1e-6: raise RuntimeError(f'checkpoint mismatch {path} {mode}.{key}')
    manifest.append({'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'condition':condition,'seed':seed,'epoch':epoch,'kind':kind,'metrics':got})
def run(epochs,seeds,device):
    task=build_task('bitwise_xor_truth_table',{'bits':4}); x=task['X'].float().to(device); y=task['Y'].float().to(device)
    CK.mkdir(parents=True,exist_ok=True); all_results=[]; manifest=[]
    for condition in CONDITIONS:
        for seed in seeds:
            paired_assertions(seed,device) if condition!='I2-A' else None
            torch.manual_seed(seed); model=model_for(condition,seed,device); opt=torch.optim.Adam(model.parameters(),lr=.01)
            init_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; initial={'edge_mask_sha256':mask_hash(model),'bias_mask_sha256':mask_hash(model,True),'layers':layer_stats(model),'activation_trace':activation_trace(model,x)}
            trajectory=[]; milestones={}; pending={}; first_bool=None; regressions=0; best_cont=None; best_bool=None
            def queue(kind,epoch,rec): pending[kind]=(epoch,copy.deepcopy(rec),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            for epoch in range(epochs+1):
                out=model(x); loss=(out-y).square().mean(); opt.zero_grad(); loss.backward(); grads=grad_summary(model); opt.step()
                if epoch!=0 and epoch%25!=0 and epoch!=epochs: continue
                ev=evaluate(model,x,y); rec={'epoch':epoch,**ev,'layer_stats':layer_stats(model),'gradient_stats':grads,'lehmer_sign':lehmer_sign(model,x)}; trajectory.append(rec)
                mse=ev['continuous']['mse']
                for m in MILESTONES:
                    if str(m) not in milestones and mse<m: milestones[str(m)]={'epoch':epoch,'metrics':ev}; queue(f'mse{m:.0e}',epoch,rec)
                if first_bool is None and ev['boolean']['exact_accuracy']==1.: first_bool={'epoch':epoch,'mse':mse}
                elif first_bool is not None and ev['boolean']['exact_accuracy']<1.: regressions+=1
                if best_cont is None or mse<best_cont['continuous']['mse']: best_cont=copy.deepcopy(rec); queue('best_continuous_mse',epoch,rec)
                if best_bool is None or ev['boolean']['exact_accuracy']>best_bool['boolean']['exact_accuracy']: best_bool=copy.deepcopy(rec); queue('best_boolean_exact',epoch,rec)
            queue('final',epochs,trajectory[-1]); model.load_state_dict({k:v.to(device) for k,v in pending['best_continuous_mse'][2].items()}); best_cont['threshold_stability']=threshold_report(model,x,y); best_cont['final_topology']=layer_stats(model)
            for kind,(ep,rec,state) in pending.items(): save_verified(CK/f'I2_{condition}_seed{seed}_{kind}.pt',state,model,x,y,rec,condition,seed,ep,kind,manifest)
            final_masks={'edge':mask_hash(model),'bias':mask_hash(model,True)}
            all_results.append({'experiment':'I2','condition':condition,'seed':seed,'epochs':epochs,'optimizer_steps':epochs+1,'initialization':initial,'final_mask_sha256':final_masks,'edge_mask_hamming':mask_hamming(model,init_state),'bias_mask_hamming':mask_hamming(model,init_state,True),'best_continuous':best_cont,'best_boolean':best_bool,'first_boolean_recovery':first_bool,'boolean_regressions_after_recovery':regressions,'milestones':milestones,'trajectory':trajectory})
            print(condition,seed,best_cont['continuous']['mse'],best_cont['boolean']['exact_accuracy'],first_bool,flush=True)
    return all_results,manifest
def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3000); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--output',default=str(OUT/'initialization_i2_results.json')); a=p.parse_args(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    shutil.rmtree(CK,ignore_errors=True); results,manifest=run(a.epochs,range(5),device)
    payload={'experiment':'I2-initialization-reliability','git_sha':None,'device':str(device),'epochs':a.epochs,'conditions':CONDITIONS,'seeds':list(range(5)),'training':{'optimizer':'Adam','lr':.01,'loss':'MSE','batch_size':256,'regularizer':None,'noise':None,'scheduler':None,'operator':'lehmer_p2'},'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'results':results,'checkpoint_manifest':manifest}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(payload,indent=2)); Path('i2_checkpoint_manifest.json').write_text(json.dumps({'experiment':'I2','checkpoints':manifest},indent=2)); print(a.output)
if __name__=='__main__': main()
