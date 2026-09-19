"""Stage B3: exact 4-bit XOR discretization-consistency experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

# These truth-table batches are small; avoiding large BLAS thread pools makes
# the repeated diagnostic evaluations substantially faster and deterministic.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402

OUT = ROOT / "research" / "operator_results"
OPS = ["lehmer_p2", "hardmax", "probabilistic_or", "softmax_value_a16"]
MILESTONES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
THRESHOLDS = [.30, .40, .45, .50, .55, .60, .70]


def metrics(out, y):
    p = out >= .5; c = p == (y >= .5)
    return {"mse": float((out-y).square().mean()), "bit_accuracy": float(c.float().mean()), "exact_accuracy": float(c.all(dim=-1).float().mean())}


def evaluate(model, x, y, threshold=.5):
    with torch.no_grad():
        cont = model(x)
        hard = model.forward_hard(x)
        # ``to_discrete`` constructs a fresh Boolean model; place it on the
        # same device as the continuous model before evaluating CUDA inputs.
        discrete = model.to_discrete(threshold).to(x.device)
        boolean = discrete(x.bool()).float()
    return {"continuous": metrics(cont,y), "hard": metrics(hard,y), "boolean": metrics(boolean,y)}


def gate_stats(model):
    result=[]
    for i, layer in enumerate(model.expectation_layers):
        g=layer.effective_gate().detach(); b=layer.actual_bias().detach(); d=g*(1-g)
        result.append({"layer":i,"gate_mean":float(g.mean()),"gate_d":float(torch.minimum(g,1-g).mean()),
          "gate_margin_mean":float((g-.5).abs().mean()),"gate_margin_lt_01":float(((g-.5).abs()<.1).float().mean()),
          "gate_margin_lt_005":float(((g-.5).abs()<.05).float().mean()),"sigmoid_deriv_mean":float(d.mean()),
          "sigmoid_deriv_median":float(d.median()),"sigmoid_deriv_p05":float(torch.quantile(d,.05)),
          "sigmoid_deriv_p95":float(torch.quantile(d,.95)),"sat_lt_1e2":float((d<1e-2).float().mean()),
          "sat_lt_1e3":float((d<1e-3).float().mean()),"sat_lt_1e4":float((d<1e-4).float().mean()),
          "sat_lt_1e6":float((d<1e-6).float().mean()),"bias_d":float(torch.minimum(b,1-b).mean()),
          "bias_margin_mean":float((b-.5).abs().mean())})
    return result


def forward_modes(model, x, modes):
    # modes indexes expectation_layers: 1 means hard-max aggregation.
    idx=0
    h = model.stem.hard_forward(x) if modes[idx] else model.stem(x); idx += 1
    for block in model.blocks:
        h1 = block.layer1.hard_forward(h) if modes[idx] else block.layer1(h); idx += 1
        h2 = block.layer2.hard_forward(h1) if modes[idx] else block.layer2(h1); idx += 1
        h = h + h2 - 2*h*h2 if block.residual_enabled else h2
    return model.head.hard_forward(h) if modes[idx] else model.head(h)


def hardening_report(model, x, y):
    n=len(model.expectation_layers); result={}
    for i in range(n):
        modes=[False]*n; modes[i]=True
        result[f"layer_{i}_only"] = metrics(forward_modes(model,x,modes),y)
    modes=[]
    for i in range(n):
        modes.append(True)
        result[f"cumulative_{i}"] = metrics(forward_modes(model,x,modes+[False]*(n-len(modes))),y)
    return result


def threshold_report(model,x,y):
    return {str(t): evaluate(model,x,y,t)["boolean"] for t in THRESHOLDS}


def lehmer_gradient_report(model,x):
    # Measure dF/dv on each layer at the best checkpoint, using the actual
    # activation entering that layer and detached upstream computation.
    reports=[]; h=x.detach(); idx=0
    layers=[]
    layers.append((model.stem,h.detach())); h=model.stem(h); idx+=1
    for block in model.blocks:
        layers.append((block.layer1,h.detach())); h=block.layer1(h); idx+=1
        layers.append((block.layer2,h.detach())); h2=block.layer2(h); h=h+h2-2*h*h2; idx+=1
    layers.append((model.head,h.detach()))
    for i,(layer,inp) in enumerate(layers):
        v=layer.contributions(inp).detach().requires_grad_(True); f=layer.or_operator(v); grad=torch.autograd.grad(f.sum(),v)[0]
        reports.append({"layer":i,"fraction_negative":float((grad<0).float().mean()),"mean_abs_negative":float(grad[grad<0].abs().mean()) if (grad<0).any() else 0.0,"mean_abs_positive":float(grad[grad>0].abs().mean()) if (grad>0).any() else 0.0,"fraction_near_zero":float((grad.abs()<1e-8).float().mean())})
    return reports


def run(epochs=3000, seeds=(0,1,2), device=None, export_dir=None):
    task=build_task("bitwise_xor_truth_table",{"bits":4});
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x,y=task["X"].to(device),task["Y"].to(device)
    all_results=[]
    export_dir = Path(export_dir) if export_dir is not None else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
    for op in OPS:
        for seed in seeds:
            torch.manual_seed(seed)
            model=SigmoidOrModernLogicGateNet(8,4,width=64,num_residual_blocks=2,or_operator=op,
                bias_initialization=lambda t: nn.init.normal_(t,mean=.5,std=.1)).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.01)
            best_cont=None; best_hard=None; best_bool=None; milestones={}; trajectory=[]; first_bool=None; lowest_wrong=None
            checkpoint_manifest=[]

            def save_verified(label, epoch, record):
                if export_dir is None:
                    return
                path = export_dir / f"B3R_{op}_seed{seed}_{label}.pt"
                torch.save(model.state_dict(), path)
                saved = torch.load(path, map_location=device, weights_only=True)
                check = SigmoidOrModernLogicGateNet(8, 4, width=64, num_residual_blocks=2,
                    or_operator=op, bias_initialization=lambda t: nn.init.normal_(t,mean=.5,std=.1)).to(device)
                check.load_state_dict(saved); check.eval()
                verify = evaluate(check, x, y)
                for mode in ("continuous", "hard", "boolean"):
                    for metric in ("mse", "bit_accuracy", "exact_accuracy"):
                        expected = record[mode][metric]
                        actual = verify[mode][metric]
                        if abs(actual - expected) > 1e-6:
                            raise RuntimeError(f"checkpoint verification failed for {path}: {mode}.{metric} {actual} != {expected}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checkpoint_manifest.append({"checkpoint": str(path), "sha256": digest, "epoch": epoch, "metrics": verify})
            # The Kaggle package contains source files only, so recreate the
            # result/checkpoint directory tree at runtime when needed.
            ckdir=OUT/"stage_b3_checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
            for epoch in range(epochs+1):
                out=model(x); loss=(out-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
                # Full hard/Boolean conversion and gate diagnostics are sampled
                # every 25 epochs; training itself remains one full-table batch
                # per epoch. Milestone checkpoints are therefore reproducible
                # diagnostic crossings at the sampling cadence.
                if epoch != 0 and epoch % 25 != 0 and epoch != epochs:
                    if epoch==epochs: break
                    continue
                ev=evaluate(model,x,y); rec={"epoch":epoch,**ev,"gate_stats":gate_stats(model),"parameter_grad_norm":float(torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])))}
                trajectory.append(rec)
                mse=ev["continuous"]["mse"]
                for m in MILESTONES:
                    if str(m) not in milestones and mse<m:
                        milestones[str(m)]=copy.deepcopy(rec); torch.save(model.state_dict(),ckdir/f"B3_{op}_seed{seed}_mse{m:.0e}.pt")
                        save_verified(f"mse{m:.0e}", epoch, rec)
                if first_bool is None and ev["boolean"]["exact_accuracy"]==1.0: first_bool={"epoch":epoch,"mse":mse}
                if first_bool is not None and first_bool["epoch"] == epoch:
                    save_verified("first_boolean_exact", epoch, rec)
                if ev["boolean"]["exact_accuracy"]<1.0 and (lowest_wrong is None or mse<lowest_wrong["mse"]): lowest_wrong={"epoch":epoch,"mse":mse,"boolean":ev["boolean"]}
                if best_cont is None or mse<best_cont["continuous"]["mse"]: best_cont=copy.deepcopy(rec); torch.save(model.state_dict(),ckdir/f"B3_{op}_seed{seed}_best_continuous.pt"); save_verified("best_continuous_mse", epoch, rec)
                if best_hard is None or ev["hard"]["exact_accuracy"]>best_hard["hard"]["exact_accuracy"] or (ev["hard"]["exact_accuracy"]==best_hard["hard"]["exact_accuracy"] and mse<best_hard["continuous"]["mse"]): best_hard=copy.deepcopy(rec); torch.save(model.state_dict(),ckdir/f"B3_{op}_seed{seed}_best_hard.pt"); save_verified("best_hard_exact", epoch, rec)
                if best_bool is None or ev["boolean"]["exact_accuracy"]>best_bool["boolean"]["exact_accuracy"] or (ev["boolean"]["exact_accuracy"]==best_bool["boolean"]["exact_accuracy"] and mse<best_bool["continuous"]["mse"]): best_bool=copy.deepcopy(rec); torch.save(model.state_dict(),ckdir/f"B3_{op}_seed{seed}_best_boolean.pt"); save_verified("best_boolean_exact", epoch, rec)
                if epoch==epochs: break
            save_verified("final", epochs, trajectory[-1])
            model.load_state_dict(torch.load(ckdir/f"B3_{op}_seed{seed}_best_continuous.pt",weights_only=True))
            best_cont.update({"threshold_stability":threshold_report(model,x,y),"hardening":hardening_report(model,x,y)})
            if op in ("lehmer_p2","probabilistic_or"): best_cont["operator_gradient_report"]=lehmer_gradient_report(model,x) if op=="lehmer_p2" else []
            all_results.append({"operator":op,"seed":seed,"epochs":epochs,"best_continuous":best_cont,"best_hard":best_hard,"best_boolean":best_bool,"first_boolean_recovery":first_bool,"lowest_mse_while_boolean_wrong":lowest_wrong,"milestones":milestones,"trajectory":trajectory,"checkpoint_manifest":checkpoint_manifest})
            print(op,seed,best_cont["continuous"]["mse"],best_cont["boolean"]["exact_accuracy"],first_bool,flush=True)
    return all_results


def main():
    import argparse
    global OPS
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=3000); p.add_argument("--operator",choices=OPS); p.add_argument("--seed",type=int); p.add_argument("--device",default=None); args=p.parse_args(); OUT.mkdir(exist_ok=True)
    if args.operator:
        old_ops=OPS; OPS=[args.operator]
    seeds=(args.seed,) if args.seed is not None else (0,1,2)
    results=run(args.epochs,seeds,args.device)
    path=OUT/(f"stage_b3_{args.operator}_seed{args.seed}.json" if args.operator and args.seed is not None else "stage_b3_results.json")
    with path.open("w") as f: json.dump({"experiment":"B3-exact-4bit-XOR","operators":OPS,"seeds":list(seeds),"epochs":args.epochs,"architecture":{"input":8,"width":64,"blocks":2,"output":4},"results":results},f,indent=2)
    print(json.dumps({"output":str(path),"runs":len(results)},indent=2))


if __name__=="__main__": main()
