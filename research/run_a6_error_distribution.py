"""A6: error-distribution polarized MSE on exhaustive 4-bit addition."""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, statistics, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from research.boolean_tasks import build_task
from research.run_a3_tolerance_loss import make_model, evaluate, forward_trace

SEEDS = (0, 1, 2, 3, 4)
STEPS = 6000
SWITCH = 2000
LAMBDA = 0.5
ARMS = ("MSE_CONTROL", "EP_MSE", "MSE_WARMUP_EP")
EVAL_STEPS = {0, *range(10, 501, 10), *range(525, STEPS + 1, 25), STEPS}
DIAG_STEPS = {0, 2000, 3000, 4000, 6000}


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); value = state[key]
        h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def masks(model):
    edges, biases = [], []
    for layer in model.expectation_layers:
        edges.append((layer.effective_gate().detach() >= .5).flatten().cpu())
        biases.append((layer.actual_bias().detach() >= .5).flatten().cpu())
    return {"edge": torch.cat(edges), "bias": torch.cat(biases)}


def topology(cur, ref):
    e = cur["edge"] != ref["edge"]; b = cur["bias"] != ref["bias"]
    return {"edge_hamming": int(e.sum()), "bias_hamming": int(b.sum()),
            "edge_0_to_1": int((~ref["edge"] & cur["edge"]).sum()),
            "edge_1_to_0": int((ref["edge"] & ~cur["edge"]).sum()),
            "bias_0_to_1": int((~ref["bias"] & cur["bias"]).sum()),
            "bias_1_to_0": int((ref["bias"] & ~cur["bias"]).sum())}


def loss_parts(output, target):
    row_mse = (output - target).square().mean(dim=1)
    mean = row_mse.mean()
    polarization = (row_mse * (1.0 - row_mse)).mean()
    return mean, polarization, row_mse


def loss_value(output, target, kind):
    mean, pol, _ = loss_parts(output, target)
    if kind == "MSE": return mean
    if kind == "EP_MSE": return mean + LAMBDA * pol
    raise ValueError(kind)


def active_kind(arm, step):
    return "MSE" if arm == "MSE_CONTROL" or (arm == "MSE_WARMUP_EP" and step <= SWITCH) else "EP_MSE"


def error_stats(row_mse):
    e = row_mse.detach()
    q = torch.quantile(e, torch.tensor([.5, .9, .95, .99], device=e.device))
    hist = torch.histc(e, bins=20, min=0, max=1)
    return {"mean": float(e.mean()), "variance": float(e.var(unbiased=False)),
            "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
            "p99": float(q[3]), "max": float(e.max()),
            "fractions": {"lt_1e-4": float((e < 1e-4).float().mean()),
                          "lt_1e-3": float((e < 1e-3).float().mean()),
                          "lt_0.01": float((e < .01).float().mean()),
                          "lt_0.05": float((e < .05).float().mean()),
                          "gt_0.25": float((e > .25).float().mean()),
                          "gt_0.50": float((e > .5).float().mean())},
            "histogram": {"bins": 20, "min": 0.0, "max": 1.0,
                          "counts": [int(v) for v in hist.cpu()]}}


def _grad_for(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)]


def _cos(a, b):
    aa = torch.cat([x.reshape(-1) for x in a]); bb = torch.cat([x.reshape(-1) for x in b])
    return float(F.cosine_similarity(aa[None], bb[None]).item()) if aa.norm() and bb.norm() else 0.0


def component_gradients(model, x, y, kind):
    out = model(x); mean, pol, _ = loss_parts(out, y)
    params = [p for p in model.parameters() if p.requires_grad]
    gm = _grad_for(mean, params); gp = _grad_for(pol, params)
    layers = []
    for i, layer in enumerate(model.expectation_layers):
        # Parameter order follows model.parameters; select by identity.
        def one(gs, p):
            for q, g in zip(params, gs):
                if q is p: return g
            return torch.zeros_like(p)
        m = one(gm, layer.raw_edge); p = one(gp, layer.raw_edge)
        layers.append({"layer": i, "mean_l2": float(m.norm()), "polarization_l2": float(p.norm()), "cosine": _cos([m], [p])})
    return {"active_loss": kind, "mean_loss": float(mean.detach()), "polarization_loss": float(pol.detach()),
            "mean_raw_edge_l2": float(torch.cat([g.reshape(-1) for i,g in enumerate(gm) if params[i] is next((l.raw_edge for l in model.expectation_layers), None)]).norm()) if False else float(torch.cat([one(gm,l.raw_edge).reshape(-1) for l in model.expectation_layers]).norm()),
            "polarization_raw_edge_l2": float(torch.cat([one(gp,l.raw_edge).reshape(-1) for l in model.expectation_layers]).norm()),
            "cosine": _cos([one(gm,l.raw_edge) for l in model.expectation_layers], [one(gp,l.raw_edge) for l in model.expectation_layers]),
            "layers": layers}


def residual_gradient_diagnostics(model, x, y, kind):
    out, traces = forward_trace(model, x)
    loss = loss_value(out, y, kind)
    records = []
    for i, xin, branch, yout in traces:
        gy = torch.autograd.grad(loss, yout, retain_graph=True)[0]
        gx = torch.autograd.grad(loss, xin, retain_graph=True)[0]
        direct = (1 - 2 * branch) * gy
        branch_grad = torch.autograd.grad(branch, xin, grad_outputs=(1 - 2 * xin) * gy, retain_graph=True)[0]
        records.append({"block": i, "grad_input_output_ratio": float(gx.norm() / gy.norm().clamp_min(1e-20)),
                        "mean_abs_input_output_ratio": float(gx.abs().mean() / gy.abs().mean().clamp_min(1e-20)),
                        "mean_abs_direct_gain": float((1 - 2 * branch).detach().abs().mean()),
                        "direct_l2": float(direct.norm()), "branch_l2": float(branch_grad.norm()),
                        "direct_over_total": float(direct.norm() / gx.norm().clamp_min(1e-20)),
                        "branch_over_total": float(branch_grad.norm() / gx.norm().clamp_min(1e-20)),
                        "direct_branch_cosine": _cos([direct], [branch_grad]),
                        "decomposition_relative_error": float((gx-direct-branch_grad).norm() / gx.norm().clamp_min(1e-20)),
                        "branch_mean_min": float(torch.minimum(branch, 1-branch).detach().mean()),
                        "branch_fraction_lt_0_1": float((branch.detach() < .1).float().mean()),
                        "branch_fraction_gt_0_9": float((branch.detach() > .9).float().mean()),
                        "branch_fraction_mid_0_4_0_6": float(((branch.detach() >= .4)&(branch.detach() <= .6)).float().mean())})
    return {"active_loss": kind, "blocks": records}


def record(model, x, y, chain, step, core, ref, arm, full=False):
    with torch.no_grad(): out = model(x); _, _, row_mse = loss_parts(out, y)
    ev = evaluate(model, x, y, chain); c = ev["continuous"]
    rec = {"step": step, "core_seconds": core, "active_loss": active_kind(arm, step),
           "native_loss": float(loss_value(out, y, active_kind(arm, step))),
           "continuous_exact": c["exact_accuracy"], "continuous_bit": c["bit_accuracy"],
           "hard_exact": ev["hard"]["exact_accuracy"], "boolean_exact": ev["boolean"]["exact_accuracy"],
           "e_inf": c["e_inf"], "mse": c["mse"], "mae": c["mae"],
           "bit_disagreement": ev["bit_disagreement_fraction"], "row_disagreement": ev["row_disagreement_fraction"],
           "mean_hamming": ev["mean_hamming_continuous_to_boolean"],
           "row_error": error_stats(row_mse), "topology": topology(masks(model), ref),
           "continuous_per_bit": c["per_bit_accuracy"], "boolean_per_bit": ev["boolean"]["per_bit_accuracy"]}
    if full:
        rec["continuous_carry_chain"] = {k: {"exact": v["exact_accuracy"], "bit": v["bit_accuracy"]} for k,v in c["carry_chain"].items()}
        rec["boolean_carry_chain"] = {k: {"exact": v["exact_accuracy"], "bit": v["bit_accuracy"]} for k,v in ev["boolean"]["carry_chain"].items()}
        rec["component_gradients"] = component_gradients(model, x, y, rec["active_loss"])
        rec["residual_gradients"] = residual_gradient_diagnostics(model, x, y, rec["active_loss"])
    return rec


def train(seed, initial, x, y, chain, device, arm):
    model = make_model(seed, device); model.load_state_dict(copy.deepcopy(initial))
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0)
    ref = masks(model); core = 0.; step_times = []; trajectory = []
    recovery = {k: None for k in ("continuous_exact","boolean_exact","stable_boolean_exact","e_inf_lt_0_25","e_inf_lt_0_10","e_inf_lt_0_05","e_inf_lt_0_01")}
    prior = {k: False for k in recovery}; regress = {k: 0 for k in recovery}
    def maybe(step):
        if step not in EVAL_STEPS: return
        full = step in DIAG_STEPS
        rec = record(model,x,y,chain,step,core,ref,arm,full); trajectory.append(rec)
        cond = {"continuous_exact": rec["continuous_exact"] >= 1., "boolean_exact": rec["boolean_exact"] >= 1.,
                "e_inf_lt_0_25": rec["e_inf"] < .25, "e_inf_lt_0_10": rec["e_inf"] < .1,
                "e_inf_lt_0_05": rec["e_inf"] < .05, "e_inf_lt_0_01": rec["e_inf"] < .01}
        for k,v in cond.items():
            if v and recovery[k] is None: recovery[k] = {"step": step, "core_seconds": core}
            if prior[k] and not v: regress[k] += 1
            prior[k] = v
    maybe(0)
    for step in range(1, STEPS+1):
        t0 = time.perf_counter(); opt.zero_grad(set_to_none=True)
        loss = loss_value(model(x), y, active_kind(arm, step)); loss.backward(); opt.step()
        dt = time.perf_counter()-t0; core += dt; step_times.append(dt); maybe(step)
    if recovery["boolean_exact"] is not None and regress["boolean_exact"] == 0:
        recovery["stable_boolean_exact"] = recovery["boolean_exact"]
    return {"seed": seed, "arm": arm, "initial_state_sha256": state_hash(initial),
            "trajectory": trajectory, "recovery": recovery, "regressions": regress,
            "stable_recovery": {k: recovery[k] is not None and regress.get(k,0)==0 for k in recovery},
            "final": trajectory[-1], "timing": {"optimizer_core_seconds": core,
            "optimizer_steps": STEPS, "examples_processed": STEPS*len(x),
            "total_wall_seconds": sum(step_times), "mean_ms_per_step": 1000*statistics.mean(step_times),
            "median_ms_per_step_after_warmup": 1000*statistics.median(step_times[10:] or step_times)},
            "checkpoint_policy": "disabled"}


def main():
    global STEPS
    ap=argparse.ArgumentParser(); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--output', default='research/operator_results/a6_error_distribution_results.json'); ap.add_argument('--seeds', default=','.join(map(str, SEEDS))); ap.add_argument('--steps', type=int, default=STEPS); args=ap.parse_args()
    STEPS = int(args.steps)
    selected_seeds = tuple(int(s) for s in args.seeds.split(',') if s.strip())
    device=torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    task=build_task('binary_addition', {'bits':4}); x=task['X'].float().to(device); y=task['Y'].float().to(device); chain=task['carry_chain_length'].to(device)
    results=[]; hashes={}
    for seed in selected_seeds:
        base=make_model(seed,device); init=snapshot(base); hashes[str(seed)]=state_hash(init)
        for arm in ARMS:
            run=train(seed,init,x,y,chain,device,arm); results.append(run); print(seed,arm,run['recovery'],flush=True)
    payload={'experiment':'A6-error-distribution-polarized-mse','git_sha':None,
      'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None},
      'capacity_gate':{'valid':True,'script':'research/capacity_a4.py','bit_accuracy':1.0,'exact_row_accuracy':1.0},
      'task':{'name':'binary_addition','bits':4,'rows':256,'target_generator_agreement':bool(torch.equal(task['target_integer'],task['target_ripple'])),'target_bit_frequency':[float(v) for v in y.mean(0)],'carry_chain_counts':{str(i):int((chain==i).sum()) for i in range(5)}},
      'architecture':{'input_dim':8,'width':64,'blocks':2,'residual_enabled':True,'output_dim':5,'operator':'lehmer_p2','initializer':'I2-B meanfield sigma2 + BIAS_ONE'},
      'losses':{'lambda':LAMBDA,'row_error':'mean((p-y)^2,dim=bits)','mse':'E[e_n]','ep_mse':'E[e_n + 0.5 e_n(1-e_n)]','arms':list(ARMS),'warmup_switch':SWITCH},
      'training':{'optimizer':'Adam','lr':.01,'weight_decay':0.0,'steps':STEPS,'batch_size':256,'evaluation_schedule':'every 10 through 500, then every 25','checkpoint_policy':'disabled'},
      'paired_initial_state_hashes':hashes,'results':results,'one_flip_oracle':{'status':'deferred','reason':'kept optional to preserve compact first A6 run; no training decisions use it'},'storage_policy':'compact trajectory and milestone diagnostics; no model checkpoints'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+'\n'); print(out)

if __name__=='__main__': main()
