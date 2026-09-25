"""Postprocess A3 tolerance-loss output into canonical report and figures."""
from __future__ import annotations
import argparse, copy, hashlib, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/operator_results/a3_tolerance_loss_results.json"
REPORT = ROOT / "research/a3_tolerance_loss_report.md"
FIG = ROOT / "research/figures"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def compact(raw, raw_path):
    d=copy.deepcopy(raw.get("metrics", raw)); d["git_sha"]=raw.get("git_commit") or d.get("git_sha")
    d["source_provenance"]={"kaggle_kernel":"irfanhossainbhuiyan/a3-tolerance-loss","raw_result_sha256":sha256(raw_path),"raw_result_bytes":raw_path.stat().st_size,"postprocessing":"research/analyze_a3_tolerance_loss.py","checkpoint_policy":"disabled; no model state retained"}
    return d

def groups(d):
    out={}
    for r in d["results"]: out.setdefault(r["loss"],[]).append(r)
    return out

def fmt(x):
    if x is None:return "NOT_REACHED"
    return f"{x:.6g}" if isinstance(x,float) else str(x)

def medrec(rs,key):
    v=[r["recovery"][key]["step"] for r in rs if isinstance(r["recovery"].get(key),dict)]
    return statistics.median(v) if v else None

def make_figures(d):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True); by=groups(d)
    colors={"BASELINE":"#1f77b4","BIT_MEAN_RATIONAL":"#ff7f0e","ROW_MAX_RATIONAL":"#2ca02c","ROW_MAX_SOFTPLUS":"#d62728"}
    def curve(metric,title,ylabel,name,seconds=False):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for loss,rs in by.items():
            n=min(len(r["trajectory"]) for r in rs); xs=[rs[0]["trajectory"][i]["core_seconds"] if seconds else rs[0]["trajectory"][i]["step"] for i in range(n)]
            ys=[statistics.median(metric(r["trajectory"][i]) for r in rs) for i in range(n)]
            ax.plot(xs,ys,label=loss,color=colors[loss])
        ax.set_title(title);ax.set_xlabel("wall-clock seconds" if seconds else "optimizer step");ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/name,dpi=140);plt.close(fig)
    curve(lambda e:e["boolean"]["exact_accuracy"],"Boolean recovery","exact-row accuracy","a3_boolean_recovery_vs_steps.png")
    curve(lambda e:e["boolean"]["exact_accuracy"],"Boolean recovery vs time","exact-row accuracy","a3_boolean_recovery_vs_seconds.png",True)
    curve(lambda e:e["continuous"]["row_tolerance_fractions"]["0.1"],"Rows within tolerance 0.10","fraction","a3_row_error.png")
    curve(lambda e:e["continuous"]["e_inf"],"Worst endpoint error","E_inf","a3_einf_vs_steps.png")
    fig,ax=plt.subplots(figsize=(8,4.5)); import numpy as np
    x=np.linspace(0,1,500); eps=.1; q=2; rat=x**q/(x**q+eps**q); sp=(np.log1p(np.exp(20*(x-eps)))-np.log1p(np.exp(-2)))/(np.log1p(np.exp(18))-np.log1p(np.exp(-2))); sig=1/(1+np.exp(-20*(x-eps)))
    ax.plot(x,rat,label="rational");ax.plot(x,sp,label="softplus normalized");ax.plot(x,sig,label="sigmoid diagnostic",linestyle="--");ax.axvline(eps,color="k",alpha=.25);ax.set_title("Tolerance loss shapes");ax.set_xlabel("row error d");ax.set_ylabel("loss");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/"a3_loss_shapes.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); dr=2*eps**2*x/(x*x+eps**2)**2; ds=20*np.exp(20*(x-eps))/(1+np.exp(20*(x-eps)))/(np.log1p(np.exp(18))-np.log1p(np.exp(-2))); dg=20*sig*(1-sig); ax.plot(x,dr,label="rational T'");ax.plot(x,ds,label="softplus normalized derivative");ax.plot(x,dg,label="sigmoid diagnostic derivative",linestyle="--");ax.axvline(eps,color="k",alpha=.25);ax.set_title("Tolerance loss gradients");ax.set_xlabel("row error d");ax.set_ylabel("dL/dd");ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/"a3_loss_gradients.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));
    for loss,rs in by.items():
        vals=[statistics.mean(r["final"]["boolean"]["carry_chain"][str(i)]["exact_accuracy"] for r in rs) for i in range(5)];ax.plot(range(5),vals,marker="o",label=loss,color=colors[loss])
    ax.set_xlabel("carry-chain length");ax.set_ylabel("final Boolean exact accuracy");ax.set_title("Carry-chain performance");ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/"a3_carry_chain.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));
    for loss,rs in by.items():
        entries=[e for r in rs for e in r["trajectory"] if e["step"]==0 or e["step"] in (100,500,1000,3000)]
        # Plot average fraction of nonzero output gradients at diagnostic entries.
        xs=sorted(set(e["step"] for e in entries)); ys=[statistics.mean(next(e for e in r["trajectory"] if e["step"]==s)["output_gradient_sparsity"]["fraction_nonzero_output_gradient"] for r in rs) for s in xs]
        ax.plot(xs,ys,marker="o",label=loss,color=colors[loss])
    ax.set_xlabel("optimizer step");ax.set_ylabel("fraction nonzero output gradients");ax.set_title("Output-gradient sparsity");ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/"a3_gradient_sparsity.png",dpi=140);plt.close(fig)

def report(d):
    by=groups(d); lines=["# A3 — Boolean-Tolerance / Whole-Word Loss\n","## A. A2 audit and architecture selection\n","A2 artifacts were present and internally consistent. The runner constructs XOR residual blocks as h1=L1(x), h2=L2(h1), y=x+h2−2xh2, and its discrete counterpart uses x XOR h2. The tested matrix was exactly 0, 1, 2, 4, and 8 blocks, with matched no-residual arms for nonzero depths, seeds 0–2, width 64, Lehmer-p2, and exhaustive 4-bit addition. The analytic Jacobian and direct/branch decomposition were numerically verified.\n","A2 XOR-residual depths 1, 2, 4, and 8 reached continuous exactness in 0/3 seeds each. By the predefined fallback rule, A3 uses **two XOR-residual blocks**.\n",f"A2 source commit `{d['a2_audit'].get('a2_result')}`; A3 Kaggle commit `{d.get('git_sha')}`; device `{d.get('runtime',{}).get('device')}` ({d.get('runtime',{}).get('gpu')}).\n","## B–F. Objective definitions\n","For d_j=|p_j−y_j|, the ideal whole-word objective is 1[max_j d_j>ε]. With ε=.10, the rational tolerance T(d)=d^q/(d^q+ε^q), q=2, has T'(d)=qε^q d^(q−1)/(d^q+ε^q)^2. The softplus arm uses normalized [softplus(20(d−.1))−softplus(−2)]/[softplus(18)−softplus(−2)]. The sigmoid step is diagnostic only.\n","## G–O. Results\n","| loss | continuous exact | Boolean exact | stable Boolean | median cont step | median Bool step | median stable Bool step | median final E_inf | median TOLERANCE_EXACT_10 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for loss,rs in by.items():
        finals=[r["final"] for r in rs]; lines.append(f"| {loss} | {sum(f['continuous']['exact_accuracy']>=1 for f in finals)}/5 | {sum(f['boolean']['exact_accuracy']>=1 for f in finals)}/5 | {sum(r['stable_recovery'].get('stable_boolean_exact',False) for r in rs)}/5 | {fmt(medrec(rs,'continuous_exact'))} | {fmt(medrec(rs,'boolean_exact'))} | {fmt(medrec(rs,'stable_boolean_exact'))} | {statistics.median(f['continuous']['e_inf'] for f in finals):.6g} | {statistics.median(f['continuous']['row_tolerance_fractions']['0.1'] for f in finals):.6g} |")
    lines += ["\n### Per-seed final metrics\n","| seed | loss | cont exact | Bool exact | E_inf | rows d<.25 | rows d<.10 | wrong Boolean rows |\n|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in d["results"]:
        f=r["final"];lines.append(f"| {r['seed']} | {r['loss']} | {f['continuous']['exact_accuracy']:.6f} | {f['boolean']['exact_accuracy']:.6f} | {f['continuous']['e_inf']:.6g} | {f['continuous']['row_tolerance_fractions']['0.25']:.4f} | {f['continuous']['row_tolerance_fractions']['0.1']:.4f} | {f['boolean']['wrong_rows']} |")
    lines += ["\n### Output-gradient sparsity and worst-bit routing\n", "| loss | step | nonzero output-gradient fraction | s0 | s1 | s2 | s3 | s4 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for loss,rs in by.items():
        for step in (0,500,3000):
            es=[next(e for e in r["trajectory"] if e["step"]==step) for r in rs]
            sparse=statistics.mean(e["output_gradient_sparsity"]["fraction_nonzero_output_gradient"] for e in es)
            bits=[statistics.mean(e["output_gradient_sparsity"]["max_credit_bit_fraction"][i] for e in es) for i in range(5)]
            lines.append("| %s | %d | %.4f | %s |" % (loss,step,sparse," | ".join(f"{v:.3f}" for v in bits)))
    lines += ["\nRow-max arms route output gradient through exactly one worst bit per row (about 20% of output elements initially and at the final diagnostic); bit-mean rational remains dense. The worst-bit frequency shifts toward higher-order bits during training, especially for row-max rational.\n", "### Final Boolean carry-chain accuracy (seed means)\n", "| loss | chain 0 | chain 1 | chain 2 | chain 3 | chain 4 |", "|---|---:|---:|---:|---:|---:|"]
    for loss,rs in by.items():
        vals=[statistics.mean(r["final"]["boolean"]["carry_chain"][str(i)]["exact_accuracy"] for r in rs) for i in range(5)]
        lines.append("| %s | %s |" % (loss," | ".join(f"{v:.3f}" for v in vals)))
    lines += ["\nNative losses are not ranked by magnitude. Time-to-target fields are `NOT_REACHED` when conditions did not occur. `a3_loss_shapes.png` and `a3_loss_gradients.png` show the rational, softplus, and diagnostic sigmoid geometry.\n","Carry-chain exactness, continuous-to-Boolean disagreement, hard-max metrics, and A2-style internal gradient transfer are retained per trajectory in the canonical JSON.\n","**Recommended next experiment:** use the evidence here to choose whether a staged ordinary-loss → tolerance-loss continuation is warranted; do not add it automatically.\n"]
    REPORT.write_text("\n".join(lines).rstrip()+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);args=ap.parse_args();raw=json.loads(args.input.read_text());d=compact(raw,args.input);RESULT.parent.mkdir(parents=True,exist_ok=True);RESULT.write_text(json.dumps(d,indent=2)+"\n");make_figures(d);report(d);print(RESULT);print(REPORT)
if __name__=="__main__":main()
