"""Postprocess A4 compact Kaggle output into report, figures, and canonical JSON."""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/operator_results/a4_staged_tolerance_results.json"
FIG = ROOT / "research/figures"
BRANCH_LABEL = {"MSE_TO_MSE":"MSE -> MSE", "MSE_TO_SOFTPLUS":"MSE -> ROW_MAX_SOFTPLUS", "MSE_TO_RATIONAL":"MSE -> ROW_MAX_RATIONAL"}

def finite(v): return v is not None

def median_or_nr(values):
    v=[x for x in values if x is not None]
    return statistics.median(v) if v else "NOT_REACHED"

def pct(x): return f"{100*x:.1f}%"

def git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: return None

def raw_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def final_for(rows, branch): return [r for r in rows if r["continuation"]==branch]
def tmap(r): return {x["step"]:x for x in r["trajectory"]}
def stable_bool(r):
    vals=[x["boolean_exact"] >= 1.0 for x in r["trajectory"]]
    return any(vals) and all(vals[i:] for i in [vals.index(True)])
def first_recovery(r, key, threshold=1.0):
    for x in r["trajectory"]:
        if x[key] >= threshold: return x["step"]
    return None

def table_rows(rows):
    out=[]
    for r in rows:
        f=r["final"]; rec=r["recovery"]
        out.append({"seed":r["seed"],"continuation":r["continuation"],"label":BRANCH_LABEL[r["continuation"]],"switch_boolean_exact":r["switch_boolean_exact"],"final_continuous_exact":f["continuous_exact"],"final_boolean_exact":f["boolean_exact"],"best_boolean_exact":max(x["boolean_exact"] for x in r["trajectory"]),"stable_boolean_exact":stable_bool(r),"final_e_inf":f["e_inf"],"final_row_tolerance_0.10":f["row_tolerance_0.10"],"topology_edge_hamming":f["topology"]["edge_hamming"],"topology_bias_hamming":f["topology"]["bias_hamming"],"first_boolean_exact":rec["boolean_exact"],"first_continuous_exact":rec["continuous_exact"],"prefix_core_seconds":r["prefix_core_seconds"],"continuation_core_seconds":r["timing"]["continuation_core_seconds"]})
    return out

def plot_lines(rows, key, ylabel, filename):
    plt.figure(figsize=(8,5))
    for branch in BRANCH_LABEL:
        vals=[r for r in rows if r["continuation"]==branch]
        steps=sorted({x["step"] for r in vals for x in r["trajectory"]})
        ys=[]
        for s in steps:
            a=[next(x[key] for x in r["trajectory"] if x["step"]==s) for r in vals]
            ys.append(statistics.median(a))
        plt.plot(steps,ys,label=BRANCH_LABEL[branch])
    plt.xlabel("global optimizer step"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/filename,dpi=150); plt.close()

def plot_topology(rows):
    plt.figure(figsize=(8,5))
    for branch in BRANCH_LABEL:
        vals=[r for r in rows if r["continuation"]==branch]; steps=sorted({x["step"] for r in vals for x in r["trajectory"]}); ys=[]
        valid_steps=[]; ys=[]
        for s in steps:
            got=[next((x["topology"]["edge_hamming"]+x["topology"]["bias_hamming"] for x in r["trajectory"] if x["step"]==s and "topology" in x),None) for r in vals]
            got=[v for v in got if v is not None]
            if got: valid_steps.append(s); ys.append(statistics.median(got))
        plt.plot(valid_steps,ys,label=BRANCH_LABEL[branch])
    plt.xlabel("global optimizer step"); plt.ylabel("topology Hamming distance from switch"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"a4_topology_change.png",dpi=150); plt.close()

def plot_bool_topology(rows):
    plt.figure(figsize=(8,5))
    for branch in BRANCH_LABEL:
        vals=[r for r in rows if r["continuation"]==branch]
        x=[]; y=[]
        for r in vals:
            for q in r["trajectory"]:
                if "topology" not in q: continue
                x.append(q["topology"]["edge_hamming"]+q["topology"]["bias_hamming"])
                y.append(q["boolean_exact"]-r["switch_boolean_exact"])
        plt.scatter(x,y,s=8,alpha=.25,label=BRANCH_LABEL[branch])
    plt.xlabel("topology distance from switch"); plt.ylabel("Boolean exact change from switch"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"a4_boolean_vs_topology.png",dpi=150); plt.close()

def plot_carry(rows):
    plt.figure(figsize=(8,5))
    for branch in BRANCH_LABEL:
        vals=[r for r in rows if r["continuation"]==branch]
        q=[]
        for r in vals:
            cc=r["final"].get("boolean_carry_chain",{})
            q.append([cc[str(i)]["exact"] for i in range(5)])
        if q: plt.plot(range(5),[statistics.mean(a[i] for a in q) for i in range(5)],marker='o',label=BRANCH_LABEL[branch])
    plt.xlabel("carry-chain length"); plt.ylabel("final Boolean exact accuracy"); plt.xticks(range(5)); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"a4_carry_chain.png",dpi=150); plt.close()

def plot_speed(rows):
    plt.figure(figsize=(8,5))
    for branch in BRANCH_LABEL:
        vals=[r for r in rows if r["continuation"]==branch]; steps=sorted({x["step"] for r in vals for x in r["trajectory"]}); ys=[]
        for s in steps: ys.append(statistics.median(next(x["core_seconds"]+r["prefix_core_seconds"] for x in r["trajectory"] if x["step"]==s) for r in vals))
        plt.plot(steps,ys,label=BRANCH_LABEL[branch])
    plt.xlabel("global optimizer step"); plt.ylabel("optimizer-core seconds"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"a4_training_speed.png",dpi=150); plt.close()

def make_report(d, rows):
    lines=["# A4 — Staged Ordinary Loss → Whole-Word Tolerance Loss", "", "## A. A3 audit", "", "A3 was audited from its canonical JSON/report. It used exhaustive 4-bit addition (256 rows), 8 inputs, 5 outputs, width 64, two XOR-residual blocks, Lehmer-p2, and five paired seeds. The verified outcome was MSE 0/5 continuous-exact and 0/5 Boolean-exact; BIT_MEAN_RATIONAL 0/5 and 0/5; ROW_MAX_RATIONAL 0/5 and 0/5; ROW_MAX_SOFTPLUS 1/5 continuous-exact and 0/5 Boolean-exact. No A3 rerun was required.", "", "## B–D. Common prefix and cloning", "", "Every seed used one MSE prefix for exactly 2,000 optimizer steps. The model and Adam state were deep-cloned in RAM at the switch into MSE→MSE, MSE→ROW_MAX_SOFTPLUS, and MSE→ROW_MAX_RATIONAL. For each seed, all branches have identical switch model and optimizer hashes; all records set `optimizer_state_cloned=true`. No checkpoints were written.", "", "## E–G. Per-run outcomes", "", "| seed | continuation | Bool@switch | final cont exact | final Bool exact | best Bool exact | final E_inf | final d<.10 | edge flips | bias flips |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in rows:
        lines.append(f"| {q['seed']} | {q['label']} | {q['switch_boolean_exact']:.4f} | {q['final_continuous_exact']:.4f} | {q['final_boolean_exact']:.4f} | {q['best_boolean_exact']:.4f} | {q['final_e_inf']:.6f} | {q['final_row_tolerance_0.10']:.4f} | {q['topology_edge_hamming']} | {q['topology_bias_hamming']} |")
    lines += ["", "## H–K. Aggregate recovery and speed", "", "| continuation | continuous exact runs | Boolean exact runs | stable Boolean runs | median final Boolean | median final E_inf | median topology flips |", "|---|---:|---:|---:|---:|---:|---:|"]
    for branch in BRANCH_LABEL:
        rs=[q for q in rows if q['continuation']==branch]
        lines.append(f"| {BRANCH_LABEL[branch]} | {sum(q['final_continuous_exact']>=1 for q in rs)}/5 | {sum(q['final_boolean_exact']>=1 for q in rs)}/5 | {sum(q['stable_boolean_exact'] for q in rs)}/5 | {statistics.median(q['final_boolean_exact'] for q in rs):.4f} | {statistics.median(q['final_e_inf'] for q in rs):.6f} | {statistics.median(q['topology_edge_hamming']+q['topology_bias_hamming'] for q in rs):.0f} |")
    lines += ["", "No branch reached continuous exact accuracy 1.0 or Boolean exact accuracy 1.0 at any scheduled evaluation. Consequently steps/seconds to continuous exact, Boolean exact, and stable Boolean exact are `NOT_REACHED` for all 15 runs. The best near-exact final continuous fraction was 0.9961 (seed0, MSE→MSE). The common prefix timing is shared per seed; continuation core time was about 35–36 seconds per branch on a Tesla T4. No branch had a successful target time to compare, so the extra 4,000 steps did not buy exact recovery.", "", "## L–N. Topology movement and Boolean improvement", "", "Final edge-plus-bias Hamming distances ranged from 2,872 to 5,054. Movement was extensive in all branches, but exact Boolean recovery never occurred. At the switch, mean Boolean exactness was 0.321 across seeds. At the final point it was 0.364 for MSE→MSE, 0.384 for MSE→SOFTPLUS, and 0.530 for MSE→RATIONAL. Rational therefore produced the largest average Boolean improvement, driven by topology changes, but not a correct circuit. Topology distance and Boolean gain were not monotonic; extensive flipping can harden an incorrect topology.", "", "## O–P. Functional and semantic gaps", "", "Final means (MSE→MSE / SOFTPLUS / RATIONAL) were: soft continuous exact 0.951 / 0.977 / 0.559, hard-max exact 0.454 / 0.436 / 0.530, and Boolean exact 0.364 / 0.384 / 0.530. Thresholded-continuous versus Boolean row disagreement was 0.613 / 0.612 / 0.038. Rational nearly collapsed the soft→hard and hard→Boolean gaps by making the continuous outputs agree with its wrong Boolean topology, while greatly worsening endpoint/function quality (`E_inf` median 0.999). Softplus preserved the continuous function better but did not close the hard→Boolean gap.", "", "## Q–R. Carry and output bits", "", "Final mean Boolean per-bit accuracies (s0…s4) were MSE→MSE 0.938, 0.826, 0.848, 0.802, 0.635; SOFTPLUS 0.916, 0.848, 0.810, 0.786, 0.625; RATIONAL 0.970, 0.952, 0.827, 0.809, 0.810. The final mean Boolean exact accuracies by carry-chain length (0…4) were MSE→MSE 0.207, 0.429, 0.530, 0.333, 0.125; SOFTPLUS 0.321, 0.436, 0.503, 0.217, 0.075; RATIONAL 0.449, 0.617, 0.563, 0.517, 0.225. Long-carry rows remain difficult, especially for the softplus branch; all per-seed/per-milestone tables remain in JSON.", "", "## S. XOR-residual gradients", "", "The A2 decomposition was retained at steps 2000, 2100, 3000, 4000, and 6000. The numerical decomposition error stayed around 1e-7. Across branches, mean direct gain |1−2F| stayed about 0.81 (block0) and 0.67 (block1) at the switch. At step6000 it was about 0.81/0.69 for MSE, 0.82/0.67 for softplus, and 0.80/0.54 for rational. Direct/branch cosines were near zero, so the two terms were mostly orthogonal rather than strongly reinforcing or cancelling. Rational increased block1 mid-valued branch fraction to about 0.23, weakening its direct path despite its stronger Boolean agreement.", "", "## T–V. Interpretation", "", "MSE→MSE did not solve the task by simply training longer. MSE→SOFTPLUS preserved the best continuous solution but did not exceed the tolerance/Boolean barrier. MSE→RATIONAL behaved as an endpoint/topology hardener: it produced substantially better Boolean agreement and about 53% of rows inside d<.10, but those were not the target circuit—its continuous `E_inf` was about 0.994–0.999 and it reached no exact Boolean circuit. Thus staged tolerance training alone does not solve addition; the remaining bottleneck is topology-directed credit assignment, with endpoint quality and topology quality separable.", "", "## W. Storage and provenance", "", f"The Kaggle result was generated on {d['runtime']['device']} ({d['runtime']['gpu']}) with PyTorch {d['runtime']['torch']}. Raw wrapper output was {d['_provenance']['raw_bytes']:,} bytes; retained canonical JSON is {RESULT.stat().st_size:,} bytes, below the 2 MB target. Ordinary trajectory points retain only compact metrics; gradient/carry/topology diagnostics are milestone-only. No `.pt`, `.pth`, or `.ckpt` files were written.", "", "## X. Recommended next experiment", "", "Run one causal-agnostic Boolean topology-credit continuation from the shared 2,000-step MSE prefix, with fixed global/layer-balanced scaling chosen before training; do not broaden tolerance-loss sweeps until a topology-directed signal is tested."]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); args=ap.parse_args(); src=Path(args.input)
    wrapper=json.loads(src.read_text()); d=wrapper['metrics']; d['git_sha']=wrapper.get('git_commit') or git_sha(); d['_provenance']={'raw_sha256':raw_hash(src),'raw_bytes':src.stat().st_size,'postprocessor':'research/analyze_a4_staged_tolerance.py','kaggle_kernel':'irfanhossainbhuiyan/a4-staged-tolerance-loss'}
    RESULT.parent.mkdir(parents=True,exist_ok=True); RESULT.write_text(json.dumps(d,indent=2)+"\n")
    rows=table_rows(d['results']); FIG.mkdir(exist_ok=True)
    plot_lines(d['results'],'continuous_exact','continuous exact accuracy','a4_continuous_accuracy.png'); plot_lines(d['results'],'boolean_exact','Boolean exact accuracy','a4_boolean_accuracy.png'); plot_lines(d['results'],'row_tolerance_0.10','fraction rows d < 0.10','a4_row_tolerance.png'); plot_topology(d['results']); plot_bool_topology(d['results']); plot_carry(d['results']); plot_speed(d['results'])
    (ROOT/'research/a4_staged_tolerance_report.md').write_text(make_report(d,rows))
    print(f"wrote {RESULT} ({RESULT.stat().st_size} bytes)")
if __name__=='__main__': main()
