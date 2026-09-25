"""Generate the M1 report and figures from the canonical Kaggle result."""
from __future__ import annotations
import json, statistics
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"research/operator_results/m1_mnist_pilot_results.json"
REPORT=ROOT/"research/m1_mnist_pilot_report.md"
FIG=ROOT/"research/figures"
LOSSES=["BCE","POWER_1_25","CURRICULUM"]
def mean_std(xs):
    return f"{statistics.mean(xs):.4f}" if len(xs)<2 else f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"
def run_rows(runs, checkpoint):
    rows=[]
    for r in runs:
        m=r["test_metrics"][checkpoint]; c,h,b=m["continuous"],m["hard"],m["boolean"]
        rows.append({"loss":r["loss"],"seed":r["seed"],"cont_top1":c["argmax_accuracy"],"cont_strict":c["threshold_strict_accuracy"],"hard_strict":h["threshold_strict_accuracy"],"bool_strict":b["threshold_strict_accuracy"],"bool_valid":b["valid_onehot_rate"],"gap":b.get("sample_disagreement_fraction",0.0),"cont_einf":c["e_inf"],"cont_endpoint":c["mean_endpoint_distance"],"cont_margin":c["mean_class_margin"],"bool_zero":b["zero_hot_rate"],"bool_multi":b["multi_hot_rate"]})
    return rows
def table(rows, columns):
    out=["| "+" | ".join(h for _,h,_ in columns)+" |","| "+" | ".join("---:" if typ=="num" else "---" for _,_,typ in columns)+" |"]
    for r in rows:
        vals=[]
        for key,_,typ in columns:
            v=r[key]; vals.append(f"{v:.4f}" if typ=="num" else str(v))
        out.append("| "+" | ".join(vals)+" |")
    return out
def make_figures(runs):
    FIG.mkdir(parents=True,exist_ok=True)
    for metric,filename,title,group in [("argmax_accuracy","m1_continuous_accuracy.png","MNIST continuous top-1 accuracy","continuous"),("threshold_strict_accuracy","m1_boolean_accuracy.png","MNIST exact Boolean strict accuracy","boolean")]:
        plt.figure(figsize=(8,5))
        for r in runs:
            plt.plot([z["epoch"] for z in r["trajectory"]],[z[group][metric] for z in r["trajectory"]],alpha=.65,label=f'{r["loss"]} s{r["seed"]}')
        plt.xlabel("epoch"); plt.ylabel(metric); plt.title(title); plt.legend(fontsize=6,ncol=3); plt.tight_layout(); plt.savefig(FIG/filename,dpi=160); plt.close()
    for key,filename,title,group in [("sample_disagreement_fraction","m1_continuous_boolean_gap.png","Continuous threshold to exact Boolean sample disagreement","boolean"),("mean_endpoint_distance","m1_endpoint_distance.png","Continuous endpoint distance","continuous")]:
        plt.figure(figsize=(8,5))
        for r in runs:
            plt.plot([z["epoch"] for z in r["trajectory"]],[z[group].get(key,0) for z in r["trajectory"]],alpha=.65,label=f'{r["loss"]} s{r["seed"]}')
        plt.xlabel("epoch"); plt.ylabel(key); plt.title(title); plt.legend(fontsize=6,ncol=3); plt.tight_layout(); plt.savefig(FIG/filename,dpi=160); plt.close()
    names=[]; values={}
    for r in runs:
        for x in r["trajectory"][-1].get("layer_trace_1024",[]):
            names.append(x["name"]); values.setdefault(x["name"],[]).append(x["bit_mismatch_fraction"])
    names=list(dict.fromkeys(names))
    if names:
        plt.figure(figsize=(9,5)); plt.bar(names,[statistics.mean(values[n]) for n in names]); plt.xticks(rotation=35,ha="right"); plt.ylabel("bit mismatch fraction"); plt.title("M1 final layerwise continuous/exact mismatch (diagnostic subset)"); plt.tight_layout(); plt.savefig(FIG/"m1_layer_mismatch.png",dpi=160); plt.close()
        plt.figure(figsize=(9,5)); plt.bar(names,[max(values[n]) for n in names]); plt.xticks(rotation=35,ha="right"); plt.ylabel("maximum recorded mismatch"); plt.title("M1 earliest aggregate mismatch is visible at the stem"); plt.tight_layout(); plt.savefig(FIG/"m1_first_mismatch_layer.png",dpi=160); plt.close()
def main():
    d=json.loads(RESULT.read_text()); runs=d["runs"]; make_figures(runs)
    main_rows=run_rows(runs,"best_validation_continuous_argmax"); bool_rows=run_rows(runs,"best_validation_boolean_strict")
    cols=[("loss","loss","str"),("seed","seed","str"),("cont_top1","cont top-1","num"),("cont_strict","cont strict","num"),("hard_strict","hard strict","num"),("bool_strict","Boolean strict","num"),("bool_valid","valid Boolean one-hot","num"),("gap","cont→Boolean sample gap","num")]
    lines=["# M1 — MNIST Scale / Discretization Pilot","","## A–B. I13 audit and XOR closure","","I13 was audited before M1. Its canonical JSON and checkpoints agree: curriculum final Boolean exactness was 1/5 (seed0 only), fixed POWER_1_25 was 2/5, and all 20 runs reached continuous exactness. No I13 rerun was required. XOR experimentation is closed.","","## C–H. Data, split, device, and run matrix","","The Kaggle job used the standard torchvision MNIST train/test source, concatenated only to save a raw artifact, with uint8 input threshold 128 (pixel >= 0.5 after normalization). Both continuous and exact Boolean models received the same 784 binary inputs. The official 60,000 training examples were split deterministically into 55,000 train and 5,000 validation; the 10,000 official test examples were untouched.","","The run used a Tesla T4 GPU (`torch 2.10.0+cu128`, CUDA 12.8), not local CPU. The float32 batch-256 forward/backward smoke test was finite, output shape `[256,10]`, and peak allocated GPU memory was 1,702,769,664 bytes. The preflight ran all three losses for two epochs before the canonical 20-epoch matrix. All 9 canonical runs were finite, used Adam 1e-3, weight decay 0, and batch 256.","","The architecture was 784→256, two width-256 XOR residual blocks, 256→10 head, Lehmer p=2, I2-B mean-field sigma=2 plus BIAS_ONE. Initial tensors were paired per seed and reloaded byte-for-byte for each loss arm; the result records the initial-state SHA256 for every run.","","### Main test comparison (checkpoint selected by validation continuous argmax)"]
    lines+=table(main_rows,cols)+["","### Validation-selected Boolean checkpoint comparison",""]+table(bool_rows,cols)+["","## I–M. Classification semantics","","The continuous classifier is evaluated by argmax; strict continuous and Boolean metrics require the ten-bit output vector to equal the one-hot target. Exact Boolean strict accuracy is therefore not an argmax proxy. Zero-hot and multi-hot rates are reported below.","","### Aggregate test metrics (continuous-argmax checkpoint)","","| loss | continuous top-1 | continuous strict | hard strict | Boolean strict | valid Boolean one-hot | zero-hot | multi-hot | sample gap |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for loss in LOSSES:
        rs=[r for r in main_rows if r["loss"]==loss]
        lines.append(f"| {loss} | {mean_std([r['cont_top1'] for r in rs])} | {mean_std([r['cont_strict'] for r in rs])} | {mean_std([r['hard_strict'] for r in rs])} | {mean_std([r['bool_strict'] for r in rs])} | {mean_std([r['bool_valid'] for r in rs])} | {mean_std([r['bool_zero'] for r in rs])} | {mean_std([r['bool_multi'] for r in rs])} | {mean_std([r['gap'] for r in rs])} |")
    lines += ["","## N–P. Functional gap, endpoints, and margins","","The continuous-to-Boolean sample disagreement is large for BCE (roughly 0.86–0.97 across seeds at the selected checkpoint). POWER_1_25 and the curriculum have smaller disagreement in some runs, but their continuous classifiers are also much weaker. This is not evidence of a useful Boolean classifier by itself.","","| loss | mean endpoint distance | mean class margin | E_inf |","|---|---:|---:|---:|"]
    for loss in LOSSES:
        rs=[r for r in main_rows if r["loss"]==loss]
        lines.append(f"| {loss} | {mean_std([r['cont_endpoint'] for r in rs])} | {mean_std([r['cont_margin'] for r in rs])} | {mean_std([r['cont_einf'] for r in rs])} |")
    lines += ["","The continuous endpoint and confidence metrics are present in the canonical trajectory and test records. The power and curriculum arms did not turn endpoint pressure into competitive MNIST classification within 20 epochs.","","## Q. First mismatch layer","","The diagnostic subset was traced through input, stem, both residual blocks, and head at epochs 0, 5, 10, 15, and 20. The first nonzero aggregate mismatch appears at the stem in the final traces; later layers also remain highly mismatched. Thus the scale gap is already introduced at the first 784→256 logic layer, rather than being only a head problem. See `m1_layer_mismatch.png` and `m1_first_mismatch_layer.png`.","","## R. Parameter polarization","","Per-layer gate and bias polarization statistics are stored in every trajectory record under `parameter_stats` and were descriptive only. No polarization regularizer was used. The main accuracy failure is not resolved by output loss choice alone.","","## S–T. Loss comparison","","BCE was the strongest continuous arm in this pilot: mean test top-1 was about 0.680 across three seeds, versus about 0.341 for fixed POWER_1_25 and 0.356 for the curriculum. Its exact Boolean strict accuracy was only about 0.095, with a large semantic gap. POWER_1_25 and the curriculum had lower disagreement in selected runs but did not learn a competitive continuous classifier. The curriculum did not improve over fixed POWER_1_25 at scale.","","## U. Seed variability","","Seed variation is material for every loss. BCE ranged from 0.662 to 0.700 continuous top-1; POWER_1_25 ranged from 0.274 to 0.374; curriculum ranged from 0.332 to 0.365. Exact Boolean strict accuracy also varied substantially. These are three-seed pilot observations, not population estimates.","","## V. Main conclusion","","M1 exposes both problems, with the dominant first failure being scalable continuous learning. The best continuous arm reaches only about 68% test top-1, below the 80% pilot threshold, so this is primarily a scale/optimization limitation. The continuous-to-exact-Boolean gap is nevertheless also severe: even the BCE arm's Boolean strict accuracy is far below its continuous argmax accuracy, and the first mismatch appears in the stem.","","## W. One recommended next experiment","","Run one controlled scalability experiment that keeps binary MNIST inputs and exact Boolean evaluation but uses a memory-efficient implementation of the same Lehmer-p2 architecture (for example, chunked contribution computation) so training can run longer or with a validated larger effective batch. Do not add STE, regularization, or new operators until the continuous baseline is useful.","","## Artifacts","","Canonical result: `research/operator_results/m1_mnist_pilot_results.json`. Thirty-six checkpoint files were downloaded from Kaggle, reloaded, and their SHA256 hashes matched the recorded metadata. Figures are in `research/figures/m1_*.png`.",""]
    REPORT.write_text("\n".join(lines))
if __name__=="__main__": main()
