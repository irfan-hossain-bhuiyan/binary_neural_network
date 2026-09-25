"""Generate the M2 report and figures from the canonical Kaggle result."""
from __future__ import annotations
import json, statistics
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"research/operator_results/m2_balanced_mnist_loss_results.json"
REPORT=ROOT/"research/m2_balanced_mnist_loss_report.md"
FIG=ROOT/"research/figures"
LOSSES=["BCE","BALANCED_BCE","BALANCED_POWER_1_25"]
def ms(xs): return f"{statistics.mean(xs):.4f}" if len(xs)<2 else f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"
def rows(runs,ck):
 out=[]
 for r in runs:
  m=r['test_metrics'][ck]; c,h,b=m['continuous'],m['hard'],m['boolean']
  out.append({'loss':r['loss'],'seed':r['seed'],'top1':c['argmax_accuracy'],'strict':c['threshold_strict_accuracy'],'bool':b['threshold_strict_accuracy'],'valid':b['valid_onehot_rate'],'zero':c['zero_hot_rate'],'gap':b.get('sample_disagreement_fraction',0),'hard':h['threshold_strict_accuracy'],'margin':c['mean_class_margin'],'einf':c['e_inf'],'endpoint':c['mean_endpoint_distance']})
 return out
def table(rs):
 o=['| loss | seed | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean one-hot | zero-hot cont | cont→Boolean gap |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rs:o.append(f"| {r['loss']} | {r['seed']} | {r['top1']:.4f} | {r['strict']:.4f} | {r['hard']:.4f} | {r['bool']:.4f} | {r['valid']:.4f} | {r['zero']:.4f} | {r['gap']:.4f} |")
 return o
def figures(runs):
 FIG.mkdir(parents=True,exist_ok=True)
 for metric,file,title,group in [('argmax_accuracy','m2_continuous_accuracy.png','M2 validation continuous top-1','continuous'),('threshold_strict_accuracy','m2_boolean_accuracy.png','M2 validation exact Boolean strict accuracy','boolean'),('zero_hot_rate','m2_zero_hot_rate.png','M2 continuous zero-hot rate','continuous')]:
  plt.figure(figsize=(8,5))
  for r in runs: plt.plot([z['epoch'] for z in r['trajectory']],[z[group][metric] for z in r['trajectory']],label=f"{r['loss']} s{r['seed']}",alpha=.65)
  plt.xlabel('epoch'); plt.ylabel(metric); plt.title(title); plt.legend(fontsize=6,ncol=3); plt.tight_layout(); plt.savefig(FIG/file,dpi=160); plt.close()
 # Target/non-target means at detailed epochs.
 plt.figure(figsize=(8,5))
 for r in runs:
  zs={z['epoch']:z for z in r['trajectory']}; e=[0,5,20,40,60]
  plt.plot(e,[zs[x]['output_distributions']['target']['mean'] for x in e],label=f"{r['loss']} s{r['seed']} target")
  plt.plot(e,[zs[x]['output_distributions']['non_target']['mean'] for x in e],linestyle='--',alpha=.6,label=f"{r['loss']} s{r['seed']} non-target")
 plt.xlabel('epoch'); plt.ylabel('output mean'); plt.title('Target vs non-target output means'); plt.legend(fontsize=5,ncol=3); plt.tight_layout(); plt.savefig(FIG/'m2_target_nontarget_outputs.png',dpi=160); plt.close()
 # Functional gap.
 plt.figure(figsize=(8,5))
 for r in runs: plt.plot([z['epoch'] for z in r['trajectory']],[z['boolean'].get('sample_disagreement_fraction',0) for z in r['trajectory']],label=f"{r['loss']} s{r['seed']}",alpha=.65)
 plt.xlabel('epoch'); plt.ylabel('sample disagreement'); plt.title('Continuous threshold to exact Boolean gap'); plt.legend(fontsize=6,ncol=3); plt.tight_layout(); plt.savefig(FIG/'m2_functional_gap.png',dpi=160); plt.close()
 # Final trace aggregate.
 names=[]; vals={}
 for r in runs:
  for x in r['trajectory'][-1].get('layer_trace_1024',[]): names.append(x['name']); vals.setdefault(x['name'],[]).append(x['bit_mismatch_fraction'])
 names=list(dict.fromkeys(names))
 if names:
  plt.figure(figsize=(9,5)); plt.bar(names,[statistics.mean(vals[n]) for n in names]); plt.xticks(rotation=35,ha='right'); plt.ylabel('bit mismatch'); plt.title('M2 final layer mismatch'); plt.tight_layout(); plt.savefig(FIG/'m2_layer_mismatch.png',dpi=160); plt.close()
def main():
 d=json.loads(RESULT.read_text()); runs=d['runs']; figures(runs); rr=rows(runs,'best_validation_continuous_argmax')
 lines=['# M2 — Balanced Boolean-Seeking MNIST Loss','', '## A. M1 audit and corrected next step','', 'M1 canonical JSON and checkpoints were audited. The validation/test aggregates reproduce the recorded BCE ≈0.680, POWER_1_25 ≈0.341, and curriculum ≈0.356 continuous top-1 results, with POWER_1_25 near-zero-hot collapse. M1 BCE was still improving at epoch 20. The M1 smoke test used only about 1.70 GB of T4 memory at batch 256, so the immediate questions were training duration and one-hot loss imbalance, not memory capacity. No M1 numerical result was changed.','', '## B–E. Setup and mathematical checks','', 'The M2 note is in `research/mnist_loss_balance.md`. For q=0.1, unbalanced power optima are p*=0.1 (alpha=2), 0.0121951 (alpha=1.5), and 0.000152393 (alpha=1.25). The balanced vector objective weights the positive target bit and all nine negative bits equally; its symmetric uninformative optimum is p*=0.5 for every alpha>1. Balanced BCE uses the analogous positive-versus-aggregate-negative weighting and does not use softmax.','', f"The synthetic gradient sanity test is stored in the result JSON at four uniform prediction values (p=0.01, .1, .5, .9). At p=.5, balanced BCE and balanced POWER have equal aggregate positive/negative gradient magnitudes; the unbalanced geometry does not. The split hash is `{d['split_indices_sha256']}`.",'', 'The run used a Tesla T4 (`torch 2.10.0+cu128`), binary MNIST inputs, the unchanged 784→256→two XOR residual blocks→256→10 Lehmer-p2 network, Adam 1e-3, batch 256, and 60 epochs. All nine runs were finite. Initial-state hashes match the corresponding M1 seed states exactly.','', '## F. Epoch-20 reproduction','', 'The M2 BCE trajectories exactly reproduce M1 at epoch 20 for all three seeds (validation continuous top-1, strict continuous, Boolean strict, and zero-hot rate match). This validates the longer continuation before interpreting epochs 21–60.','', '## G–J. 60-epoch results','', '### Test metrics (checkpoint selected by validation continuous top-1)','']
 lines+=table(rr)+['','### Aggregate test metrics','', '| loss | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean | zero-hot cont | sample gap |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for loss in LOSSES:
  x=[r for r in rr if r['loss']==loss]; lines.append(f"| {loss} | {ms([r['top1'] for r in x])} | {ms([r['strict'] for r in x])} | {ms([r['hard'] for r in x])} | {ms([r['bool'] for r in x])} | {ms([r['valid'] for r in x])} | {ms([r['zero'] for r in x])} | {ms([r['gap'] for r in x])} |")
 lines += ['', '## K–L. Zero-hot and output distributions','', 'Balanced BCE reduces the continuous zero-hot rate from roughly 0.74–0.78 at M1 epoch 20 to about 0.008–0.011 at epoch 60. Balanced POWER_1_25 similarly stays near 0.015–0.025. Original BCE remains zero-hot-heavy at about 0.48–0.54 after 60 epochs. Target outputs rise while non-target outputs fall for both balanced arms; the target/non-target trajectories are shown in `m2_target_nontarget_outputs.png`.','', '## M–N. Continuous and Boolean comparison','', 'Balanced BCE is the strongest continuous result: mean test top-1 is about 0.848 across seeds, exceeding the 80% pilot criterion. Original BCE reaches about 0.793, while balanced POWER_1_25 reaches about 0.780. Despite the stronger continuous classifier, exact Boolean strict accuracy remains near zero for balanced BCE and around one percent for balanced POWER. This is a severe parameter/topology discretization gap, not a zero-hot output problem alone.','', '## O. Functional discretization gap','', 'The sample disagreement between thresholded continuous outputs and exact Boolean outputs remains high: approximately 0.92 on average for BCE, 0.99 for balanced BCE, and 0.96 for balanced POWER at the selected continuous checkpoints. The Boolean network is therefore not implementing the learned classifier, even when the continuous classifier is useful.','', '## P. First mismatch layer','', 'The detailed traces at epochs 0, 20, 40, and 60 continue to show the first nonzero mismatch at the stem on the fixed 1024-image validation subset. Later logic layers also remain mismatched. Longer training and balanced output losses improve continuous learning but do not remove the earliest stem-level semantic divergence. See `m2_layer_mismatch.png`.','', '## Q–S. Interpretation','', 'BCE was undertrained in M1: its validation accuracy continued to rise, and M2 improves the mean test top-1 from about 0.680 to about 0.793. One-hot imbalance explains much of the unbalanced POWER collapse: balancing reduces zero-hot outputs dramatically. However, balanced POWER_1_25 still trails BCE in continuous top-1 and does not preserve a meaningful Boolean advantage. Balanced BCE is the best continuous baseline in this study, but its exact Boolean output remains poor. Thus one-hot imbalance explains the power-loss scaling failure, while the remaining continuous-to-Boolean gap is a separate internal discretization problem.','', '## T. One recommended next experiment','', 'Run one controlled post-training discretization study on the balanced-BCE checkpoint, using the existing exact Boolean evaluation and layer mismatch trace to test a single Boolean-aware continuation method. Do not change the architecture or output encoding until that gap is characterized.','', '## Checkpoints and artifacts','', 'The canonical JSON stores all trajectories, distributions, gradient sanity, test evaluations, and hashes. Thirty-six selected/final checkpoints were downloaded and rehashed successfully. The runner did not emit separate epoch-20/epoch-40 checkpoint files; those epochs are fully recorded in the trajectories, while selected and final checkpoints are preserved.','']
 REPORT.write_text('\n'.join(lines))
if __name__=='__main__': main()
