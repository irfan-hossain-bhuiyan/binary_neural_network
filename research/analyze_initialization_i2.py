"""Summarize and plot the archived I2 initialization experiment."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'research/operator_results/initialization_i2_results.json'
FIG=ROOT/'research/figures'; FIG.mkdir(parents=True,exist_ok=True)

def main():
    payload=json.loads(RESULT.read_text()); rows=payload['results']; conditions=list(payload['conditions'])
    summary=[]
    for c in conditions:
        rs=[r for r in rows if r['condition']==c]; epochs=[r['first_boolean_recovery']['epoch'] for r in rs if r['first_boolean_recovery']]
        summary.append({'condition':c,'boolean_recovery_count':len(epochs),'boolean_recovery_rate':len(epochs)/len(rs),'median_recovery_epoch':sorted(epochs)[len(epochs)//2] if epochs else None,'median_best_mse':sorted(r['best_continuous']['continuous']['mse'] for r in rs)[len(rs)//2],'mean_final_boolean_exact':sum(r['trajectory'][-1]['boolean']['exact_accuracy'] for r in rs)/len(rs),'bad_basins':sum(r['trajectory'][-1]['boolean']['exact_accuracy']<1 for r in rs)})
    out=ROOT/'research/operator_results/initialization_i2_summary.json'; out.write_text(json.dumps({'experiment':'I2-summary','summary':summary,'paired_sigma':{c:[{'seed':s,'sigma2_recovered':next(r for r in rows if r['condition']==c and r['seed']==s)['first_boolean_recovery'] is not None,'sigma4_recovered':next(r for r in rows if r['condition']==('I2-C' if c=='I2-B' else 'I2-E') and r['seed']==s)['first_boolean_recovery'] is not None} for s in range(5)] for c in ('I2-B','I2-D')}},indent=2))
    # Continuous-loss trajectories, one faint line per seed and a median line.
    plt.figure(figsize=(9,5))
    for c in conditions:
        rs=[r for r in rows if r['condition']==c]; curves=[[(x['epoch'],x['continuous']['mse']) for x in r['trajectory']] for r in rs]
        for curve in curves: plt.plot([x[0] for x in curve],[max(x[1],1e-12) for x in curve],alpha=.18)
        xs=[x[0] for x in curves[0]]; ys=[sorted(max(curve[i][1],1e-12) for curve in curves)[len(curves)//2] for i in range(len(xs))]
        plt.plot(xs,ys,label=c,linewidth=2)
    plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('continuous MSE'); plt.title('I2 continuous optimization'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'initialization_i2_training_curves.png',dpi=160); plt.close()
    plt.figure(figsize=(8,4)); plt.bar([x['condition'] for x in summary],[x['boolean_recovery_rate'] for x in summary]); plt.ylim(0,1); plt.ylabel('Boolean recovery rate'); plt.title('I2 exact Boolean recovery'); plt.tight_layout(); plt.savefig(FIG/'initialization_i2_recovery.png',dpi=160); plt.close()
    plt.figure(figsize=(7,5))
    for c in conditions:
        rs=[r for r in rows if r['condition']==c]; plt.scatter([r['best_continuous']['continuous']['mse'] for r in rs],[r['best_boolean']['boolean']['exact_accuracy'] for r in rs],label=c,s=55)
    plt.xscale('log'); plt.xlabel('best continuous MSE'); plt.ylabel('best Boolean exact accuracy'); plt.title('I2 continuous optimization vs Boolean recovery'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'initialization_i2_mse_vs_boolean.png',dpi=160); plt.close()
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
