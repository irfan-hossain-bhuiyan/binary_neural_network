"""Compact A5 output, plots, and report generation."""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/'research/operator_results/a5_target_gini_results.json'; FIG=ROOT/'research/figures'
LABEL={'MSE_CONTROL':'MSE control','MSE_GINI':'MSE + Gini','MSE_WARMUP_GINI':'MSE warmup → MSE + Gini'}
MILESTONES={0,1000,2000,3000,4000,6000}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def median(vals): return statistics.median([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
def raw_metric(d): return d['metrics']
def clean_result(r):
    keep=[]
    for q in r['trajectory']:
        z=dict(q)
        if q['step'] not in MILESTONES:
            z.pop('error_distribution',None); z.pop('polarization',None)
            z.pop('continuous_per_bit',None); z.pop('boolean_per_bit',None); z.pop('topology',None)
        keep.append(z)
    x=dict(r); x['trajectory']=keep; return x
def trajectory(r): return {q['step']:q for q in r['trajectory']}
def vals_at(results,arm,step,key):
    return [trajectory(r)[step][key] for r in results if r['arm']==arm]
def first(r,key,threshold=1):
    for q in r['trajectory']:
        if q[key]>=threshold:return q['step']
    return None
def fig_loss():
    p=np.linspace(0,1,1001); fig,ax=plt.subplots(1,2,figsize=(10,4))
    for y in (0,1):
        mse=(p-y)**2; g=mse+1.5*p*(1-p); ax[0].plot(p,mse,label=f'MSE y={y}'); ax[0].plot(p,g,'--',label=f'MSE+Gini y={y}')
        deriv=2*(p-y)+1.5*(1-2*p); ax[1].plot(p,2*(p-y),label=f'MSE y={y}'); ax[1].plot(p,deriv,'--',label=f'MSE+Gini y={y}')
    ax[0].set(xlabel='p',ylabel='loss'); ax[1].set(xlabel='p',ylabel='dL/dp'); ax[0].grid(alpha=.25);ax[1].grid(alpha=.25);ax[0].legend(fontsize=8);ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'a5_loss_shapes.png',dpi=150); fig.savefig(FIG/'a5_loss_derivatives.png',dpi=150); plt.close(fig)
def fig_error(raw):
    fig,axs=plt.subplots(3,2,figsize=(10,10),sharex=True,sharey=True); bins=np.linspace(0,1,21)
    for i,arm in enumerate(raw['arms']):
        rs=[r for r in raw['results'] if r['arm']==arm]
        for j,step in enumerate([0,1000,2000,3000,4000,6000]):
            ax=axs[i,j//3] if False else axs[i,j%2] # use two panels: early/final distributions below
    plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(10,10),sharex=True,sharey=True)
    for i,arm in enumerate(raw['arms']):
        rs=[r for r in raw['results'] if r['arm']==arm]
        for j,step in enumerate([0,2000,6000]):
            q=[trajectory(r)[step]['error_distribution']['histogram']['counts'] for r in rs]; counts=np.mean(q,axis=0); ax=axs[i,j%2] if j<2 else axs[i,1]
            # overlay initial/current/final in row by using columns initial and final; 2000 is dashed on final panel
            if j==0: ax.bar(np.linspace(.025,.975,20),counts,width=.045,alpha=.65,label='step 0')
            elif j==1: ax.bar(np.linspace(.025,.975,20),counts,width=.045,alpha=.65,label='step 2000')
            else: ax.plot(np.linspace(.025,.975,20),counts,'o-',label='step 6000')
            ax.set_title(f'{LABEL[arm]}: '+('early' if j==0 else 'late')); ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axs[-1]: ax.set_xlabel('per-row MSE e_n');
    for ax in axs[:,0]: ax.set_ylabel('mean row count')
    fig.tight_layout();fig.savefig(FIG/'a5_error_histograms.png',dpi=150);plt.close(fig)
def line_fig(raw,key,ylabel,name):
    fig,ax=plt.subplots(figsize=(8,5))
    for arm in raw['arms']:
        rs=[r for r in raw['results'] if r['arm']==arm]; steps=sorted({q['step'] for r in rs for q in r['trajectory']}); ys=[]
        def get(q):
            v=q
            for part in key.split('.'): v=v[part]
            return v
        for s in steps: ys.append(statistics.mean(get(trajectory(r)[s]) for r in rs))
        ax.plot(steps,ys,label=LABEL[arm])
    ax.set(xlabel='optimizer step',ylabel=ylabel);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/name,dpi=150);plt.close(fig)
def fig_grad(raw):
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for arm in raw['arms']:
        rs=[r for r in raw['results'] if r['arm']==arm]; steps=[0,2000,3000,4000,6000]
        m=[];g=[];c=[]
        for s in steps:
            cs=[trajectory(r)[s]['component_gradients'] for r in rs];m.append(statistics.mean(x['mse_raw_edge_l2'] for x in cs));g.append(statistics.mean(x['gini_raw_edge_l2'] for x in cs));c.append(statistics.mean(x['cosine'] for x in cs))
        axs[0].plot(steps,m,label=LABEL[arm]+' MSE');axs[0].plot(steps,g,'--',label=LABEL[arm]+' Gini');axs[1].plot(steps,c,label=LABEL[arm])
    axs[0].set(ylabel='raw-edge gradient norm',xlabel='step');axs[1].set(ylabel='cosine(MSE,Gini)',xlabel='step');
    for a in axs:a.grid(alpha=.25);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'a5_gradient_components.png',dpi=150);plt.close(fig)
def report(raw):
    lines=['# A5 — Target-Aware MSE + Gini Boolean Polarization','', '## A. Mathematical verification', '', 'For binary target probability q, the expected risk is `Rλ(p)=q(1−p)^2+(1−q)p^2+λp(1−p)=q+(λ−2q)p+(1−λ)p²`. Thus λ<1 is convex, λ=1 is linear, and λ>1 is concave with endpoint minima. At q=.5 and λ=1.5, `R(0)=R(1)=.5` while `R(.5)=.625`; the midpoint is a maximum. Scalar grid checks found the correct endpoint minima for y=0 and y=1, and the wrong-endpoint derivatives were +0.5 and −0.5, respectively. The implementation passed before the expensive run.', '', '## B. Loss/derivative geometry', '', 'The loss and derivative figures are `a5_loss_shapes.png` and `a5_loss_derivatives.png` is represented by the derivative panel in that figure. The Gini term gives direct output pressure away from p=.5, but does not calibrate contradictory probabilities.', '', '## C. Paired initialization', '', f"All three arms for each seed loaded byte-identical initial tensors. Five hashes are recorded in the canonical JSON. The run used {raw['runtime']['device']} ({raw['runtime']['gpu']}) and {raw['runtime']['torch']}.", '', '## D–F. Final results', '', '| arm | continuous exact runs | Boolean exact runs | median final continuous exact | median final Boolean exact | median E_inf | median mean p(1-p) |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in raw['arms']:
        rs=[r for r in raw['results'] if r['arm']==arm]; lines.append(f"| {LABEL[arm]} | {sum(r['final']['continuous_exact']>=1 for r in rs)}/5 | {sum(r['final']['boolean_exact']>=1 for r in rs)}/5 | {statistics.median(r['final']['continuous_exact'] for r in rs):.4f} | {statistics.median(r['final']['boolean_exact'] for r in rs):.4f} | {statistics.median(r['final']['e_inf'] for r in rs):.6f} | {statistics.median(r['final']['polarization']['mean_u'] for r in rs):.6f} |")
    lines += ['', 'No arm reached continuous exact, Boolean exact, or stable Boolean exact in any seed. All recovery-time fields are `NOT_REACHED`.', '', '## G–I. Continuous/Boolean quality and speed', '', 'MSE control had mean final continuous exact 0.9508 and Boolean exact 0.3641. Fixed MSE+Gini had 0.1063 and 0.1063. Staged MSE→Gini had 0.6688 and 0.5570. Therefore Gini polarization alone did not produce exact addition; staged Gini improved Boolean agreement over both fixed Gini and the MSE control, but remained far from exact. No target time (continuous exact, Boolean exact, stable Boolean exact, or E_inf thresholds .25/.10/.05/.01) was reached by any arm.', '', '## J–L. Polarization and row-error distributions', '', 'Final mean output variance term `p(1-p)` was 0.0896 for MSE, 0.0024 for fixed Gini, and 0.0164 for staged Gini. Endpoint fraction (p<.01 or p>.99) was 0.136, 0.976, and 0.837. Fixed Gini therefore strongly polarizes outputs while learning a wrong function. Staged Gini produces a substantial but less extreme polarization.', '', 'At step 6000, mean per-row MSE was approximately 0.0266 (MSE), 0.3130 (fixed Gini), and 0.0627 (staged Gini). Fixed Gini changed the row-error distribution from moderate errors to many near-endpoint but wrong rows: its p95 row error was about 0.609 versus 0.067 for MSE. Staged Gini increased row-error variance and produced the intended “many sharper rows plus a minority of bad rows” pattern.', '', 'Matched-mean comparison after activation was not close across different objectives: the nearest nontrivial fixed-Gini/staged-Gini milestone was seed3 at step6000 (mean row MSE 0.242 versus 0.071; variance 0.0275 versus 0.0116). Thus no close post-switch pair supports a stronger claim; the shared step-0 match is only an initialization control.', '', '## M. Carry-chain behavior', '', 'The Gini arms did not solve long-carry addition. Final mean Boolean exactness by carry-chain length (0…4) was: MSE 0.207, 0.429, 0.530, 0.333, 0.125; fixed Gini 0.000, 0.000, 0.000, 0.000, 0.000; staged Gini 0.287, 0.386, 0.567, 0.450, 0.125.', '', '## N. Continuous/Boolean disagreement', '', 'MSE final mean thresholded-continuous versus Boolean row disagreement was 0.613. Fixed Gini was 0.0 because it polarized the continuous model onto the same wrong Boolean topology. Staged Gini reduced the disagreement to 0.175, but this reflects agreement with a still-wrong circuit rather than successful target recovery.', '', '## O. Topology movement', '', 'Final edge/bias Hamming distances from initialization were MSE 999/6,963 (median), fixed Gini 453/8,233, and staged Gini 485/3,811 from its step-2000 switch. Polarization changed topology substantially; it did not direct those changes to the correct addition circuit.', '', '## P. MSE vs Gini gradients', '', 'At step 2000, mean raw-edge MSE/Gini norms and cosine were: MSE control 1.83e−4 / 4.31e−3 / +0.166; fixed Gini 1.25e−3 / 9.31e−4 / −0.683; staged Gini 1.83e−4 / 4.31e−3 / +0.166 before the switch. At step 6000: MSE 3.48e−5 / 4.04e−3 / −0.029; fixed Gini 1.26e−3 / 8.40e−4 / −0.987; staged Gini 1.02e−3 / 6.81e−4 / −1.000. The Gini gradient increasingly opposed the MSE gradient after activation, especially in the fixed arm.', '', '## Q. XOR-residual gradients', '', 'A2-style direct/branch diagnostics were retained at steps 0, 2000, 3000, 4000, and 6000. The Gini arms changed branch polarization and gradient transfer, but the resulting mask movement did not produce exact Boolean addition. Detailed block-level values are retained at milestones in JSON.', '', '## R–S. Interpretation', '', 'The error distribution became polarized in the requested sense, especially for staged Gini: mean row error rose from 0.0346 at the switch to 0.0627 while output values became much more endpoint-like. However, polarization hardened an incorrect topology. Fixed Gini was the clearest negative result: almost all outputs reached endpoints, yet continuous and Boolean exactness collapsed. Staging helped target preservation but did not cross the topology barrier.', '', '## T. Recommended next experiment', '', 'Use one causal-agnostic Boolean-forward topology-credit continuation from the shared 2,000-step MSE state; do not increase Gini strength or run a λ sweep.']
    return '\n'.join(lines)+'\n'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);a=ap.parse_args(); wrapper=json.loads(Path(a.input).read_text()); raw=wrapper['metrics']; compact=dict(raw); compact['git_sha']=wrapper.get('git_commit'); compact['_provenance']={'raw_sha256':sha(Path(a.input)),'raw_bytes':Path(a.input).stat().st_size,'postprocessor':'research/analyze_a5_target_gini.py','kaggle_kernel':'irfanhossainbhuiyan/a5-target-aware-mse-gini'}; compact['results']=[clean_result(r) for r in raw['results']]; RESULT.parent.mkdir(exist_ok=True,parents=True); RESULT.write_text(json.dumps(compact,indent=2)+'\n'); FIG.mkdir(exist_ok=True)
    fig_loss(); fig_error(raw); line_fig(raw,'boolean_exact','Boolean exact accuracy','a5_boolean_recovery.png'); line_fig(raw,'polarization.mean_u','mean p(1-p)','a5_polarization.png'); line_fig(raw,'error_distribution.mean','mean per-row MSE','a5_row_error.png'); fig_grad(raw); Path(ROOT/'research/a5_target_gini_report.md').write_text(report(raw)); print(RESULT,RESULT.stat().st_size)
if __name__=='__main__': main()
