# A5 — Target-Aware MSE + Gini Boolean Polarization

## A. Mathematical verification

For binary target probability q, the expected risk is `Rλ(p)=q(1−p)^2+(1−q)p^2+λp(1−p)=q+(λ−2q)p+(1−λ)p²`. Thus λ<1 is convex, λ=1 is linear, and λ>1 is concave with endpoint minima. At q=.5 and λ=1.5, `R(0)=R(1)=.5` while `R(.5)=.625`; the midpoint is a maximum. Scalar grid checks found the correct endpoint minima for y=0 and y=1, and the wrong-endpoint derivatives were +0.5 and −0.5, respectively. The implementation passed before the expensive run.

## B. Loss/derivative geometry

The loss and derivative figures are `a5_loss_shapes.png` and `a5_loss_derivatives.png` is represented by the derivative panel in that figure. The Gini term gives direct output pressure away from p=.5, but does not calibrate contradictory probabilities.

## C. Paired initialization

All three arms for each seed loaded byte-identical initial tensors. Five hashes are recorded in the canonical JSON. The run used cuda (Tesla T4) and 2.10.0+cu128.

## D–F. Final results

| arm | continuous exact runs | Boolean exact runs | median final continuous exact | median final Boolean exact | median E_inf | median mean p(1-p) |
|---|---:|---:|---:|---:|---:|---:|
| MSE control | 0/5 | 0/5 | 0.9297 | 0.3047 | 0.656747 | 0.081344 |
| MSE + Gini | 0/5 | 0/5 | 0.0898 | 0.0898 | 0.999973 | 0.000761 |
| MSE warmup → MSE + Gini | 0/5 | 0/5 | 0.6602 | 0.6094 | 0.999869 | 0.011197 |

No arm reached continuous exact, Boolean exact, or stable Boolean exact in any seed. All recovery-time fields are `NOT_REACHED`.

## G–I. Continuous/Boolean quality and speed

MSE control had mean final continuous exact 0.9508 and Boolean exact 0.3641. Fixed MSE+Gini had 0.1063 and 0.1063. Staged MSE→Gini had 0.6688 and 0.5570. Therefore Gini polarization alone did not produce exact addition; staged Gini improved Boolean agreement over both fixed Gini and the MSE control, but remained far from exact. No target time (continuous exact, Boolean exact, stable Boolean exact, or E_inf thresholds .25/.10/.05/.01) was reached by any arm.

## J–L. Polarization and row-error distributions

Final mean output variance term `p(1-p)` was 0.0896 for MSE, 0.0024 for fixed Gini, and 0.0164 for staged Gini. Endpoint fraction (p<.01 or p>.99) was 0.136, 0.976, and 0.837. Fixed Gini therefore strongly polarizes outputs while learning a wrong function. Staged Gini produces a substantial but less extreme polarization.

At step 6000, mean per-row MSE was approximately 0.0266 (MSE), 0.3130 (fixed Gini), and 0.0627 (staged Gini). Fixed Gini changed the row-error distribution from moderate errors to many near-endpoint but wrong rows: its p95 row error was about 0.609 versus 0.067 for MSE. Staged Gini increased row-error variance and produced the intended “many sharper rows plus a minority of bad rows” pattern.

Matched-mean comparison after activation was not close across different objectives: the nearest nontrivial fixed-Gini/staged-Gini milestone was seed3 at step6000 (mean row MSE 0.242 versus 0.071; variance 0.0275 versus 0.0116). Thus no close post-switch pair supports a stronger claim; the shared step-0 match is only an initialization control.

## M. Carry-chain behavior

The Gini arms did not solve long-carry addition. Final mean Boolean exactness by carry-chain length (0…4) was: MSE 0.207, 0.429, 0.530, 0.333, 0.125; fixed Gini 0.000, 0.000, 0.000, 0.000, 0.000; staged Gini 0.287, 0.386, 0.567, 0.450, 0.125.

## N. Continuous/Boolean disagreement

MSE final mean thresholded-continuous versus Boolean row disagreement was 0.613. Fixed Gini was 0.0 because it polarized the continuous model onto the same wrong Boolean topology. Staged Gini reduced the disagreement to 0.175, but this reflects agreement with a still-wrong circuit rather than successful target recovery.

## O. Topology movement

Final edge/bias Hamming distances from initialization were MSE 999/6,963 (median), fixed Gini 453/8,233, and staged Gini 485/3,811 from its step-2000 switch. Polarization changed topology substantially; it did not direct those changes to the correct addition circuit.

## P. MSE vs Gini gradients

At step 2000, mean raw-edge MSE/Gini norms and cosine were: MSE control 1.83e−4 / 4.31e−3 / +0.166; fixed Gini 1.25e−3 / 9.31e−4 / −0.683; staged Gini 1.83e−4 / 4.31e−3 / +0.166 before the switch. At step 6000: MSE 3.48e−5 / 4.04e−3 / −0.029; fixed Gini 1.26e−3 / 8.40e−4 / −0.987; staged Gini 1.02e−3 / 6.81e−4 / −1.000. The Gini gradient increasingly opposed the MSE gradient after activation, especially in the fixed arm.

## Q. XOR-residual gradients

A2-style direct/branch diagnostics were retained at steps 0, 2000, 3000, 4000, and 6000. The Gini arms changed branch polarization and gradient transfer, but the resulting mask movement did not produce exact Boolean addition. Detailed block-level values are retained at milestones in JSON.

## R–S. Interpretation

The error distribution became polarized in the requested sense, especially for staged Gini: mean row error rose from 0.0346 at the switch to 0.0627 while output values became much more endpoint-like. However, polarization hardened an incorrect topology. Fixed Gini was the clearest negative result: almost all outputs reached endpoints, yet continuous and Boolean exactness collapsed. Staging helped target preservation but did not cross the topology barrier.

## T. Recommended next experiment

Use one causal-agnostic Boolean-forward topology-credit continuation from the shared 2,000-step MSE state; do not increase Gini strength or run a λ sweep.
