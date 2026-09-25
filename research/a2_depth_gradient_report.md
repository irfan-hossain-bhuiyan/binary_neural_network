# A2 — XOR-Residual Depth / Gradient Propagation

## A. A1 baseline and audit

A1 is the latest completed benchmark before A2. Its canonical JSON, report, and task tests agreed; no A1 rerun was required. A1 selected **MSE** for A2 because all requested recovery rates were zero, while MSE had the strongest task-independent final continuous result (highest continuous exact rate and lowest E_inf). Native objective magnitudes were not used.

A2 Kaggle commit: `b711fcae2d5ebb432a2ee10d8e6882d0760e4b30`; device `cuda` (Tesla T4). Raw result SHA256: `e869f0c08235243d3f81aa6a544241fd875a02e2d6c2293dce9b47bc475d6003`.

## B. Architecture matrix

All runs use input 8, width 64, output 5, Lehmer-p2, I2-B mean-field sigma=2 plus BIAS_ONE, full-batch Adam (lr=.01, 3000 updates), and seeds 0–2. Depths are 0, 1, 2, 4, and 8 two-layer width-preserving blocks. Depth 0 has no residual factorial; positive depths have matched XOR_RESIDUAL and NO_RESIDUAL arms. No checkpoints were retained.

## C. XOR residual Jacobian

For y=x+F(x)-2xF(x), J_y=diag(1−2F)+diag(1−2x)J_F. The direct skip gain is D_skip=diag(1−2F): it is approximately +1 when F≈0, −1 when F≈1, and vanishes near F=.5. Thus this is a signed, state-dependent residual path rather than an additive identity. The runner verifies g_x=g_direct+g_branch numerically at initialization and diagnostic checkpoints.

## D–F. Gradient propagation and cancellation

For each block the JSON records input/output activation-gradient norms, transfer ratios, cosine, direct and branch norms, direct/branch cosine, and relative reconstruction error. It also records per-layer parameter gradient norms and gradient/parameter ratios. The direct/branch cosine diagnoses reinforcement versus cancellation; values near −1 indicate cancellation.

## G–H. Depth and polarization

Initial and final branch statistics include mean min(F,1−F), fractions below .1/above .9, and fraction near .5. These can be compared with mean |1−2F| to test whether polarization strengthens the direct XOR path.

## I–M. Recovery and carry structure

The tables below use task-independent continuous, hard-max, and exact Boolean metrics. Carry-chain groups and per-output-bit metrics remain in the canonical JSON for every evaluation.

| blocks | residual | cont exact runs | Boolean exact runs | stable Boolean runs | median cont step | median Bool step | min initial transfer |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | XOR_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | NA |
| 1 | XOR_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.414973646402359 |
| 1 | NO_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.16982915997505188 |
| 2 | XOR_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.41996365785598755 |
| 2 | NO_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.10935327410697937 |
| 4 | XOR_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.22185949981212616 |
| 4 | NO_RESIDUAL | 1/3 | 0/3 | 0/3 | 1975 | NOT_REACHED | 0.09964537620544434 |
| 8 | XOR_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.27175435423851013 |
| 8 | NO_RESIDUAL | 0/3 | 0/3 | 0/3 | NOT_REACHED | NOT_REACHED | 0.06290566176176071 |

### Initial/final activation-gradient decomposition (means over seed and block)

| blocks | residual | initial transfer | final transfer | initial direct/total | final direct/total | initial branch/total | final branch/total | initial direct-branch cosine | final direct-branch cosine | initial mean |1−2F| | final mean |1−2F| |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | XOR_RESIDUAL | 0.414687 | 0.874781 | 0.935135 | 0.610597 | 0.235323 | 0.783035 | 0.141253 | 0.00673508 | 0.382197 | 0.622828 |
| 1 | NO_RESIDUAL | 0.172302 | 0.789511 | N/A | N/A | 1 | 1 | N/A | N/A | N/A | N/A |
| 2 | XOR_RESIDUAL | 0.478127 | 0.901104 | 0.996017 | 0.757332 | 0.0964107 | 0.619167 | -0.0141454 | 0.00950329 | 0.358473 | 0.730525 |
| 2 | NO_RESIDUAL | 0.124528 | 0.873842 | N/A | N/A | 1 | 1 | N/A | N/A | N/A | N/A |
| 4 | XOR_RESIDUAL | 0.34891 | 0.990674 | 0.99039 | 0.835121 | 0.0908603 | 0.495298 | 0.00723794 | 0.0292642 | 0.347183 | 0.826621 |
| 4 | NO_RESIDUAL | 0.13281 | 0.951006 | N/A | N/A | 1 | 1 | N/A | N/A | N/A | N/A |
| 8 | XOR_RESIDUAL | 0.396201 | 0.994073 | 0.995732 | 0.940418 | 0.041175 | 0.22614 | 0.00390219 | 0.0701906 | 0.339097 | 0.903644 |
| 8 | NO_RESIDUAL | 0.14167 | 0.108066 | N/A | N/A | 1 | 1 | N/A | N/A | N/A | N/A |

For NO_RESIDUAL rows, direct-path columns are `None` and the branch is the entire path. Residual decomposition errors are approximately machine precision; the individual values are retained in the canonical JSON.

### Final Boolean accuracy by carry-chain length

| blocks | residual | chain 0 | chain 1 | chain 2 | chain 3 | chain 4 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | XOR_RESIDUAL | 0.1523 | 0.1526 | 0.0778 | 0.0000 | 0.0000 |
| 1 | XOR_RESIDUAL | 0.4815 | 0.3012 | 0.0833 | 0.0278 | 0.0000 |
| 1 | NO_RESIDUAL | 0.1852 | 0.2811 | 0.2500 | 0.1389 | 0.0000 |
| 2 | XOR_RESIDUAL | 0.2716 | 0.3976 | 0.5556 | 0.3750 | 0.1667 |
| 2 | NO_RESIDUAL | 0.3292 | 0.3333 | 0.2000 | 0.2222 | 0.2917 |
| 4 | XOR_RESIDUAL | 0.1358 | 0.3534 | 0.2833 | 0.3611 | 0.2917 |
| 4 | NO_RESIDUAL | 0.1317 | 0.0602 | 0.0778 | 0.0000 | 0.0000 |
| 8 | XOR_RESIDUAL | 0.3580 | 0.2771 | 0.2167 | 0.2361 | 0.2500 |
| 8 | NO_RESIDUAL | 0.0329 | 0.0161 | 0.0000 | 0.0000 | 0.0000 |

### Final Boolean accuracy by output bit

| blocks | residual | s0 | s1 | s2 | s3 | s4 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | XOR_RESIDUAL | 0.5833 | 0.7292 | 0.6068 | 0.5312 | 0.8281 |
| 1 | XOR_RESIDUAL | 1.0000 | 0.5911 | 0.7422 | 0.6458 | 0.8594 |
| 1 | NO_RESIDUAL | 0.5833 | 0.6771 | 0.8203 | 0.7344 | 0.8490 |
| 2 | XOR_RESIDUAL | 0.8958 | 0.9167 | 0.8203 | 0.7643 | 0.6562 |
| 2 | NO_RESIDUAL | 0.9167 | 0.7969 | 0.7279 | 0.6393 | 0.7917 |
| 4 | XOR_RESIDUAL | 1.0000 | 0.7865 | 0.8880 | 0.7305 | 0.5755 |
| 4 | NO_RESIDUAL | 0.6224 | 0.6432 | 0.5065 | 0.5352 | 0.6081 |
| 8 | XOR_RESIDUAL | 1.0000 | 0.9167 | 0.9167 | 0.5612 | 0.5951 |
| 8 | NO_RESIDUAL | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4896 |

Recovery timestamps are `NOT_REACHED` when a condition did not occur; they are never replaced by step 3000. The canonical trajectories include per-evaluation MSE/MAE/E_inf, continuous/hard/Boolean accuracy, wrong rows/bits, carry groups, per-bit metrics, functional disagreement, internal mismatch traces, timing, and diagnostics at steps 0, 100, 500, 1000, 2000, 3000 (plus any first recovery checkpoint).

## N. Interpretation

Use `a2_recovery_vs_depth.png`, `a2_gradient_by_layer.png`, `a2_direct_skip_gain.png`, `a2_branch_polarization.png`, and `a2_carry_chain_accuracy.png` for the depth/residual comparison. A residual benefit is supported only if it improves task-independent recovery and/or gradient transfer relative to the matched no-residual arm. If both modes fail together, the limiting mechanism is more likely depth/operator optimization than the skip. If direct gain is weak early and increases as F polarizes, the data support a weak-early/strong-late XOR path.

**Recommended subsequent experiment:** introduce the planned threshold-aware loss on the unchanged A1-selected MSE baseline, using the A2 diagnostics to determine whether its pressure should target the first failing semantic layer. Do not change depth, operator, initialization, or residual structure in that follow-up.
