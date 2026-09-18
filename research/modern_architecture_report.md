# Modern XOR-Residual Boolean Network: Research Report

**Report date:** 2026-09-17
**Working branch:** `research-modern-xor-residual`
**Experiment:** M001, seed 0, 4-bit smoke run
**Evidence status:** two completed, SHA-verified 2000-epoch Kaggle runs of the same M001 setup; the corrected run adds hard-max evaluation. No truth-table, no-residual, 16-bit, or multi-seed experiment has been run.

## Executive summary

This research phase corrects the previous topology: the historical `MultiLayerLogicGateNet` remains intact, and a separate modern network now has width-preserving XOR residual blocks. The modern graph is input → stem → two blocks of logic → logic → XOR with block input → output head. The matching discrete network uses the same skip locations and bitwise XOR.

The continuous/discrete topology has CPU tests that exhaust Boolean rows for a residual block and a complete tiny network. All 21 repository CPU tests passed after the implementation. A 4-bit M001 run completed 2000 epochs on a Kaggle Tesla T4 in 38.6 minutes. It became operationally near-Boolean early, but readiness did not mean task success: at epoch 5 the thresholded model had only 7.2% exact accuracy. At epoch 2000 the continuous model reached 95.6% bit accuracy and 83.6% exact accuracy; the thresholded Boolean model reached 91.5% bit and 70.6% exact accuracy. The continuous-to-discrete exact-accuracy gap was 13.0 percentage points.

Both residual branches were mostly polarized by the end. Their mean local direct XOR gains, `|1-2F(x)|`, were 0.978 and 0.990. Backpropagation transfer ratios from block output to input were above 1 for this diagnostic batch, so the earlier report's attenuation wording was reversed and has been corrected. There is no same-depth no-residual control, so these measurements do not establish that residuals improve gradient flow or task performance.

The first run did not calculate continuous hard-max accuracy. A corrected run on the same training configuration evaluated soft, hard-max, and Boolean inference on the same final checkpoint. Accuracy decreased in that order, showing that both aggregation and parameter thresholding contribute descriptively. The corrected SHA was verified.

## Architecture and implementation

### Boolean neuron

For each output unit `i` and input edge `j`:

```text
z_ij = w_ij AND (x_j XOR b_ij)
n_i  = OR_j z_ij
```

`w=0` removes an edge; `w=1` selects it. `b=0` selects the input literal and `b=1` selects its negation. The outer reduction is OR, with no output inversion.

The differentiable literal uses `x + b - 2xb`, and the selected contribution is `w * literal`. The existing soft aggregation is retained for training; this phase adds no OR surrogate.

### Modern graph

For block input `x`:

```text
h1 = LogicLayer1(x)
h2 = LogicLayer2(h1)
y  = x XOR h2
```

The continuous skip is `y = x + h2 - 2*x*h2`; the discrete skip is `y = x ^ h2`. Only width-matched `W → W → W` blocks receive a skip. No projection, adapter, padding, or truncation is used.

The implemented modern graph is:

```text
input → stem → [64→64 → 64→64 → XOR skip] × 2 → head
```

For the 4-bit smoke, the actual dimensions are `8→64`, four `64→64` block layers, then `64→4`. This gives 6 logic layers, 17,152 possible selected edges, 34,304 weight and bias scalars, and 6 learnable temperatures: 34,310 trainable scalars total. The 16-bit version would have 19,456 possible edges and 38,918 trainable scalars.

The deep control is supported by `residual_enabled=False`; it retains the same layers and parameter shapes. It has not been trained.

### Conversion and invariant tests

`ModernLogicGateNet.to_discrete()` creates a separate `DiscreteModernLogicGateNet`, thresholds the effective weights and biases at 0.5, and copies each layer into the same stem/block/head positions. It does not mutate the training model. The discrete residual blocks contain bitwise XOR at the same locations.

CPU tests check all four XOR endpoint combinations, all 8 Boolean inputs to a 3-wide residual block, and all 8 rows of a tiny two-block modern network with Boolean parameters and hard-max continuous layers. They also check that the no-residual control has identical parameter shapes. The full test command was:

```bash
pytest -q research/tests
```

Result: **21 passed**. A one-epoch local harness smoke also exercised the modern model, conversion, and block diagnostics. This verifies implementation invariants; it does not establish trained-model correctness.

## M001 experiment design

| Setting | Value |
|---|---|
| Task | sampled bitwise XOR, 4 input operand bits → 4 output bits |
| Seed | 0 |
| Data | 20,000 sampled operand pairs; seeded 80/20 random train/validation split (16,000 / 4,000) |
| Topology | stem width 64; 2 residual blocks, 2 logic layers each; 4-output head |
| Optimizer / loss | Adam, learning rate 0.01; MSE |
| Batch size | 256 |
| Budget | minimum configured 300; maximum 2000; metrics checked every 25 epochs (plus early diagnostic points) |
| Regularization | recovered `regularization_factory2`: `disc_lambda=0.5`, `tau_lambda=0.3`, patience 15, minimum error 0.01, isolate on plateau |
| Plateau constraint | Gaussian weight noise, std 0.3, patience 15, minimum delta 0.01 |
| Temperature | learnable per logic layer, initialized to 1.0; no new annealing policy |
| Conversion threshold | 0.5 |
| Readiness rule | `D_w≤0.01`, `D_b≤0.01`, `w_corner_05≥0.95`, `b_corner_05≥0.95` |
| Hardware / runtime | Kaggle Tesla T4; 2,316.4 seconds (38.6 minutes) |
| Packaged revision | `091faa02bd60a8cb2f53fe6efb800fb3c5c9cb6f`, Kaggle kernel version 11; returned SHA matched |
| Corrected evaluation revision | `1f25331357476462c947b7269a47c8ebada920c9`, Kaggle kernel version 12; returned SHA matched; 2140.1 seconds |

Layer initializers were specified explicitly, using normal distributions with these means: stem 1.0; block 0 layer 1: 0.0; block 0 layer 2: 1.0; block 1 layer 1: 0.0; block 1 layer 2: 1.0; head: 0.0. Bias initialization mean was 1.0 at every layer. The initial effective parameter means after `[0,1]` clamping were not equal to those raw initializer means; measured weight means by layer were 0.599, 0.270, 0.174, 0.154, 0.237, and 0.103. Initial bias means were 0.678, 0.684, 0.694, 0.656, 0.680, and 0.676.

The dataset is sampled, rather than an exhaustive 8-bit input truth table. Validation metrics are therefore on 4,000 held-out sampled pairs, not every possible pair. The split is seeded through the configured seed and PyTorch RNG.

## Training trajectory

The archived complete trajectory is in [`091faa0_M001_modern_4bit_preliminary.json`](../kaggle/results/091faa0_M001_modern_4bit_preliminary.json). It contains an entry per epoch. The plotted discrete values are thresholded diagnostic outputs until the corresponding checkpoint meets the readiness rule; readiness itself is recomputed each epoch and is not monotonic.

![M001 training task loss and regularization loss](figures/m001_training_losses.png)

![Continuous and thresholded validation accuracies; hollow markers are not ready](figures/m001_accuracy_trajectory.png)

The trajectory plot above is from kernel version 11 and does not include
hard-max values. Kernel version 12 evaluated the final checkpoint three ways:

| Inference mode | Validation bit accuracy | Validation exact accuracy |
|---|---:|---:|
| Continuous soft aggregation | 0.95625 | 0.83575 |
| Continuous hard-max | 0.9388125 | 0.77025 |
| Exact discrete Boolean | 0.914625 | 0.70550 |

Soft-to-hard gaps are 1.744 bit-accuracy points and 6.550 exact-accuracy
points. Hard-to-Boolean gaps are 2.419 and 6.475 points. Thus the pattern is
`soft > hard > Boolean`: both aggregation and thresholding contribute
descriptively, with similar exact-accuracy gaps. Exact accuracy is nonlinear,
so these differences are not additive causal effects. The corrected-run
artifact is
[`1f25331_M001_modern_4bit_corrected.json`](../kaggle/results/1f25331_M001_modern_4bit_corrected.json).

![Weight/bias distance and corner fractions](figures/m001_polarization_readiness.png)

| Epoch | Task loss | Regularization loss | Continuous bit acc. | Continuous exact acc. | Thresholded bit acc. | Thresholded exact acc. | `D_w` | `D_b` | Ready? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | — | — | 0.4991 | 0.0585 | 0.5009 | 0.0608 | 0.08560 | 0.08541 | No |
| 1 | 0.2905 | 0.5891 | 0.4991 | 0.0585 | 0.5009 | 0.0608 | 0.01224 | 0.00331 | No |
| 5 | 0.1336 | 0.0696 | 0.8334 | 0.4505 | 0.5046 | 0.0723 | 0.00787 | 0.00419 | Yes |
| 25 | 0.0801 | 0.0308 | 0.9074 | 0.6638 | 0.6376 | 0.1923 | 0.00370 | 0.00139 | Yes |
| 50 | 0.0665 | 1.4880 | 0.9100 | 0.6800 | 0.7085 | 0.2855 | 0.01134 | 0.00143 | **No** |
| 100 | 0.0624 | 0.0067 | 0.9146 | 0.6880 | 0.7376 | 0.3253 | 0.00117 | 0.00024 | Yes |
| 300 | 0.0477 | 1.4410 | 0.9404 | 0.7753 | 0.8534 | 0.5233 | 0.00160 | 0.00025 | Yes |
| 1000 | 0.0378 | 1.4269 | 0.9493 | 0.8158 | 0.8737 | 0.5903 | 0.00111 | 0.00044 | Yes |
| 1500 | 0.0321 | 0.0024 | 0.9566 | 0.8353 | 0.8707 | 0.5643 | 0.00037 | 0.00007 | Yes |
| 2000 | 0.0300 | 1.4032 | 0.9563 | 0.8358 | 0.9146 | 0.7055 | 0.00068 | 0.00007 | Yes |

The first operational readiness checkpoint was epoch 5. The model was not ready at epoch 50, then was ready again at epoch 100 and at epoch 2000. This matters: “first ready” is a trajectory event, not a permanent state guarantee. At epoch 5, polarization was already sufficient under the operational thresholds while thresholded exact accuracy was only 7.2%; this directly demonstrates that polarization alone is not task success.

Task accuracy continued to improve well after first readiness. Continuous exact accuracy reached a sampled-checkpoint maximum of 0.84 at epoch 1200, and was 0.83575 at epoch 2000. It was still moving near the max budget, but it had largely plateaued from epoch 1200 onward and was non-monotonic. No best-task or best-ready model checkpoint was saved; the JSON preserves metrics, not model weights. The requested checkpoint selection strategy remains an implementation gap.

The regularization loss is highly variable, with large spikes at epochs such as 50, 300, 1000, 1750, and 2000. The task loss is reported separately and declines overall. The total objective should not be read as task error.

The run used the full configured 2000-epoch maximum. It did not implement a validation-driven convergence stop; `min_epochs=300` and `check_every=25` are configuration/measurement settings, while stopping was fixed at the maximum. The trace therefore shows behavior at the ceiling but does not claim formal convergence.

At the final checkpoint, `w_corner_05=0.99720`, `b_corner_05=0.99959`, `D_w=0.000685`, and `D_b=0.0000662`. The final learned temperatures ranged from 0.0270 to 0.1382 (taus 7.24 to 36.99). This is substantial polarization, but the exact Boolean model still did not match continuous exact accuracy.

## XOR residual behavior and gradients

The following activation summaries use the final soft model over the 4,000 validation rows. “Middle” is activation in `(0.25, 0.75)`; “near 0/1” means `≤0.05` / `≥0.95`. Entropy is mean binary entropy in bits.

| Block | Signal | Mean | Variance | Near 0 | Near 1 | Middle | Entropy (bits) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | input `x` | 0.914 | 0.076 | 0.080 | 0.907 | 0.007 | 0.014 |
| 1 | first layer `h1` | 0.887 | 0.096 | 0.106 | 0.878 | 0.014 | 0.021 |
| 1 | branch `F(x)=h2` | 0.782 | 0.163 | 0.200 | 0.764 | 0.022 | 0.039 |
| 1 | XOR output `y` | 0.240 | 0.172 | 0.737 | 0.216 | 0.029 | 0.051 |
| 2 | input `x` | 0.240 | 0.172 | 0.737 | 0.216 | 0.029 | 0.051 |
| 2 | first layer `h1` | 0.868 | 0.108 | 0.122 | 0.858 | 0.017 | 0.034 |
| 2 | branch `F(x)=h2` | 0.951 | 0.043 | 0.043 | 0.947 | 0.009 | 0.024 |
| 2 | XOR output `y` | 0.758 | 0.171 | 0.215 | 0.732 | 0.035 | 0.069 |

The residual branches are mostly near Boolean values, especially block 2. The branch-threshold flip fractions are 0.7884 (block 1) and 0.9530 (block 2); for these thresholded values, the fraction of output bits differing from the block input is the same. Those measurements describe substantial representation changes; they do not say whether the changes help the task.

The direct derivative diagnostic is `|1-2F(x)|`, calculated from the continuous branch output at the final checkpoint:

| Block | Mean | Median | p10 | p25 | p75 | p90 | `<0.1` | `<0.25` | `>0.75` | `>0.9` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9777 | 0.9996 | 0.9904 | 0.9983 | 0.9999 | 1.0000 | 0.0027 | 0.0094 | 0.9701 | 0.9643 |
| 2 | 0.9897 | 0.9981 | 0.9926 | 0.9959 | 0.9994 | 0.9999 | 0.0012 | 0.0066 | 0.9907 | 0.9904 |

![Direct residual gain and measured activation gradients](figures/m001_residual_gradients.png)

These final gains show that the local direct derivative was usually near magnitude 1 rather than near zero. They do **not** prove that the full network’s gradients were preserved: the nonlinear branch derivatives and downstream layers still affect the total gradient, and there is no no-residual control for comparison.

Backpropagation travels from block output `y` to block input `x`. On a 512-row validation diagnostic batch, measured backward gradient transfer was:

| Block | `||grad_input|| / ||grad_output||` | `mean_abs_grad_input / mean_abs_grad_output` | Input/output gradient norms |
|---|---:|---:|---:|---:|
| 1 | 4.047 | 6.831 | 0.0608 / 0.0150 |
| 2 | 1.149 | 2.611 | 0.0150 / 0.0131 |

The gradients were larger at the block input than output. Because backpropagation goes output-to-input, these ratios are above 1 and do not show backward attenuation across the measured block activations. Magnitudes remain small in absolute terms, especially by the second block. This single-model measurement cannot establish a residual advantage without the matched control.

The first run recorded only `abs(1-2F(x))`, not signed `1-2F(x)`, so positive versus negative direct derivatives were not counted. Absolute gain near 1 is consistent with either `F≈0` and a positive path or `F≈1` and a sign-reversed path. Signed statistics remain unmeasured.

As a Boolean flip mask, the branch is mostly 1 on these activations: block 1 mask-1 fraction 0.7884 (mask-0 0.2116), block 2 mask-1 fraction 0.9530 (mask-0 0.0470). The output differs from its input at those rates. A mask near 1 behaves approximately like NOT on those bits; this describes the learned transformation without judging it.

Named parameter-gradient mean magnitudes at the final epoch were: stem `3.38e-4`; block 1 layer 1 `2.45e-5`; block 1 layer 2 `1.34e-5`; block 2 layer 1 `7.92e-6`; block 2 layer 2 `3.79e-6`; head `6.70e-5`. The small gradients in later block logic layers are a potential failure mode worth examining in the next approved experiment.

## Boolean circuit size and semantic comparison

At epoch 2000 the thresholded network selected 3,756 of 17,152 possible edges (21.90%). There were 2,593 selected negated-polarity edges (69.04% of selected edges).

| Logic layer | Shape | Selected edges | Selected fraction | Selected negated edges |
|---|---:|---:|---:|---:|
| Stem | 8→64 | 306 | 59.77% | 205 |
| Block 1, layer 1 | 64→64 | 1,108 | 27.05% | 785 |
| Block 1, layer 2 | 64→64 | 713 | 17.41% | 508 |
| Block 2, layer 1 | 64→64 | 631 | 15.41% | 395 |
| Block 2, layer 2 | 64→64 | 971 | 23.71% | 677 |
| Head | 64→4 | 27 | 10.55% | 23 |

The corrected run separates the preliminary 13.025-point soft-to-Boolean exact gap into a 6.550-point soft-to-hard gap and a 6.475-point hard-to-Boolean gap. The conversion invariants pass exactly at Boolean endpoints in local exhaustive tests. At this near-corner checkpoint, thresholding small residual parameter fractions and their propagation through later layers may explain some of the hard-to-Boolean difference. Model weights were not saved, so this could not be probed layer by layer.

1. Continuous model with existing soft aggregation.
2. Continuous model with hard-max logic layers.
3. Separate discrete Boolean model.

The corrected run was Kaggle kernel version 12 from commit `1f25331357476462c947b7269a47c8ebada920c9`; its returned SHA matched. It completed 2000 epochs in 2140.1 seconds on a Tesla T4. Its training configuration and seed match the first run; the corrected revision adds hard-max evaluation.

## What the results support—and what they do not

Supported by current evidence:

- The modern topology is implemented independently from the historical model, including an exact discrete counterpart.
- Continuous XOR at Boolean endpoints and small-network hard-max/discrete equivalence pass exhaustive CPU checks.
- On the sampled 4-bit task, the model learned substantially above chance and polarized strongly under the recovered training setup.
- Task metrics improved after first readiness, confirming that training should not stop merely when the weights polarize.
- The residual branches ended near Boolean corners; their local direct XOR gain was usually close to 1.
- The preliminary thresholded model learned XOR better than chance but underperformed the continuous model on validation accuracy.

Not established:

- Whether continuous hard-max matches the discrete result exactly at the ready checkpoint. Their exact accuracies differ by 6.475 points although endpoint tests pass; no model checkpoint is available for deeper mismatch analysis.
- Whether XOR residuals improve gradients or task learning. There is no matched no-residual run.
- Whether 16-bit XOR is learnable under this setup. No 16-bit run was started.
- Whether these metrics reproduce across seeds. Only seed 0 was run.
- Whether readiness thresholds are scientifically optimal. They are bookkeeping thresholds only.
- Whether the best continuous or best ready checkpoint would outperform the final epoch. Model-state checkpoints were not saved.

## Files and Git record

The implementation and archived experiment history are preserved on `research-modern-xor-residual`. The old `MultiLayerLogicGateNet` and `DiscreteMultiLayerLogicGateNet` remain available as the historical baseline.

- `layers.py`: continuous XOR helper and `XorResidualLogicBlock`.
- `models.py`: new `ModernLogicGateNet`, retaining `MultiLayerLogicGateNet`.
- `discrete_logic_net.py`: `DiscreteXorResidualLogicBlock` and `DiscreteModernLogicGateNet`.
- `research/run_experiment.py`: selectable modern model, readiness trajectory, residual activation/gain/gradient diagnostics, and hard-max comparison in the corrected commit.
- `research/tests/test_modern_xor_residual.py`: CPU invariants.
- `research/configs/M001_modern_4bit.json`: exact 4-bit M001 configuration.
- `research/modern_architecture_research.md`: append-only experiment notebook.
- `research/plot_modern_architecture_report.py`: matplotlib source for all plots in this report.
- `research/figures/`: generated PNG figures.

Commits, oldest first: `e67e2d7` architecture notes/config; `3351c57` modern continuous/discrete models; `b85e717` equivalence tests; `d0b09af` instrumentation/initialization record; `091faa0` readiness trajectory; `b8d2626` hard-max comparison; `1f25331` preliminary M001 result record.

The preliminary JSON is an ignored Kaggle artifact, preserved locally under `kaggle/results/`. The figures can be rebuilt with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python research/plot_modern_architecture_report.py
```

## Current research position

M001’s 4-bit run shows partial task learning, strong Boolean polarization, and a material soft-to-discrete gap. Readiness by itself was an unreliable proxy for task success, and the run continued learning after it first became ready. The branch measurements are consistent with a strong local XOR skip derivative, but actual activation gradients still attenuate through the blocks. The data do not yet tell us whether the residual architecture is better than an equally deep network without residuals.

The corrected soft/hard/Boolean comparison is complete and appended to the canonical notebook. It shows that both the continuous aggregation and thresholding matter on this run. No exact truth-table or matched no-residual experiment has been run. Those are the next controlled questions; no temperature, regularization, stochastic-inference, or MNIST experiment has been run.


## M001-R: exhaustive 4-bit truth table

M001-R changes the task protocol to all 256 4-bit operand pairs, keeping the
modern width-64 residual network and its M001 training settings. It is a
complete function-recovery test, not a held-out generalization test. The run
used seed 0, 2000 epochs, a Tesla T4 (62.02 s), and the SHA-verified package
`6a33da88923b466e6498ef940d46b02742fc82e2` (Kaggle kernel v13).

| Inference on all 256 rows | Bit accuracy | Exact-row accuracy |
|---|---:|---:|
| Continuous soft | 0.86816 | 0.58594 |
| Continuous hard-max | 0.64746 | 0.16406 |
| Exact Boolean | 0.64746 | 0.16797 |

The Boolean network exactly matches 43/256 rows and selects 4,088/17,152
possible edges. It does not recover the function. At epoch 0, D_w=0.08629 and
D_b=0.08571. Parameter binarization first passed the operational threshold
at epoch 175; final D_w=0.002761 and D_b=0.001725, but discrete exact
accuracy was only 0.16797. The best continuous exact accuracy was 0.61719 at
epoch 1725; the best Boolean exact accuracy among binarized checkpoints was
0.18359 at epoch 1975.

Soft exact accuracy is 42.19 percentage points higher than hard-max, whereas
hard-max and Boolean differ by only 0.39 points. This shifts the immediate
interpretation toward the current continuous aggregation semantics as the
larger gap on the exhaustive task endpoint. It does not establish that the
soft aggregation is the only issue or explain the failure to learn the full
truth table. The M001-R JSON and checkpoint remain archived locally at
`kaggle/results/6a33da8_M001-R_seed0.json` and
`kaggle/results/6a33da8_M001-R_seed0.pt`.

Block 0/1 mean signed direct XOR gain is -0.8073/-0.7183; mean absolute gain
is 0.9561/0.8827. Their mask-one and output-flip fractions are 0.9005/0.8698.
Backward activation-gradient norm transfer `||grad_input||/||grad_output||`
is 3.984/0.535 (mean-absolute transfer 5.556/0.886). Layer parameter gradient
means are uneven, with first/last ratio 0.000738. These measurements are
descriptive; the matched control is needed to isolate residual effects.

The next run is M001-NR: same layers, initialization, optimizer, training
budget, exact truth table, and seed, with residual XOR disabled as the only
change. Temperature and regularization experiments remain queued until this
control is recorded.


## M001-NR: matched no-residual control

M001-NR uses the exact M001-R truth table and six logic layers. Its only model
change is disabling both residual XOR operations. The run used seed 0, 2000
epochs, Tesla T4 (70.05 s), Kaggle v14; packaged and returned SHA matched
`feb9db19e6bf4be4ae73699f5bca183c78683d5a`.

| Final inference | Residual bit / exact | No-residual bit / exact |
|---|---:|---:|
| Continuous soft | 0.86816 / 0.58594 | 0.81738 / 0.37891 |
| Continuous hard-max | 0.64746 / 0.16406 | 0.69336 / 0.17969 |
| Thresholded Boolean | 0.64746 / 0.16797 | 0.60840 / 0.11719 |

No model recovers the complete truth table. The control reached 0.90625
continuous exact accuracy at epoch 900, then finished at 0.37891. Its best
ready Boolean checkpoint reached 0.31250 exact at epoch 550; the residual
model's best ready Boolean checkpoint reached 0.18359 at epoch 1975. Yet at
the final checkpoint, the residual model is higher on soft and Boolean exact
accuracy. This checkpoint sensitivity and disagreement make the one-seed
comparison mixed.

Parameter binarization first passed the operational threshold at epoch 548 in
M001-NR, versus 175 in M001-R. Final M001-NR D_w=0.008991 and D_b=0.002193;
its circuit has 4,259 selected edges of 17,152 possible (24.83%). M001-R has
4,088 (23.83%). Neither selected-edge count corresponds to successful
function recovery.

![M001-R and M001-NR truth-table exact-accuracy trajectories](figures/m001_truth_table_residual_control.png)

The residual model's direct signed XOR gain means are -0.8073/-0.7183, with
absolute means 0.9561/0.8827. Backward activation-gradient norm transfer
`||grad_input||/||grad_output||` is 3.984/0.535. In M001-NR, skip operations
are absent, so direct XOR skip gain is not an active path; the measured
branch-like diagnostic proxy has signed means -0.8840/-0.8593, while activation
gradient norm transfer is 1.194/1.472. The named first/last parameter-gradient
ratio is 8.264 for M001-R and 0.926 for M001-NR. These final-batch summaries
are descriptive and do not explain the trajectory or establish causality.

The controlled result does not show a consistent residual advantage: residuals
win at final metrics, while no-residual wins at best observed checkpoints.
Continue with the isolated M002 temperature-policy test on the residual
architecture before deciding whether to spend on larger tasks. Full trajectories
and diagnostics are retained in the two archived JSON files.


## M002: fixed temperature

M002 fixes every logic-layer temperature at 1 (`learnable_tau=false`) and sets
`tau_lambda=0`, leaving the weight/bias regularizer and plateau noise active.
All other settings match M001-R. The run used seed 0, 2000 epochs, Tesla T4
(55.61 s), Kaggle v15; packaged and returned SHA verified
`f8e998fdb4948d48f7f24122de2534874e83116a`.

| Inference | Bit accuracy | Exact accuracy |
|---|---:|---:|
| Continuous soft | 0.53223 | 0.08984 |
| Continuous hard-max | 0.50000 | 0.06250 |
| Boolean | 0.50000 | 0.06250 |

Despite near-chance task results (chance exact = 0.0625), parameters were
strongly polarized: the readiness predicate first passed at epoch 303 and
final D_w/D_b were 0.000670/0.000106. The circuit selected 2,982 of 17,152
possible edges. The best continuous exact accuracy was 0.12891 at epoch 1400.
This demonstrates the separation between parameter binarization and function
recovery in this run.

Final branch direct-gain means were +0.6563/+0.5965 (absolute means
0.6563/0.6045); branch flip fractions were 0.0000/0.0619. Backward gradient
transfer ratios were 3.036/0.957 by L2 norm and 2.783/0.828 by mean absolute
gradient.

The result suggests the learned temperature policy matters under this recipe,
but a single seed cannot establish a general necessity. M003 will hold the
temperature fixed and remove only explicit weight/bias discretization
regularization.


## M003: no explicit discretization regularization

M003 changes only the regularization loss from M002 to none. Temperature stays
fixed at 1; plateau noise and all other settings remain. Seed 0, 2000 epochs,
Tesla T4 (50.25 s), Kaggle v16; exact package SHA verified:
`62f17059b1dc66094f1cb4e633ccc1d6585c6218`.

| Inference | Bit accuracy | Exact accuracy |
|---|---:|---:|
| Continuous soft | 0.70605 | 0.21484 |
| Continuous hard-max | 0.50000 | 0.06250 |
| Thresholded Boolean, diagnostic only | 0.50000 | 0.06250 |

The best soft exact accuracy was 0.31641 at epoch 1325. The final task loss
was 0.23733; regularization loss was zero. Final D_w=0.003329, but
D_b=0.05030 and b_corner_05=0.82579, so `PARAMETER_BINARIZED` was never met.
The Boolean score remains diagnostic only. Parameter entropy means were
0.00984 for weights and 0.14244 for biases. The thresholded diagnostic
circuit selected 4,588 of 17,152 possible edges.

Residual branch direct-gain means were +0.96345/+0.86606 (absolute
+0.96345/+0.89734), with flip fractions 0/0.03125. Backprop gradient transfer
was 4.033/1.503 by L2 norm and 3.677/1.425 by mean absolute gradient.

M003 outperforms M002 modestly in continuous task metrics, but all exact
Boolean outputs remain near chance and no full truth-table recovery occurred.
Across the experiments the current soft aggregation often scores far above
hard-max/Boolean inference, making OR semantics a leading question for the
next design decision. No claim is made that a new OR surrogate will solve the
problem.

The planned phase stops here. We did not run M004, P001, MNIST D001, or P004;
the exhaustive XOR task has not been recovered, and the human should choose
whether the next single hypothesis isolates plateau noise or tests a new OR
semantics.


## Interpretation correction: M002 and M003 are fixed-T experiments

The earlier description of M002 as “removing temperature dynamics” was
incorrect. The existing operator computes `softmax(tau*z)` with
`tau=1/T`; its Boolean/max limit is `T→0+` (or `tau→+∞`). M002 fixed the old
softmax at T=1, which remains smoothed. Preserve its numerical results as a
negative control for fixed T=1. M003 inherits that branch and likewise does
not test a temperature-free architecture.

The intended temperature-free operator is separately defined as
`r=softplus(theta)`, `g=tanh(r)`,
`S(a,r)=<softmax(a*r), a*tanh(r)>`. M002b tests whether this same positive
edge strength can jointly select edges and sharpen the softmax, without a
separate temperature. The existing regularizer is omitted because its bounded
weight/tau penalties do not apply; this also removes its coupled bias penalty
and is an acknowledged experimental consequence.


## M002b: temperature-free self-sharpening operator

The intended temperature-free layer is implemented separately from the old
layer:

```text
a = x + b - 2*x*b
r = softplus(theta) > 0
g = tanh(r)
logits = a*r
value = a*g
output = sum(softmax(logits) * value)
```

There is no temperature or tau variable. In the asymptotic regime, r near zero
suppresses a connection, and large r makes g approach one while sharpening the
softmax over active literals. Bias/polarity remains independently
parameterized in [0,1]. The residual graph is the same two-block modern graph;
its Boolean conversion thresholds g and b at 0.5 and keeps both residual
positions.

The alternating initial gate targets are 0.75, 0.25, 0.75, 0.25, 0.75, 0.25
for stem, block0.layer1, block0.layer2, block1.layer1, block1.layer2 and head.
Mapping `r=atanh(g)` and `theta=log(exp(r)-1)` gives high-gate
`r=0.972955, theta=0.498197` and low-gate `r=0.255413,
theta=-1.234451`. Bias initialization remains the M001-R raw normal mean 1.

Operator CPU tests cover all-zero literals, an active selected edge as R grows
through 1, 2, 5, 10 and 20, inactive selected literals, randomized Boolean OR
limits, initialization mapping, absence of temperature/tau parameters, and
residual-preserving conversion. The full CPU suite passed: **29 tests**. A
2-epoch CPU integration smoke also passed, confirming the model trains and its
serialized trajectory has no temperature/tau fields.

M002b used the exact 256-row XOR table, seed 0, Adam 0.01, MSE, batch 256,
2000 epochs, no explicit regularizer and no plateau noise. It ran on a Tesla
T4 for 49.90 seconds as Kaggle kernel v17. Exact packaged/returned SHA:
`c555781c38113b7e3590c75ecfb5068412d70caf`.

| Final inference | Bit accuracy | Exact accuracy |
|---|---:|---:|
| Continuous temperature-free soft | 1.0000 | 1.0000 |
| Continuous hard-max | 0.8545 | 0.4688 |
| Thresholded Boolean, diagnostic | 0.7305 | 0.2812 |

Soft exact accuracy first reached 1.0 at epoch 525 and remained 1.0 at all 60
recorded evaluations afterward. The best discrete diagnostic was 0.28125 at
epoch 1175. The Boolean model was not parameter-binarized: final D_g=0.08588
and g corner05=0.56145, although biases were highly polarized (D_b=0.00580,
b corner05=0.98292). Therefore perfect continuous truth-table fitting is not
yet exact Boolean function recovery.

At the final checkpoint, 56.15% of effective gates are within 0.05 of 0 or 1.
The sharpening is uneven: block0.layer1 has 70.0% of its gates at or below
0.05, while block0.layer2 has 76.7% at or above 0.95. Internal theta gradient
means are between 6.9e-8 and 3.5e-7; the smallest is in block0.layer2, which
has the strongest high-gate population. Task loss also falls to 0.000479, so
this correlation does not establish gradient saturation as a cause.

The direct XOR gains have signed means -0.6052 and +0.3656 by block, with
absolute means 0.9713 and 0.8794. Flip fractions are 0.8164 and 0.2875. The
backward activation gradient transfer ratios are above one in both blocks:
1.850/1.287 by L2 norm and 2.016/1.765 by mean absolute gradient.

![M002b accuracy, polarization, gates and theta gradients](figures/m002b_temperature_free_trajectory.png)

Relative to M001-R, M002b reaches perfect continuous truth-table accuracy
much earlier (epoch 525 versus M001-R's best 0.61719 at epoch 1725), but the
deterministic Boolean result is still not ready for a like-for-like quality
claim. M001-R final Boolean exact was 0.16797 and its best ready checkpoint
was 0.18359; M002b's 0.28125 Boolean score is diagnostic only.

This is partial evidence for the intended self-sharpening behavior: r develops
near-zero and large values and the soft operator learns XOR. Global edge
polarization and exact Boolean recovery remain incomplete, with a large
53.13-point soft-to-hard exact gap. No MNIST, stochastic circuit sampling, or
new OR operator was run. The next single experiment should replicate this
committed config with another seed before changing the operator or adding
training interventions.
