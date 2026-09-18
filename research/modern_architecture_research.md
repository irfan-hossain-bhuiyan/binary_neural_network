# Modern XOR-Residual Boolean Network Research

## Architecture specification

### Discrete neuron

z_ij = w_ij AND (x_j XOR b_ij)
n_i = OR_j z_ij

### Continuous literal

xor(x,b) = x + b - 2xb

### XOR residual block

h1 = L1(x)
h2 = L2(h1)
y = XOR(x,h2)

### Dimension rule

Direct XOR residuals are allowed only when dimensions match.
No learned projection is used in the initial architecture.

### Discretization philosophy

The discrete Boolean network is the actual target model.
The continuous model is an optimization surrogate.

Discrete evaluation is considered scientifically meaningful
only after weights and biases are sufficiently close to {0,1}.

The initial operational definition of `DISCRETIZATION_READY` is D_w <= 0.01,
D_b <= 0.01, w_corner_05 >= 0.95, and b_corner_05 >= 0.95. These are
experiment bookkeeping thresholds, not established scientific constants.

## Experiment index

| ID | Git SHA | hypothesis | task | topology | seed | result |
|---|---|---|---|---|---|---|
| M001 | 1f25331357476462c947b7269a47c8ebada920c9 | XOR residual blocks can train, polarize, and discretize faithfully | sampled bitwise_xor, 4-bit | stem 64, two 64-wide two-layer XOR blocks, head | 0 | 2000 epochs; soft/hard/Boolean exact 0.83575/0.77025/0.70550; ready first at 5 |

## M001 — Modern XOR-residual baseline

### Hypothesis

A width-preserving XOR residual topology can learn bitwise XOR, polarize its
parameters under the recovered discretization-oriented training setup, and
produce a discrete network consistent with its continuous hard-max graph.

### Exact code state

Architecture/test source SHA: `b85e717`; completed corrected training package:
`1f25331357476462c947b7269a47c8ebada920c9` (Kaggle kernel version 12; returned
SHA verified).

### Architecture

ModernLogicGateNet: input -> stem -> two 64->64 logic layers plus elementwise
XOR with the block input (for each of two blocks) -> output head. Direct skips
are only applied at width 64. The no-residual control uses the same layers,
shapes, initialization sequence, and dimensions with only those XOR operations
disabled. No OR surrogate, adapter, projection, or threshold change is used.

For the 4-bit smoke (input 8, output 4), the six layer dimensions are
8->64, 64->64, 64->64, 64->64, 64->64, 64->4. Initializer means by index are:
stem 1.0; block0.layer0 0.0; block0.layer1 1.0; block1.layer0 0.0;
block1.layer1 1.0; head 0.0. Bias initialization mean is 1.0 for every layer.
The 4-bit graph has 17,152 possible selected edges, 34,304 weight/bias scalar
parameters, and 6 learnable temperatures (34,310 trainable scalars). The
16-bit graph has 19,456 possible selected edges, 38,912 weight/bias scalars,
and 6 temperatures (38,918 trainable scalars).

Discrete conversion copies thresholded effective weights and biases into a new
DiscreteModernLogicGateNet and keeps the same residual positions.

### Training configuration

Completed seed-0 sampled 4-bit run. The corrected evaluation used the same
final checkpoint for soft, hard-max, and discrete inference. The dataset uses
20,000 seeded operand pairs and an 80/20 train/validation split. No 16-bit run
has been started.

The configured alternating initializer is listed by explicit layer index in
the M001 config. Bias initialization remains the recovered baseline setting.

### Discretization-readiness definition

Ready iff D_w <= 0.01 AND D_b <= 0.01 AND w_corner_05 >= 0.95 AND
b_corner_05 >= 0.95. Threshold 0.5 is used for conversion. Before readiness,
thresholded measurements are labeled diagnostic_only and do not support a
claim about discretization success or failure.

### Training trajectory

Recorded for every epoch: task/regularization/total loss, accuracy at
measurement checkpoints, D_w/D_b, corner fractions, readiness, diagnostic
discrete accuracies, temperature, and tau. Full JSON is archived below.

### Residual-block behavior

Recorded for both blocks on the final validation pass: input, first logic
output, residual branch, XOR output distributions and thresholded flip rates.

### Gradient behavior

The first run recorded absolute direct gain only; it did not retain the signed
direct derivative. Named parameter gradients and actual activation gradients
are in the report. The corrected evaluation adds hard-max predictions but
does not retrain or change those gradient measurements.

### Continuous performance

See completed result below.

### Boolean polarization

First ready at epoch 5; readiness was false again at epoch 50. Final D_w and
D_b were 0.000685 and 0.0000662. Polarization alone was not task success.

### Discrete performance after readiness

Completed at the epoch-2000 parameter-binarized checkpoint. Soft, hard-max,
and exact Boolean predictions were evaluated on the same 4,000 validation
rows. Earlier thresholded outputs remain diagnostic only.

### Continuous/discrete gap

Hard-max remained above the Boolean result by 6.475 percentage points in exact
accuracy. Endpoint equivalence tests pass. The near-corner checkpoint weights
were not saved, so the remaining threshold/margin effects could not be probed
per layer after training.

### Result

The model learned and polarized, but the final discrete result remained below
the continuous soft and hard-max results. See the complete three-way result
and interpretation below; this is partial function recovery, not success.

### Interpretation

Both continuous aggregation and parameter thresholding contributed to the
observed exact-accuracy gaps. This does not establish whether XOR residuals
help relative to a matched no-residual topology.

### Open questions

Does F(x) move toward 0/1 and keep the direct XOR gradient path usable? Does the
residual model outperform the same deep no-residual topology? Does either model
reach readiness while retaining task performance?


### M001 preliminary 4-bit run (measurement incomplete)

The first committed Kaggle run used code SHA `091faa02bd60a8cb2f53fe6efb800fb3c5c9cb6f`,
seed 0, 20,000 sampled examples, 80/20 train/test split, and all 2000 epochs.
Returned SHA matched. The original artifact is
`kaggle/results/091faa0_M001_modern_4bit_preliminary.json`. It recorded epoch-0
continuous exact accuracy 0.0585, bit accuracy 0.499125, D_w 0.08560, D_b
0.08541; first operational readiness was epoch 5. At epoch 2000, validation
continuous exact/bit accuracy was 0.83575/0.95625, thresholded discrete exact/bit
accuracy was 0.7055/0.914625, D_w 0.000685, D_b 0.0000662, and 3,756 of 17,152
possible edges were selected. Task loss was 0.03004; reported regularization
loss was 1.4032. The model was still improving in continuous exact accuracy
from 0.0585 at epoch 0 to 0.83575 at the budget ceiling. The direct-gain means
were 0.9777 and 0.9897 in blocks 0 and 1; block branch flip fractions were
0.7884 and 0.9530. These are preliminary only: this run omitted the continuous
hard-max prediction measurement needed to validate agreement with the discrete
graph, so it is not the final M001 comparison. No 16-bit or no-residual run was
started from this incomplete measurement.

That statement records the decision at the time of the preliminary run. The
corrected M001 comparison has since completed and is recorded below; the
matched control and truth-table experiment remain unrun.

Full narrative, trajectory figures, block statistics, limitations, and the
current stopping point are in [`modern_architecture_report.md`](modern_architecture_report.md).

### M001 corrected three-way evaluation

#### Parent experiment

The preliminary M001 run at commit `091faa02bd60a8cb2f53fe6efb800fb3c5c9cb6f`;
same seed, data, initialization, optimizer, regularization, architecture, and
training budget.

#### Hypothesis

Separate the validation gap from continuous soft aggregation to hard-max, and
from hard-max continuous inference to the exact Boolean network.

#### Single changed variable

Evaluation instrumentation only: measure a cloned continuous hard-max model.
Training code path and configuration were otherwise unchanged.

#### Architecture

Modern width-64 stem, two two-layer width-64 XOR residual blocks, and four-bit
head; exact discrete counterpart retains both residual skip positions.

#### Dataset

20,000 sampled 4-bit operand pairs, seed 0, 16,000 train and 4,000 validation
examples. This is a held-out sampled split, not exhaustive truth-table
recovery.

#### Training configuration

M001 config, 2000 epochs, Adam at 0.01, MSE, batch size 256, learned per-layer
temperature, recovered `regularization_factory2` and plateau noise. Tesla T4
runtime 2140.1 seconds. Kaggle kernel version 12. Packaged and returned SHA both
`1f25331357476462c947b7269a47c8ebada920c9`.

#### Metrics

| Inference on same final checkpoint | Bit accuracy | Exact accuracy |
|---|---:|---:|
| Continuous soft aggregation | 0.95625 | 0.83575 |
| Continuous hard-max | 0.9388125 | 0.77025 |
| Exact discrete Boolean model | 0.914625 | 0.70550 |

Soft-to-hard gap: 1.744 bit-accuracy points and 6.550 exact-accuracy points.
Hard-max-to-Boolean gap: 2.419 bit-accuracy points and 6.475 exact-accuracy
points. Total soft-to-Boolean gap: 4.163 and 13.025 points, respectively.
At this checkpoint D_w=0.000685, D_b=0.0000662, w_corner_05=0.99720, and
b_corner_05=0.99959. The separate Boolean circuit selected 3,756 of 17,152
possible edges.

#### Result

The ordered result is `soft > hard-max > Boolean` for both bit and exact
accuracy. Returned SHA was verified. The corrected artifact is
`kaggle/results/1f25331_M001_modern_4bit_corrected.json`; the earlier result
remains preserved as `kaggle/results/091faa0_M001_modern_4bit_preliminary.json`.

#### Interpretation

Both the continuous aggregation choice and the parameter-to-Boolean conversion
contribute to the observed gap. The soft-to-hard change is larger than the
hard-to-Boolean change in exact accuracy by 0.075 percentage points (6.550 vs
6.475), so their sizes are similar in this run. Since exact accuracy is a
nonlinear metric, these gaps are descriptive and are not additive causal
effects. The pattern is consistent with contributions from both soft
aggregation and thresholding; it does not prove their isolated causal impact.

All four near-Boolean equivalence tests still pass, so no endpoint topology
mismatch was found. The remaining hard-max-to-Boolean discrepancy at
near-corner, not exactly Boolean parameters may arise from small parameter
deviations being amplified by later layers. The training checkpoint was not
saved, preventing a layerwise margin analysis. Future runs must save selected
continuous and parameter-binarized checkpoints.

#### What this does NOT prove

This sampled run does not establish exact 256-row XOR truth-table recovery,
generalization, reproducibility across seeds, or a benefit from residuals. No
matched no-residual model was trained. It does not prove whether the soft
surrogate or thresholding is the dominant cause in other configurations.

#### Next question

Build the exact 256-row 4-bit XOR task and compare modern residual against the
same-depth no-residual control at seed 0. Preserve the current M001 setup for
that matched pair and record signed skip derivatives plus correctly directed
backpropagation transfer ratios.
