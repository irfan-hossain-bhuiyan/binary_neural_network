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
| M001 | b85e717 | XOR residual blocks can train and reach a faithful Boolean solution | bitwise_xor, 4-bit smoke then 16-bit | stem 64, two 64-wide two-layer XOR blocks, head | 0 | implementation/tests only; training not yet run |

## M001 — Modern XOR-residual baseline

### Hypothesis

A width-preserving XOR residual topology can learn bitwise XOR, polarize its
parameters under the recovered discretization-oriented training setup, and
produce a discrete network consistent with its continuous hard-max graph.

### Exact code state

Code SHA: `b85e717` (source and tests); exact packaged HEAD is to be recorded
with returned Kaggle metadata after the remote run.

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

Pending run. Planned task is bitwise_xor, initially 4-bit with seed 0, then
16-bit only if the smoke run validates training. Maximum budget is 2000 epochs;
trajectory checks every 25 epochs. Dataset uses the task's seeded sampled
train/test split (`num_samples` and `train_ratio` are recorded in config).

The configured alternating initializer is listed by explicit layer index in
the M001 config. Bias initialization remains the recovered baseline setting.

### Discretization-readiness definition

Ready iff D_w <= 0.01 AND D_b <= 0.01 AND w_corner_05 >= 0.95 AND
b_corner_05 >= 0.95. Threshold 0.5 is used for conversion. Before readiness,
thresholded measurements are labeled diagnostic_only and do not support a
claim about discretization success or failure.

### Training trajectory

Pending run. Required fields: epoch, task/regularization/total loss, continuous
bit/exact accuracy, D_w/D_b, corner fractions, readiness, diagnostic discrete
accuracies, temperature, and tau.

### Residual-block behavior

Pending run. Per block, measure input, first logic output, residual branch, and
XOR output: mean, variance, near-zero, near-one, middle fractions, and binary
entropy. Also report XOR flip fraction and output/input difference fraction.

### Gradient behavior

Pending run. For each residual block report abs(1-2F(x)) mean, median, p10,
p25, p75, p90, and fractions below 0.1/0.25 and above 0.75/0.9. Record
parameter gradient means/norms by stem, each logic layer, each block input/output,
and head, including first/last ratio and activation gradient entering/leaving
blocks where retained gradients permit.

### Continuous performance

Pending run.

### Boolean polarization

Pending run. Report readiness epoch and D_w/D_b trajectory; polarization alone
is not success.

### Discrete performance after readiness

Pending run. Compare continuous soft, continuous hard-max, and exact Boolean
predictions on validation data after readiness. Pre-ready discrete results are
diagnostic_only.

### Continuous/discrete gap

Pending run. Investigate any mismatch between hard-max and Boolean predictions
at a ready checkpoint as a possible conversion/implementation defect.

### Result

Not run. Architecture correctness tests passed locally; no research result is
claimed yet.

### Interpretation

Pending experiment.

### Open questions

Does F(x) move toward 0/1 and keep the direct XOR gradient path usable? Does the
residual model outperform the same deep no-residual topology? Does either model
reach readiness while retaining task performance?
