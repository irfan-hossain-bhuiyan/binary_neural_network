# Differentiable OR Operator Research

## Research objective

Find a differentiable aggregation operator that:

1. trains reliably,
2. preserves useful gradients,
3. respects Boolean OR at Boolean endpoints,
4. minimizes fractional endpoint loopholes,
5. makes low continuous error predictive of low Boolean error,
6. scales to large fan-in,
7. has reasonable compute cost.

## Central hypothesis

The useful criterion is not merely similarity to OR. The critical property is
functional discretization consistency: low continuous error on a complete
Boolean function should predict low error after the same model is hardened.
Functionally unused parameters may remain fractional.

## Scope and preserved context

This notebook is a separate operator track for the modern XOR-residual
architecture. The historical shallow model and M001/M002/M002b/M003 records
remain in `modern_architecture_research.md` and are not rewritten. In
particular, M002b remains an important partial-positive baseline: it reached
continuous exact accuracy 1.0 on the 256-row four-bit XOR table while its final
hard-max exact accuracy was 0.46875 and thresholded Boolean exact accuracy was
0.28125. This motivates measuring operator semantics independently before
spending neural-network training budget.

## Candidate definitions

The common input is `v=a*g`, where `a` is the existing continuous XOR literal
and `g=sigmoid(raw_edge)`. Every candidate reduces the final fan-in dimension.
Candidate formulas and initial theoretical classifications are in
[`operator_candidates.md`](operator_candidates.md). Implementations are in
[`or_surrogates.py`](or_surrogates.py).

## Analytic and numerical screening

`analyze_or_operators.py` evaluates float64 forward and backward behavior at
fan-ins 2 through 784. It records gradient distributions, width scaling,
negative partial derivatives, duplicate sensitivity, endpoint accumulation,
and fractional endpoint searches. Results are archived at
`operator_results/operator_stress_results.json`; plots are under
`figures/or_*.png`.

The first screening run is intentionally not a neural-network experiment.
This separates mathematical semantics and gradient behavior from optimizer,
initialization and residual-topology effects.

## Initial findings from the stress run

- Hard max is the semantic control: it is Boolean exact, duplicate invariant,
  monotone and endpoint rigid, but its gradient is winner-only and nonsmooth.
- Lehmer means preserve Boolean endpoints and are idempotent, but the
  derivative can be negative for weak coordinates. This is a possible edge
  competition mechanism, not a proof of useful optimization.
- Probabilistic OR, Einstein and Hamacher are monotone and Boolean exact, but
  repeated moderate values accumulate toward one. At width 784 with all
  inputs equal to 0.5, probabilistic OR is effectively one and its gradient is
  numerically zero. This is a fractional endpoint loophole in approximate
  operation even though its exact zero endpoint is rigid.
- The bounded Łukasiewicz sum is retained as a negative control: `[0.5, 0.5]`
  maps exactly to one despite neither input being one.
- Finite-alpha softmax weighted value is a useful historical bridge because
  score and value have the same ordering, but it is not exact Boolean OR at
  finite alpha.

The numerical endpoint search is evidence for counterexamples, not a theorem.
Any candidate that reaches an endpoint with high fractionality is recorded in
the JSON archive for later review.

## Research questions

1. Does a monotone t-conorm optimize better than a nonmonotone Lehmer mean?
2. Does endpoint rigidity predict functional discretization consistency?
3. How does fan-in change gradient dilution or fractional accumulation?
4. Can a tiny Boolean task expose continuous-to-discrete counterexamples
   before the modern residual network is trained?

## Queued protocol

After review of this property report, survivors will be screened on complete
truth tables (`identity4`, `not4`, `or4`, `and4`, `xor2`, `majority5`) before
the exact four-bit XOR network comparison. No MNIST or stochastic circuit
experiment is started in this phase.

## Stage B implementation

Stage B uses a shared sigmoid edge parameterization:

```text
g = sigmoid(raw_edge)
a = x + b - 2xb
v = a*g
output = OR_OPERATOR(v)
```

The configurable layer is `SigmoidOrLogicLayer` and the modern graph is
`SigmoidOrModernLogicGateNet`. It has no temperature, tau, temperature
regularizer, old weight regularizer, or plateau noise. `to_discrete()` creates
a separate `DiscreteModernLogicGateNet`; it does not mutate the training
model. XOR residual positions are unchanged.

Endpoint tests cover hard max, Lehmer p=1 and p=2, log-hazard,
probabilistic OR, and the complete tiny modern graph. The full test suite has
67 passing tests.

## Stage B0 — direct optimization

The direct CPU test optimized one active literal toward target 0 and target 1
for each promoted operator, with seeds 0, 1 and 2, Adam learning rate 0.05,
and 400 epochs. All candidates reached the target-0 endpoint. For target 1,
the final mean loss was approximately `8.3e-4` and the thresholded Boolean
output was correct for every promoted operator and seed. This confirms that
the edge logits and candidate reductions can move in the required direction,
but it is not yet evidence of a useful multilayer circuit.

## Stage B1 — simple Boolean functions

The B1 CPU screen trained one shared-shape sigmoid-edge layer on complete
truth tables for `or4`, `identity4`, and `not4`, with seeds 0, 1 and 2, Adam
learning rate 0.01, MSE, 600 epochs, and no regularization or noise. The
operators were hard max, Lehmer p=1, Lehmer p=2, log-hazard,
probabilistic OR, and softmax-value alpha=16.

| task | operator | best continuous MSE by seed | continuous exact | Boolean exact | classification |
|---|---|---|---|---|---|
| OR4 | hard max | .0937, .0931, .0694 | .8125 each | .875 each | optimization failure/partial |
| OR4 | Lehmer p=1 | .2231, .2171, .2209 | .875 each | .875 each | optimization failure/partial |
| OR4 | Lehmer p=2 | .2153, .2121, .2157 | .875 each | .875 each | optimization failure/partial |
| OR4 | log-hazard | .1162, .1205, .0206 | .875, .875, 1.0 | .875, .875, 1.0 | seed-sensitive |
| OR4 | probabilistic OR | .00221, .00220, .00223 | 1.0 each | 1.0 each | best OR4 screen |
| OR4 | softmax alpha=16 | .0942, .1222, .00572 | .8125, .875, 1.0 | .875, .875, 1.0 | seed-sensitive |
| identity4 | Lehmer p=1 | .00459, .00460, .00465 | 1.0 each | 1.0 each | exact classification |
| identity4 | Lehmer p=2 | .00344, .00355, .00352 | 1.0 each | 1.0 each | exact classification |
| identity4 | log-hazard | .00364, .00371, .00370 | 1.0 each | 1.0 each | exact classification |
| identity4 | probabilistic OR | .00411, .00416, .00412 | 1.0 each | 1.0 each | exact classification |
| identity4 | softmax alpha=16 | .00356, .00368, .00367 | 1.0 each | 1.0 each | exact classification |
| NOT4 | Lehmer p=1 | .00463, .00470, .00464 | 1.0 each | 1.0 each | exact classification |
| NOT4 | Lehmer p=2 | .00343, .00337, .00341 | 1.0 each | 1.0 each | exact classification |
| NOT4 | log-hazard | .00365, .00363, .00362 | 1.0 each | 1.0 each | exact classification |
| NOT4 | probabilistic OR | .00410, .00406, .00409 | 1.0 each | 1.0 each | exact classification |
| NOT4 | softmax alpha=16 | .00354, .00361, .00362 | 1.0 each | 1.0 each | exact classification |

The B1 training curves and continuous-MSE versus Boolean-error plots are in
`figures/stage_b1_consistency_*.png`. The complete machine-readable record,
including ten-epoch trajectories, milestone lookup, gate margins, sigmoid
derivative saturation and raw-edge gradient norms, is in
`operator_results/stage_b_results.json`. No B1 run reached the `1e-3` or lower
MSE milestones; the `1e-2` milestone was reached at the final epoch for the
successful truth-table classifications.

### Stage B classification

- Hard max: semantic control, but optimization was weak on OR4 and NOT4.
- Lehmer p=1 and p=2: reliable exact classification for identity/NOT, but
  OR4 numerical optimization was poor at this fixed budget.
- Log-hazard: promising but seed-sensitive on OR4; no evidence yet of
  numerical convergence to zero.
- Probabilistic OR: strongest OR4 B1 result and exact Boolean classification,
  but its continuous loss plateaued near `2.2e-3`, consistent with the Stage-A
  fractional accumulation warning.
- Softmax alpha=16: useful historical control and seed-sensitive behavior;
  it remains semantically non-OR at finite alpha.

These are screening results, not a final operator ranking. The main four-bit
XOR network and B2 tasks have not been launched.

## Correction: Lehmer B1 OR4 numerical failure

The original B1 run used a masked ratio implementation that evaluated
`num/den` even when `den == 0`. Although `torch.where` selected the zero
branch in the forward pass, autograd could retain a `0/0` NaN. Lehmer p=1 and
p=2 OR4 trajectories showed this failure around epoch 50 and then collapsed;
those results are retained as superseded diagnostics, not as valid operator
rankings.

The ratio now uses a safe denominator before division:

```python
mask = den > 0
safe_den = torch.where(mask, den, torch.ones_like(den))
ratio = num / safe_den
return torch.where(mask, ratio, torch.zeros_like(ratio))
```

Zero and mixed-row backward regression tests now pass for Lehmer p=1, Lehmer
p=2, log-hazard and odds-weighted means.

### Lehmer OR4 rerun

Only the contaminated cases were rerun: Lehmer p=1 and p=2, OR4, seeds 0, 1
and 2, with the original 600-epoch configuration. No NaN gradients occurred.
The corrected runs continued improving after epoch 50, but p=1 remained at
best MSE `0.0702–0.1180` and exact accuracy `0.875` for all seeds. Lehmer p=2
reached best MSE `0.00684` for seed 2 and reached continuous, hard and Boolean
exact accuracy `1.0` at epoch 370; seeds 0 and 1 remained at exact accuracy
`0.875` with best MSE around `0.118`. Thus the NaN bug invalidated the earlier
collapse diagnosis, but it did not by itself make Lehmer p=1 reliably solve
OR4 under this budget. The rerun record is
`operator_results/stage_b1_lehmer_or4_rerun.json`.

The next stage is B2 compositional truth-table tasks after reviewing this
corrected result. B3 four-bit XOR and MNIST remain pending.

## Stage B2 — compositional truth tables

B2 used the same sigmoid-edge layer and modern graph for every operator:

```text
input -> width 16 stem -> one two-layer XOR-residual block -> head
```

Each task used its complete truth table, Adam with learning rate `0.01`, MSE,
no regularization, no noise, no temperature, 1000 epochs, and seeds 0, 1 and
2. Tasks were `xor2`, `majority5`, `parity4`, `full_adder` and
`multiplexer4`. The machine-readable archive is
`operator_results/stage_b2_results.json`; consistency plots are
`figures/stage_b2_consistency_*.png`.

### Best exact accuracies by task and seed

| task | hard max | Lehmer p=1 | Lehmer p=2 | log-hazard | probabilistic OR | softmax α=16 |
|---|---|---|---|---|---|---|
| XOR2 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | .50, .50, .25 | 1.0, 1.0, 1.0 |
| majority5 | .9375, .96875, .6875 | .8125, .9375, .9375 | .96875, .84375, .875 | .96875 each | 1.0, .50, 1.0 | .84375, .8125, .84375 |
| parity4 | .75, .875, .9375 | .5625, .50, .50 | .5625, .6875, .50 | .625, .625, .50 | 1.0 each | .75, .875, .50 |
| full_adder | .875, .875, 1.0 | .25, .75, .75 | .625, .875, 1.0 | .50, .625, .50 | .375, .875, .0 | .375, .875, .625 |
| multiplexer4 | .8125, .96875, .75 | .8125, .5625, .84375 | 1.0, .90625, 1.0 | .953125, .796875, .9375 | 1.0, .50, .50 | .9375, .765625, .78125 |

### Interpretation

- XOR2 was solved by every semantic/smooth control except probabilistic OR,
  whose best continuous MSE was low but whose thresholded Boolean circuit was
  wrong for all three seeds. This is an early example of continuous loss and
  Boolean function disagreement.
- Probabilistic OR solved parity4 for all seeds and majority5 for two seeds,
  but was unstable on full adder and multiplexer. Its many-fractional-input
  accumulation remains a real compositional risk.
- Lehmer p=2 was the strongest Lehmer candidate: it reached exact
  multiplexer recovery for seeds 0 and 2 and full-adder recovery for seed 2.
  It did not solve parity4 reliably.
- Lehmer p=1 was weaker than p=2 on these compositional tasks and did not
  reach exact full-adder or multiplexer recovery.
- Log-hazard gave stable majority5 results near `.96875`, but did not solve
  parity4 or full adder.
- Hard max remains a useful semantic/gradient control: it occasionally
  recovers exact functions, but has high seed variability and does not
  dominate smooth candidates.

### Threshold stability

Every best checkpoint was evaluated at thresholds `0.3, 0.4, 0.45, 0.5,
0.55, 0.6, 0.7`. XOR2 solutions from hard max, Lehmer, log-hazard and
softmax remained exact across all thresholds. For the harder tasks, most
checkpoints did not remain exact across the interval; probabilistic OR was
especially threshold-sensitive on full adder and multiplexer. Threshold
stability is therefore useful evidence of discretization margin, not merely a
post-hoc threshold choice.

## Stage B3 — exact 4-bit XOR on Kaggle

B3 used the complete 256-row `bitwise_xor_truth_table` with the canonical
`8 -> 64 -> two XOR-residual blocks -> 4` network. The four operators were
Lehmer p=2, hard max, probabilistic OR, and softmax-value alpha=16. All runs
used Adam (`lr=0.01`), MSE, batch size 256, 3000 epochs, no temperature,
regularizer, or plateau noise. The verified CUDA result is archived at
`operator_results/stage_b3_kaggle_v6.json` and was produced by Git SHA
`a23ec4d900bf034129bd9a91158b1bd76e3f29c9` on Kaggle kernel version 6.

| operator | seed | minimum continuous MSE | continuous exact | hard exact | Boolean exact | first Boolean recovery |
|---|---:|---:|---:|---:|---:|---|
| Lehmer p=2 | 0 | 2.11e-4 | 1.000 | 1.000 | 1.000 | epoch 375, MSE .0757 |
| Lehmer p=2 | 1 | 2.06e-2 | 1.000 | .500 | .500 | not reached |
| Lehmer p=2 | 2 | 2.18e-4 | 1.000 | 1.000 | 1.000 | epoch 375, MSE .0784 |
| hard max | 0 | 6.32e-2 | .570 | .570 | .500 | not reached |
| hard max | 1 | 8.84e-2 | .621 | .621 | .375 | not reached |
| hard max | 2 | 6.32e-2 | .500 | .500 | .500 | not reached |
| probabilistic OR | 0 | 9.83e-5 | 1.000 | .063 | 1.000 | epoch 200, MSE .0945 |
| probabilistic OR | 1 | 1.00e-4 | 1.000 | .063 | 1.000 | epoch 200, MSE .0959 |
| probabilistic OR | 2 | 9.67e-5 | 1.000 | .063 | 1.000 | epoch 200, MSE .0984 |
| softmax value alpha=16 | 0 | 1.20e-2 | 1.000 | 1.000 | .500 | not reached |
| softmax value alpha=16 | 1 | 1.37e-2 | .984 | .734 | .734 | not reached |
| softmax value alpha=16 | 2 | 5.85e-2 | .898 | .484 | .156 | not reached |

### B3 interpretation

Lehmer p=2 is the strongest deterministic candidate but remains seed
sensitive: two of three runs recover the exact Boolean XOR function, while
the third reaches continuous exact accuracy without recovering the Boolean
function. This is positive but not yet reliable evidence of functional
discretization consistency. Probabilistic OR reaches continuous and
thresholded Boolean exactness for all three seeds, but its hard-max outputs
are poor (`.0625` exact); this separates the aggregation-semantic gap from
the parameter-threshold result and confirms that classification alone is not
enough. Hard max has the expected semantic control behavior but weak
optimization. Softmax alpha=16 remains both less stable and less
discretization-consistent than the leading candidates.

The B3 Kaggle execution initially exposed three infrastructure bugs, all
fixed in separate commits and preserved in history: direct-script import
path (`88ab5a2`), missing packaged output directories (`8882f1d`), and CUDA
device placement for the freshly created discrete model (`a23ec4d`). The
first two failed kernel versions remain archived by Kaggle; version 6 is the
first successful run.

B3 is complete. The next experiment should be chosen after reviewing the
seed-1 Lehmer p=2 failure and the probabilistic-OR hard/Boolean divergence;
MNIST and further operator changes remain deferred.
