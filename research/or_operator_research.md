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
