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

