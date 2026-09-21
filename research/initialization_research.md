# Mean-Field Initialization Research

## Motivation

This phase tests initialization only. The OR operator remains Lehmer
`p=2`, with the existing sigmoid edge mapping, XOR residual topology, loss,
optimizer, and Boolean conversion unchanged.

## Current initialization (CURRENT_BASELINE)

The Stage-B model is `SigmoidOrModernLogicGateNet` in `models.py` and
`SigmoidOrLogicLayer` in `layers.py`.

For each logic layer, the current B3 constructor initializes raw edges as a
constant logit from alternating effective gate values (default `0.75, 0.25`):

```python
raw_edge = logit(gate_initialization)
g = sigmoid(raw_edge)
```

The B3 experiment overrides the bias initializer with:

```python
bias_raw ~ Normal(0.5, 0.1**2)
b = leaky_clamp(bias_raw, 0, 1, 0.1)
```

The effective Boolean threshold is `g >= 0.5` and `b >= 0.5`.

## Mean-field derivation

For fan-in `m`, selected-edge probability `s`, and polarity-one probability
`q`, the mean-field zero-probability map is:

```text
F(p) = [1 - s*(1-q+(2q-1)*p)]^m
```

Requiring `F(0.5)=0.5` gives the stable numerical target:

```text
s_target(m) = -2*expm1(-ln(2)/m)
```

The expected selected fan-in approaches `2 ln(2) = 1.38629` as `m` grows.
For `m=8`, `s_target=0.1659919`; for `m=64`, `s_target=0.0215440`.

## Gaussian threshold-matched initializer

Raw edge logits use:

```text
r ~ Normal(mu_m, sigma^2)
mu_m = sigma * Phi^{-1}(s_target(m))
g = sigmoid(r)
```

Thus `P(g>=0.5)=P(r>=0)=s_target` without Bernoulli initialization.
The implementation is `research/meanfield_initialization.py`.

The tested scales are `sigma=2,4,6`; sigma 4 is the primary candidate.
`BIAS_ONE` uses the existing leaky-clamp mapping with
`bias_raw ~ Normal(1.0, 0.1**2)`. `BIAS_CURRENT` preserves the B3 baseline.

## I0 initializer-only statistics

The diagnostic used 32 independent seeds. Observed selection fractions were
close to the target:

| fan-in | sigma | target selected | observed selected | observed fan-in | mean g |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 0.16599 | 0.17090 | 1.367 | 0.2317 |
| 8 | 4 | 0.16599 | 0.17090 | 1.367 | 0.1886 |
| 8 | 6 | 0.16599 | 0.17090 | 1.367 | 0.1780 |
| 64 | 2 | 0.02154 | 0.02139 | 1.369 | 0.0656 |
| 64 | 4 | 0.02154 | 0.02139 | 1.369 | 0.0326 |
| 64 | 6 | 0.02154 | 0.02139 | 1.369 | 0.0263 |

The threshold probability is independent of sigma as designed. Larger sigma
reduces the mean fractional gate value but increases sigmoid saturation.

## I1 plain-chain propagation

The diagnostic plain chain uses width 64, depth 12, controlled binary inputs,
and no residual connections. It is a diagnostic of the mean-field heuristic,
not a replacement architecture. Results are stored in
`research/operator_results/initialization_i0_i1.json`.

The original single-chain result is retained, but it is not treated as a
rejection of the theory. At expected selected fan-in about 1.38, graph-level
variance is large. The early empirical trajectory is qualitatively
consistent with the predicted oscillatory trajectory, so replication is
required before judging the hypothesis.

The continuous XOR has an additional issue: `a=x+b-2xb` has
`da/dx=1-2b`. A narrow `Normal(0.5,0.1)` bias distribution therefore
suppresses input sensitivity even though its threshold polarity probability
is about one half.

## I1R — replicated propagation and continuous-bias analysis

`research/analyze_meanfield_i1r.py` adds the requested replicated diagnostic
for `CURRENT`, `ONE`, and `BALANCED_POLARIZED` bias distributions and edge
sigmas 2, 4, and 6. It records Boolean propagation bands, continuous
activation quantiles, selected-fan-in histograms, and bias signal-gain
statistics. The output is:

`research/operator_results/initialization_i1r.json`

The checked-in run is a smoke-scale execution (4 network realizations,
1024 Boolean rows, 16 continuous rows) because the full 64-realization,
8192-row continuous sweep is computationally large. The script defaults to
the required 64 realizations and 8192 Boolean rows for a full rerun.

The smoke result confirms the intended separation: `BIAS_CURRENT` has mean
`|1-2b|` about `0.159`, `BIAS_ONE` about `0.920`, and
`BIAS_BALANCED_POLARIZED` about `0.902`, while the threshold polarity of the
balanced distribution remains about one half. This supports measuring the
continuous bias distribution separately from Boolean polarity probability.

## I2 training factorial

Not run yet. It must remain a separate test of edge initialization
(`CURRENT_EDGE` vs mean-field Gaussian sigma 4) and bias initialization
(`BIAS_CURRENT` vs `BIAS_ONE`) after the propagation diagnostics are reviewed.

## Files

- `research/meanfield_initialization.py`
- `research/analyze_meanfield_initialization.py`
- `research/tests/test_meanfield_initialization.py`
- `research/operator_results/initialization_i0_i1.json`
