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

With the present finite-width implementation, Boolean zero fractions drift
upward through the chain rather than remaining exactly at 0.5. For example,
at input `p0=0.5`, the depth-12 zero fraction was approximately `0.614` with
`BIAS_CURRENT` and `0.684` with `BIAS_ONE`. This indicates that the simple
independence map does not fully predict the actual finite network; the bias-one
hypothesis is not yet supported by this diagnostic.

## I2 training factorial

Not run yet. It must remain a separate 2x2 test of edge initialization
(`CURRENT_EDGE` vs mean-field Gaussian sigma 4) and bias initialization
(`BIAS_CURRENT` vs `BIAS_ONE`) after the propagation diagnostics are reviewed.

## Files

- `research/meanfield_initialization.py`
- `research/analyze_meanfield_initialization.py`
- `research/tests/test_meanfield_initialization.py`
- `research/operator_results/initialization_i0_i1.json`

