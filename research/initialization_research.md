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

## I1R smoke

The earlier smoke artifact is preserved as
`research/operator_results/initialization_i1r_smoke.json`. It used four
network realizations, 1024 Boolean rows, and 16 continuous rows and is not
used for final conclusions.

## I1R full

The canonical full run is stored at
`research/operator_results/initialization_i1r_full.json`. It ran on CUDA with
64 plain-chain seeds and 64 residual-network seeds, Boolean batch 8192,
continuous batch 512, depth 12, all seven `p0` values, sigmas 2/4/6 for the
plain chain, and sigmas 2/4 for the residual network. The CUDA Boolean smoke
comparison passed exactly.

### Boolean balance and theory

The measured edge selection probability was about `0.02197`, close to the
target `0.021544`, with fan-in frequencies approximately:

```text
K=0       0.239
K=1       0.355
K=2       0.244
K=3       0.109
K>=4      0.053
```

`CURRENT` and `BALANCED_POLARIZED` had measured polarity probability about
`0.499` and therefore share the ideal `q=0.5` Boolean mean-field curve.
`ONE` had `q=1.0` and followed the predicted alternating trajectory. The
empirical Boolean means stayed near one half after replication, but individual
depth ratios had high variance when the denominator was close to zero; signed
rho values are retained in the JSON rather than summarized as a guaranteed
contraction theorem.

### Continuous signal health

The exact bias mapping produced these literal signal statistics:

| bias | mean `|1-2b|` | literal/input variance ratio |
|---|---:|---:|
| CURRENT | 0.163 | 0.042 |
| ONE | 0.920 | 0.860 |
| BALANCED_POLARIZED | 0.898 | 0.863 |

Thus CURRENT and BALANCED_POLARIZED have almost identical Boolean polarity
probabilities but very different continuous XOR signal preservation. This is
the intended paired initialization control.

For the plain chain, increasing sigma raised continuous activation means and
variance while leaving the initial threshold topology paired. Sigma changes
the fractional gates and sigmoid derivative statistics, not the expected
threshold mask.

### Exact residual-network trace

The residual diagnostic now uses a separate exact discrete network and the
single `discrete_residual_trace()` implementation. At sigma 4, the final
head Boolean zero fraction was approximately `0.536` for CURRENT and
BALANCED_POLARIZED and `0.518` for ONE. Continuous head means were about
`0.280`, `0.286`, and `0.285` respectively. These are not expected to match
Boolean one probabilities numerically; they measure different computations.

The head remains systematically low in the continuous trace, especially for
sigma 2. No head-specific correction is introduced in I1R.

### I1R decision

The full diagnostic supports carrying the following configurations into a
future initialization-only training comparison, while retaining the current
baseline:

```text
CURRENT_EDGE + CURRENT
MEANFIELD sigma2 + ONE
MEANFIELD sigma4 + ONE
MEANFIELD sigma2 + BALANCED_POLARIZED
MEANFIELD sigma4 + BALANCED_POLARIZED
```

Sigma 6 is not promoted automatically because its sigmoid derivatives are
more saturated. I2 training has not been started.

## Files

- `research/meanfield_initialization.py`
- `research/analyze_meanfield_initialization.py`
- `research/tests/test_meanfield_initialization.py`
- `research/operator_results/initialization_i0_i1.json`
