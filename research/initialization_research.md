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

## I2 — initialization reliability training

I2 trained the fixed Lehmer-p2 modern network on the complete 256-row XOR
truth table using the historical inclusive loop (3001 optimizer updates for
`epochs=3000`). The five conditions and five seeds were run on Kaggle CUDA.
All checkpoints were reloaded and reevaluated before being included in the
manifest; 100 checkpoint files were downloaded and their SHA-256 hashes
matched the remote manifest.

| condition | Boolean recovery | median recovery epoch | median best MSE | mean final Boolean exact | bad basins |
|---|---:|---:|---:|---:|---:|
| I2-A historical current/current | 4/5 | 500 | 0.000211 | 0.900 | 1 |
| I2-B MF sigma2 + ONE | 1/5 | 400 | 0.001259 | 0.848 | 4 |
| I2-C MF sigma4 + ONE | 0/5 | — | 0.019390 | 0.541 | 5 |
| I2-D MF sigma2 + balanced polarized | 1/5 | 425 | 0.006141 | 0.803 | 4 |
| I2-E MF sigma4 + balanced polarized | 0/5 | — | 0.024392 | 0.470 | 5 |

The historical baseline recovered the exact Boolean function for four of five
seeds. The sparse mean-field initializers did not improve recovery in this
experiment. Sigma 2 was consistently less damaging than sigma 4, while sigma
4 produced more Boolean-like initial gates but substantially worse training
basins. The one successful MF sigma2/ONE run recovered at epoch 400; the one
successful MF sigma2/balanced run recovered at epoch 425.

The paired sigma assertions passed: sigma2 and sigma4 used identical initial
thresholded edge masks for each matched seed, while their continuous gate
values differed. The paired edge topology therefore does not explain the
sigma2/sigma4 outcome; the difference is continuous parameter geometry and
sigmoid saturation. The CURRENT and BALANCED_POLARIZED bias streams likewise
preserved matched polarity masks in the paired initializer tests.

Post-processing swept the saved Boolean checkpoints over thresholds from
`0.20` through `0.80` in steps of `0.025`, using the complete truth table.
The successful baseline circuits were functionally stable over intervals
`[0.475, 0.775]`, `[0.475, 0.750]`, `[0.500, 0.800]`, and `[0.275, 0.525]`
for seeds 0, 2, 3, and 4 respectively. The single successful MF sigma2/ONE
circuit (seed 4) was exact on `[0.425, 0.650]`; the single successful MF
sigma2/BALANCED_POLARIZED circuit (seed 3) was exact on `[0.500, 0.800]`.
The unsuccessful Boolean checkpoints had no exact threshold in this sweep.
These are functional margins of saved circuits, not thresholds selected for
training or model selection.

These results do not show that mean-field initialization is useless in
general. They show that the proposed sparse target fan-in and polarized bias
conditions are not a drop-in improvement for this fixed architecture and
training recipe. The historical dense edge initialization remains the best
candidate for the next decision.

### I2 artifacts

- `research/operator_results/initialization_i2_results.json`
- `research/operator_results/initialization_i2_summary.json`
- `research/analyze_i2_threshold_stability.py`
- `research/operator_results/initialization_i2_checkpoints/`
- `research/figures/initialization_i2_training_curves.png`
- `research/figures/initialization_i2_recovery.png`
- `research/figures/initialization_i2_mse_vs_boolean.png`

## I2F — missing initialization factorial cells

I2F completed the four cells omitted from the first initialization factorial using the same historical B3 loop (3001 optimizer updates), Lehmer p=2, Adam at 0.01, and the complete 256-row XOR table. The Kaggle run used commit `811bf4f3aec6c88270166dadc6d082dde01e5975`; all 76 exported checkpoints match the SHA-256 manifest.

| condition | recovery | median best MSE | mean final Boolean exact |
|---|---:|---:|---:|
| Historical edge + ONE (I2-F) | 2/5 | 0.001813 | 0.8711 |
| Historical edge + BALANCED_POLARIZED (I2-G) | 0/5 | 0.097530 | 0.3633 |
| MF sigma2 + CURRENT (I2-H) | 1/5 | 0.002054 | 0.7625 |
| MF sigma4 + CURRENT (I2-I) | 2/5 | 0.019036 | 0.6688 |

Adding the missing cells does not change the earlier conclusion: the historical edge/current-bias baseline remains the strongest fixed recipe. Bias near one can work with historical edges in some seeds, while balanced-polarized bias is poor here. Mean-field edges remain seed-sensitive and generally worse than the historical dense pattern. These results are archived separately from I2.
