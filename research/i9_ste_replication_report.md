# I9 — Generalize Exact-Boolean STE Without Causal Knowledge

## A. Metadata and stop rule

I8 metadata handling was corrected in the I9 runner. Controls now record:
`active_losses=[top8_row_bce]`, `ste_auxiliary_active=false`, `lambda=0`, and
`rho=0`. STE records include the active Boolean loss, variant, global gradient
calibration, and lambda. I9 uses global raw-edge norms across all six logic
layers; no causal layer, edge, or error row is used to choose lambda.

The required seed3 generic reproduction was negative, so the broad seed
training replication was stopped as instructed. Seeds 0–4 were still verified
without training; seed4 received its control-only continuation.

## B–D. Parent verification and exact forward

| seed | parent MSE | continuous exact | parent Boolean exact | wrong rows | generic lambda |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.4636e-3 | 1.0 | .6875 | 80 | .8670 |
| 1 | 1.2591e-3 | 1.0 | .6875 | 80 | .05217 |
| 2 | 1.6137e-3 | 1.0 | .8750 | 32 | .7367 |
| 3 | 2.9311e-4 | 1.0 | .9921875 | 2 | 3.4151 |
| 4 | 1.5298e-4 | 1.0 | 1.0 | 0 | 0 |

Both STE variants matched `model.to_discrete(.5)` at every stage for every
verified parent. No forward mismatch was observed.

## E–G. Global calibration and gradient coverage

For seed3:

```text
G_task_global = 4.6192e-3
G_bool_global = 6.7629e-4
lambda        = 3.4151
lambda*G_bool/G_task = 0.500000
```

The generic lambda is much smaller than the I8 causal-layer lambda (`11.02`). The canonical calibration target is `rho=.5`; the recorded seed3 ratio is `0.500000`.
Seed3 Boolean-gradient coverage by layer was approximately:
`.00586, .00024, .00024, 0, .00537, .00391`; only 28 raw-edge parameters
received a nonzero sigmoid-STE gradient. The global task/Boolean gradient
cosine was `.197`.

The constant-slope variant was diagnostic only; its seed3 lambda was `0.5042` and it was not trained.

Across parent diagnostics, sigmoid-STE global coverage was:

```text
seed0  .04297   seed1 .06641   seed2 .02344   seed3 .00586   seed4 0
```

Seed4 had `L_bool=0` and `G_bool=0`, so lambda was correctly set to zero.

The redundant-active diagnostic found nonzero fractions of exact Boolean OR
neurons with at least two active causes (seed3: approximately 8.6%–18.5%
depending on layer). This confirms the product-OR backward has potential
redundancy-induced zero-gradient regions; I9 did not alter that backward.

## H–I. Seed3 generic reproduction

| arm | first Boolean exact | final Boolean wrong rows | final MSE |
|---|---:|---:|---:|
| control | never | 2 | 7.4935e-5 |
| generic STE_SIGMOID | never | 2 | 9.1586e-5 |

The generic STE moved the tracked repair edge upward temporarily, but it did
not cross the Boolean threshold or recover the function. The failed rows
remained present through step 3000. This fails the I9 gate for broad
replication.

The edge-level sign at step zero was still useful: the Boolean gradient on
`block1.layer2[62,10]` was negative. Global calibration nevertheless diluted
the causal layer's influence relative to the I8 causal-layer calibration.

## J. Seed4 stability control

Seed4 began Boolean exact with `L_bool=0` and zero Boolean gradient. Its
control continuation remained Boolean exact through 3000 steps. No STE
continuation was run, as required by the early-stop design.

## K–P. Interpretation

I9 does not establish generalization. It establishes a precise limitation:

```text
Exact Boolean STE can repair the seed3 basin when its auxiliary gradient is
calibrated on the causally relevant layer, but one global all-edge norm ratio
does not reproduce that success.
```

The result is not evidence against Boolean STE itself. It shows that global
layer aggregation is poorly matched to sparse Boolean-gradient coverage:
large gradients in unrelated layers determine lambda, while the few gates
that can change the failing topology receive insufficient effective pressure.

The next single experiment should test **layer-balanced, causal-agnostic STE
calibration** (equalize Boolean/task gradient ratios per expectation layer,
without inspecting wrong rows or a known repair edge). No such follow-up was
run in I9. Constant STE, bias STE, curricula, and new operator experiments
remain unimplemented.
