# I10 — Causal-Agnostic Layer-Balanced Boolean STE

## A. Calibration

I10 computes task and Boolean STE gradients independently at step 0, then
freezes one multiplier per expectation layer. No wrong row, repair edge, or
repair layer is used in calibration. The Boolean gradients are applied only
to raw edge parameters; biases receive task gradients only.

For seed 3, with `rho=.50`:

| layer | task norm | Boolean norm | lambda | scaled/task | coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.8925e-3 | 6.5460e-4 | 1.4456 | 0.5000 | 0.00586 |
| 1 | 8.0008e-4 | 1.1374e-5 | 35.1720 | 0.5000 | 0.00024 |
| 2 | 1.8472e-3 | 1.6536e-4 | 5.5854 | 0.5000 | 0.00024 |
| 3 | 4.9803e-4 | 0 | 0 | 0 | 0 |
| 4 | 4.7320e-4 | 2.1462e-5 | 11.0242 | 0.5000 | 0.00537 |
| 5 | 3.6374e-3 | 3.0472e-5 | 59.6843 | 0.5000 | 0.00391 |

The layer-4 value is an emergent result of per-layer calibration, not a
selected hyperparameter. Effective pressure is recorded per reached raw edge
in the JSON.

## B. Forward and gradient verification

Both `STE_SIGMOID` and diagnostic `STE_CONST025` Boolean forwards exactly
matched `to_discrete(.5)` at every traced stage. The manual compositor makes
one Adam update per step. Non-edge parameters receive task gradients only.
The focused tests cover synthetic zero-gradient layers, calibration ratios,
bias-gradient isolation, and one-step optimizer behavior.

At seed 3 step 0, the known repair edge was recorded only as a diagnostic:

```text
task gradient:       +2.7021e-6
Boolean gradient:    -1.1182e-5
composed gradient:   -1.2057e-4
```

This confirms the composed direction is initially favorable, without using
that edge to choose any multiplier.

## C. Seed 3 gate result

| method | calibration | first Boolean exact | final wrong rows | final MSE |
|---|---|---:|---:|---:|
| I8 STE | known layer | ~400 | 0 | ~1.0184e-4 |
| I9 STE | global | never | 2 | 9.1586e-5 |
| I10 STE | per-layer generic | never | 128 | 6.0001e-2 |
| I10 control | none | never | 2 | 7.4935e-5 |

I10 therefore fails the required seed-3 gate. The Boolean auxiliary begins
with the desired local gradient scale, but the continuous function degrades
rapidly: the STE run reaches 67 wrong rows by evaluation step 750 and 128 by
step 1000. The tracked edge remains negative (`-4.675` at step 3000), so the
known one-bit repair does not occur.

The I10 STE run had no exact Boolean recovery, no recovery streak, and no
threshold interval containing `.5` at the final checkpoint. Its final edge
and bias mask displacement from the parent was 420 and 5,473 bits. The
control displacement was 304 and 4,339 bits while retaining the original two
wrong rows.

## D. Replication decision

The seed-3 gate failed, so seeds 0–2 were not trained. This follows the
specified stop rule; no broad recovery-rate claim is made. Seed 4 was not
continued in I10 because the gate failed; I9 already established its
zero-Boolean-gradient stability control.

## E. Interpretation

Layer-balanced norm matching is not sufficient by itself. Equalizing the
initial per-layer Boolean/task norm ratio gives very large multipliers in
sparse-gradient layers, and the resulting fixed composition damages the
continuous solution before a Boolean topology transition occurs. This is a
negative result for the proposed static layer-balanced composition, not for
Boolean STE in general.

The next single experiment should measure a bounded or task-preserving
layer-balanced composition (with an explicitly controlled update budget),
without using causal knowledge. That follow-up was not run in I10.
