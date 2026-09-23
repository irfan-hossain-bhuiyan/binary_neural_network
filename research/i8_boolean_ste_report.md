# I8 — Exact Boolean Forward / Straight-Through Gate Gradient

## A. Scope and provenance

I8 starts from the verified I2-B seed3 best-continuous checkpoint. I7 and all
earlier results were preserved. The run uses the unchanged Lehmer p=2
continuous task, fresh Adam at `lr=.01`, no weight decay, and no bias STE,
regularization, homotopy, or operator changes. The canonical result is
`operator_results/i8_boolean_ste_results.json`; checkpoint metadata includes
the parent hash, STE variant, lambda, Git SHA, and reload verification.

The research tree was checked after execution: no archives, bytecode, or
`__pycache__` directories remain.

## B. Exact-forward equivalence

Both `STE_SIGMOID` and `STE_CONST025` use a custom autograd function whose
forward returns the exact thresholded gate and whose backward supplies only the
specified surrogate derivative. Exhaustive 256-row tests matched
`model.to_discrete(.5)` at every stage:

```text
input, stem, block0.layer1, block0.layer2, block0.residual,
block1.layer1, block1.layer2, block1.residual, head: 0 mismatches
```

The Boolean STE loss is therefore exactly the Boolean bit-error MSE in the
forward pass.

## C–G. Gradient diagnostics and choice

At the known repair edge `block1.layer2.raw_edge[62,10]`:

| quantity | task p2 | STE_SIGMOID | STE_CONST025 |
|---|---:|---:|---:|
| raw value | -5.850245 | — | — |
| sigmoid gate | .0028709 | — | — |
| gradient | +2.7021e-6 | -1.1182e-5 | -9.7656e-4 |
| calibrated lambda | — | 11.02419 | .0591982 |
| combined gradient | — | -1.2057e-4 | -5.5109e-5 |

Both STE variants have the desired negative repair direction. The selected
variant was `STE_SIGMOID`: it preserves the ordinary sigmoid slope while the
global layer-4 calibration makes its combined repair gradient negative. The
constant-slope variant is retained in the diagnostic result but was not
trained. Initial gradient ratios are 0.50 by construction (`G_task=4.7320e-4`
on layer 4).

The Boolean auxiliary gradient on the causal layer is nonzero, while the
reported block1.layer1 predecessor gradients are effectively zero at the
parent. Head bit-2 receives a nonzero Boolean gradient.

## H–M. Training trajectories

| arm | first Boolean exact | final MSE | final Boolean wrong rows | final repair-edge raw |
|---|---:|---:|---:|---:|
| control, p2 top8 BCE | never | 7.4935e-5 | 2 | -9.7650 |
| p2 top8 BCE + STE_SIGMOID | step 400 | 1.0184e-4 | 0 | -1.7569 |

The control reproduces the I4/I7 behavior: continuous exactness improves while
rows 239 and 255 remain wrong. The STE arm reaches Boolean exactness at the
step-400 evaluation. It temporarily has MSE `1.9285e-3` at recovery, then
continues improving to `1.0184e-4` at step 3000. Continuous exact accuracy is
1.0 at the final checkpoint, and Boolean exact accuracy remains 1.0.

The repair edge moves substantially upward (`-5.85 → -1.76`) but never crosses
zero. Thus the known I6 one-bit repair did not occur.

## N–O. Topology transition

The first observed recovery interval is step 300→400:

```text
Boolean wrong rows: 2 → 0
edge changes by layer: [4, 5, 14, 13, 13, 0]
bias changes by layer: [31, 259, 267, 215, 248, 5]
```

`block1.layer2[62,10]` did not flip. The interval contains 13 other edge
changes in block1.layer2, including `[2,10]`, `[44,10]`, and `[50,10]`, plus
large normal task-loss-driven bias changes. No unique one-bit replacement was
isolated; the STE found a different exact Boolean topology through coordinated
threshold transitions.

## P. Conclusion

I8 solves the credit-assignment barrier for this near-miss: exact Boolean
forward values plus a surrogate gate derivative produce a Boolean-exact
network where ordinary p2 training remains two rows wrong. The result is
stronger than merely hardening the operator: the auxiliary loss directly
trains the discrete semantics while preserving a valid continuous p2 model.

The next single experiment should replicate the selected all-gate
`STE_SIGMOID` continuation over a small set of failed and successful I2 seeds
with the same provenance controls. A bias STE or causal-layer-only restriction
was not added in I8.
