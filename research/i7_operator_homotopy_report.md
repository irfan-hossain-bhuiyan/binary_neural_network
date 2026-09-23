# I7 — Lehmer-to-Max Operator Homotopy

## Scope and provenance

I7 starts from the verified `I2-B seed3 best_continuous_mse` checkpoint
(`SHA-256` is stored in `i7_operator_homotopy_results.json`). No weights,
initialization, architecture, bias mapping, residual, or gate parameterization
was changed. Every continuation used fresh Adam (`lr=0.01`, no weight decay)
and the top-8-row BCE task loss. Operator type and Lehmer `p` are stored
outside the state dict and are restored explicitly when checkpoints are
loaded.

The research tree was checked at completion: no archive files, Python bytecode,
or `__pycache__` directories remain. The existing I3–I6 results and
checkpoints were preserved. The I5 Markdown row for `margin4_rho0.50` retains
the canonical values (`lambda=1.7898797`, final MSE `1.4678e-4`, E_inf
`.2484`, two wrong rows); its JSON was not modified.

## No-training p sweep

The parent is continuously exact through `p=3`. At `p=4`, it first loses
continuous exactness (two rows), while its thresholded Boolean network is
unchanged.

| operator | continuous MSE | E_inf | continuous exact | continuous wrong rows | Boolean exact |
|---|---:|---:|---:|---:|---:|
| Lehmer p=2 | 2.9311e-4 | .2521 | 1.0000 | 0 | .9921875 |
| Lehmer p=2.5 | 3.9341e-4 | .3350 | 1.0000 | 0 | .9921875 |
| Lehmer p=3 | 6.3591e-4 | .4401 | 1.0000 | 0 | .9921875 |
| Lehmer p=4 | 1.2026e-3 | .6043 | .9921875 | 2 | .9921875 |
| Lehmer p=6 | 1.7220e-3 | .7109 | .9921875 | 2 | .9921875 |
| Lehmer p=8 | 1.8489e-3 | .7314 | .9921875 | 2 | .9921875 |
| hard max | 1.8409e-3 | .7378 | .9921875 | 2 | .9921875 |

For rows 239 and 255, bit 2 rises from approximately `.251` at p=2 to
`.738` under hard max, while the target is zero. The Boolean output remains
one because parameter threshold masks do not depend on p.

The JSON also includes the per-layer causal traces for these rows, including
the block1 layer1/layer2/residual source indices and head bit 2, so the first
semantic divergence can be inspected without reconstructing the sweep.

## Continuation results

| arm | final operator | final continuous MSE | final continuous exact | final Boolean wrong rows | repair edge raw value | recovered? |
|---|---|---:|---:|---:|---:|---|
| p2 control | Lehmer p=2 | 7.4935e-5 | 1.0000 | 2 | -9.765 | no |
| fixed p4 | Lehmer p=4 | 2.5265e-4 | 1.0000 | 2 | -7.584 | no |
| direct hard max | hard max | 4.8830e-4 | 1.0000 | 2 | -6.527 | no |
| homotopy | hard max transition | non-finite at step 2250 | non-finite | 112 at transition | -8.703 | no |

The p2 and p4 arms preserve continuous exactness but never change the two
Boolean failures. Direct hard max remains numerically finite at the end and
also retains both failures; its continuous exactness briefly dips at an
evaluation point but returns. The homotopy is not successful: the p=4→8
segment sharply degrades the learned function (64 wrong Boolean rows by step
1500 and 112 by step 2250), and the first hard-max evaluation is non-finite.
The non-finite event is recorded rather than silently treated as success.

## Semantic evaluation

Every checkpoint was evaluated under p=2, p=4, p=8, hard max, and the exact
Boolean network. The p2 control becomes increasingly specialized to p2:
p2 remains exact, while p4/p8/hard max remain at the two-row Boolean-near
miss. The p4 arm becomes exact under p4 but not under p8/hard max. Direct
hard max is hard-max-compatible at its own output level most of the time, but
its parameter-threshold Boolean circuit remains unchanged. Homotopy loses
all semantic views as p grows and does not reach a Boolean-compatible basin.

All eight saved checkpoints (final and best-Boolean for each arm) were loaded
into fresh models, restored with their recorded operator metadata, and
reevaluated. Stored SHA-256 hashes and finite metrics matched; the homotopy
checkpoint at the non-finite transition is explicitly marked as such.

## Repair edge and topology

The I6 candidate `block1.layer2.raw_edge[62,10]` starts at approximately
`-5.850` (`g≈.00287`, Boolean bit 0). It never crosses zero:

* p2 moves it monotonically outward to about `-9.765`;
* p4 moves it to about `-7.584`;
* direct hard max moves it to about `-6.527`;
* homotopy moves it to about `-8.703` before destabilizing.

Thus no observed Boolean recovery or topology transition repaired the I6
one-bit edit. All arms exhibit mask changes during optimization, but the
changes do not include the required positive crossing. The p2 task gradient on
the repair edge is already tiny and decreases; direct hard max reports zero
repair-edge gradient at the sampled diagnostics, consistent with winner-take-
all starvation on this parameter.

## Effective contributors and gradient behavior

The corrected concentration diagnostic excludes all-zero contribution rows
from normalized weights and reports their valid fraction. At initialization,
the stem has mean effective contributor count about 3.18 for Lehmer p=2 and
2.37 for p=4, versus exactly 1 for hard max. At final p2, the stem mean is
about 2.42; at final p4 it is about 2.40. Hard max is exactly one selected
contributor by definition. The homotopy becomes increasingly concentrated,
but its task gradients do not move the repair edge toward the needed Boolean
state and the run destabilizes at the hard-max switch.

## Conclusion

I7 does **not** show that operator hardening repairs the two-row Boolean
error. Fixed p4 and direct hard max preserve the near-miss; gradual homotopy
is worse and becomes numerically unstable. The result supports the I6
classification: the learned p2 basin is a continuous/discrete upstream
semantic mismatch, and simply replacing Lehmer with a harder OR does not
provide a useful path to the one-bit Boolean repair.

The next single experiment should therefore be a discrete-aware gradient
method targeted at the identified causal subgraph (for example a carefully
isolated straight-through Boolean objective), with the current p2 model and
all existing diagnostics retained as controls. It was **not** implemented in
I7.
