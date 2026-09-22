# I5 — Gate Discretization Regularization

## Scope and provenance

I5 continued the verified `I2-B seed3 best_continuous_mse` checkpoint.  The
parent was re-evaluated before training: continuous MSE was
`2.9311285e-4`, continuous exact accuracy was `1.0`, and the thresholded
Boolean network had exactly two wrong rows, `[239, 255]`.

The fixed task loss was top-8-row binary cross entropy.  Four arms were run
for 3000 fresh-Adam steps at `lr=0.01`, with no weight decay, clipping,
scheduler, noise, or bias regularization:

| arm | regularizer | rho | calibrated lambda | final MSE | final E_inf | final Boolean wrong rows |
|---|---|---:|---:|---:|---:|---:|
| `polar_rho0.10` | layer-balanced `mean(4*g*(1-g))` | .10 | 0.2866260 | 1.4267e-4 | 0.2441 | 2 |
| `polar_rho0.50` | layer-balanced `mean(4*g*(1-g))` | .50 | 1.4331301 | 0.1616e-3 | 0.2669 | 2 |
| `margin4_rho0.10` | layer-balanced `mean((relu(4-|r|)/4)^2)` | .10 | 0.3579759 | 1.4552e-4 | 0.2354 | 2 |
| `margin4_rho0.50` | layer-balanced `mean((relu(4-|r|)/4)^2)` | .50 | 1.4678e-4 | 0.2484 | 2 |

The task/regularizer gradient ratios at step zero were respectively `.10`,
`.50`, `.10`, and `.50` (within floating point tolerance).  Each arm has 17
evaluation points at the requested steps and four SHA-256-verified
checkpoints (`best_E_inf`, `best_task`, `best_boolean`, `final`).

## Cleanup and infrastructure

Removed research Python caches and archive files.  The archive search now
returns no files and no `research/**/__pycache__` directories remain.  The
`.gitignore` now covers Python caches and research archive extensions without
ignoring scientific result directories.

The I4 runner was corrected so separate arm invocations merge by `(seed, arm)`
instead of overwriting the canonical JSON.  The canonical I4 JSON still
contains the three seed-3 arms; no missing historical trajectories were
fabricated.  I5 used separate raw arm files and a deterministic merge into
`operator_results/i5_gate_regularization_results.json`.

## Trajectories and topology

No I5 arm reached Boolean exactness.  All remained at two wrong rows through
step 3000.  The difficult output bit on rows 239 and 255 stayed around:

| arm | row 239 bit 2 | row 255 bit 2 |
|---|---:|---:|
| parent | .2509 | .2521 |
| polar .10 | .2441 | .2431 |
| polar .50 | .2669 | .2666 |
| margin4 .10 | .2354 | .2353 |
| margin4 .50 | .2484 | .2475 |

The focused regularizers improve average endpoint metrics, especially at
rho=.10, but they do not cross the Boolean topology barrier.  The two rows
remain in the top-8 worst-row set at the final evaluation for every arm.

There were no recovery intervals and therefore no decisive topology crossing
to localize.  Edge-mask changes were sparse (typically only a few bits per
evaluation), while unregularized bias parameters crossed threshold much more
often.  Thus the regularizer did not directly produce a successful edge
topology transition.

## Gate polarization

The polarization regularizer strongly reduced the layer-mean gate distance
`D_g`:

| arm | initial mean `D_g` | final mean `D_g` | final fraction `|r|>4` |
|---|---:|---:|---:|
| polar .10 | .0307 | .00189 | .9903 |
| polar .50 | .0307 | .00130 | .9929 |
| margin4 .10 | .0307 | .00877 | .8876 |
| margin4 .50 | .0307 | .00868 | .8773 |

The finite-margin formulation still made gates decisively Boolean-like, but
stopped pushing most logits beyond `|r|=4`.  The continuing polarization
formulation produced lower `D_g` without improving Boolean recovery.  This is
evidence that gate polarization alone is insufficient: the model can harden
the wrong functional topology.

At step 500, the scaled regularizer gradient was no longer fixed at its
initial ratio.  For rho=.10 the observed global ratios were about `.43`
(polarization) and `.25` (margin); for rho=.50 they were about `.66` and
`.60`.  Per-layer cosines had mixed signs, including strongly negative values
in later layers, so regularization was not uniformly aligned with the task.

## Comparison with I4 control

The I4 top-8-row BCE control ended at MSE `7.4935e-5`, E_inf `.1795`, and
two wrong Boolean rows.  I5 regularization reduced gate ambiguity but did not
improve the discrete result; its best E_inf was `.2354` (`margin4_rho0.10`),
which is worse than the I4 control.  The rho=.50 arms also degraded endpoint
quality relative to rho=.10.

## Conclusion

For this already-learned seed-3 function, gate-only regularization did not
repair the remaining Boolean errors.  The strongest observed effect was
parameter polarization, not functional recovery.  The failed rows remained
at a stable fractional output near `.24–.27`, and the sparse edge-mask changes
did not change the wrong topology.  The finite-margin regularizer is preferable
to unbounded polarization if a later experiment needs hardening without
unnecessary logit growth, but neither should become a default from this
single continuation.

The next single experiment should inspect or target the specific internal
path producing output bit 2 on rows 239/255, rather than adding stronger
global gate pressure.  No bias regularization, initialization sweep, or MNIST
run was started.

## Artifacts

- Results: `research/operator_results/i5_gate_regularization_results.json`
- Checkpoints: `research/operator_results/i5_gate_regularization_checkpoints/`
- Figures: `research/figures/i5_mse.png`, `i5_einf.png`,
  `i5_boolean_wrong_rows.png`, `i5_gate_distance.png`
- Runner: `research/run_i5_gate_regularization.py`
- Tests: `research/tests/test_i5_regularizers.py`

