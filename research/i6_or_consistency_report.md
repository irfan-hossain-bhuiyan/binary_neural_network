# I6 — Lehmer versus Boolean OR consistency

## A. Cleanup and provenance

The research tree was checked before I6. No research archives, Python bytecode,
or `__pycache__` directories remain. The generated root archive `research2.zip`
was also removed. I5 results, checkpoints, figures, and notes were preserved.

The I4 runner now merges separately executed arms by `(seed, arm)` rather than
overwriting the canonical result. No missing historical I4 trajectory was
reconstructed.

The I5 Markdown table was corrected without changing its JSON. In particular,
`margin4_rho0.50` is now shown as lambda `1.78987973695`, final MSE
`1.4677807e-4`, E_inf `.24840765`, and two wrong rows.

## B. Lehmer bound and threshold implication

For nonnegative contributions,

```text
F(v) = sum(v^3) / sum(v^2)
     = sum((v_i^2 / sum(v_j^2)) * v_i)
```

when the denominator is nonzero. It is therefore a weighted average of the
contributions and satisfies:

```text
0 <= F(v) <= max(v).
```

The all-zero case is defined as zero. The unit tests verify the bound over
random vectors and verify that `F>.5` implies `max(v)>.5`. Consequently a
relaxed-zero/hard-one threshold violation can only be:

```text
F < .5 and max(v) >= .5.
```

The diagnostic reconstructs every layer's contributions and asserts that the
reconstructed Lehmer output agrees with the actual layer output to numerical
tolerance.

## C. Parent forensic result

The verified parent is the I2-B seed-3 best-continuous checkpoint. Its head
output bit 2 is:

| row | target | continuous Lehmer F | continuous-input M | hard-path output | Boolean output |
|---:|---:|---:|---:|---:|---:|
| 239 | 0 | .250872 | .400716 | .737752 | 1 |
| 255 | 0 | .252138 | .399572 | .737755 | 1 |

The largest continuous head contribution is source neuron 62. Its exact
Boolean edge is active on both rows, but its continuous source differs from
the exact discrete source. The hard propagation produces a much larger source
and contribution:

| row | source | continuous source | exact source | continuous v | hard-path v | gate |
|---:|---:|---:|---:|---:|---:|---:|
| 239 | 62 | .596108 | 0 | .400716 | .737752 | .992137 |
| 255 | 62 | .596108 | 0 | .399572 | .737755 | .992137 |

The classification is therefore **TYPE U (upstream mismatch)**, not a
head-local Lehmer threshold violation. The head itself has no `F<.5, M>=.5`
violation on these rows; its `M` is below .5 when evaluated on the relaxed
continuous source.

The top sixteen parent head contributions for row 239 are saved in the JSON
and plotted in `figures/i6_head_contributions.png`. Source 62 is the only
exact Boolean-active head edge among the dominant entries. Row 255 has the
same culprit source.

## D. Violation counts

Parent violation counts `(F<.5, M>=.5)` by layer were:

| layer | count | fraction |
|---|---:|---:|
| stem | 0 | 0 |
| block0.layer1 | 75 | .0143 |
| block0.layer2 | 394 | .0240 |
| block1.layer1 | 151 | .0092 |
| block1.layer2 | 712 | .0435 |
| head | 0 | 0 |

The two failed output rows are not themselves violation rows at the head. The
first causal divergence is the relaxed block1 residual feeding source 62:
the continuous residual is about `.596` while the exact discrete residual is
zero. The backward dominant path is recorded for both rows in the JSON,
through block1.layer2 output 62, block1.layer1 output 15, and block0
predecessors.

## E. Discrete repair search

The parent Boolean circuit has two wrong rows, `[239, 255]`.

- All 128 single head edge/bias flips were tested. None produced an exact
  circuit.
- The top 24 head candidates were searched pairwise. No exact two-bit head
  repair exists.
- A causal-subgraph search tested 32 upstream edge/bias candidates.

The minimum discovered repair is **one upstream bit**:

```text
layer:       block1.layer2 (expectation layer 4)
parameter:   edge[out=62, in=10]
old Boolean: 0
new Boolean: 1
wrong rows:  0
wrong bits:  0
```

The parent raw edge is approximately `-5.85` (`g≈.00287`), so this is a
substantial discrete topology decision rather than a near-threshold numerical
tie. The edit is a diagnostic only; no model parameters were changed by the
search.

## F. Threshold-consistency continuation

The causal trace selected only `block1.layer2` for the new semantic
regularizer. The head was not regularized because its relaxed contribution
does not satisfy the violation predicate on the failed rows.

The regularizer was:

```text
mean((F.detach()<.5) * relu(M-.5)^2)
```

over layer 4. At step zero:

```text
task edge-gradient norm = 4.7320116e-4
TC edge-gradient norm   = 4.8281241e-4
lambda                  = 0.09800932
initial gradient ratio  = 0.10
```

Both arms used the same parent, fresh Adam at `lr=.01`, top-8-row BCE, and
3000 steps.

| arm | final MSE | final E_inf | final Boolean wrong rows | first Boolean recovery |
|---|---:|---:|---:|---|
| control top-8 BCE | `7.4935e-5` | `.17948` | 2 | none |
| TC on block1.layer2 | `6.4145e-5` | `.17047` | 2 | none |

The TC arm improved continuous endpoint metrics but did not repair the Boolean
function. Its block1.layer2 violation count increased from 712 to 1558 while
the mean violation gap decreased. The regularizer therefore reduced the size
of many violations without crossing the specific upstream topology bit
`[62,10]`; it did not solve the causal mismatch.

The decisive edge remained `g≈0.000058` at the TC final checkpoint, more
negative than the parent. This confirms that the semantic penalty, as defined,
did not provide pressure toward the needed Boolean edge because the offending
source is an upstream continuous/discrete mismatch rather than an active
`F<.5, M>=.5` head contribution.

## G. Conclusion and next experiment

I6 identifies a concrete one-bit Boolean repair, but the repair is upstream
of the head and is not found by the proposed local threshold-consistency loss.
The current two-row failure is therefore a structural continuous-versus-exact
discrete mismatch, not insufficient gate polarization and not a simple head
Lehmer dilution case.

The next single experiment should be a **causal upstream consistency loss** on
the block1 residual/source-62 path, using the exact discrete activation as a
detached target for the continuous block1 layer output. It should be tested
only after reviewing this forensic result. No bias regularization, generic
polarization sweep, or MNIST run was started.

## Artifacts

- Results and forensic data: `research/operator_results/i6_or_consistency_results.json`
- Checkpoints: `research/operator_results/i6_or_consistency_checkpoints/`
- Runner: `research/run_i6_or_consistency.py`
- Tests: `research/tests/test_i6_or_consistency.py`
- Figures: `i6_head_contributions.png`, `i6_head_contributions_255.png`,
  `i6_violation_by_layer.png`, `i6_rows239_255_F_vs_M.png`
