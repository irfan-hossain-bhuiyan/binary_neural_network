# B3D — Forensic analysis status

## Scope

B3D is intended to analyze the exact 256-row XOR circuits without changing
the architecture or retraining: Lehmer p=2 seeds 0, 1 and 2, plus
probabilistic-OR seeds 0, 1 and 2. The analysis implementation is
`analyze_b3_circuits.py` and is designed to produce contribution, gradient
sign, circuit topology, functional-layer, hardening, threshold-margin and
probabilistic-accumulation diagnostics.

## Checkpoint provenance problem

The verified B3 result is the CUDA Kaggle version 6 artifact
`operator_results/stage_b3_kaggle_v6.json`, produced by commit
`a23ec4d900bf034129bd9a91158b1bd76e3f29c9`. Its canonical checkpoint files
are not available in the downloaded Kaggle output. The Kaggle bootstrap only
copies files from `/kaggle/working/repo/artifacts/checkpoints`; B3 wrote its
checkpoints to `/kaggle/working/repo/research/operator_results/
stage_b3_checkpoints` instead.

There are local files with similar names, but they are stale checkpoints from
an earlier local run. They do not represent the verified CUDA B3 result. For
example:

| operator/seed | local checkpoint MSE | verified B3 MSE |
|---|---:|---:|
| Lehmer p=2, seed 1 | 0.0154101 | 0.0206271 |
| probabilistic OR, seed 0 | 0.2122013 | 0.0000983 |

These differences are large enough that using the local files would give a
different circuit and an invalid forensic conclusion. The analyzer therefore
checks each checkpoint's continuous MSE against the archived Kaggle result
and stops on a mismatch. The current guard correctly stops at Lehmer p=2
seed 1 rather than silently analyzing the wrong model.

## What is established without the missing weights

The result-level B3 evidence remains unchanged: Lehmer p=2 succeeds for
seeds 0 and 2 and stalls at nonzero numerical loss for seed 1; probabilistic
OR reaches continuous and thresholded Boolean exactness in all three Kaggle
runs while its same-parameter hard-max evaluation is poor. These facts do
not identify the first divergent layer, gradient-sign cause, or contribution
accumulation mechanism.

## Required next action

Run a checkpoint-export-only reproduction of the exact committed B3 job, with
the same configuration and seed set, copying each saved `.pt` file into
`/kaggle/working/artifacts/checkpoints` before the bootstrap removes the
source tree. Then download and verify all six files against the archived B3
metrics, and run `analyze_b3_circuits.py`. This is infrastructure recovery,
not a new operator or architecture hypothesis; no B3D conclusions should be
drawn until it is done.

The requested B3D figures are intentionally not treated as valid outputs yet.
The analyzer may have generated exploratory files from stale checkpoints
locally, but they must not be archived or cited as B3D evidence.

## B3R replication and verified forensic analysis

B3R was run as a separate experiment on Kaggle kernel version 2 from Git SHA
`382df82a87fb79f2138ca45c6d6aca0e96daca4e`. It used the exact B3 training
configuration and exported 65 checkpoints. Every downloaded checkpoint hash
matches the B3R manifest in `operator_results/stage_b3r_results.json`.

B3R is a strong qualitative reproduction: all 12 minimum-MSE and Boolean
results match the historical B3 values to the recorded precision. This makes
the seed-specific Lehmer comparison and the probabilistic-OR mechanism
appropriate for forensic study, while keeping B3 and B3R as separate records.

### Lehmer p=2

Seeds 0 and 2 recover the complete Boolean XOR function. Their functional
threshold interval is `[0.20, 0.80]`, so the result is robust across the
tested threshold range. Every single-layer hardening test remains exact.

Seed 1 reaches continuous exact accuracy but its best numerical MSE remains
`0.020627`; its thresholded circuit is wrong on 128 of 256 rows and only
output bit 0 is exact. The wrong rows are therefore a systematic single-bit
failure, rather than diffuse errors across all four output bits. Hardening
`block0.layer2` gives exact accuracy `.6875`, the worst individual layer
result, while hardening `block1.layer1` or `block1.layer2` gives `1.0`.

At the best checkpoints, the Lehmer p=2 negative-gradient fractions are:

| layer | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| stem | .527 | .565 | .529 |
| block0.layer1 | .124 | .935 | .153 |
| block0.layer2 | .058 | .218 | .001 |
| block1.layer1 | .029 | .902 | .043 |
| block1.layer2 | .001 | .203 | .003 |
| head | .470 | .588 | .464 |

Seed 1 is therefore in a distinct loser-suppression basin: the first logic
layer of block 0 and both layers of block 1 retain a majority of negative
Lehmer contribution derivatives, unlike the two successful seeds. This is a
strong correlation with failure, not a causal proof. The parameter-level
thresholded circuit Hamming distances are large (`4741`, `4685`, and `5030`
between the three seed pairs), so seed success is not explained by a single
shared wiring pattern.

### Probabilistic OR

All three seeds retain exact thresholded XOR and have functional threshold
interval `[0.275, 0.80]`. Their same-parameter hard-max exact accuracy is only
`.0625`, and hardening either second residual-layer (`block0.layer2` or
`block1.layer2`) collapses to `.0625` while hardening the surrounding layers
stays exact.

The contribution traces show why. In block 0 layer 1, representative output
neurons have probabilistic-OR values near `.9998` while their largest single
contribution is only about `.18`; in block 0 layer 2, the aggregate is exactly
`1.0` with many contributions around `.5` and no single max near one. This is
fractional accumulation across the 64-wide fan-in. Replacing the aggregate by
max destroys the continuous computation, but thresholding the gates still
produces the correct Boolean topology. In these checkpoints every gate is
selected (`g >= .5`), while polarity bits remain mixed, showing that full
parameter polarization is not required for functional Boolean recovery.

### Interpretation

B3R supports three separate notions: parameter polarization, hard-operator
equivalence, and functional discretization. Probabilistic OR has poor
hard-operator equivalence and weak polarization, yet exact thresholded
function recovery. Lehmer p=2 has much stronger hard/Boolean agreement when
it enters a successful basin, but seed 1 remains an optimization-basin
failure at nonzero numerical loss. These results do not justify changing the
operator or optimizer yet.

The B3R machine-readable forensic output is
`operator_results/b3r_forensic_analysis.json`. Figures are in `figures/`:
`b3_lehmer_seed_comparison.png`, `b3_layer_hardening.png`,
`b3_functional_threshold_margin.png`, and `b3_prob_or_accumulation.png`.
