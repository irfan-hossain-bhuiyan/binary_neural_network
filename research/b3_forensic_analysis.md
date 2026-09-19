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
