# I4 Worst-Case Endpoint Optimization

I3 did not show that BCE universally guarantees Boolean recovery. Seed 2 recovered under mean BCE, seed 4 was already exact, and seed 3 retained two wrong rows. I4 tests whether concentrating BCE on rare hard bits or rows fixes that seed-3 basin.

| arm | first Boolean recovery | final MSE | final E_inf | final wrong rows | final row239 bit2 | final row255 bit2 |
|---|---:|---:|---:|---:|---:|---:|
| mean_bce | none | 0.00020052 | 0.320502 | 2 | 0.320502 | 0.320138 |
| top16_bit_bce | none | 7.59775e-05 | 0.19427 | 2 | 0.19427 | 0.193744 |
| top8_row_bce | none | 7.4935e-05 | 0.179478 | 2 | 0.179478 | 0.16627 |

## Interpretation

All three seed-3 arms retained the same two Boolean errors for all 3000 steps. Top-16 bit BCE and top-8 row BCE reduced the difficult target-zero output more than mean BCE, but neither crossed the discrete topology needed for recovery. The result is therefore an optimization-basin/topology barrier, not merely dilution under mean reduction.

Mask hashes and edge/bias bit flips are recorded at every evaluation. No Boolean-recovery interval or decisive threshold crossing exists in this run.

At steps 0, 100, and 500, hard-row and other-row gradient norms and per-layer cosine similarities are recorded. The hard-row gradients are not absent; their effect is mixed with the rest of the table and does not produce a threshold transition under these pure losses.

Secondary seed-2 validation was not run. No hybrid loss was introduced.
