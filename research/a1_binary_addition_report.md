# A1 — Exact Binary Addition Benchmark

## Audit and provenance

The latest completed experiment before A1 was M2. Its canonical JSON and report were retained; no scientific rerun was performed during the cleanup or A1 setup. A1 ran on Kaggle with the recorded source commit and no model checkpoints.

Kaggle source commit: `0a7b6bd879bb18bc5c79b935ba34ca5d86baca82`
Raw result SHA256: `54686c2e4d74c9e827d345cf32384c7425f7579dd5c0d0dbacc8a5aeb01d0391`
Device: `cuda` (Tesla T4)

## Task

A1 exhaustively evaluates unsigned 4-bit addition. Inputs are `[a0,a1,a2,a3,b0,b1,b2,b3]`, outputs are `[s0,s1,s2,s3,s4]`, and all 256 rows are used. Integer arithmetic and independent ripple-carry equations agreed exactly. XOR has no cross-bit dependency; addition requires recursively propagated carries, so higher output bits have longer logical paths.

Target bit frequencies: `[0.5, 0.5, 0.5, 0.5, 0.46875]`. Carry-chain counts: `{'0': 81, '1': 83, '2': 60, '3': 24, '4': 8}`. Full-adder smoke test: `{'rows': 8, 'passed': True}`.

## Adam scale diagnostic

The 10× MSE update/global-norm ratio was `1.29235`. This is not a training arm; it shows that this implementation's first Adam update is not perfectly scale-invariant, so speed comparisons remain tied to the fixed loss definitions and learning rate.

## Final results

| loss | continuous exact runs | Boolean exact runs | stable Boolean runs | median final E_inf | median ms/step |
|---|---:|---:|---:|---:|---:|
| MSE | 0/5 | 0/5 | 0/5 | 0.693133 | 9.095 |
| BCE | 0/5 | 0/5 | 0/5 | 0.811083 | 9.170 |
| POWER_1_5 | 0/5 | 0/5 | 0/5 | 0.908779 | 9.169 |
| POWER_1_25 | 0/5 | 0/5 | 0/5 | 0.970657 | 9.205 |

No run reached continuous exact, hard exact, Boolean exact, or any requested `E_inf` threshold. All steps-to-target and seconds-to-target fields are therefore `NOT_REACHED`; the canonical JSON retains right-censored recovery fields.

## Per-seed/per-loss outcomes

| seed | loss | continuous exact | hard exact | Boolean exact | E_inf | wrong Boolean rows |
|---:|---|---:|---:|---:|---:|---:|
| 0 | MSE | 0.992188 | 0.441406 | 0.277344 | 0.693133 | 185 |
| 0 | BCE | 0.906250 | 0.339844 | 0.269531 | 0.819065 | 187 |
| 0 | POWER_1_5 | 0.929688 | 0.179688 | 0.125000 | 0.948285 | 224 |
| 0 | POWER_1_25 | 0.718750 | 0.355469 | 0.355469 | 0.970657 | 165 |
| 1 | MSE | 0.902344 | 0.332031 | 0.316406 | 0.886137 | 175 |
| 1 | BCE | 0.898438 | 0.386719 | 0.363281 | 0.794896 | 163 |
| 1 | POWER_1_5 | 0.816406 | 0.308594 | 0.308594 | 0.935159 | 177 |
| 1 | POWER_1_25 | 0.859375 | 0.617188 | 0.515625 | 0.985131 | 124 |
| 2 | MSE | 0.921875 | 0.570312 | 0.562500 | 0.651425 | 112 |
| 2 | BCE | 0.847656 | 0.617188 | 0.613281 | 0.766081 | 99 |
| 2 | POWER_1_5 | 0.792969 | 0.554688 | 0.476562 | 0.880249 | 134 |
| 2 | POWER_1_25 | 0.824219 | 0.414062 | 0.414062 | 0.913973 | 150 |
| 3 | MSE | 0.910156 | 0.488281 | 0.414062 | 0.90001 | 150 |
| 3 | BCE | 0.761719 | 0.281250 | 0.281250 | 0.811083 | 184 |
| 3 | POWER_1_5 | 0.765625 | 0.296875 | 0.203125 | 0.846914 | 204 |
| 3 | POWER_1_25 | 0.722656 | 0.574219 | 0.574219 | 0.929008 | 109 |
| 4 | MSE | 0.949219 | 0.437500 | 0.421875 | 0.617436 | 148 |
| 4 | BCE | 0.792969 | 0.734375 | 0.722656 | 0.821281 | 71 |
| 4 | POWER_1_5 | 0.828125 | 0.414062 | 0.359375 | 0.908779 | 164 |
| 4 | POWER_1_25 | 0.703125 | 0.507812 | 0.460938 | 0.976833 | 138 |

## Speed and recovery

The 20 runs each used 3000 full-batch Adam updates and 768,000 examples. Median post-warmup step times were approximately 9.1–9.2 ms on the Kaggle T4. Since no run crossed a target, training speed changes throughput but does not produce a recovery winner. See `a1_continuous_recovery_speed.png`, `a1_boolean_recovery_speed.png`, `a1_einf_vs_steps.png`, and `a1_einf_vs_seconds.png`.

## Carry-chain and output-bit difficulty

Final Boolean accuracy is grouped by carry-chain length in `a1_carry_chain_accuracy.png`; per-output-bit accuracy is in `a1_per_output_bit_accuracy.png`. The canonical JSON stores both continuous and Boolean carry groups and per-bit metrics for every run.

## Continuous/Boolean gap and internal mismatch

Each trajectory records bit disagreement, row disagreement, mean Hamming distance, and selected internal traces. The layer plot is `a1_layer_discretization_gap.png`; the first nonzero layer in each selected trace is the earliest semantic mismatch.

## Final wrong-row forensic data

Complete records for every final Boolean-wrong row—including A, B, sum, carry-chain length, continuous output, hard output, Boolean output, and target—are retained in `results[*].final_wrong_rows` in the canonical JSON. This avoids duplicating large tables in Markdown while preserving all exhaustive forensic data.

## Conclusion

Four-bit addition was substantially harder than the prior bitwise XOR benchmark under the unchanged width-64 architecture and 3000-step budget. The continuous function was not solved exactly, so the result does not isolate a pure discretization failure: it demonstrates a continuous optimization/compositional-capacity challenge first. MSE produced the best final continuous exact accuracy in this fixed comparison; BCE and the power losses did not recover exact addition.

**Recommended next experiment:** a single continuous-learning diagnostic on the same 4-bit addition task that preserves the architecture and compares a longer training budget or optimizer schedule before attempting 8-bit addition.
