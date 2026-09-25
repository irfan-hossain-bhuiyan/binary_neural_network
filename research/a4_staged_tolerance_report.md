# A4 — Staged Ordinary Loss → Whole-Word Tolerance Loss

## A. A3 audit

A3 was audited from its canonical JSON/report. It used exhaustive 4-bit addition (256 rows), 8 inputs, 5 outputs, width 64, two XOR-residual blocks, Lehmer-p2, and five paired seeds. The verified outcome was MSE 0/5 continuous-exact and 0/5 Boolean-exact; BIT_MEAN_RATIONAL 0/5 and 0/5; ROW_MAX_RATIONAL 0/5 and 0/5; ROW_MAX_SOFTPLUS 1/5 continuous-exact and 0/5 Boolean-exact. No A3 rerun was required.

## B–D. Common prefix and cloning

Every seed used one MSE prefix for exactly 2,000 optimizer steps. The model and Adam state were deep-cloned in RAM at the switch into MSE→MSE, MSE→ROW_MAX_SOFTPLUS, and MSE→ROW_MAX_RATIONAL. For each seed, all branches have identical switch model and optimizer hashes; all records set `optimizer_state_cloned=true`. No checkpoints were written.

## E–G. Per-run outcomes

| seed | continuation | Bool@switch | final cont exact | final Bool exact | best Bool exact | final E_inf | final d<.10 | edge flips | bias flips |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | MSE -> MSE | 0.2031 | 0.9961 | 0.2539 | 0.2812 | 0.523487 | 0.0352 | 373 | 2499 |
| 0 | MSE -> ROW_MAX_SOFTPLUS | 0.2031 | 0.9922 | 0.3008 | 0.3125 | 0.634468 | 0.0000 | 459 | 3634 |
| 0 | MSE -> ROW_MAX_RATIONAL | 0.2031 | 0.8008 | 0.6562 | 0.6562 | 0.998529 | 0.6562 | 465 | 3263 |
| 1 | MSE -> MSE | 0.3281 | 0.9258 | 0.3047 | 0.3281 | 0.889341 | 0.0000 | 397 | 2598 |
| 1 | MSE -> ROW_MAX_SOFTPLUS | 0.3281 | 0.9766 | 0.2539 | 0.3281 | 0.794677 | 0.0000 | 544 | 4053 |
| 1 | MSE -> ROW_MAX_RATIONAL | 0.3281 | 0.5469 | 0.5430 | 0.5430 | 0.998773 | 0.5430 | 350 | 3238 |
| 2 | MSE -> MSE | 0.5391 | 0.9297 | 0.5625 | 0.5625 | 0.591503 | 0.0000 | 404 | 3052 |
| 2 | MSE -> ROW_MAX_SOFTPLUS | 0.5391 | 0.9805 | 0.5586 | 0.5586 | 0.867527 | 0.0000 | 479 | 4342 |
| 2 | MSE -> ROW_MAX_RATIONAL | 0.5391 | 0.5625 | 0.5625 | 0.5625 | 0.998849 | 0.5625 | 333 | 3242 |
| 3 | MSE -> MSE | 0.1250 | 0.9180 | 0.2969 | 0.4141 | 0.805779 | 0.0508 | 345 | 2907 |
| 3 | MSE -> ROW_MAX_SOFTPLUS | 0.1250 | 0.9453 | 0.4062 | 0.4375 | 0.884628 | 0.1367 | 555 | 4103 |
| 3 | MSE -> ROW_MAX_RATIONAL | 0.1250 | 0.4062 | 0.4062 | 0.4062 | 0.998728 | 0.4062 | 348 | 3346 |
| 4 | MSE -> MSE | 0.4141 | 0.9844 | 0.4023 | 0.4258 | 0.656747 | 0.0898 | 511 | 3005 |
| 4 | MSE -> ROW_MAX_SOFTPLUS | 0.4141 | 0.9922 | 0.3984 | 0.4141 | 0.530561 | 0.0352 | 579 | 4026 |
| 4 | MSE -> ROW_MAX_RATIONAL | 0.4141 | 0.4805 | 0.4805 | 0.4805 | 0.993138 | 0.4805 | 755 | 4299 |

## H–K. Aggregate recovery and speed

| continuation | continuous exact runs | Boolean exact runs | stable Boolean runs | median final Boolean | median final E_inf | median topology flips |
|---|---:|---:|---:|---:|---:|---:|
| MSE -> MSE | 0/5 | 0/5 | 0/5 | 0.3047 | 0.656747 | 3252 |
| MSE -> ROW_MAX_SOFTPLUS | 0/5 | 0/5 | 0/5 | 0.3984 | 0.794677 | 4605 |
| MSE -> ROW_MAX_RATIONAL | 0/5 | 0/5 | 0/5 | 0.5430 | 0.998728 | 3694 |

No branch reached continuous exact accuracy 1.0 or Boolean exact accuracy 1.0 at any scheduled evaluation. Consequently steps/seconds to continuous exact, Boolean exact, and stable Boolean exact are `NOT_REACHED` for all 15 runs. The best near-exact final continuous fraction was 0.9961 (seed0, MSE→MSE). The common prefix timing is shared per seed; continuation core time was about 35–36 seconds per branch on a Tesla T4. No branch had a successful target time to compare, so the extra 4,000 steps did not buy exact recovery.

## L–N. Topology movement and Boolean improvement

Final edge-plus-bias Hamming distances ranged from 2,872 to 5,054. Movement was extensive in all branches, but exact Boolean recovery never occurred. At the switch, mean Boolean exactness was 0.321 across seeds. At the final point it was 0.364 for MSE→MSE, 0.384 for MSE→SOFTPLUS, and 0.530 for MSE→RATIONAL. Rational therefore produced the largest average Boolean improvement, driven by topology changes, but not a correct circuit. Topology distance and Boolean gain were not monotonic; extensive flipping can harden an incorrect topology.

## O–P. Functional and semantic gaps

Final means (MSE→MSE / SOFTPLUS / RATIONAL) were: soft continuous exact 0.951 / 0.977 / 0.559, hard-max exact 0.454 / 0.436 / 0.530, and Boolean exact 0.364 / 0.384 / 0.530. Thresholded-continuous versus Boolean row disagreement was 0.613 / 0.612 / 0.038. Rational nearly collapsed the soft→hard and hard→Boolean gaps by making the continuous outputs agree with its wrong Boolean topology, while greatly worsening endpoint/function quality (`E_inf` median 0.999). Softplus preserved the continuous function better but did not close the hard→Boolean gap.

## Q–R. Carry and output bits

Final mean Boolean per-bit accuracies (s0…s4) were MSE→MSE 0.938, 0.826, 0.848, 0.802, 0.635; SOFTPLUS 0.916, 0.848, 0.810, 0.786, 0.625; RATIONAL 0.970, 0.952, 0.827, 0.809, 0.810. The final mean Boolean exact accuracies by carry-chain length (0…4) were MSE→MSE 0.207, 0.429, 0.530, 0.333, 0.125; SOFTPLUS 0.321, 0.436, 0.503, 0.217, 0.075; RATIONAL 0.449, 0.617, 0.563, 0.517, 0.225. Long-carry rows remain difficult, especially for the softplus branch; all per-seed/per-milestone tables remain in JSON.

## S. XOR-residual gradients

The A2 decomposition was retained at steps 2000, 2100, 3000, 4000, and 6000. The numerical decomposition error stayed around 1e-7. Across branches, mean direct gain |1−2F| stayed about 0.81 (block0) and 0.67 (block1) at the switch. At step6000 it was about 0.81/0.69 for MSE, 0.82/0.67 for softplus, and 0.80/0.54 for rational. Direct/branch cosines were near zero, so the two terms were mostly orthogonal rather than strongly reinforcing or cancelling. Rational increased block1 mid-valued branch fraction to about 0.23, weakening its direct path despite its stronger Boolean agreement.

## T–V. Interpretation

MSE→MSE did not solve the task by simply training longer. MSE→SOFTPLUS preserved the best continuous solution but did not exceed the tolerance/Boolean barrier. MSE→RATIONAL behaved as an endpoint/topology hardener: it produced substantially better Boolean agreement and about 53% of rows inside d<.10, but those were not the target circuit—its continuous `E_inf` was about 0.994–0.999 and it reached no exact Boolean circuit. Thus staged tolerance training alone does not solve addition; the remaining bottleneck is topology-directed credit assignment, with endpoint quality and topology quality separable.

## W. Storage and provenance

The Kaggle result was generated on cuda (Tesla T4) with PyTorch 2.10.0+cu128. Raw wrapper output was 1,070,677 bytes; retained canonical JSON is 1,014,322 bytes, below the 2 MB target. Ordinary trajectory points retain only compact metrics; gradient/carry/topology diagnostics are milestone-only. No `.pt`, `.pth`, or `.ckpt` files were written.

## X. Recommended next experiment

Run one causal-agnostic Boolean topology-credit continuation from the shared 2,000-step MSE prefix, with fixed global/layer-balanced scaling chosen before training; do not broaden tolerance-loss sweeps until a topology-directed signal is tested.
