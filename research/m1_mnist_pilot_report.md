# M1 — MNIST Scale / Discretization Pilot

## A–B. I13 audit and XOR closure

I13 was audited before M1. Its canonical JSON and checkpoints agree: curriculum final Boolean exactness was 1/5 (seed0 only), fixed POWER_1_25 was 2/5, and all 20 runs reached continuous exactness. No I13 rerun was required. XOR experimentation is closed.

## C–H. Data, split, device, and run matrix

The Kaggle job used the standard torchvision MNIST train/test source, concatenated only to save a raw artifact, with uint8 input threshold 128 (pixel >= 0.5 after normalization). Both continuous and exact Boolean models received the same 784 binary inputs. The official 60,000 training examples were split deterministically into 55,000 train and 5,000 validation; the 10,000 official test examples were untouched.

The run used a Tesla T4 GPU (`torch 2.10.0+cu128`, CUDA 12.8), not local CPU. The float32 batch-256 forward/backward smoke test was finite, output shape `[256,10]`, and peak allocated GPU memory was 1,702,769,664 bytes. The preflight ran all three losses for two epochs before the canonical 20-epoch matrix. All 9 canonical runs were finite, used Adam 1e-3, weight decay 0, and batch 256.

The architecture was 784→256, two width-256 XOR residual blocks, 256→10 head, Lehmer p=2, I2-B mean-field sigma=2 plus BIAS_ONE. Initial tensors were paired per seed and reloaded byte-for-byte for each loss arm; the result records the initial-state SHA256 for every run.

### Main test comparison (checkpoint selected by validation continuous argmax)
| loss | seed | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean one-hot | cont→Boolean sample gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCE | 0 | 0.7002 | 0.2077 | 0.2087 | 0.0595 | 0.2034 | 0.9217 |
| POWER_1_25 | 0 | 0.3497 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CURRICULUM | 0 | 0.3651 | 0.0271 | 0.0364 | 0.0937 | 0.1459 | 0.1196 |
| BCE | 1 | 0.6622 | 0.1775 | 0.2753 | 0.1810 | 0.3318 | 0.8568 |
| POWER_1_25 | 1 | 0.3001 | 0.0175 | 0.0175 | 0.0303 | 0.0426 | 0.0187 |
| CURRICULUM | 1 | 0.3333 | 0.0175 | 0.0175 | 0.0417 | 0.0565 | 0.0326 |
| BCE | 2 | 0.6775 | 0.2220 | 0.2461 | 0.0433 | 0.0859 | 0.9707 |
| POWER_1_25 | 2 | 0.3741 | 0.0304 | 0.0310 | 0.1271 | 0.2409 | 0.2252 |
| CURRICULUM | 2 | 0.3696 | 0.0882 | 0.1115 | 0.2026 | 0.3405 | 0.2699 |

### Validation-selected Boolean checkpoint comparison

| loss | seed | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean one-hot | cont→Boolean sample gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCE | 0 | 0.4393 | 0.0000 | 0.0000 | 0.0697 | 0.3479 | 0.8714 |
| POWER_1_25 | 0 | 0.0990 | 0.0000 | 0.0000 | 0.1032 | 1.0000 | 1.0000 |
| CURRICULUM | 0 | 0.3455 | 0.0271 | 0.0271 | 0.0937 | 0.1459 | 0.1205 |
| BCE | 1 | 0.6528 | 0.1658 | 0.2753 | 0.1860 | 0.3453 | 0.8472 |
| POWER_1_25 | 1 | 0.1028 | 0.0000 | 0.0000 | 0.0549 | 0.3824 | 1.0000 |
| CURRICULUM | 1 | 0.2966 | 0.0175 | 0.0175 | 0.0417 | 0.0565 | 0.0326 |
| BCE | 2 | 0.5882 | 0.1074 | 0.2002 | 0.0467 | 0.1200 | 0.9669 |
| POWER_1_25 | 2 | 0.3664 | 0.0304 | 0.0310 | 0.1271 | 0.2409 | 0.2252 |
| CURRICULUM | 2 | 0.3696 | 0.0882 | 0.1115 | 0.2026 | 0.3405 | 0.2699 |

## I–M. Classification semantics

The continuous classifier is evaluated by argmax; strict continuous and Boolean metrics require the ten-bit output vector to equal the one-hot target. Exact Boolean strict accuracy is therefore not an argmax proxy. Zero-hot and multi-hot rates are reported below.

### Aggregate test metrics (continuous-argmax checkpoint)

| loss | continuous top-1 | continuous strict | hard strict | Boolean strict | valid Boolean one-hot | zero-hot | multi-hot | sample gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | 0.6800 ± 0.0191 | 0.2024 ± 0.0227 | 0.2434 ± 0.0334 | 0.0946 ± 0.0753 | 0.2070 ± 0.1230 | 0.0493 ± 0.0328 | 0.7436 ± 0.1546 | 0.9164 ± 0.0571 |
| POWER_1_25 | 0.3413 ± 0.0377 | 0.0160 ± 0.0153 | 0.0162 ± 0.0155 | 0.0525 ± 0.0664 | 0.0945 ± 0.1286 | 0.8998 ± 0.1383 | 0.0057 ± 0.0098 | 0.0813 ± 0.1250 |
| CURRICULUM | 0.3560 ± 0.0198 | 0.0443 ± 0.0383 | 0.0551 ± 0.0497 | 0.1127 ± 0.0821 | 0.1810 ± 0.1452 | 0.8046 ± 0.1606 | 0.0144 ± 0.0155 | 0.1407 ± 0.1200 |

## N–P. Functional gap, endpoints, and margins

The continuous-to-Boolean sample disagreement is large for BCE (roughly 0.86–0.97 across seeds at the selected checkpoint). POWER_1_25 and the curriculum have smaller disagreement in some runs, but their continuous classifiers are also much weaker. This is not evidence of a useful Boolean classifier by itself.

| loss | mean endpoint distance | mean class margin | E_inf |
|---|---:|---:|---:|
| BCE | 0.1013 ± 0.0027 | 0.1164 ± 0.0066 | 0.9738 ± 0.0011 |
| POWER_1_25 | 0.0211 ± 0.0050 | 0.0171 ± 0.0136 | 0.9958 ± 0.0013 |
| CURRICULUM | 0.0208 ± 0.0064 | 0.0263 ± 0.0135 | 0.9967 ± 0.0008 |

The continuous endpoint and confidence metrics are present in the canonical trajectory and test records. The power and curriculum arms did not turn endpoint pressure into competitive MNIST classification within 20 epochs.

## Q. First mismatch layer

The diagnostic subset was traced through input, stem, both residual blocks, and head at epochs 0, 5, 10, 15, and 20. The first nonzero aggregate mismatch appears at the stem in the final traces; later layers also remain highly mismatched. Thus the scale gap is already introduced at the first 784→256 logic layer, rather than being only a head problem. See `m1_layer_mismatch.png` and `m1_first_mismatch_layer.png`.

## R. Parameter polarization

Per-layer gate and bias polarization statistics are stored in every trajectory record under `parameter_stats` and were descriptive only. No polarization regularizer was used. The main accuracy failure is not resolved by output loss choice alone.

## S–T. Loss comparison

BCE was the strongest continuous arm in this pilot: mean test top-1 was about 0.680 across three seeds, versus about 0.341 for fixed POWER_1_25 and 0.356 for the curriculum. Its exact Boolean strict accuracy was only about 0.095, with a large semantic gap. POWER_1_25 and the curriculum had lower disagreement in selected runs but did not learn a competitive continuous classifier. The curriculum did not improve over fixed POWER_1_25 at scale.

## U. Seed variability

Seed variation is material for every loss. BCE ranged from 0.662 to 0.700 continuous top-1; POWER_1_25 ranged from 0.274 to 0.374; curriculum ranged from 0.332 to 0.365. Exact Boolean strict accuracy also varied substantially. These are three-seed pilot observations, not population estimates.

## V. Main conclusion

M1 exposes both problems, with the dominant first failure being scalable continuous learning. The best continuous arm reaches only about 68% test top-1, below the 80% pilot threshold, so this is primarily a scale/optimization limitation. The continuous-to-exact-Boolean gap is nevertheless also severe: even the BCE arm's Boolean strict accuracy is far below its continuous argmax accuracy, and the first mismatch appears in the stem.

## W. One recommended next experiment

The immediate next questions are training duration and output-loss imbalance,
not GPU-memory capacity. The smoke test used only about 1.70 GB of allocated
T4 memory at batch 256. M2 therefore extends training and tests a mathematically
balanced one-hot loss while keeping the architecture, Lehmer-p2 operator, and
binary input unchanged. Do not add STE, regularization, or new operators until
the continuous baseline is useful.

## Artifacts

Canonical result: `research/operator_results/m1_mnist_pilot_results.json`. Thirty-six checkpoint files were downloaded from Kaggle, reloaded, and their SHA256 hashes matched the recorded metadata. Figures are in `research/figures/m1_*.png`.
