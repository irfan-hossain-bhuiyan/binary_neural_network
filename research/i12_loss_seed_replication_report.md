# I12 — Five-Seed Clean Loss Replication

## A. I11 audit and correction

The canonical I11 JSON and checkpoints were audited before I12. Four clean eta=0 arms agreed with their checkpoints. The MSE seed3 eta=0 JSON trajectory was correct, but its recorded final checkpoint had been overwritten and its SHA did not match; that single arm was rerun with the original configuration and the canonical JSON/checkpoint metadata were repaired. No other I11 arm was retrained.

The I11 report was corrected: POWER_1_5 is better at eta=.1, while POWER_1_25 is better at eta=.2 and .4. This is a tradeoff, not monotonic dominance by smaller alpha.

Kaggle execution was attempted previously but the API was unavailable from this environment (DNS/network access). I12 therefore ran locally with one CPU thread per concurrent arm; no remote result is claimed.

## B. Configuration and paired initialization

All 20 runs used the exact 256-row XOR truth table, I2-B (mean-field sigma=2 + BIAS_ONE), Lehmer p=2, Adam lr=.01, zero weight decay, full batch, and 3000 optimizer steps. Each seed's four losses reloaded one byte-identical initial state.

| seed | initial state SHA256 | paired arms |
|---:|---|:---:|
| 0 | `d8fad20680124095e77b8921161097c7a4a6f49033840fba05d37b3243d5ef6a` | yes |
| 1 | `e9dda60f4dc96cfbe1a83f7cc4b89d77baed27ea78617440138d8c8053f531cd` | yes |
| 2 | `c21d073cc2c86a097786e45f730910bb970dab1d2400254f2a7489577bed59f4` | yes |
| 3 | `9fe12ddd7c1e003c794d990533a7ecc44d5d729c7f6b3f5018b95c8243bad031` | yes |
| 4 | `7ebc3a89d2db1815cd5525e9e34ce908ecdebbb98770f9b603ae4926d7b4ef03` | yes |

## C. Aggregate results (denominator 5 seeds)

| loss | continuous exact | hard exact | Boolean exact | stable Boolean exact | median final E_inf | median endpoint distance | median first Boolean step |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE | 5/5 | 2/5 | 1/5 | 1/5 | 0.14692 | 0.0242506 | 500 |
| BCE | 5/5 | 2/5 | 2/5 | 2/5 | 0.204907 | 0.00688428 | 1000 |
| POWER_1_5 | 5/5 | 1/5 | 1/5 | 1/5 | 0.27861 | 0.0130655 | 750 |
| POWER_1_25 | 5/5 | 2/5 | 2/5 | 2/5 | 0.228069 | 0.0229206 | 525 |

## D. Per-seed final results

| seed | loss | continuous exact | hard exact | Boolean exact | wrong rows | final E_inf | first Boolean exact |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MSE | 1.0000 | 0.8750 | 0.6875 | 80 | 0.358948 | — |
| 0 | BCE | 1.0000 | 0.9688 | 0.9375 | 16 | 0.355557 | — |
| 0 | POWER_1_5 | 1.0000 | 1.0000 | 1.0000 | 0 | 0.00801292 | 750 |
| 0 | POWER_1_25 | 1.0000 | 1.0000 | 1.0000 | 0 | 0.00308979 | 300 |
| 1 | MSE | 1.0000 | 0.9375 | 0.6875 | 80 | 0.116052 | — |
| 1 | BCE | 1.0000 | 0.6562 | 0.6562 | 88 | 0.292623 | — |
| 1 | POWER_1_5 | 1.0000 | 0.7500 | 0.7500 | 64 | 0.27861 | — |
| 1 | POWER_1_25 | 1.0000 | 0.5000 | 0.5000 | 128 | 0.279398 | — |
| 2 | MSE | 1.0000 | 1.0000 | 0.8750 | 32 | 0.14692 | — |
| 2 | BCE | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0032332 | 500 |
| 2 | POWER_1_5 | 1.0000 | 0.8750 | 0.8750 | 32 | 0.0799961 | — |
| 2 | POWER_1_25 | 1.0000 | 0.7031 | 0.6250 | 96 | 0.228069 | — |
| 3 | MSE | 1.0000 | 0.9922 | 0.9922 | 2 | 0.25218 | — |
| 3 | BCE | 1.0000 | 0.7500 | 0.7500 | 64 | 0.204907 | — |
| 3 | POWER_1_5 | 1.0000 | 0.9766 | 0.9766 | 6 | 0.312095 | — |
| 3 | POWER_1_25 | 1.0000 | 1.0000 | 1.0000 | 0 | 0.00376177 | 750 |
| 4 | MSE | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0181608 | 500 |
| 4 | BCE | 1.0000 | 1.0000 | 1.0000 | 0 | 0.00289749 | 1500 |
| 4 | POWER_1_5 | 1.0000 | 0.8125 | 0.8125 | 48 | 0.365666 | — |
| 4 | POWER_1_25 | 1.0000 | 0.7500 | 0.7500 | 64 | 0.300505 | — |

## E. Endpoint, threshold, and parameter diagnostics

| loss | median confidence | median endpoint distance | final exact threshold intervals containing .5 | median gate D | median bias D |
|---|---:|---:|---|---:|---:|
| MSE | 0.975749 | 0.0242506 | 1/5 ([0.30,0.70]) | 0.0290665 | 0.166689 |
| BCE | 0.993116 | 0.00688428 | 2/5 ([0.30,0.70], [0.30,0.70]) | 0.0223698 | 0.17766 |
| POWER_1_5 | 0.986935 | 0.0130655 | 1/5 ([0.30,0.70]) | 0.0253429 | 0.176708 |
| POWER_1_25 | 0.977079 | 0.0229206 | 2/5 ([0.30,0.70], [0.30,0.70]) | 0.0235432 | 0.17602 |

The clean seed3 POWER_1_25 result was reproducible: Boolean exact first appeared at step 750 in I11 and I12, with final Boolean exact 1.0 and E_inf about 0.00376. Among successful Boolean runs, POWER_1_25 had median E_inf 0.00343 versus 0.00307 for BCE and 0.00801 for POWER_1_5; across all seeds its median is worse because three seeds failed to reach the Boolean basin. Lower-alpha endpoint sharpening is therefore beneficial conditional on success, but it does not guarantee the topology transition.

## F. Interpretation and MNIST decision

Every run reached continuous exact accuracy in this clean function-recovery setting, but Boolean recovery was much more seed-sensitive. POWER_1_25 reached stable Boolean exactness in 2/5 seeds (0 and 3), POWER_1_5 in 1/5, MSE in 1/5, and BCE in 2/5. POWER_1_25 therefore reproduced the strong seed3 result but did not generalize to the required 3/5 adequate or 4/5 strong threshold. POWER_1_5 was better than POWER_1_25 on the final Boolean score in three seeds, but it was not clearly more stable for continuous learning because all 20 runs reached continuous exactness.

The predefined MNIST decision rule does not pass for POWER_1_25: stable Boolean exact is 2/5. No I13 run was started. If one final XOR experiment is required before MNIST, the smallest justified study is a three-seed paired POWER_1_5→POWER_1_25 curriculum using the same clean setup; otherwise the evidence is insufficient to choose a curriculum. MNIST should wait for that decision.

The results do not support claiming POWER_1_25 is universally superior. In this five-seed clean-XOR study it produces the most endpoint-like outputs when it succeeds, but its optimization basin is not yet reliable enough for the MNIST go decision.
