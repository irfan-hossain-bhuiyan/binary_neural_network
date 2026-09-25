# I13 — POWER_1_5 → POWER_1_25 Curriculum

## A–C. I12 audit and paired initial states

The canonical I12 JSON contains 20 runs. All 20 reached continuous exact accuracy 1.0. Stable Boolean exact counts are MSE 1/5, BCE 2/5, POWER_1_5 1/5, and POWER_1_25 2/5. Every I12 checkpoint reload/SHA check passed before I13; no I12 training was rerun.

Each I13 seed reproduced its I12 initial-state SHA256 exactly. The curriculum used the same I2-B initialization, clean 256-row XOR table, Lehmer p=2, Adam lr=.01, zero weight decay, one uninterrupted optimizer state, and 3000 updates.

## D–E. Trigger and alpha schedule

| seed | initial hash | first continuous exact | trigger step | alpha at first Boolean exact | curriculum not triggered |
|---:|---|---:|---:|---:|---:|
| 0 | `d8fad20680124095e77b8921161097c7a4a6f49033840fba05d37b3243d5ef6a` | 400 | 450 | 1.35 | no |
| 1 | `e9dda60f4dc96cfbe1a83f7cc4b89d77baed27ea78617440138d8c8053f531cd` | 350 | 400 | — | no |
| 2 | `c21d073cc2c86a097786e45f730910bb970dab1d2400254f2a7489577bed59f4` | 575 | 625 | — | no |
| 3 | `9fe12ddd7c1e003c794d990533a7ecc44d5d729c7f6b3f5018b95c8243bad031` | 450 | 500 | — | no |
| 4 | `7ebc3a89d2db1815cd5525e9e34ce908ecdebbb98770f9b603ae4926d7b4ef03` | 600 | 650 | — | no |

The trigger required three consecutive 25-step continuous-exact evaluations and was reached for every seed. Alpha then annealed linearly from 1.5 to 1.25 over 500 updates and stayed at 1.25. Adam was not reset.

## F–G. Curriculum trajectories

The canonical JSON stores every 25-step continuous, hard-max, Boolean, endpoint, confidence, current-alpha loss, and topology record. The figures show alpha, Boolean accuracy, and E_inf trajectories for each seed.

## H. Per-seed comparison

| seed | fixed 1.5 Boolean | fixed 1.25 Boolean | curriculum Boolean | curriculum first exact | trigger step |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 750 | 450 |
| 1 | 0.7500 | 0.5000 | 0.7500 | — | 400 |
| 2 | 0.8750 | 0.6250 | 0.8750 | — | 625 |
| 3 | 0.9766 | 1.0000 | 0.9766 | — | 500 |
| 4 | 0.8125 | 0.7500 | 0.8125 | — | 650 |

## I–L. Aggregate comparison

| method | continuous exact | Boolean exact | stable Boolean exact | median E_inf | median endpoint distance | median first Boolean step |
|---|---:|---:|---:|---:|---:|---:|
| fixed POWER_1_5 | 5/5 | 1/5 | 1/5 | 0.27861 | 0.0130655 | 750 |
| fixed POWER_1_25 | 5/5 | 2/5 | 2/5 | 0.228069 | 0.0229206 | 525.0 |
| I13 curriculum | 5/5 | 1/5 | 1/5 | 0.298144 | 0.0106846 | 750 |

## M. Threshold robustness and N. topology changes

All final threshold sweeps are stored in JSON under `threshold_robustness`. The exact interval containing .5 was:

| seed | final exact threshold interval containing .5 | final mean gate D | final mean bias D |
|---:|---|---:|---:|
| 0 | [0.30,0.70] | 0.0226149 | 0.16493 |
| 1 | none | 0.0193115 | 0.186126 |
| 2 | none | 0.0236769 | 0.178314 |
| 3 | none | 0.0240983 | 0.162246 |
| 4 | none | 0.0243349 | 0.178781 |

Topology Hamming changes and per-evaluation edge/bias changes are stored in the trajectory. The anneal-window deltas were:

| seed | edge Hamming change during anneal | bias Hamming change during anneal | final edge Hamming from init | final bias Hamming from init |
|---:|---:|---:|---:|---:|
| 0 | 57 | 602 | 485 | 6283 |
| 1 | 111 | 548 | 659 | 6269 |
| 2 | 91 | 300 | 693 | 6288 |
| 3 | 54 | 463 | 546 | 6164 |
| 4 | 41 | 392 | 645 | 6253 |

## O–Q. Conclusion

The curriculum did not repair a seed where both fixed power losses failed. Its final Boolean results matched the fixed POWER_1_5 results in this run: seed0 1.0, seed1 .75, seed2 .875, seed3 .9765625, seed4 .8125. Stable Boolean exactness was therefore 1/5, versus 1/5 for fixed POWER_1_5 and 2/5 for fixed POWER_1_25.

For context, I12 BCE was stable Boolean exact in 2/5 with median E_inf 0.2049, and I12 MSE was stable in 1/5 with median E_inf 0.1469. I13's median E_inf was 0.2981; its lower median endpoint distance was not accompanied by improved Boolean topology.

The POWER_1_5 → POWER_1_25 hypothesis is not supported as a basin-improving method in this five-seed study. Alpha sharpening did not produce additional topology recovery; it mostly preserved the basin selected during the POWER_1_5 phase.

## R–S. MNIST recommendation

XOR experimentation complete; next experiment is an MNIST pilot. Use fixed POWER_1_25, the I13 curriculum, and BCE as the minimum comparison arms. The curriculum is not established as superior, so MNIST should be treated as a scale/generalization experiment rather than a claim that Boolean discretization is solved.
