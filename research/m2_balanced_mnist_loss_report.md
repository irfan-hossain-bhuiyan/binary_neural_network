# M2 — Balanced Boolean-Seeking MNIST Loss

## A. M1 audit and corrected next step

M1 canonical JSON and checkpoints were audited. The validation/test aggregates reproduce the recorded BCE ≈0.680, POWER_1_25 ≈0.341, and curriculum ≈0.356 continuous top-1 results, with POWER_1_25 near-zero-hot collapse. M1 BCE was still improving at epoch 20. The M1 smoke test used only about 1.70 GB of T4 memory at batch 256, so the immediate questions were training duration and one-hot loss imbalance, not memory capacity. No M1 numerical result was changed.

## B–E. Setup and mathematical checks

The M2 note is in `research/mnist_loss_balance.md`. For q=0.1, unbalanced power optima are p*=0.1 (alpha=2), 0.0121951 (alpha=1.5), and 0.000152393 (alpha=1.25). The balanced vector objective weights the positive target bit and all nine negative bits equally; its symmetric uninformative optimum is p*=0.5 for every alpha>1. Balanced BCE uses the analogous positive-versus-aggregate-negative weighting and does not use softmax.

The synthetic gradient sanity test is stored in the result JSON at four uniform prediction values (p=0.01, .1, .5, .9). At p=.5, balanced BCE and balanced POWER have equal aggregate positive/negative gradient magnitudes; the unbalanced geometry does not. The split hash is `e0f7388f936a6d979c1f68c9170ba74dd1cbdb0bfdf0a8432d30d492a4831fb0`.

The run used a Tesla T4 (`torch 2.10.0+cu128`), binary MNIST inputs, the unchanged 784→256→two XOR residual blocks→256→10 Lehmer-p2 network, Adam 1e-3, batch 256, and 60 epochs. All nine runs were finite. Initial-state hashes match the corresponding M1 seed states exactly.

## F. Epoch-20 reproduction

The M2 BCE trajectories exactly reproduce M1 at epoch 20 for all three seeds (validation continuous top-1, strict continuous, Boolean strict, and zero-hot rate match). This validates the longer continuation before interpreting epochs 21–60.

## G–J. 60-epoch results

### Test metrics (checkpoint selected by validation continuous top-1)

| loss | seed | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean one-hot | zero-hot cont | cont→Boolean gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | 0 | 0.7857 | 0.4031 | 0.1822 | 0.0097 | 0.0576 | 0.5419 | 0.9932 |
| BALANCED_BCE | 0 | 0.8490 | 0.4174 | 0.0211 | 0.0001 | 0.0001 | 0.0101 | 0.9975 |
| BALANCED_POWER_1_25 | 0 | 0.7794 | 0.2749 | 0.0313 | 0.0104 | 0.0131 | 0.0162 | 0.9358 |
| BCE | 1 | 0.7914 | 0.4561 | 0.1831 | 0.0750 | 0.1815 | 0.4808 | 0.9128 |
| BALANCED_BCE | 1 | 0.8401 | 0.3518 | 0.0097 | 0.0000 | 0.0004 | 0.0080 | 0.9990 |
| BALANCED_POWER_1_25 | 1 | 0.7835 | 0.2831 | 0.0130 | 0.0024 | 0.0031 | 0.0178 | 0.9909 |
| BCE | 2 | 0.8032 | 0.4161 | 0.2642 | 0.1254 | 0.3037 | 0.5184 | 0.8602 |
| BALANCED_BCE | 2 | 0.8534 | 0.3876 | 0.0011 | 0.0000 | 0.0001 | 0.0111 | 1.0000 |
| BALANCED_POWER_1_25 | 2 | 0.7769 | 0.2987 | 0.0221 | 0.0040 | 0.0068 | 0.0250 | 0.9515 |

### Aggregate test metrics

| loss | cont top-1 | cont strict | hard strict | Boolean strict | valid Boolean | zero-hot cont | sample gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | 0.7934 ± 0.0089 | 0.4251 ± 0.0276 | 0.2098 ± 0.0471 | 0.0700 ± 0.0580 | 0.1809 ± 0.1231 | 0.5137 ± 0.0308 | 0.9221 ± 0.0670 |
| BALANCED_BCE | 0.8475 ± 0.0068 | 0.3856 ± 0.0328 | 0.0106 ± 0.0100 | 0.0000 ± 0.0001 | 0.0002 ± 0.0002 | 0.0097 ± 0.0016 | 0.9988 ± 0.0013 |
| BALANCED_POWER_1_25 | 0.7799 ± 0.0033 | 0.2856 ± 0.0121 | 0.0221 ± 0.0092 | 0.0056 ± 0.0042 | 0.0077 ± 0.0051 | 0.0197 ± 0.0047 | 0.9594 ± 0.0284 |

## K–L. Zero-hot and output distributions

Balanced BCE reduces the continuous zero-hot rate from roughly 0.74–0.78 at M1 epoch 20 to about 0.008–0.011 at epoch 60. Balanced POWER_1_25 similarly stays near 0.015–0.025. Original BCE remains zero-hot-heavy at about 0.48–0.54 after 60 epochs. Target outputs rise while non-target outputs fall for both balanced arms; the target/non-target trajectories are shown in `m2_target_nontarget_outputs.png`.

## M–N. Continuous and Boolean comparison

Balanced BCE is the strongest continuous result: mean test top-1 is about 0.848 across seeds, exceeding the 80% pilot criterion. Original BCE reaches about 0.793, while balanced POWER_1_25 reaches about 0.780. Despite the stronger continuous classifier, exact Boolean strict accuracy remains near zero for balanced BCE and around one percent for balanced POWER. This is a severe parameter/topology discretization gap, not a zero-hot output problem alone.

## O. Functional discretization gap

The sample disagreement between thresholded continuous outputs and exact Boolean outputs remains high: approximately 0.92 on average for BCE, 0.99 for balanced BCE, and 0.96 for balanced POWER at the selected continuous checkpoints. The Boolean network is therefore not implementing the learned classifier, even when the continuous classifier is useful.

## P. First mismatch layer

The detailed traces at epochs 0, 20, 40, and 60 continue to show the first nonzero mismatch at the stem on the fixed 1024-image validation subset. Later logic layers also remain mismatched. Longer training and balanced output losses improve continuous learning but do not remove the earliest stem-level semantic divergence. See `m2_layer_mismatch.png`.

## Q–S. Interpretation

BCE was undertrained in M1: its validation accuracy continued to rise, and M2 improves the mean test top-1 from about 0.680 to about 0.793. One-hot imbalance explains much of the unbalanced POWER collapse: balancing reduces zero-hot outputs dramatically. However, balanced POWER_1_25 still trails BCE in continuous top-1 and does not preserve a meaningful Boolean advantage. Balanced BCE is the best continuous baseline in this study, but its exact Boolean output remains poor. Thus one-hot imbalance explains the power-loss scaling failure, while the remaining continuous-to-Boolean gap is a separate internal discretization problem.

## T. One recommended next experiment

Run one controlled post-training discretization study on the balanced-BCE checkpoint, using the existing exact Boolean evaluation and layer mismatch trace to test a single Boolean-aware continuation method. Do not change the architecture or output encoding until that gap is characterized.

## Checkpoints and artifacts

The canonical JSON stores all trajectories, distributions, gradient sanity, test evaluations, and hashes. Thirty-six selected/final checkpoints were downloaded and rehashed successfully. The runner did not emit separate epoch-20/epoch-40 checkpoint files; those epochs are fully recorded in the trajectories, while selected and final checkpoints are preserved.
