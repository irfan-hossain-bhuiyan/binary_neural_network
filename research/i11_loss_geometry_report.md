# I11 — Boolean-Seeking Output Loss Geometry

## Scope

This is the staged seed-3 factorial: five losses × four exact repeated-label noise levels. The architecture, I2-B initializer, optimizer, and 3000-step budget are fixed. Kaggle was attempted but its API was unavailable from this workspace, so seeds 0–4 replication was not run. The repeated 10-copy objective is evaluated through exact per-row label-count weights; this is algebraically identical to the expanded 2560-example mean loss.

## A. Population loss geometry

For repeated labels with $q=P(y=1\mid x)$, MSE and BCE have $p^*=q$. For $L_\alpha=|p-y|^\alpha$, $1<\alpha\le2$, stationarity gives $q(1-p)^{\alpha-1}=(1-q)p^{\alpha-1}$ and therefore $\operatorname{logit}(p^*)=\operatorname{logit}(q)/(\alpha-1)$. MAE selects 1 for $q>.5$, 0 for $q<.5$, and is non-unique at $q=.5$.

| q | MSE | BCE | power 1.5 | power 1.25 | MAE |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | non-unique |
| 0.55 | 0.5500 | 0.5500 | 0.5990 | 0.6905 | 1.0 |
| 0.60 | 0.6000 | 0.6000 | 0.6923 | 0.8351 | 1.0 |
| 0.70 | 0.7000 | 0.7000 | 0.8448 | 0.9674 | 1.0 |
| 0.80 | 0.8000 | 0.8000 | 0.9412 | 0.9961 | 1.0 |
| 0.90 | 0.9000 | 0.9000 | 0.9878 | 0.9998 | 1.0 |
| 0.95 | 0.9500 | 0.9500 | 0.9972 | 1.0000 | 1.0 |

Scalar sigmoid-parameter optimization matched the analytic optima to numerical precision for q=.6, .8, and .9. At q=.5, MAE converged to the arbitrary initialization-dependent value .5, as expected.

## B. Gradient geometry

The generated gradient plot includes MSE, BCE, power losses, and the non-primary sigmoid-step diagnostic (`tau=.05,.10`). At p=.99, the corrupted-label/correct-label gradient ratios were:

| loss | ratio at p=.99 |
|---|---:|
| MSE | 99.000 |
| BCE | 99.000 |
| POWER_1_5 | 9.950 |
| POWER_1_25 | 3.154 |
| MAE | 1.000 |

BCE and MSE both have a 99× bad-label influence ratio at this point, although BCE has much stronger absolute endpoint gradients. Power 1.5 reduces the ratio to 9.95, power 1.25 to 3.15, and MAE to 1.

## C. Paired initialization

All 20 seed-3 arms used the same I2-B initial state hash:

`9fe12ddd7c1e003c794d990533a7ecc44d5d729c7f6b3f5018b95c8243bad031`

Checkpoint reload verification passed for every saved best-continuous, best-Boolean, and final checkpoint.

## D. Seed-3 clean and noisy results

| loss | eta | clean MSE | clean Boolean exact | wrong rows | mean confidence | mean endpoint distance |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 0.0 | 0.002718 | 0.7500 | 64 | 0.9812 | 0.0188 |
| BCE | 0.1 | 0.010385 | 0.7500 | 64 | 0.8982 | 0.1018 |
| BCE | 0.2 | 0.040005 | 0.7500 | 64 | 0.8000 | 0.2000 |
| BCE | 0.4 | 0.160003 | 0.2422 | 194 | 0.6000 | 0.4000 |
| MAE | 0.0 | 0.052844 | 0.3359 | 170 | 0.8939 | 0.0763 |
| MAE | 0.1 | 0.062047 | 0.3438 | 168 | 0.8812 | 0.0809 |
| MAE | 0.2 | 0.068824 | 0.4062 | 152 | 0.8784 | 0.0726 |
| MAE | 0.4 | 0.093878 | 0.4062 | 152 | 0.8520 | 0.0776 |
| MSE | 0.0 | 0.000293 | 0.9922 | 2 | 0.9873 | 0.0127 |
| MSE | 0.1 | 0.014110 | 0.6797 | 82 | 0.8862 | 0.1130 |
| MSE | 0.2 | 0.040007 | 0.5625 | 112 | 0.8000 | 0.2000 |
| MSE | 0.4 | 0.160002 | 0.1055 | 229 | 0.6000 | 0.4000 |
| POWER_1_25 | 0.0 | 0.000005 | 1.0000 | 0 | 0.9977 | 0.0023 |
| POWER_1_25 | 0.1 | 0.001546 | 0.8906 | 28 | 0.9856 | 0.0144 |
| POWER_1_25 | 0.2 | 0.003385 | 0.8555 | 37 | 0.9723 | 0.0277 |
| POWER_1_25 | 0.4 | 0.029012 | 0.6797 | 82 | 0.8309 | 0.1687 |
| POWER_1_5 | 0.0 | 0.000621 | 0.9766 | 6 | 0.9917 | 0.0083 |
| POWER_1_5 | 0.1 | 0.000582 | 0.9766 | 6 | 0.9835 | 0.0165 |
| POWER_1_5 | 0.2 | 0.006890 | 0.7148 | 73 | 0.9236 | 0.0764 |
| POWER_1_5 | 0.4 | 0.094681 | 0.2539 | 191 | 0.6923 | 0.3077 |

Clean eta=0 Boolean results:

- `POWER_1_25`: exact Boolean recovery (0 wrong rows).
- `POWER_1_5`: 6 wrong rows.
- `MSE`: 2 wrong rows.
- `BCE`: 64 wrong rows.
- `MAE`: 170 wrong rows.

Under noise, POWER_1_5 was better at eta=.1 (0.9765625 exact, 6 wrong rows, versus POWER_1_25 at 0.890625 exact and 28 wrong rows). POWER_1_25 was better at eta=.2 (0.85546875 versus 0.71484375) and eta=.4 (0.6796875 versus 0.25390625). This is an optimization-hardness/Boolean-sharpening tradeoff, not monotonic dominance by the smaller exponent. BCE and MSE moved toward their conditional probabilities. At eta=.4, all methods suffered substantial clean-function degradation.

## E. Theoretical versus observed confidence

The scalar confidence predictions are $1-\eta$ for MSE/BCE, sharper for power losses, and 1 for MAE below 50% noise. The deep shared network follows this direction but does not attain the scalar values exactly. Power 1.25 gives the highest clean confidence among the tested arms at eta=0, .1, and .2; at eta=.4 its confidence falls because the shared circuit cannot satisfy all contradictory rows while preserving the clean function.

## F. Interpretation

1. MSE and BCE estimate conditional probabilities in contradictory data; BCE is not inherently endpoint-seeking.
2. Lower power exponents sharpen the population optimum toward Boolean majority and produced the strongest staged clean Boolean result.
3. MAE has the desired population endpoint optimum but was unstable and substantially worse on the clean XOR circuit in this basin.
4. BCE showed strong endpoint pressure but also strong sensitivity to corrupted labels; its noisy runs degraded sharply.
5. The staged result supports POWER_1_25 as the preferred candidate for a later loss-focused replication, but seed robustness is not established.

## G. Required stopping point

No STE, regularization, homotopy, curriculum, or MNIST experiment was run. A full five-seed × four-noise replication remains the next validation step if loss geometry is pursued further.
