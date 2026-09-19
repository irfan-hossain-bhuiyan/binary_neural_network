# OR operator candidate property table

This is the pre-training semantic classification for the operators in the
operator study. The table distinguishes **strict parameter endpoint
rigidity** from **functional Boolean safety**. An exact one produced by an
already-selected contribution can remain one after thresholding even when
other, functionally irrelevant contributions are fractional. Numerical
stress results are recorded in `operator_results/operator_stress_results.json`.

| operator | formula | exact Boolean OR | strict F=0 parameter rigidity | strict F=1 parameter rigidity | functional Boolean safety | monotonic | duplicate/idempotent | smooth | negative partials | zero-gradient/plateau risk | singularity risk | width-sensitive | associative | expected discretization behavior |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| hard max | `max(v)` | yes | yes | yes | yes | yes | no | no | winner-only gradient | no | no | yes | semantic reference; sparse gradients |
| Lehmer p=.5,1,2,4 | `sum(v^(p+1))/sum(v^p)` | yes | yes | yes in the exact interior semantics | yes for the Boolean result | no for weak coordinates | yes | interior | yes | denominator/all-zero branch | p<1 near zero | no | no | strong edge competition; check nonmonotone optimization |
| odds weighted | `sum(v^2/(1-v))/sum(v/(1-v))` | Boolean endpoints | yes | no strict rigidity, but an exact-one contribution is functionally safe | yes | no | interior | no | near-one domination | near one | no | no | endpoint loophole and possible gradient explosion |
| log-hazard weighted | `h=-log(1-v); sum(hv)/sum(h)` | Boolean endpoints | yes | no strict rigidity, but an exact-one contribution is functionally safe | yes | no | interior | no | near-one domination | near one | no | no | smoother than odds, still endpoint loophole |
| probabilistic OR | `1-prod(1-v)` | yes | yes | no strict rigidity; exact-one contribution is functionally safe | yes | no | interior | no | gradient vanishes as other values grow | no | strong | yes | correct endpoint semantics, fractional accumulation |
| Einstein OR | fold `(a+b)/(1+ab)` | yes | yes | no strict rigidity; exact-one contribution is functionally safe | yes | no | smooth | no | repeated-input saturation | no | strong | yes | monotone but width can create near-one outputs |
| Hamacher λ=0,.5,1 | fold `(a+b-(2-λ)ab)/(1-(1-λ)ab)` | yes | yes | no strict rigidity; exact-one contribution is functionally safe | yes | generally no | interior | no | bounded-sum saturation | λ=0 boundary | strong | yes | fuzzy controls; test width accumulation |
| Łukasiewicz bounded sum | `min(1,sum(v))` | yes | yes | no strict rigidity and no general functional safety | yes | no | no | no | exact plateau | no | strong | no | explicit endpoint-rigidity negative control |
| softmax weighted value α=1,4,16 | `sum(softmax(αv)*v)` | no at finite α | yes | no | can be negative | no | yes | possible | no | no | no | no | useful score/value bridge, not exact OR |

Sparsemax and entmax are queued as optional weighting controls. Ordinary
normalized power means and unnormalized log-sum-exp are excluded from the
primary set because they do not reproduce Boolean OR exactly at finite
parameters.

## Tests used

The stress script tests widths 2, 4, 8, 16, 64, 256 and 784 under zero,
single-winner, equal, Boolean, uniform, beta and near-endpoint regimes. It
records forward quantiles, gradient quantiles and norms, negative-gradient
fractions, top-gradient concentration, NaN/Inf counts, duplicate accumulation,
and an endpoint search from random fractional initializations. It does not
train a neural network.
