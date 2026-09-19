# OR operator candidate property table

This is the pre-training semantic classification for the operators in the
operator study. “Endpoint rigid” refers to the aggregate value itself; it does
not require functionally unused network parameters to be binary. Numerical
stress results are recorded in `operator_results/operator_stress_results.json`.

| operator | formula | exact Boolean OR | F=0 rigid | F=1 rigid | monotonic | duplicate/idempotent | smooth | negative partials | zero-gradient/plateau risk | singularity risk | width-sensitive | associative | expected discretization behavior |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| hard max | `max(v)` | yes | yes | yes | yes | yes | no | no | winner-only gradient | no | no | yes | semantic reference; sparse gradients |
| Lehmer p=.5,1,2,4 | `sum(v^(p+1))/sum(v^p)` | yes | yes | yes in the exact interior semantics | no for weak coordinates | yes | interior | yes | denominator/all-zero branch | p<1 near zero | no | no | strong edge competition; check nonmonotone optimization |
| odds weighted | `sum(v^2/(1-v))/sum(v/(1-v))` | Boolean endpoints | yes | no: any exact one dominates | yes | no | interior | no | near-one domination | near one | no | no | endpoint loophole and possible gradient explosion |
| log-hazard weighted | `h=-log(1-v); sum(hv)/sum(h)` | Boolean endpoints | yes | no: exact one dominates | yes | no | interior | no | near-one domination | near one | no | no | smoother than odds, still endpoint loophole |
| probabilistic OR | `1-prod(1-v)` | yes | yes | no: any exact one is enough | yes | no | interior | no | gradient vanishes as other values grow | no | strong | yes | correct endpoint semantics, fractional accumulation |
| Einstein OR | fold `(a+b)/(1+ab)` | yes | yes | no: exact one is absorbing | yes | no | smooth | no | repeated-input saturation | no | strong | yes | monotone but width can create near-one outputs |
| Hamacher λ=0,.5,1 | fold `(a+b-(2-λ)ab)/(1-(1-λ)ab)` | yes | yes | no for an exact one | yes | generally no | interior | no | bounded-sum saturation | λ=0 boundary | strong | yes | fuzzy controls; test width accumulation |
| Łukasiewicz bounded sum | `min(1,sum(v))` | yes | yes | no: `.5+.5=1` | yes | no | no | no | exact plateau | no | strong | no | explicit endpoint-rigidity negative control |
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

