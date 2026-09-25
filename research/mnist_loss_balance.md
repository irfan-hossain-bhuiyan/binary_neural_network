# One-hot loss balance for MNIST

For one output bit with positive marginal probability (q=P(y_j=1)), the
power loss has population risk

\[
R_\alpha(p)=q(1-p)^\alpha+(1-q)p^\alpha.
\]

For (1<\alpha\le2), setting the derivative to zero gives

\[
q(1-p)^{\alpha-1}=(1-q)p^{\alpha-1},
\qquad
p_\alpha^*=\frac{q^{1/(\alpha-1)}}{q^{1/(\alpha-1)}+(1-q)^{1/(\alpha-1)}}.
\]

For an approximately balanced ten-class one-hot target, (q\approx0.1):

| alpha | uninformative optimum (p_\alpha^*) |
|---:|---:|
| 2 | 0.1 |
| 1.5 | 0.0121951 |
| 1.25 | 0.000152393 |

Thus unbalanced POWER(_{1.25}) strongly rewards the all-zero output before
features provide class-specific evidence. This is an output-imbalance effect,
not evidence that the power loss is intrinsically unable to learn MNIST.

For a sample whose target class is (c), define the balanced vector loss

\[
L_{\alpha,\mathrm{balanced}}
=|1-p_c|^\alpha+\frac1{K-1}\sum_{j\ne c}|p_j|^\alpha,
\qquad K=10.
\]

The positive bit and all negative bits therefore have equal aggregate weight.
For an uninformative symmetric predictor (p_1=\cdots=p_K=p), its expected
risk is proportional to

\[
(1-p)^\alpha+p^\alpha,
\]

whose unique optimum is (p^*=0.5) for every \(\alpha>1\). Balancing removes
the unconditional all-zero attractor; feature learning must decide which bit
should move above or below one half.

Balanced BCE uses the same vector weighting:

\[
L_{\mathrm{BBCE}}=-\log(p_c)-\frac1{K-1}\sum_{j\ne c}\log(1-p_j),
\]

with probabilities clamped to \([10^{-7},1-10^{-7}]).
