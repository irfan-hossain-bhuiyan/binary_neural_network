# A3 — Boolean-Tolerance / Whole-Word Loss

## A. A2 audit and architecture selection

A2 artifacts were present and internally consistent. The runner constructs XOR residual blocks as h1=L1(x), h2=L2(h1), y=x+h2−2xh2, and its discrete counterpart uses x XOR h2. The tested matrix was exactly 0, 1, 2, 4, and 8 blocks, with matched no-residual arms for nonzero depths, seeds 0–2, width 64, Lehmer-p2, and exhaustive 4-bit addition. The analytic Jacobian and direct/branch decomposition were numerically verified.

A2 XOR-residual depths 1, 2, 4, and 8 reached continuous exactness in 0/3 seeds each. By the predefined fallback rule, A3 uses **two XOR-residual blocks**.

A2 source commit `research/operator_results/a2_depth_gradient_results.json`; A3 Kaggle commit `a320fe5f61b7f7774209ae2782534c51563cfadc`; device `cuda` (Tesla T4).

## B–F. Objective definitions

For d_j=|p_j−y_j|, the ideal whole-word objective is 1[max_j d_j>ε]. With ε=.10, the rational tolerance T(d)=d^q/(d^q+ε^q), q=2, has T'(d)=qε^q d^(q−1)/(d^q+ε^q)^2. The softplus arm uses normalized [softplus(20(d−.1))−softplus(−2)]/[softplus(18)−softplus(−2)]. The sigmoid step is diagnostic only.

## G–O. Results

| loss | continuous exact | Boolean exact | stable Boolean | median cont step | median Bool step | median stable Bool step | median final E_inf | median TOLERANCE_EXACT_10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | 0/5 | 0/5 | 0/5 | NOT_REACHED | NOT_REACHED | NOT_REACHED | 0.693133 | 0.0351562 |
| BIT_MEAN_RATIONAL | 0/5 | 0/5 | 0/5 | NOT_REACHED | NOT_REACHED | NOT_REACHED | 0.999171 | 0.00390625 |
| ROW_MAX_RATIONAL | 0/5 | 0/5 | 0/5 | NOT_REACHED | NOT_REACHED | NOT_REACHED | 0.997804 | 0.613281 |
| ROW_MAX_SOFTPLUS | 1/5 | 0/5 | 0/5 | 2350 | NOT_REACHED | NOT_REACHED | 0.660307 | 0.195312 |

### Per-seed final metrics

| seed | loss | cont exact | Bool exact | E_inf | rows d<.25 | rows d<.10 | wrong Boolean rows |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | BASELINE | 0.992188 | 0.277344 | 0.693133 | 0.3906 | 0.0352 | 185 |
| 0 | BIT_MEAN_RATIONAL | 0.003906 | 0.003906 | 0.999197 | 0.0039 | 0.0039 | 255 |
| 0 | ROW_MAX_RATIONAL | 0.671875 | 0.671875 | 0.997884 | 0.6719 | 0.6719 | 84 |
| 0 | ROW_MAX_SOFTPLUS | 1.000000 | 0.527344 | 0.488653 | 0.6797 | 0.1992 | 121 |
| 1 | BASELINE | 0.902344 | 0.316406 | 0.886137 | 0.0938 | 0.0000 | 175 |
| 1 | BIT_MEAN_RATIONAL | 0.003906 | 0.003906 | 0.999253 | 0.0039 | 0.0039 | 255 |
| 1 | ROW_MAX_RATIONAL | 0.613281 | 0.613281 | 0.998232 | 0.6133 | 0.6133 | 99 |
| 1 | ROW_MAX_SOFTPLUS | 0.917969 | 0.152344 | 0.839415 | 0.6836 | 0.0000 | 217 |
| 2 | BASELINE | 0.921875 | 0.562500 | 0.651425 | 0.5430 | 0.0000 | 112 |
| 2 | BIT_MEAN_RATIONAL | 0.015625 | 0.015625 | 0.999147 | 0.0156 | 0.0156 | 252 |
| 2 | ROW_MAX_RATIONAL | 0.667969 | 0.667969 | 0.997407 | 0.6680 | 0.6680 | 85 |
| 2 | ROW_MAX_SOFTPLUS | 0.957031 | 0.699219 | 0.639347 | 0.8008 | 0.6875 | 77 |
| 3 | BASELINE | 0.910156 | 0.414062 | 0.90001 | 0.6016 | 0.0508 | 150 |
| 3 | BIT_MEAN_RATIONAL | 0.003906 | 0.003906 | 0.9991 | 0.0039 | 0.0039 | 255 |
| 3 | ROW_MAX_RATIONAL | 0.570312 | 0.570312 | 0.997804 | 0.5703 | 0.5703 | 110 |
| 3 | ROW_MAX_SOFTPLUS | 0.953125 | 0.410156 | 0.734195 | 0.7539 | 0.1953 | 151 |
| 4 | BASELINE | 0.949219 | 0.421875 | 0.617436 | 0.6562 | 0.0586 | 148 |
| 4 | BIT_MEAN_RATIONAL | 0.007812 | 0.007812 | 0.999171 | 0.0078 | 0.0078 | 254 |
| 4 | ROW_MAX_RATIONAL | 0.703125 | 0.457031 | 0.987003 | 0.6484 | 0.5156 | 139 |
| 4 | ROW_MAX_SOFTPLUS | 0.960938 | 0.054688 | 0.660307 | 0.4727 | 0.0000 | 242 |

### Output-gradient sparsity and worst-bit routing

| loss | step | nonzero output-gradient fraction | s0 | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | 0 | 1.0000 | 0.175 | 0.092 | 0.274 | 0.181 | 0.277 |
| BASELINE | 500 | 1.0000 | 0.032 | 0.095 | 0.334 | 0.375 | 0.165 |
| BASELINE | 3000 | 1.0000 | 0.036 | 0.152 | 0.203 | 0.309 | 0.299 |
| BIT_MEAN_RATIONAL | 0 | 1.0000 | 0.169 | 0.075 | 0.272 | 0.168 | 0.316 |
| BIT_MEAN_RATIONAL | 500 | 1.0000 | 0.126 | 0.187 | 0.194 | 0.244 | 0.250 |
| BIT_MEAN_RATIONAL | 3000 | 1.0000 | 0.130 | 0.177 | 0.151 | 0.234 | 0.309 |
| ROW_MAX_RATIONAL | 0 | 0.2000 | 0.175 | 0.092 | 0.274 | 0.181 | 0.277 |
| ROW_MAX_RATIONAL | 500 | 0.2000 | 0.034 | 0.173 | 0.336 | 0.420 | 0.037 |
| ROW_MAX_RATIONAL | 3000 | 0.2000 | 0.095 | 0.147 | 0.180 | 0.272 | 0.305 |
| ROW_MAX_SOFTPLUS | 0 | 0.2000 | 0.175 | 0.092 | 0.274 | 0.181 | 0.277 |
| ROW_MAX_SOFTPLUS | 500 | 0.2000 | 0.027 | 0.210 | 0.348 | 0.392 | 0.023 |
| ROW_MAX_SOFTPLUS | 3000 | 0.2000 | 0.191 | 0.126 | 0.188 | 0.280 | 0.216 |

Row-max arms route output gradient through exactly one worst bit per row (about 20% of output elements initially and at the final diagnostic); bit-mean rational remains dense. The worst-bit frequency shifts toward higher-order bits during training, especially for row-max rational.

### Final Boolean carry-chain accuracy (seed means)

| loss | chain 0 | chain 1 | chain 2 | chain 3 | chain 4 |
|---|---:|---:|---:|---:|---:|
| BASELINE | 0.291 | 0.451 | 0.533 | 0.333 | 0.125 |
| BIT_MEAN_RATIONAL | 0.015 | 0.005 | 0.000 | 0.008 | 0.000 |
| ROW_MAX_RATIONAL | 0.652 | 0.831 | 0.460 | 0.133 | 0.000 |
| ROW_MAX_SOFTPLUS | 0.499 | 0.393 | 0.273 | 0.158 | 0.150 |

Native losses are not ranked by magnitude. Time-to-target fields are `NOT_REACHED` when conditions did not occur. `a3_loss_shapes.png` and `a3_loss_gradients.png` show the rational, softplus, and diagnostic sigmoid geometry.

Carry-chain exactness, continuous-to-Boolean disagreement, hard-max metrics, and A2-style internal gradient transfer are retained per trajectory in the canonical JSON.

**Recommended next experiment:** use the evidence here to choose whether a staged ordinary-loss → tolerance-loss continuation is warranted; do not add it automatically.
