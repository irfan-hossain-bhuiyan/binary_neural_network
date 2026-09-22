# Endpoint Consistency for Lehmer-p2 Boolean Networks

## Scope

This note separates an exact endpoint argument from finite-error training
observations.  It concerns functional discretization: irrelevant fractional
parameters need not become Boolean.

## Continuous XOR

For `x,b in [0,1]`,

```text
X(x,b) = x + b - 2xb.
```

Its endpoint equations factor as

```text
X = 0  iff  x=b=0 or x=b=1,
X = 1  iff  (x,b)=(0,1) or (x,b)=(1,0).
```

The centered identity is

```text
X(x,b) - 1/2 = -2(x-1/2)(b-1/2).
```

Thus, away from exact half ties, thresholding the continuous XOR agrees
with XOR of the separately thresholded inputs.

## Edge contribution

`v = a*g`, with `a=X(x,b)` and `g=sigmoid(r)`.  In the closure of the
parameter space, `g=0` and `g=1` correspond to `r -> -infinity` and
`r -> +infinity`.  Consequently `v=1` requires `a=1,g=1`; `v=0` means
`a=0` or `g=0`.  For finite raw logits, exact gate endpoints are generally
not attained, so zero loss may be an infimum approached by increasingly
large logits rather than a finite stationary point.

## Lehmer p=2 endpoints

For nonzero `v`,

```text
F(v) = sum(v_i^3)/sum(v_i^2)
     = sum((v_i^2/sum(v_j^2))*v_i).
```

It is therefore a weighted average and lies in `[0,1]`.  `F=0` iff every
`v_i=0`.  `F=1` iff at least one contribution is one and every contribution
with positive weight is one; zero contributions may remain irrelevant.
This is the useful endpoint-rigidity property of Lehmer p=2.

## Residual endpoints

The same XOR endpoint argument shows that `X(x,F)` can be exactly Boolean
only when both inputs are Boolean endpoints.  This preserves the functional
induction when the preceding layer is already discretization-consistent.

## Conditional soundness statement

For a fixed Boolean input row, suppose the continuous graph is evaluated in
the endpoint closure and every output of every visited logic/residual stage
is exactly Boolean.  Backward induction then gives functional agreement:

* a logic output one forces a contribution one, hence an active Boolean
  literal and edge;
* a logic output zero has only zero contributions, each of which is safe
  because its edge is absent or its literal is false;
* an XOR residual preserves the thresholded Boolean relation.

This is a conditional endpoint-soundness argument, not a theorem that
finite training must reach the closure.  The unresolved step for a general
finite network is proving that final zero numerical loss forces all needed
intermediate stages into this endpoint regime.  The I3 continuation and
tiny-network search test that gap empirically.

## Finite error

MSE, BCE, and classification accuracy are not interchangeable endpoint
certificates.  I3 records MAE, maximum absolute error `E_inf`, p95/p99
endpoint error, wrong Boolean rows, layerwise mismatches, and a conservative
Lehmer zero-side safety bound `F < 1/(4n)` for each fan-in `n`.
