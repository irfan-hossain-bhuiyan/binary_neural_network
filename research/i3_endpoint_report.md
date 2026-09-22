# I3 Endpoint Continuation Analysis

The continuation starts from verified I2-B best-continuous checkpoints.

| seed | arm | final MSE | final BCE | final E_inf | final Boolean wrong rows |
|---:|---|---:|---:|---:|---:|
| 2 | mse | 0.000962293 | 0.0161553 | 0.130798 | 32 |
| 2 | bce | 8.0828e-09 | 6.50558e-05 | 0.000423836 | 0 |
| 4 | mse | 3.27438e-07 | 0.000567251 | 0.00097692 | 0 |
| 4 | bce | 2.62609e-09 | 5.06853e-05 | 7.6047e-05 | 0 |
| 3 | mse | 7.05914e-05 | 0.00181071 | 0.179186 | 2 |
| 3 | bce | 0.00020052 | 0.000830908 | 0.320502 | 2 |

The same checkpoint and learning rate were used for both arms; Adam state was reset identically because the parent checkpoints did not contain optimizer state.

A zero-loss endpoint argument is conditional: it applies to exact endpoint values in the closure of the sigmoid parameterization. Finite low loss alone does not force every internal gate or output to an endpoint.

## Endpoint diagnostics

- seed 2 mse: worst layer mismatch `block1.residual` = `0.0390015`, mean zero-safety fraction `0.160665`, certified `False`.
- seed 2 bce: worst layer mismatch `block1.residual` = `0.108337`, mean zero-safety fraction `0.360697`, certified `False`.
- seed 4 mse: worst layer mismatch `block1.residual` = `0.158142`, mean zero-safety fraction `0.236867`, certified `False`.
- seed 4 bce: worst layer mismatch `block1.residual` = `0.0908813`, mean zero-safety fraction `0.425873`, certified `False`.
- seed 3 mse: worst layer mismatch `block1.residual` = `0.146545`, mean zero-safety fraction `0.26416`, certified `False`.
- seed 3 bce: worst layer mismatch `block1.residual` = `0.134033`, mean zero-safety fraction `0.385661`, certified `False`.

The tiny direct search (`i3_endpoint_search.json`) found no sampled counterexample at tolerance 1e-10 for fan-ins 1, 2, and 4; this is diagnostic evidence only, not a proof.
