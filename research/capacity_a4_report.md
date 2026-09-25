# Representational capacity gate: exact 4-bit addition

## Result

The implemented discrete architecture can represent unsigned 4-bit addition.
The construction in [`capacity_a4.py`](capacity_a4.py) programs the actual
`DiscreteModernLogicGateNet(8, 5, width=64, num_residual_blocks=2)` and was
evaluated on all 256 rows.  It obtains Boolean bit accuracy **1.0** and exact
row accuracy **1.0**.  This is an existence proof for the current model class,
not a claim that ordinary optimization finds the construction.

## Construction

Each discrete layer is an OR of selected `(input XOR bias)` literals.  The
64-wide stem provides constants, raw input literals, and their complements.
The first residual block uses

```text
a XOR (b) = a XOR b
0 XOR (NOT u1 OR ... OR NOT uk) = NOT(u1 AND ... AND uk)
```

to produce the four pairwise XOR bits and NAND monomials for the ripple carries.
The next block ORs complements of those NANDs to form `c1`, `c2`, and `c3`,
then XORs each carry into the corresponding higher sum bit.  Three final NAND
features form the majority carry `c4`, which the head ORs after complementing.
Unused width coordinates are left at zero.  The construction is deliberately
larger than a minimal circuit; only existence inside the exact architecture is
required.

## Independent target check

The task generator checks integer arithmetic and ripple-carry equations before
the construction is evaluated.  Both references agree on all 256 rows.

## Implication for A6

The capacity confound is removed.  Subsequent failures on this architecture
can be investigated as optimization or topology-credit failures.  A6 is
therefore authorized by the capacity gate.

Reproduce with:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python research/capacity_a4.py
```
