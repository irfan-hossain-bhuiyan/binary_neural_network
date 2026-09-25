"""Explicit exact Boolean 4-bit adder construction for the current model.

This is a capacity gate, not a trained experiment.  It programs the actual
``DiscreteModernLogicGateNet`` with OR-of-XOR literals and XOR residuals and
checks every row of the exhaustive 4-bit addition table.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from discrete_logic_net import DiscreteModernLogicGateNet
from research.boolean_tasks import build_binary_addition


WIDTH = 64


def select(layer, out: int, inputs: list[int], biases: list[int] | None = None) -> None:
    """Set one OR row to selected input XOR literals."""
    if biases is None:
        biases = [0] * len(inputs)
    if len(inputs) != len(biases):
        raise ValueError("inputs/biases length mismatch")
    for i, b in zip(inputs, biases):
        layer.weight[out, i] = True
        layer.bias[out, i] = bool(b)


def clear(net: DiscreteModernLogicGateNet) -> None:
    for layer in net.expectation_layers:
        layer.weight.zero_()
        layer.bias.zero_()


def make_capacity_net() -> DiscreteModernLogicGateNet:
    net = DiscreteModernLogicGateNet(8, 5, width=WIDTH, num_residual_blocks=2,
                                     residual_enabled=True)
    clear(net)
    stem = net.stem
    b0l1, b0l2 = net.blocks[0].layer1, net.blocks[0].layer2
    b1l1, b1l2 = net.blocks[1].layer1, net.blocks[1].layer2
    head = net.head

    # Stem coordinates: 0=constant zero, 1=constant one, 2..9=x[0..7],
    # 10..17=not x[0..7].  A tautology selects x and not-x.
    select(stem, 1, [0, 0], [0, 1])
    for i in range(8):
        select(stem, 2 + i, [i])
        select(stem, 10 + i, [i], [1])
    # The first block's residual input must also carry the a_i literals at the
    # coordinates where its XOR outputs are stored.
    for i in range(4):
        select(stem, 26 + i, [i])

    # Block 0 layer 1 copies the useful stem features.  Layer 2 is zero for
    # these rows, so the XOR residual preserves them.
    for i in range(18):
        select(b0l1, i, [i])
    for i in range(4):
        select(b0l1, 26 + i, [2 + i])

    # XOR(a_i,b_i) in coordinates 26..29: input is a_i and branch selects b_i.
    for i in range(4):
        select(b0l2, 26 + i, [2 + 4 + i])

    # NAND monomials for carries c1, c2, c3.  The residual input at these
    # coordinates is zero, so the branch output is the NAND itself.
    terms: list[list[int]] = [
        [0, 4],  # c1
        [1, 5], [0, 4, 1], [0, 4, 5],  # c2
        [2, 6],  # c3, expansion of majority(a2,b2,c2)
        [2, 1, 5], [2, 0, 4, 1], [2, 0, 4, 5],
        [6, 1, 5], [6, 0, 4, 1], [6, 0, 4, 5],
    ]
    for row, term in zip(range(34, 45), terms):
        # Terms are written in raw input indices 0..7; layer-1 stores those
        # literals at coordinates 2..9.
        select(b0l2, row, [2 + i for i in term], [1] * len(term))

    # Block 0 residual output preserves all copied features and places the
    # NAND terms at 34..44.
    # Block 1 layer 1 copies raw/features and ORs negated NAND terms into c1..c3.
    for i in range(18):
        select(b1l1, i, [i])
    for row, term_row in [(45, [34]), (46, [35, 36, 37]),
                          (47, list(range(38, 45)))]:
        select(b1l1, row, term_row, [1] * len(term_row))

    # Sums s0..s3.  The block input carries XOR(a,b) at 26..29; branch adds
    # c1,c2,c3 to the upper three bits via XOR residual.
    select(b1l2, 27, [45])
    select(b1l2, 28, [46])
    select(b1l2, 29, [47])

    # Final carry c4 = majority(a3,b3,c3), represented by three NAND terms.
    select(b1l2, 48, [5, 9], [1, 1])
    select(b1l2, 49, [5, 47], [1, 1])
    select(b1l2, 50, [9, 47], [1, 1])

    # Head copies sum bits and ORs the negated final-carry NAND terms.
    for out, src in enumerate(range(26, 30)):
        select(head, out, [src])
    select(head, 4, [48, 49, 50], [1, 1, 1])
    return net


def verify() -> dict:
    task = build_binary_addition({"bits": 4})
    net = make_capacity_net()
    with torch.no_grad():
        got = net(task["X"].bool())
    expected = task["Y"].bool()
    bit = float((got == expected).float().mean())
    rows = float((got == expected).all(dim=1).float().mean())
    if bit != 1.0 or rows != 1.0:
        bad = torch.nonzero((got != expected).any(dim=1)).flatten().tolist()
        raise AssertionError(f"capacity construction failed: bit={bit}, rows={rows}, bad={bad[:20]}")
    return {"bit_accuracy": bit, "exact_row_accuracy": rows,
            "rows": int(task["X"].shape[0]), "architecture": {
                "input_dim": 8, "width": WIDTH, "blocks": 2,
                "output_dim": 5, "residual_enabled": True,
            }}


if __name__ == "__main__":
    result = verify()
    out = Path(__file__).with_name("operator_results") / "capacity_a4_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
