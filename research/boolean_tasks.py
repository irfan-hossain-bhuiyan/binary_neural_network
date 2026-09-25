"""Boolean benchmark task registry.

Each builder returns a task dict::

    {
        "task": str,               # task name
        "task_params": dict,       # parameters used
        "input_dim": int,
        "output_dim": int,
        "eval_mode": str,          # "full_truth_table" | "train_test_split"
        "X": FloatTensor,          # Boolean 0/1 inputs, shape (N, input_dim)
        "Y": FloatTensor,          # Boolean 0/1 outputs, shape (N, output_dim)
        "description": str,
    }

Small enumerable tasks return the COMPLETE truth table (deterministic,
no sampling). Only `bitwise_xor` samples (kept for regression against the
original baseline) and is reproducible via an explicit seed.
"""

from __future__ import annotations

import itertools

import torch

FULL_TRUTH_TABLE = "full_truth_table"
TRAIN_TEST_SPLIT = "train_test_split"


def _all_binary_rows(n_bits: int) -> torch.Tensor:
    """All 2**n binary rows, MSB-first, as float tensor."""
    rows = list(itertools.product([0.0, 1.0], repeat=n_bits))
    return torch.tensor(rows, dtype=torch.float32)


def _task_dict(task, params, X, Y, eval_mode, description, **extra) -> dict:
    return {
        "task": task,
        "task_params": dict(params),
        "input_dim": X.shape[1],
        "output_dim": Y.shape[1],
        "eval_mode": eval_mode,
        "X": X,
        "Y": Y,
        "description": description,
        **extra,
    }


def build_identity(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 4))
    X = _all_binary_rows(n)
    return _task_dict("identity", {"n": n}, X, X.clone(), FULL_TRUTH_TABLE,
                      f"{n}-bit identity y=x")


def build_not(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 4))
    X = _all_binary_rows(n)
    return _task_dict("not", {"n": n}, X, 1.0 - X, FULL_TRUTH_TABLE,
                      f"{n}-bit bitwise NOT")


def build_and(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 4))
    X = _all_binary_rows(n)
    Y = (X.sum(dim=1, keepdim=True) == n).float()
    return _task_dict("and", {"n": n}, X, Y, FULL_TRUTH_TABLE,
                      f"{n}-input AND")


def build_or(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 4))
    X = _all_binary_rows(n)
    Y = (X.sum(dim=1, keepdim=True) >= 1).float()
    return _task_dict("or", {"n": n}, X, Y, FULL_TRUTH_TABLE,
                      f"{n}-input OR")


def build_xor2(params: dict, seed: int = 0) -> dict:
    X = _all_binary_rows(2)
    Y = ((X[:, 0] != X[:, 1])).float().unsqueeze(1)
    return _task_dict("xor2", {}, X, Y, FULL_TRUTH_TABLE, "2-input XOR gate")


def build_parity(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 4))
    X = _all_binary_rows(n)
    Y = (X.sum(dim=1, keepdim=True) % 2).float()
    return _task_dict("parity", {"n": n}, X, Y, FULL_TRUTH_TABLE,
                      f"{n}-bit parity (XOR of all inputs)")


def build_majority(params: dict, seed: int = 0) -> dict:
    n = int(params.get("n", 5))
    if n % 2 == 0:
        raise ValueError(f"majority requires odd n, got {n}")
    X = _all_binary_rows(n)
    Y = ((X.sum(dim=1, keepdim=True) * 2 > n)).float()
    return _task_dict("majority", {"n": n}, X, Y, FULL_TRUTH_TABLE,
                      f"{n}-input majority (1 iff sum > n/2)")


def build_multiplexer4(params: dict, seed: int = 0) -> dict:
    # Input order: [s0, s1, d0, d1, d2, d3]; sel = s0*2 + s1; y = d_sel.
    X = _all_binary_rows(6)
    sel = (X[:, 0] * 2 + X[:, 1]).long()
    d = X[:, 2:6]
    Y = d[torch.arange(X.shape[0]), sel].unsqueeze(1)
    return _task_dict("multiplexer4", {}, X, Y, FULL_TRUTH_TABLE,
                      "4-to-1 multiplexer: 2 select + 4 data bits")


def build_full_adder(params: dict, seed: int = 0) -> dict:
    # Inputs [a, b, carry_in]; outputs [sum, carry_out].
    X = _all_binary_rows(3)
    a, b, cin = X[:, 0], X[:, 1], X[:, 2]
    s = ((a + b + cin) % 2).unsqueeze(1)
    # carry_out = (a&b) | (a&cin) | (b&cin), computed on 0/1 floats:
    cout = (((a * b) + (a * cin) + (b * cin)) >= 1).float().unsqueeze(1)
    Y = torch.cat([s, cout], dim=1)
    return _task_dict("full_adder", {}, X, Y, FULL_TRUTH_TABLE,
                      "1-bit full adder: sum = a^b^cin, cout = majority")


def binary_addition_targets_integer(X: torch.Tensor, bits: int) -> torch.Tensor:
    """Build LSB-first addition targets with integer arithmetic."""
    weights = 2 ** torch.arange(bits, dtype=torch.long, device=X.device)
    a = (X[:, :bits].to(torch.long) * weights).sum(dim=1)
    b = (X[:, bits:2 * bits].to(torch.long) * weights).sum(dim=1)
    total = a + b
    output_weights = 2 ** torch.arange(bits + 1, dtype=torch.long, device=X.device)
    return ((total.unsqueeze(1) // output_weights) % 2).to(torch.float32)


def binary_addition_targets_ripple(X: torch.Tensor, bits: int) -> torch.Tensor:
    """Build LSB-first addition targets with ripple-carry Boolean equations."""
    a = X[:, :bits].bool()
    b = X[:, bits:2 * bits].bool()
    carry = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
    output = []
    for i in range(bits):
        output.append(a[:, i] ^ b[:, i] ^ carry)
        carry = (a[:, i] & b[:, i]) | (a[:, i] & carry) | (b[:, i] & carry)
    output.append(carry)
    return torch.stack(output, dim=1).to(torch.float32)


def binary_addition_carry_chain_lengths(X: torch.Tensor, bits: int) -> torch.Tensor:
    """Return the longest generated-carry chain for each LSB-first row.

    A chain starts at a bit that generates an outgoing carry and counts that
    carry plus consecutive propagate bits at higher positions. Rows with no
    generated carry have chain length zero.
    """
    a = X[:, :bits].bool()
    b = X[:, bits:2 * bits].bool()
    generate = a & b
    propagate = a ^ b
    lengths = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for start in range(bits):
        length = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        active = generate[:, start].clone()
        length[active] = 1
        for position in range(start + 1, bits):
            active = active & propagate[:, position]
            length[active] += 1
        lengths = torch.maximum(lengths, length)
    return lengths


def build_binary_addition(params: dict, seed: int = 0) -> dict:
    """Complete truth table for unsigned ``bits``-bit binary addition."""
    bits = int(params.get("bits", 4))
    if bits < 1 or bits > 8:
        raise ValueError(f"binary addition supports 1..8 bits, got {bits}")
    X = _all_binary_rows(2 * bits)
    Y_integer = binary_addition_targets_integer(X, bits)
    Y_ripple = binary_addition_targets_ripple(X, bits)
    if not torch.equal(Y_integer, Y_ripple):
        raise AssertionError("integer and ripple-carry addition targets disagree")
    return _task_dict(
        "binary_addition",
        {"bits": bits, "input_order": "a0..a(bits-1),b0..b(bits-1)",
         "output_order": "s0..s(bits)"},
        X,
        Y_integer,
        FULL_TRUTH_TABLE,
        f"unsigned {bits}-bit addition with LSB-first inputs and outputs",
        carry_chain_length=binary_addition_carry_chain_lengths(X, bits),
        target_integer=Y_integer,
        target_ripple=Y_ripple,
    )


def build_compare_unsigned(params: dict, seed: int = 0) -> dict:
    bits = int(params.get("bits", 3))
    X = _all_binary_rows(2 * bits)
    weights = 2 ** torch.arange(bits - 1, -1, -1, dtype=torch.float32)
    A = (X[:, :bits] * weights).sum(dim=1)
    B = (X[:, bits:] * weights).sum(dim=1)
    Y = ((A > B)).float().unsqueeze(1)
    return _task_dict("compare_unsigned", {"bits": bits}, X, Y,
                      FULL_TRUTH_TABLE,
                      f"unsigned {bits}-bit comparison: 1 iff A > B")


def build_bitwise_xor(params: dict, seed: int = 0) -> dict:
    """Sampled two-operand bitwise XOR (original baseline task).

    Reproducible via explicit seed; evaluated with a train/test split
    (unlike the full-truth-table tasks).
    """
    bits = int(params.get("bits", 4))
    num_samples = int(params.get("num_samples", 20000))
    gen = torch.Generator().manual_seed(seed)
    max_val = 2 ** bits
    a = torch.randint(0, max_val, (num_samples,), generator=gen)
    b = torch.randint(0, max_val, (num_samples,), generator=gen)
    powers = 2 ** torch.arange(bits - 1, -1, -1)

    def to_bits(v: torch.Tensor) -> torch.Tensor:
        return ((v.unsqueeze(1) & powers) != 0).float()

    X = torch.cat([to_bits(a), to_bits(b)], dim=1)
    Y = to_bits(a ^ b)
    return _task_dict(
        "bitwise_xor",
        {"bits": bits, "num_samples": num_samples},
        X, Y, TRAIN_TEST_SPLIT,
        f"bitwise XOR of two {bits}-bit numbers ({num_samples} samples)",
        train_ratio=float(params.get("train_ratio", 0.8)),
    )


def build_bitwise_xor_truth_table(params: dict, seed: int = 0) -> dict:
    """Complete truth table for bitwise XOR (function recovery, not generalization)."""
    bits = int(params.get("bits", 4))
    if bits < 1 or bits > 8:
        raise ValueError(f"truth-table XOR supports 1..8 bits, got {bits}")
    X = _all_binary_rows(2 * bits)
    Y = torch.logical_xor(X[:, :bits].bool(), X[:, bits:].bool()).float()
    return _task_dict(
        "bitwise_xor_truth_table", {"bits": bits}, X, Y, FULL_TRUTH_TABLE,
        f"complete {2 ** (2 * bits)}-row {bits}-bit XOR truth table; function recovery",
    )


TASK_BUILDERS = {
    "identity": build_identity,
    "not": build_not,
    "and": build_and,
    "or": build_or,
    "xor2": build_xor2,
    "parity": build_parity,
    "majority": build_majority,
    "multiplexer4": build_multiplexer4,
    "full_adder": build_full_adder,
    "binary_addition": build_binary_addition,
    "compare_unsigned": build_compare_unsigned,
    "bitwise_xor": build_bitwise_xor,
    "bitwise_xor_truth_table": build_bitwise_xor_truth_table,
}


def list_tasks() -> list[str]:
    return sorted(TASK_BUILDERS)


def build_task(name: str, params: dict | None = None, seed: int = 0) -> dict:
    if name not in TASK_BUILDERS:
        raise ValueError(f"Unknown task {name!r}. Known: {list_tasks()}")
    return TASK_BUILDERS[name](params or {}, seed=seed)
