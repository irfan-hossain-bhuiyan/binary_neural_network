"""Tests for the Boolean task generators (CPU-only).

Each test compares the generator output against an independent pure-Python
reference implementation of the Boolean function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boolean_tasks import build_task


def _rows(X):
    return [tuple(int(v) for v in row) for row in X.tolist()]


def test_identity():
    t = build_task("identity", {"n": 4})
    assert t["input_dim"] == 4 and t["output_dim"] == 4
    assert t["X"].tolist() == t["Y"].tolist()
    assert len(_rows(t["X"])) == 16


def test_not():
    t = build_task("not", {"n": 4})
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        assert y == tuple(1 - b for b in x)


def test_and():
    t = build_task("and", {"n": 4})
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        assert y == (1 if all(x) else 0,), x


def test_or():
    t = build_task("or", {"n": 4})
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        assert y == (1 if any(x) else 0,), x


def test_xor2():
    t = build_task("xor2", {})
    assert sorted(_rows(t["X"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        assert y == (x[0] ^ x[1],), x


def test_parity():
    for n in (4, 8):
        t = build_task("parity", {"n": n})
        assert len(_rows(t["X"])) == 2 ** n
        for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
            assert y == (sum(x) % 2,), (n, x)


def test_majority():
    for n in (5, 7):
        t = build_task("majority", {"n": n})
        for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
            assert y == (1 if sum(x) * 2 > n else 0,), (n, x)


def test_multiplexer4():
    t = build_task("multiplexer4", {})
    assert t["input_dim"] == 6 and t["output_dim"] == 1
    assert len(_rows(t["X"])) == 64
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        s0, s1, d0, d1, d2, d3 = x
        assert y == ([d0, d1, d2, d3][s0 * 2 + s1],), x


def test_full_adder():
    t = build_task("full_adder", {})
    assert t["input_dim"] == 3 and t["output_dim"] == 2
    assert len(_rows(t["X"])) == 8
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        a, b, cin = x
        assert y == (a ^ b ^ cin, 1 if a + b + cin >= 2 else 0), x


def test_compare_unsigned():
    t = build_task("compare_unsigned", {"bits": 3})
    assert t["input_dim"] == 6 and t["output_dim"] == 1
    assert len(_rows(t["X"])) == 64
    for x, y in zip(_rows(t["X"]), _rows(t["Y"])):
        aval = x[0] * 4 + x[1] * 2 + x[2]
        bval = x[3] * 4 + x[4] * 2 + x[5]
        assert y == (1 if aval > bval else 0,), x


def test_bitwise_xor_reproducible():
    t1 = build_task("bitwise_xor", {"bits": 4, "num_samples": 500}, seed=0)
    t2 = build_task("bitwise_xor", {"bits": 4, "num_samples": 500}, seed=0)
    assert t1["X"].tolist() == t2["X"].tolist()
    assert t1["Y"].tolist() == t2["Y"].tolist()
    assert t1["input_dim"] == 8 and t1["output_dim"] == 4
    for x, y in zip(_rows(t1["X"]), _rows(t1["Y"])):
        a = x[0] * 8 + x[1] * 4 + x[2] * 2 + x[3]
        b = x[4] * 8 + x[5] * 4 + x[6] * 2 + x[7]
        c = a ^ b
        assert y == ((c >> 3) & 1, (c >> 2) & 1, (c >> 1) & 1, c & 1), x


def test_bitwise_xor_truth_table_is_exact_and_complete():
    from boolean_tasks import FULL_TRUTH_TABLE

    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    assert task["input_dim"] == 8 and task["output_dim"] == 4
    assert task["eval_mode"] == FULL_TRUTH_TABLE
    assert task["X"].shape == (256, 8) and task["Y"].shape == (256, 4)
    rows = _rows(task["X"])
    assert len(set(rows)) == 256
    for x, y in zip(rows, _rows(task["Y"])):
        a = sum(bit << (3 - i) for i, bit in enumerate(x[:4]))
        b = sum(bit << (3 - i) for i, bit in enumerate(x[4:]))
        value = a ^ b
        expected = tuple((value >> shift) & 1 for shift in (3, 2, 1, 0))
        assert y == expected, (x, y, expected)


def test_all_values_binary():
    import boolean_tasks as bt

    for name in bt.list_tasks():
        params = {}
        if name in ("identity", "not", "and", "or", "parity"):
            params = {"n": 3}
        elif name == "majority":
            params = {"n": 3}
        elif name == "compare_unsigned":
            params = {"bits": 2}
        elif name == "bitwise_xor":
            params = {"bits": 2, "num_samples": 64}
        elif name == "bitwise_xor_truth_table":
            params = {"bits": 2}
        t = build_task(name, params, seed=1)
        vals = set(t["X"].tolist().__str__().replace("[", "").replace("]", "").replace(",", "").split())
        assert vals <= {"0.0", "1.0", "0", "1", ".0"}, (name, vals)
        assert t["X"].shape[1] == t["input_dim"]
        assert t["Y"].shape[1] == t["output_dim"]
