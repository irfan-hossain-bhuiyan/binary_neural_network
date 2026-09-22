import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from layers import xor  # noqa: E402
from research.or_surrogates import LehmerMean  # noqa: E402


def test_continuous_xor_endpoint_and_centered_identity():
    endpoints = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    expected = torch.tensor([0., 1., 1., 0.])
    assert torch.equal(xor(endpoints[:, 0], endpoints[:, 1]), expected)
    x = torch.rand(128, dtype=torch.float64)
    b = torch.rand(128, dtype=torch.float64)
    lhs = xor(x, b) - .5
    rhs = -2 * (x - .5) * (b - .5)
    assert torch.allclose(lhs, rhs)


def test_xor_threshold_homomorphism_away_from_ties():
    x = torch.rand(1000)
    b = torch.rand(1000)
    out = xor(x, b) >= .5
    expected = (x >= .5) ^ (b >= .5)
    # Exclude exact half ties; random float samples almost surely have none.
    assert torch.equal(out, expected)


def test_lehmer_p2_endpoint_rigidity_small_vectors():
    op = LehmerMean(p=2.0)
    v = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    assert torch.equal(op(v), torch.tensor([0., 1., 1., 1.]))
    fractional = torch.tensor([[.2, .8], [.5, .5]])
    out = op(fractional)
    assert torch.all((out > 0) & (out < 1))
