import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from or_surrogates import HardMax, LehmerMean


def test_lehmer_order_and_max_bound():
    torch.manual_seed(7)
    v = torch.rand(32, 64)
    l2 = LehmerMean(2)(v)
    l4 = LehmerMean(4)(v)
    l8 = LehmerMean(8)(v)
    maximum = v.max(dim=-1).values
    assert torch.all(l2 <= l4 + 1e-6)
    assert torch.all(l4 <= l8 + 1e-6)
    assert torch.all(l8 <= maximum + 1e-6)


def test_effective_contributors_do_not_increase_with_p():
    torch.manual_seed(8)
    v = torch.rand(32, 64)
    values = []
    for p in (2, 4, 8):
        weights = v.pow(p)
        alpha = weights / weights.sum(dim=-1, keepdim=True)
        values.append((1 / alpha.square().sum(dim=-1)).mean())
    assert values[1] <= values[0] + 1e-6
    assert values[2] <= values[1] + 1e-6


def test_unique_max_gets_closer_at_p8():
    v = torch.tensor([[0.91, 0.61, 0.4, 0.2]])
    maximum = v.max(dim=-1).values
    assert (maximum - LehmerMean(8)(v)).item() < (maximum - LehmerMean(2)(v)).item()


def test_hardmax_operator_is_explicit_and_reloadable():
    v = torch.tensor([[0.1, 0.7, 0.3]])
    op = HardMax()
    assert op.name == "hardmax"
    assert torch.equal(op(v), v.max(dim=-1).values)
