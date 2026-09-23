import torch

from research.run_i6_or_consistency import (lehmer_from_contributions,
                                            tc_regularizer,
                                            threshold_consistency_penalty)


def test_lehmer_is_bounded_by_maximum():
    v = torch.rand(128, 17)
    f = lehmer_from_contributions(v)
    assert torch.all(f <= v.max(dim=-1).values + 1e-7)
    assert torch.all(f >= -1e-7)


def test_positive_lehmer_output_implies_positive_max_threshold():
    v = torch.rand(128, 17)
    f = lehmer_from_contributions(v)
    m = v.max(dim=-1).values
    assert torch.all((f > .5) <= (m > .5))


def test_threshold_violation_penalty_is_one_sided_and_differentiable():
    assert threshold_consistency_penalty(torch.tensor([.25]), torch.tensor([.75])) > 0
    assert threshold_consistency_penalty(torch.tensor([.25]), torch.tensor([.40])) == 0
    assert threshold_consistency_penalty(torch.tensor([.75]), torch.tensor([.80])) == 0
    # The helper is exercised through a tiny real network layer so the max
    # contribution remains connected to autograd.
    from research.run_i5_gate_regularization import make_model

    net = make_model()
    x = torch.rand(3, 8)
    r = tc_regularizer(net, x, (4,))
    assert r.requires_grad
    r.backward()
    assert any(layer.raw_edge.grad is not None for layer in net.expectation_layers)
