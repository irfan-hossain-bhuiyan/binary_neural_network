import torch

from research.run_i5_gate_regularization import make_model, regularizer


def test_polar_endpoints_and_midpoint():
    assert torch.allclose(4 * torch.tensor([0., .5, 1.]) * (1 - torch.tensor([0., .5, 1.])), torch.tensor([0., 1., 0.]))


def test_margin4_values():
    r = torch.tensor([0., 2., 4., -4., 6.])
    value = (torch.relu(4 - r.abs()) / 4).square()
    assert torch.allclose(value, torch.tensor([1., .25, 0., 0., 0.]))


def test_regularizers_are_layer_balanced():
    net = make_model()
    with torch.no_grad():
        for layer in net.expectation_layers:
            layer.raw_edge.fill_(0.)
    assert torch.isclose(regularizer(net, "polar"), torch.tensor(1.0))
    assert torch.isclose(regularizer(net, "margin4"), torch.tensor(1.0))
