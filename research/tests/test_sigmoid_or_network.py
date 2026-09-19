import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from layers import SigmoidOrLogicLayer
from models import SigmoidOrModernLogicGateNet


def test_boolean_endpoints_for_exact_or_candidates():
    candidates = ["hardmax", "lehmer_p1", "lehmer_p2", "log_hazard", "probabilistic_or"]
    v = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    for name in candidates:
        layer = SigmoidOrLogicLayer(2, 1, name, gate_initialization=0.999)
        with torch.no_grad():
            layer.raw_edge.fill_(20.0)
            layer.bias.zero_()
        out = layer.or_operator(v.unsqueeze(1) * 0 + v.unsqueeze(1))
        assert torch.allclose(out.flatten(), torch.tensor([0.0, 1.0, 1.0, 1.0]), atol=1e-5), name


def test_tiny_modern_network_discrete_graph_matches_boolean_continuous_hard_path():
    torch.manual_seed(4)
    model = SigmoidOrModernLogicGateNet(2, 1, width=2, num_residual_blocks=1, or_operator="hardmax")
    for layer in model.expectation_layers:
        with torch.no_grad():
            layer.raw_edge.copy_(torch.where(torch.rand_like(layer.raw_edge) > .5, torch.tensor(20.), torch.tensor(-20.)))
            layer.bias.copy_(torch.where(torch.rand_like(layer.bias) > .5, torch.tensor(20.), torch.tensor(-20.)))
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    continuous = model.forward_hard(x) >= .5
    discrete = model.to_discrete().forward(x.bool())
    assert torch.equal(continuous, discrete)
