"""CPU invariants for the modern XOR residual architecture."""
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from layers import xor, XorResidualLogicBlock
from discrete_logic_net import DiscreteXorResidualLogicBlock
from models import ModernLogicGateNet
from research.run_experiment import parameter_binarized


def _truth_rows(n):
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=n)))


def _set_layer_pair(continuous, discrete):
    with torch.no_grad():
        for cl, dl in zip(continuous, discrete):
            w = torch.randint(0, 2, cl.weight.shape).bool()
            b = torch.randint(0, 2, cl.bias.shape).bool()
            cl.weight.copy_(w.float()); cl.bias.copy_(b.float())
            dl.weight.copy_(w); dl.bias.copy_(b)


def test_continuous_xor_boolean_endpoints():
    rows = _truth_rows(2)
    assert torch.equal(xor(rows[:, :1], rows[:, 1:]), (rows[:, :1].bool() ^ rows[:, 1:].bool()).float())


def test_residual_block_matches_discrete_for_all_boolean_rows():
    c = XorResidualLogicBlock(3, use_softmax=False)
    d = DiscreteXorResidualLogicBlock(3)
    _set_layer_pair([c.layer1, c.layer2], [d.layer1, d.layer2])
    rows = _truth_rows(3)
    assert torch.equal(c(rows), d(rows).float())


def test_entire_modern_network_matches_discrete_truth_table():
    c = ModernLogicGateNet(input_dim=3, output_dim=2, width=3, num_residual_blocks=2,
                           use_softmax=False, learnable_tau=False)
    d = c.to_discrete()
    # Exact parameter correspondence, then enumerate every input row.
    _set_layer_pair(c.expectation_layers, d.expectation_layers)
    rows = _truth_rows(3)
    assert torch.equal(c(rows), d(rows).float())
    assert [type(b).__name__ for b in c.blocks] == ["XorResidualLogicBlock"] * 2
    assert c.stem.out_features == c.width == c.head.in_features


def test_no_residual_control_has_same_parameter_shapes():
    r = ModernLogicGateNet(4, 2, width=5, num_residual_blocks=2, residual_enabled=True)
    n = ModernLogicGateNet(4, 2, width=5, num_residual_blocks=2, residual_enabled=False)
    assert [tuple(x.weight.shape) for x in r.expectation_layers] == [tuple(x.weight.shape) for x in n.expectation_layers]
    assert len(r.expectation_layers) == len(n.expectation_layers) == 6


def test_parameter_binarized_is_only_a_corner_predicate():
    assert parameter_binarized({"D_w": 0.005, "D_b": 0.005,
                                "w_corner_05": 0.96, "b_corner_05": 0.97})
    assert not parameter_binarized({"D_w": 0.005, "D_b": 0.005,
                                    "w_corner_05": 0.96, "b_corner_05": 0.94})
    assert not parameter_binarized({"D_w": 0.5, "D_b": 0.5,
                                    "w_corner_05": 1.0, "b_corner_05": 1.0})


def test_temperature_free_model_has_boolean_conversion_and_residual_topology():
    from models import TemperatureFreeModernLogicGateNet
    model = TemperatureFreeModernLogicGateNet(
        input_dim=4, output_dim=2, width=5, num_residual_blocks=2,
        gate_initializations=[0.75, 0.25, 0.75, 0.25, 0.75, 0.25])
    discrete = model.to_discrete()
    assert len(model.expectation_layers) == len(discrete.expectation_layers) == 6
    assert [b.residual_enabled for b in model.blocks] == [True, True]
    for layer, dlayer in zip(model.expectation_layers, discrete.expectation_layers):
        gate, bias = layer.to_discrete()
        assert torch.equal(gate, dlayer.weight)
        assert torch.equal(bias, dlayer.bias)
    init = model.expectation_layers[0]
    assert torch.allclose(init.effective_gate(), torch.full_like(init.theta, 0.75), atol=1e-6)
    assert not any("temperature" in name or "tau" in name for name, _ in model.named_parameters())


def test_temperature_free_layer_zeroes_all_zero_literals_for_arbitrary_strength():
    from layers import TemperatureFreeLogicLayer
    layer = TemperatureFreeLogicLayer(4, 2, gate_initialization=0.6)
    with torch.no_grad():
        layer.theta.copy_(torch.tensor([[-10.0, -2.0, 0.5, 10.0], [4.0, -6.0, 2.0, -1.0]]))
        layer.bias.zero_()
    assert torch.allclose(layer(torch.zeros(3, 4)), torch.zeros(3, 2), atol=1e-8)


def test_temperature_free_softmax_strength_self_sharpens_to_one():
    from layers import TemperatureFreeLogicLayer
    layer = TemperatureFreeLogicLayer(5, 1, gate_initialization=0.5)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    observed = []
    with torch.no_grad():
        layer.bias.zero_()
        eps_r = 1e-3
        for selected_r in (1.0, 2.0, 5.0, 10.0, 20.0):
            raw_r = torch.full_like(layer.theta, eps_r)
            raw_r[0, 0] = selected_r
            layer.theta.copy_(torch.log(torch.expm1(raw_r)))
            observed.append(layer(x).item())
    assert observed == sorted(observed)
    assert observed[-1] > 0.999


def test_temperature_free_selected_but_inactive_literal_outputs_zero():
    from layers import TemperatureFreeLogicLayer
    layer = TemperatureFreeLogicLayer(3, 1, gate_initialization=0.5)
    with torch.no_grad():
        layer.theta.fill_(torch.log(torch.expm1(torch.tensor(20.0))).item())
        layer.bias.zero_()
    assert torch.equal(layer(torch.zeros(1, 3)), torch.zeros(1, 1))


def test_temperature_free_layer_approaches_boolean_or_for_polarized_strengths():
    from layers import TemperatureFreeLogicLayer
    torch.manual_seed(112)
    layer = TemperatureFreeLogicLayer(16, 1, gate_initialization=0.5)
    selected = torch.tensor([1, 3, 5, 9, 12, 15])
    with torch.no_grad():
        layer.bias.zero_()
        r = torch.full_like(layer.theta, 1e-3)
        r[:, selected] = 20.0
        layer.theta.copy_(torch.log(torch.expm1(r)))
    x = torch.randint(0, 2, (512, 16)).float()
    expected = x[:, selected].bool().any(dim=-1, keepdim=True).float()
    assert (layer(x) - expected).abs().max() < 0.002


def test_temperature_free_gate_initialization_maps_through_inverse_softplus_atanh():
    from models import TemperatureFreeModernLogicGateNet
    model = TemperatureFreeModernLogicGateNet(
        4, 2, width=3, num_residual_blocks=2,
        gate_initializations=[0.75, 0.25, 0.75, 0.25, 0.75, 0.25])
    for layer, expected in zip(model.expectation_layers,
                               [0.75, 0.25, 0.75, 0.25, 0.75, 0.25]):
        assert torch.allclose(layer.effective_gate(), torch.full_like(layer.theta, expected), atol=1e-6)
        r = torch.atanh(torch.tensor(expected))
        theta = torch.log(torch.expm1(r))
        assert torch.allclose(layer.theta, torch.full_like(layer.theta, theta), atol=1e-6)
    assert not any("temperature" in n or "tau" in n for n, _ in model.named_parameters())
