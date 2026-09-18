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
