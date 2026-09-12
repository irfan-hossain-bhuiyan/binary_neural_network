"""Boundary tests for the continuous -> Boolean conversion (CPU-only).

Verifies that parameters exactly at 0/1 map through to_discrete() to the
expected discrete tensors, and that hand-constructed discrete gates behave
as native Boolean logic. No architecture mathematics is changed here.
"""

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from discrete_logic_net import DiscreteMultiLayerLogicGateNet, DiscreteOrNorGateLayer
from initializers import ConstantInitWrapper
from models import MultiLayerLogicGateNet


def _tiny_net() -> MultiLayerLogicGateNet:
    return MultiLayerLogicGateNet(
        input_dim=2,
        layer_dims=(2, 1),
        use_softmax=True,
        grad_scalar=False,
        odd_initialization=ConstantInitWrapper(0.0),
        even_initialization=ConstantInitWrapper(0.0),
        bias_initialization=ConstantInitWrapper(0.0),
    )


def test_discrete_or_gate():
    layer: Any = DiscreteOrNorGateLayer(in_features=2, out_features=1)
    layer.weight.copy_(torch.tensor([[True, True]]))
    layer.bias.copy_(torch.tensor([[False, False]]))
    X = torch.tensor([[False, False], [False, True], [True, False], [True, True]])
    assert layer(X).tolist() == [[False], [True], [True], [True]]


def test_discrete_not_gate():
    layer: Any = DiscreteOrNorGateLayer(in_features=1, out_features=1)
    layer.weight.copy_(torch.tensor([[True]]))
    layer.bias.copy_(torch.tensor([[True]]))
    X = torch.tensor([[False], [True]])
    assert layer(X).tolist() == [[True], [False]]


def test_to_discrete_copies_exact_binary_params():
    net = _tiny_net()
    layers: list[Any] = list(net.expectation_layers)
    with torch.no_grad():
        layers[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        layers[0].bias.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        layers[1].weight.copy_(torch.tensor([[1.0, 1.0]]))
        layers[1].bias.copy_(torch.tensor([[0.0, 0.0]]))
    disc = net.to_discrete(threshold=0.5)
    assert isinstance(disc, DiscreteMultiLayerLogicGateNet)
    dlayers: list[Any] = list(disc.expectation_layers)
    assert dlayers[0].weight.tolist() == [[True, False], [False, True]]
    assert dlayers[0].bias.tolist() == [[False, False], [True, True]]
    assert dlayers[1].weight.tolist() == [[True, True]]
    assert dlayers[1].bias.tolist() == [[False, False]]


def test_to_discrete_threshold_boundary():
    net = _tiny_net()
    layers: list[Any] = list(net.expectation_layers)
    with torch.no_grad():
        for layer in layers:
            layer.weight.fill_(0.49)
            layer.bias.fill_(0.51)
    disc = net.to_discrete(threshold=0.5)
    dlayers: list[Any] = list(disc.expectation_layers)
    for layer in dlayers:
        assert not layer.weight.any().item()
        assert layer.bias.all().item()


def test_discretize_snaps_raw_params():
    net = _tiny_net()
    layers: list[Any] = list(net.expectation_layers)
    with torch.no_grad():
        layers[0].weight.fill_(0.2)
        layers[0].bias.fill_(0.8)
    MultiLayerLogicGateNet.discretize(net, threshold=0.5)
    assert bool((layers[0].weight <= 0).all().item())
    assert bool((layers[0].bias >= 1).all().item())
