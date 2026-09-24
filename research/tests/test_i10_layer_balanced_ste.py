import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from boolean_tasks import build_task
from run_i10_layer_balanced_ste import (
    all_grads,
    calibration,
    compose_gradients,
    layer_lambdas,
    load_parent,
    manual_optimizer_step,
    task_loss,
)
from run_i8_boolean_ste import bool_loss


def test_synthetic_layer_calibration_is_balanced_and_zero_safe():
    task = [torch.tensor([3.0, 4.0]), torch.tensor([2.0])]
    boolean = [torch.tensor([1.0, 0.0]), torch.zeros(1)]
    lambdas = layer_lambdas(task, boolean)
    assert abs(lambdas[0] * boolean[0].norm() / task[0].norm() - 0.5) < 1e-7
    assert lambdas[1] == 0.0


def test_seed3_per_layer_calibration_and_seed4_zero_boolean_gradient():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    seed3, _, _ = load_parent(3)
    cal = calibration(seed3, x, y)
    rows = cal["variants"]["STE_SIGMOID"]["layers"]
    for row in rows:
        if row["bool_norm"] > 1e-12:
            assert abs(row["scaled_ratio"] - 0.5) < 1e-5
    assert 10.0 < cal["lambdas"][4] < 12.0

    seed4, _, _ = load_parent(4)
    cal4 = calibration(seed4, x, y)
    assert cal4["lambdas"] == [0.0] * 6
    assert cal4["variants"]["STE_SIGMOID"]["bool_global_norm"] == 0.0


def test_bias_gradients_are_task_only_in_composition():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    net, _, _ = load_parent(3)
    cal = calibration(net, x, y)
    params = list(net.parameters())
    task_grads = all_grads(net, task_loss(net, x, y), params)
    bool_grads = [g for g in torch.autograd.grad(
        bool_loss(net, x, y, "STE_SIGMOID"),
        [layer.raw_edge for layer in net.expectation_layers],
        retain_graph=True,
        allow_unused=True,
    )]
    bool_grads = [g if g is not None else torch.zeros_like(layer.raw_edge)
                  for g, layer in zip(bool_grads, net.expectation_layers)]
    composed = compose_gradients(net, task_grads, bool_grads, cal["lambdas"])
    for parameter, task_gradient, final_gradient in zip(params, task_grads, composed):
        if parameter.ndim == 2 and any(parameter is layer.bias for layer in net.expectation_layers):
            assert torch.equal(task_gradient, final_gradient)


def test_manual_composition_performs_one_optimizer_step():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    net, _, _ = load_parent(3)
    cal = calibration(net, x, y)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.0)
    before = {id(p): p.detach().clone() for p in net.parameters()}
    manual_optimizer_step(net, optimizer, x, y, "STE_SIGMOID", cal["lambdas"])
    assert all(len(state) > 0 for state in optimizer.state.values())
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    assert any(not torch.equal(before[id(p)], p) for p in net.parameters())
