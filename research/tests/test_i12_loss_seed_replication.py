"""Focused I12 regression tests."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from boolean_tasks import build_task
from run_i11_loss_geometry import state_hash, weighted_loss
from run_i12_loss_seed_replication import (
    loss_value,
    threshold_robustness,
    make_model,
)
from run_initialization_i2 import model_for


def test_shared_losses_have_expected_values_and_gradients():
    p = torch.tensor([[.25, .75]], requires_grad=True)
    y = torch.tensor([[0.0, 1.0]])
    assert torch.isclose(loss_value(p, y, "MSE"), torch.tensor(.0625))
    assert torch.isclose(loss_value(p, y, "POWER_1_5"), torch.tensor(.125))
    assert torch.isclose(loss_value(p, y, "POWER_1_25"), torch.tensor(.25 ** 1.25))
    for name in ("MSE", "BCE", "POWER_1_5", "POWER_1_25"):
        q = p.detach().clone().requires_grad_(True)
        loss_value(q, y, name).backward()
        assert torch.isfinite(q.grad).all()


def test_clean_unique_and_repeated_objectives_match():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"], task["Y"]
    output = torch.linspace(.05, .95, y.numel()).reshape_as(y).requires_grad_(True)
    repeated_y = y.repeat_interleave(10, dim=0)
    for name in ("MSE", "BCE", "POWER_1_5", "POWER_1_25"):
        unique = output.detach().clone().requires_grad_(True)
        counts_one = y * 10
        counts_zero = (1 - y) * 10
        weighted = weighted_loss(unique, y, counts_one, counts_zero, name)
        expanded = output.detach().repeat_interleave(10, dim=0).requires_grad_(True)
        expanded_value = loss_value(expanded, repeated_y, name)
        assert torch.allclose(weighted, expanded_value, atol=1e-6, rtol=1e-5)
        weighted.backward(); expanded_value.backward()
        assert torch.allclose(unique.grad, expanded.grad.reshape(256, 10, 4).sum(dim=1), atol=1e-6, rtol=1e-5)


def test_i2b_initial_state_is_paired_by_seed():
    for seed in range(5):
        a = model_for("I2-B", seed, torch.device("cpu"))
        b = make_model()
        initial = {k: v.detach().cpu().clone() for k, v in a.state_dict().items()}
        b.load_state_dict(initial)
        assert state_hash(initial) == state_hash({k: v.detach().cpu() for k, v in b.state_dict().items()})


def test_threshold_sweep_contains_canonical_threshold():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    net = model_for("I2-B", 0, torch.device("cpu"))
    result = threshold_robustness(net, task["X"], task["Y"])
    assert set(result) == {"0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70"}
    assert "exact_accuracy" in result["0.50"]
