import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from boolean_tasks import build_task
from run_i11_loss_geometry import (
    REPEATS,
    scalar_optimum,
    scalar_risk,
    noisy_dataset,
    weighted_loss,
    make_model,
    state_hash,
)
from run_initialization_i2 import model_for


def test_population_optima_match_closed_forms():
    assert scalar_optimum(.8, "MSE") == .8
    assert scalar_optimum(.8, "BCE") == .8
    assert abs(scalar_optimum(.8, "POWER_1_5", 1.5) - .9411764706) < 1e-9
    assert abs(scalar_optimum(.8, "POWER_1_25", 1.25) - .9961089494) < 1e-9
    assert scalar_optimum(.8, "MAE") == 1.0
    assert scalar_optimum(.5, "MAE") is None


def test_weighted_repeated_objective_equals_expanded_objective():
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    clean_x, clean_y = task["X"], task["Y"]
    _, expanded_y, ones, zeros = noisy_dataset(clean_x, clean_y, .2, 3)
    output = torch.rand_like(clean_y).clamp(.01, .99)
    expanded_output = output.repeat_interleave(REPEATS, dim=0)
    for loss in ("MSE", "BCE", "POWER_1_5", "POWER_1_25", "MAE"):
        weighted = weighted_loss(output, clean_y, ones, zeros, loss)
        if loss == "MSE":
            values = (expanded_output - expanded_y).square()
        elif loss == "BCE":
            values = torch.nn.functional.binary_cross_entropy(expanded_output, expanded_y, reduction="none")
        elif loss == "MAE":
            values = (expanded_output - expanded_y).abs()
        else:
            alpha = 1.5 if loss == "POWER_1_5" else 1.25
            values = (expanded_output - expanded_y).abs().pow(alpha)
        assert torch.allclose(weighted, values.mean(), atol=1e-7)


def test_i2b_initial_state_is_byte_identical_when_reloaded():
    a = model_for("I2-B", 3, torch.device("cpu"))
    b = model_for("I2-B", 3, torch.device("cpu"))
    assert state_hash(a.state_dict()) == state_hash(b.state_dict())


def test_scalar_risk_has_expected_minimum_for_power_loss():
    p_star = scalar_optimum(.9, "POWER_1_25", 1.25)
    left = float(scalar_risk(p_star - .00005, .9, "POWER_1_25", 1.25))
    center = float(scalar_risk(p_star, .9, "POWER_1_25", 1.25))
    right = float(scalar_risk(p_star + (1.0 - p_star) / 2, .9, "POWER_1_25", 1.25))
    assert center < left and center < right
