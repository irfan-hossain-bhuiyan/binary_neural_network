import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from boolean_tasks import build_task
from run_i9_ste_replication import generic_diagnostics, load_parent


def test_global_calibration_uses_all_layers_and_hits_half_ratio():
    net, _, _ = load_parent(3)
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    d = generic_diagnostics(net, task["X"].float(), task["Y"].float())
    v = d["variants"]["STE_SIGMOID"]
    assert abs(v["ratio"] - 0.5) < 1e-5
    assert len(v["layers"]) == 6


def test_boolean_exact_parent_has_zero_sigmoid_ste_gradient():
    net, _, _ = load_parent(4)
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    d = generic_diagnostics(net, task["X"].float(), task["Y"].float())
    assert d["variants"]["STE_SIGMOID"]["bool_loss"] == 0.0
    assert d["variants"]["STE_SIGMOID"]["bool_global_norm"] == 0.0
