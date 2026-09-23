import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_i8_boolean_ste import _ConstantSTE, _SigmoidSTE  # noqa: E402


def test_ste_forward_is_exact_hard_gate_and_gradients_have_expected_scale():
    for fn, expected in ((_SigmoidSTE, None), (_ConstantSTE, 0.25)):
        raw = torch.tensor([-2.0, 2.0], requires_grad=True)
        out = fn.apply(raw)
        assert torch.equal(out, torch.tensor([0.0, 1.0]))
        out.sum().backward()
        if expected is not None:
            assert torch.allclose(raw.grad, torch.tensor([expected, expected]))
        else:
            assert raw.grad[0] > 0 and raw.grad[1] > 0


def test_boolean_ste_forward_matches_parent_discrete_network():
    from run_i8_boolean_ste import forward_equivalence, load_parent, make_model
    from boolean_tasks import build_task

    net, _, _ = load_parent()
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    for variant in ("STE_SIGMOID", "STE_CONST025"):
        result = forward_equivalence(net, task["X"].float(), variant)
        assert result["all_equal"]
        assert result["total_mismatch"] == 0
    for seed in (3, 11, 29):
        torch.manual_seed(seed)
        random_net = make_model()
        for variant in ("STE_SIGMOID", "STE_CONST025"):
            assert forward_equivalence(random_net, task["X"].float(), variant)["all_equal"]


def test_simple_gate_gradient_moves_wrong_zero_gate_upward():
    raw = torch.tensor(-1.0, requires_grad=True)
    gate = _ConstantSTE.apply(raw)
    loss = (gate - 1).square()
    loss.backward()
    assert raw.grad < 0


def test_simple_gate_gradient_moves_wrong_one_gate_downward():
    raw = torch.tensor(1.0, requires_grad=True)
    gate = _ConstantSTE.apply(raw)
    loss = gate.square()
    loss.backward()
    assert raw.grad > 0
