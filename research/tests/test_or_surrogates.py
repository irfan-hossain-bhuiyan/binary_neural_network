import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from or_surrogates import (
    EinsteinOr,
    HamacherOr,
    LehmerMean,
    LukasiewiczOr,
    ProbabilisticOr,
    SoftmaxWeightedValue,
    OddsWeightedMean,
    LogHazardWeightedMean,
    get_operator,
)


from or_surrogates import candidate_factories


@pytest.mark.parametrize("name", list(candidate_factories()))
def test_operator_shape_and_endpoints(name):
    op = get_operator(name)
    v = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    out = op(v)
    assert out.shape == (4,)
    assert out[0].item() == pytest.approx(0.0, abs=1e-10)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["hardmax", "lehmer_p1", "probabilistic_or", "einstein_or", "hamacher_l0", "hamacher_l0.5", "hamacher_l1", "lukasiewicz_bounded_sum"])
def test_boolean_or_candidates_are_exact_on_truth_rows(name):
    v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.float64)
    assert torch.allclose(get_operator(name)(v), torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_lehmer_derivative_formula_p1():
    v = torch.tensor([[0.2, 0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    out = LehmerMean(1)(v)
    grad = torch.autograd.grad(out.sum(), v)[0]
    F = out.detach()
    expected = (2 * v.detach() - F) / v.detach().sum(dim=-1, keepdim=True)
    assert torch.allclose(grad, expected, atol=1e-10)


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 4.0])
def test_lehmer_general_derivative_formula(p):
    v = torch.tensor([[0.2, 0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    out = LehmerMean(p)(v)
    grad = torch.autograd.grad(out.sum(), v)[0]
    powers = v.detach().pow(p)
    expected = v.detach().pow(p - 1) / powers.sum(dim=-1, keepdim=True) * (
        (p + 1) * v.detach() - p * out.detach().unsqueeze(-1)
    )
    assert torch.allclose(grad, expected, atol=1e-9)


def test_probabilistic_or_formula():
    v = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    assert torch.allclose(ProbabilisticOr()(v), 1 - (1 - v).prod(dim=-1))


def test_lukasiewicz_fractional_endpoint_counterexample():
    assert LukasiewiczOr()(torch.tensor([[0.5, 0.5]])) == pytest.approx(1.0)


def test_softmax_value_derivative_matches_autograd():
    v = torch.tensor([[0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    alpha = 4.0
    out = SoftmaxWeightedValue(alpha)(v)
    grad = torch.autograd.grad(out.sum(), v)[0]
    p = torch.softmax(alpha * v.detach(), dim=-1)
    expected = p * (1 + alpha * (v.detach() - out.detach().unsqueeze(-1)))
    assert torch.allclose(grad, expected, atol=1e-10)


@pytest.mark.parametrize("op", [LehmerMean(1), LehmerMean(2), LogHazardWeightedMean(), OddsWeightedMean()])
def test_ratio_operators_have_finite_gradients_on_zero_and_mixed_rows(op):
    v = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]], dtype=torch.float64, requires_grad=True)
    op(v).sum().backward()
    assert torch.isfinite(v.grad).all()


@pytest.mark.parametrize("op", [EinsteinOr(), HamacherOr(0), HamacherOr(0.5), HamacherOr(1)])
def test_fold_operators_are_boolean_on_boolean_inputs(op):
    v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.float64)
    assert torch.equal(op(v), torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64))
