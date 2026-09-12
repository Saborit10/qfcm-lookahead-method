"""
Unit tests para las relajaciones lineales de core.relaxation.
"""

import pytest
import torch

from core.relaxation import (
    apply_relaxed_activation,
    get_function_data,
    lower_relaxation,
    overapproximation_from_relaxations,
    upper_relaxation,
)
from core.symbolic import induced_state_space
from experimental.tensor_functions import sigmoid, tanh

RELAXATION_CASES = [
    ("sigmoid", sigmoid, -2.0, 2.0),
    ("sigmoid", sigmoid, 0.1, 3.0),
    ("tanh", tanh, -2.0, 2.0),
    ("tanh", tanh, -3.0, 0.1),
]


class TestGetFunctionData:
    @pytest.mark.parametrize("activation_function", ["sigmoid", "tanh"])
    def test_returns_four_callables(self, activation_function):
        func, derivative, lower_step, upper_step = get_function_data(
            activation_function, phi=1.0
        )
        for callable_ in (func, derivative, lower_step, upper_step):
            assert callable(callable_)

    def test_unknown_function_raises(self):
        with pytest.raises(KeyError):
            get_function_data("relu", phi=1.0)


class TestLowerRelaxation:
    @pytest.mark.parametrize("name,func,lo,hi", RELAXATION_CASES)
    def test_line_is_below_function(self, name, func, lo, hi):
        n_concepts = 3
        inputs = torch.tensor([[lo] * n_concepts, [hi] * n_concepts])
        m, b = lower_relaxation(inputs, name, phi=1.0)

        assert m.shape == (n_concepts,)
        assert b.shape == (n_concepts,)

        xs = torch.linspace(lo, hi, 501).unsqueeze(0)
        line = m.view(n_concepts, 1) * xs + b.view(n_concepts, 1)
        assert torch.all(line <= func(xs) + 1e-4)

    def test_shape_assert(self):
        with pytest.raises(AssertionError):
            lower_relaxation(torch.zeros(3, 4), "sigmoid", phi=1.0)
        with pytest.raises(AssertionError):
            lower_relaxation(torch.zeros(4), "sigmoid", phi=1.0)


class TestUpperRelaxation:
    @pytest.mark.parametrize("name,func,lo,hi", RELAXATION_CASES)
    def test_line_is_above_function(self, name, func, lo, hi):
        n_concepts = 3
        inputs = torch.tensor([[lo] * n_concepts, [hi] * n_concepts])
        m, b = upper_relaxation(inputs, name, phi=1.0)

        assert m.shape == (n_concepts,)
        assert b.shape == (n_concepts,)

        xs = torch.linspace(lo, hi, 501).unsqueeze(0)
        line = m.view(n_concepts, 1) * xs + b.view(n_concepts, 1)
        assert torch.all(line >= func(xs) - 1e-4)

    def test_shape_assert(self):
        with pytest.raises(AssertionError):
            upper_relaxation(torch.zeros(3, 4), "sigmoid", phi=1.0)
        with pytest.raises(AssertionError):
            upper_relaxation(torch.zeros(4), "sigmoid", phi=1.0)


class TestApplyRelaxedActivation:
    def test_output_shapes(self):
        n_concepts = 3
        state = induced_state_space(n_concepts)
        inputs = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        result, lower_m, lower_b, upper_m, upper_b = apply_relaxed_activation(
            state, inputs, "sigmoid", phi=1.0
        )

        assert result.shape == (2, n_concepts, 2 * n_concepts + 1)
        assert lower_m.shape == (n_concepts,)
        assert lower_b.shape == (n_concepts,)
        assert upper_m.shape == (n_concepts,)
        assert upper_b.shape == (n_concepts,)

    def test_zero_state_embeds_b_intercepts(self):
        n_concepts = 3
        state = torch.zeros(2, n_concepts, 2 * n_concepts + 1)
        inputs = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        result, lower_m, lower_b, upper_m, upper_b = apply_relaxed_activation(
            state, inputs, "sigmoid", phi=1.0
        )

        assert torch.all(result[0, :, 0] <= lower_b)
        assert torch.all(result[1, :, 0] >= upper_b)
        assert torch.allclose(result[:, :, 1:], torch.zeros_like(result[:, :, 1:]), atol=1e-30)

    def test_shape_asserts(self):
        with pytest.raises(AssertionError):
            apply_relaxed_activation(torch.zeros(2, 2, 5), torch.zeros(2, 3), "sigmoid")
        with pytest.raises(AssertionError):
            apply_relaxed_activation(torch.zeros(2, 3, 7), torch.zeros(2, 2), "sigmoid")


class TestOverapproximation:
    def test_zero_when_relaxations_coincide(self):
        inputs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        m = torch.zeros(2)
        b = torch.zeros(2)
        assert torch.all(
            overapproximation_from_relaxations(inputs, m, b, m, b) == 0.0
        )

    def test_non_negative_for_valid_interval(self):
        inputs = torch.tensor([[0.1, 0.2], [0.9, 1.1]])
        lower_m = torch.zeros(2)
        lower_b = torch.zeros(2)
        upper_m = torch.ones(2)
        upper_b = torch.ones(2)
        overapprox = overapproximation_from_relaxations(
            inputs, lower_m, lower_b, upper_m, upper_b
        )
        assert torch.all(overapprox >= 0.0)
