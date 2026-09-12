"""
Unit tests para las métricas de comparación de core.metrics.
"""

import pytest
import torch

from core.metrics import (
    average_gap,
    convergence_iteration,
    covering,
    is_inside_state,
    is_inside_state_series,
    margin,
    series_intersection,
    series_is_nested,
    series_is_strictly_contracting,
    state_intersection,
    state_series_from_samples,
    states_intersection,
)


class TestIsInsideState:
    def test_contained(self):
        s1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        s2 = torch.tensor([[-1.0, -0.5], [2.0, 1.5]])
        assert is_inside_state(s1, s2)

    def test_not_contained(self):
        s1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        s2 = torch.tensor([[0.5, 0.0], [1.0, 1.0]])
        assert not is_inside_state(s1, s2)

    def test_equal(self):
        s = torch.tensor([[0.0, -1.0], [1.0, 1.0]])
        assert is_inside_state(s, s)

    def test_tolerance(self):
        s1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        s2 = torch.tensor([[0.25, 0.25], [0.75, 0.75]])
        assert not is_inside_state(s1, s2)
        assert is_inside_state(s1, s2, tolerance=0.25)

    def test_different_concepts_assert(self):
        s1 = torch.zeros(2, 2)
        s2 = torch.zeros(2, 3)
        try:
            is_inside_state(s1, s2)
        except AssertionError:
            return
        raise AssertionError("debería lanzar AssertionError")


class TestIsInsideStateSeries:
    def test_contained(self):
        s1 = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]]])
        s2 = torch.tensor([[[-1.0, -1.0], [2.0, 2.0]], [[-1.0, -1.0], [1.0, 1.0]]])
        assert is_inside_state_series(s1, s2)

    def test_not_contained(self):
        s1 = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        s2 = torch.tensor([[[0.1, 0.0], [1.0, 1.0]]])
        assert not is_inside_state_series(s1, s2)

    def test_equal(self):
        s = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])
        assert is_inside_state_series(s, s)


class TestStateSeriesFromSamples:
    def test_shape_and_envelope(self):
        samples = torch.tensor(
            [
                [[0.1, 0.2], [0.5, 0.8], [0.3, 0.6]],
                [[0.4, 0.1], [0.9, 0.4], [0.7, 0.2]],
                [[0.2, 0.7], [0.3, 0.5], [0.8, 0.9]],
                [[0.3, 0.3], [0.6, 0.9], [0.2, 0.4]],
            ]
        )
        series = state_series_from_samples(samples)
        assert series.shape == (3, 2, 2)

        expected_min = torch.min(samples, dim=0).values
        expected_max = torch.max(samples, dim=0).values
        assert torch.equal(series[:, 0, :], expected_min)
        assert torch.equal(series[:, 1, :], expected_max)

    def test_single_sample(self):
        samples = torch.tensor([[[0.5, 0.5]]])
        series = state_series_from_samples(samples)
        assert torch.equal(series[0, 0], torch.tensor([0.5, 0.5]))
        assert torch.equal(series[0, 1], torch.tensor([0.5, 0.5]))


class TestCovering:
    def test_constant_width_is_one(self):
        series = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.2, 0.2], [1.2, 1.2]]]
        )
        assert covering(series) == pytest.approx(1.0)

    def test_shrinking_width(self):
        series = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.25, 0.25], [0.75, 0.75]]]
        )
        assert covering(series) == pytest.approx(0.5)

    def test_custom_iteration(self):
        series = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [2.0, 2.0]],
                [[0.0, 0.0], [0.5, 0.5]],
            ]
        )
        assert covering(series, iteration=1) == pytest.approx(2.0)
        assert covering(series, iteration=2) == pytest.approx(0.5)


class TestAverageGap:
    def test_zero_gap(self):
        state_series = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        simulations = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]])
        assert average_gap(state_series, simulations) == pytest.approx(0.0)

    def test_positive_gap(self):
        state_series = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        simulations = torch.tensor([[[0.2, 0.3]], [[0.8, 0.7]]])
        assert average_gap(state_series, simulations) == pytest.approx(0.25)


class TestIntersections:
    def test_states_intersection(self):
        s1 = torch.tensor([[0.25, -1.0], [0.75, 1.0]])
        s2 = torch.tensor([[0.0, 0.25], [0.5, 0.75]])
        inter = states_intersection(s1, s2)
        assert inter.shape == (2, 2)
        assert torch.all(inter[0] <= torch.maximum(s1[0], s2[0]))
        assert torch.all(inter[1] >= torch.minimum(s1[1], s2[1]))

    def test_state_intersection(self):
        s1 = torch.tensor([[0.25, -1.0], [0.75, 1.0]])
        s2 = torch.tensor([[0.0, 0.25], [0.5, 0.75]])
        inter = state_intersection(s1, s2)
        assert torch.all(inter[0] <= torch.maximum(s1[0], s2[0]))
        assert torch.all(inter[1] >= torch.minimum(s1[1], s2[1]))

    def test_series_intersection(self):
        s1 = torch.tensor([[[0.25], [0.75]], [[0.0], [1.0]]])
        s2 = torch.tensor([[[0.0], [0.5]], [[0.5], [1.0]]])
        inter = series_intersection(s1, s2)
        assert torch.all(inter[:, 0, :] <= torch.maximum(s1[:, 0, :], s2[:, 0, :]))
        assert torch.all(inter[:, 1, :] >= torch.minimum(s1[:, 1, :], s2[:, 1, :]))


class TestMargin:
    def test_contained_series_non_negative(self):
        s1 = torch.tensor([[[0.0], [1.0]], [[0.1], [0.9]]])
        s2 = torch.tensor([[[-1.0], [2.0]], [[0.0], [1.0]]])
        assert margin(s1, s2) >= 0

    def test_equal_series_zero(self):
        s = torch.tensor([[[0.0], [1.0]], [[0.2], [0.8]]])
        assert margin(s, s) == pytest.approx(0.0)

    def test_outside_series_negative(self):
        s1 = torch.tensor([[[0.0], [1.0]]])
        s2 = torch.tensor([[[0.5], [0.6]]])
        assert margin(s1, s2) < 0


class TestConvergenceIteration:
    def test_finds_first_convergent(self):
        series = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.0], [0.3]],
                [[0.0], [0.05]],
                [[0.0], [0.01]],
            ]
        )
        assert convergence_iteration(series, epsilon=0.1) == 2

    def test_immediate_convergence(self):
        series = torch.tensor([[[0.0, 0.0], [0.05, 0.05]]])
        assert convergence_iteration(series, epsilon=0.1) == 0

    def test_no_convergence(self):
        series = torch.tensor([[[0.0], [0.5]], [[0.0], [0.6]]])
        assert convergence_iteration(series, epsilon=0.1) is None

    def test_all_concepts_must_converge(self):
        series = torch.tensor([[[0.0, 0.0], [0.05, 0.5]]])
        assert convergence_iteration(series, epsilon=0.1) is None


class TestSeriesIsNested:
    def test_shrinking_series(self):
        series = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.25, 0.25], [0.75, 0.75]]]
        )
        assert series_is_nested(series)

    def test_constant_series(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[0.0], [1.0]], [[0.0], [1.0]]]
        )
        assert series_is_nested(series)

    def test_growing_series(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[0.0], [2.0]]]
        )
        assert not series_is_nested(series)

    def test_lower_bound_grows(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[-0.5], [1.0]]]
        )
        assert not series_is_nested(series)

    def test_tolerance(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[-0.1], [1.1]]]
        )
        assert not series_is_nested(series)
        assert series_is_nested(series, tolerance=0.1)

    def test_single_iteration_trivially_nested(self):
        series = torch.tensor([[[0.0], [1.0]]])
        assert series_is_nested(series)


class TestSeriesIsStrictlyContracting:
    def test_strictly_shrinking(self):
        series = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]]]
        )
        assert series_is_strictly_contracting(series)

    def test_plateau_not_strict(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[0.0], [1.0]]]
        )
        assert not series_is_strictly_contracting(series)

    def test_growing_not_strict(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[0.0], [2.0]]]
        )
        assert not series_is_strictly_contracting(series)

    def test_epsilon_required_decrease(self):
        series = torch.tensor(
            [[[0.0], [1.0]], [[0.0], [0.9]]]
        )
        assert series_is_strictly_contracting(series, epsilon=0.0)
        assert not series_is_strictly_contracting(series, epsilon=0.2)

    def test_single_iteration_is_not_strict(self):
        series = torch.tensor([[[0.0], [1.0]]])
        assert not series_is_strictly_contracting(series)
