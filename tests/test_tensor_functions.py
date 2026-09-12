"""
Unit tests para las funciones de activación y relajación base de
experimental.tensor_functions.
"""

import pytest
import torch

from experimental.tensor_functions import (
    line_from_points_vectorized,
    lower_quasi_nonlinear_sigmoid_step,
    lower_quasi_nonlinear_tanh_step,
    quasi_nonlinear_sigmoid,
    quasi_nonlinear_sigmoid_derivative,
    quasi_nonlinear_tanh,
    quasi_nonlinear_tanh_derivative,
    sigmoid,
    tanh,
    tangent_to_function_vectorized,
    upper_quasi_nonlinear_sigmoid_step,
    upper_quasi_nonlinear_tanh_step,
)

GRID = torch.linspace(-4.0, 4.0, 129)


class TestActivations:
    def test_sigmoid_matches_torch(self):
        assert torch.allclose(sigmoid(GRID), torch.sigmoid(GRID))

    def test_sigmoid_in_unit_interval_and_monotonic(self):
        values = sigmoid(GRID)
        assert torch.all(values > 0)
        assert torch.all(values < 1)
        assert torch.all(torch.diff(values) > 0)

    def test_sigmoid_symmetry(self):
        assert torch.allclose(sigmoid(GRID) + sigmoid(-GRID), torch.ones_like(GRID))

    def test_tanh_matches_torch(self):
        assert torch.allclose(tanh(GRID), torch.tanh(GRID))


class TestQuasiNonlinear:
    def test_sigmoid_scaled_by_phi(self):
        phi = 0.5
        assert torch.allclose(
            quasi_nonlinear_sigmoid(phi)(GRID), phi * torch.sigmoid(GRID)
        )

    def test_tanh_scaled_by_phi(self):
        phi = 0.3
        assert torch.allclose(quasi_nonlinear_tanh(phi)(GRID), phi * torch.tanh(GRID))

    def test_sigmoid_derivative_formula(self):
        phi = 1.0
        s = sigmoid(GRID)
        assert torch.allclose(
            quasi_nonlinear_sigmoid_derivative(phi)(GRID), phi * s * (1 - s)
        )

    def test_sigmoid_derivative_finite_differences(self):
        x = torch.linspace(-3.0, 3.0, 61)
        h = 1e-4
        numeric = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
        analytic = quasi_nonlinear_sigmoid_derivative(1.0)(x)
        assert torch.allclose(numeric, analytic, atol=1e-3)

    def test_tanh_derivative_formula(self):
        phi = 0.5
        t = tanh(GRID)
        assert torch.allclose(
            quasi_nonlinear_tanh_derivative(phi)(GRID), phi * (1 - t * t)
        )

    def test_tanh_derivative_finite_differences(self):
        x = torch.linspace(-3.0, 3.0, 61)
        h = 1e-4
        numeric = (tanh(x + h) - tanh(x - h)) / (2 * h)
        analytic = quasi_nonlinear_tanh_derivative(1.0)(x)
        assert torch.allclose(numeric, analytic, atol=1e-3)


class TestLineFromPoints:
    def test_line_passes_through_points(self):
        p1 = torch.tensor([[0.0, -1.0], [1.0, 2.0]])
        p2 = torch.tensor([[2.0, 1.0], [5.0, -1.0]])
        m, b = line_from_points_vectorized(p1, p2)
        assert m.shape == (2,)
        assert b.shape == (2,)
        assert torch.allclose(m * p1[0] + b, p1[1])
        assert torch.allclose(m * p2[0] + b, p2[1])

    def test_degenerate_points_yield_constant_line(self):
        p = torch.tensor([[1.5, -2.0], [0.25, 0.75]])
        m, b = line_from_points_vectorized(p, p)
        assert torch.all(m == 0)
        assert torch.allclose(b, p[1])

    def test_shape_asserts(self):
        with pytest.raises(AssertionError):
            line_from_points_vectorized(torch.zeros(2, 2), torch.zeros(2, 3))


class TestTangentToFunction:
    def test_tangent_touches_function_at_point(self):
        x = torch.tensor([0.5, -1.0])
        m, b = tangent_to_function_vectorized(x, sigmoid, quasi_nonlinear_sigmoid_derivative(1.0))
        assert torch.allclose(m * x + b, sigmoid(x))
        assert torch.allclose(m, sigmoid(x) * (1 - sigmoid(x)))


class TestSigmoidSteps:
    def test_lower_step_fixed_point_tangency(self):
        l = torch.tensor([-2.0])
        u = torch.tensor([2.0])
        step = lower_quasi_nonlinear_sigmoid_step(1.0)
        z = l.clone()
        for _ in range(2000):
            z = step(z, u)
        slope = (sigmoid(u) - sigmoid(z)) / (u - z)
        derivative = sigmoid(z) * (1 - sigmoid(z))
        assert torch.allclose(slope, derivative, atol=1e-5)

    def test_upper_step_fixed_point_tangency(self):
        l = torch.tensor([-2.0])
        u = torch.tensor([2.0])
        step = upper_quasi_nonlinear_sigmoid_step(1.0)
        z = u.clone()
        for _ in range(2000):
            z = step(z, l)
        slope = (sigmoid(z) - sigmoid(l)) / (z - l)
        derivative = sigmoid(z) * (1 - sigmoid(z))
        assert torch.allclose(slope, derivative, atol=1e-5)


class TestTanhSteps:
    def test_lower_step_fixed_point_tangency(self):
        l = torch.tensor([-2.0])
        u = torch.tensor([2.0])
        step = lower_quasi_nonlinear_tanh_step(1.0)
        z = l.clone()
        for _ in range(2000):
            z = step(z, u)
        slope = (tanh(u) - tanh(z)) / (u - z)
        derivative = 1 - tanh(z) * tanh(z)
        assert torch.allclose(slope, derivative, atol=1e-4)

    def test_upper_step_fixed_point_tangency(self):
        l = torch.tensor([-2.0])
        u = torch.tensor([2.0])
        step = upper_quasi_nonlinear_tanh_step(1.0)
        z = u.clone()
        for _ in range(2000):
            z = step(z, l)
        slope = (tanh(z) - tanh(l)) / (z - l)
        derivative = 1 - tanh(z) * tanh(z)
        assert torch.allclose(slope, derivative, atol=1e-4)
