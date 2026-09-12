"""
Unit tests para las utilidades de core.utils.
"""

import pytest
import torch

from core.utils import (
    downwards_rounding,
    generate_input_samples,
    generate_inputs,
    generate_random_matrix,
    get_default_device,
    outwards_rounding,
    upwards_rounding,
)


class TestGetDefaultDevice:
    def test_returns_torch_device(self):
        device = get_default_device()
        assert isinstance(device, torch.device)

    def test_matches_cuda_availability(self):
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert get_default_device() == expected

    def test_cuda_only_when_available(self):
        assert get_default_device().type == (
            "cuda" if torch.cuda.is_available() else "cpu"
        )


class TestGenerateInputs:
    def test_shape(self):
        tensor = generate_inputs(n_concepts=4, l=-1.0, u=1.0)
        assert tensor.shape == (2, 4)

    def test_lower_and_upper_rows(self):
        tensor = generate_inputs(n_concepts=3, l=-0.5, u=0.5)
        assert torch.all(tensor[0] == -0.5)
        assert torch.all(tensor[1] == 0.5)

    def test_dtype(self):
        tensor = generate_inputs(n_concepts=2, l=0.0, u=1.0)
        assert tensor.dtype == torch.get_default_dtype()

    def test_default_device_when_none(self):
        tensor = generate_inputs(n_concepts=2, l=0.0, u=1.0)
        assert tensor.device == get_default_device()

    def test_explicit_device(self):
        tensor = generate_inputs(n_concepts=2, l=0.0, u=1.0, device="cpu")
        assert tensor.device == torch.device("cpu")


class TestGenerateInputSamples:
    def test_shape(self):
        samples = generate_input_samples(n_samples=10, n_concepts=3, l=-1.0, u=1.0)
        assert samples.shape == (10, 3)

    def test_within_interval(self):
        l, u = -0.3, 0.7
        samples = generate_input_samples(n_samples=100, n_concepts=5, l=l, u=u)
        assert torch.all(samples >= l)
        assert torch.all(samples <= u)

    def test_default_device_when_none(self):
        samples = generate_input_samples(n_samples=10, n_concepts=3, l=-1.0, u=1.0)
        assert samples.device == get_default_device()

    def test_reproducible_with_seed(self):
        torch.manual_seed(42)
        a = generate_input_samples(n_samples=5, n_concepts=3, l=-1.0, u=1.0)
        torch.manual_seed(42)
        b = generate_input_samples(n_samples=5, n_concepts=3, l=-1.0, u=1.0)
        assert torch.equal(a, b)


class TestGenerateRandomMatrix:
    def test_shape_diagonal_and_bounds(self):
        W = generate_random_matrix(n_concepts=10, density=0.5)
        assert W.shape == (10, 10)
        assert torch.all(torch.diagonal(W) == 0)
        assert torch.all(W[W != 0] >= -1.0)
        assert torch.all(W[W != 0] <= 1.0)

    def test_has_nonzero_entries(self):
        W = generate_random_matrix(n_concepts=10, density=0.5)
        assert (W != 0).sum().item() > 0

    def test_reproducible_with_seed(self):
        torch.manual_seed(7)
        a = generate_random_matrix(n_concepts=8, density=0.3)
        torch.manual_seed(7)
        b = generate_random_matrix(n_concepts=8, density=0.3)
        assert torch.equal(a, b)

    def test_full_density_selects_all_off_diagonal(self):
        torch.manual_seed(1234)
        W = generate_random_matrix(n_concepts=6, density=1.0)
        assert (W != 0).sum().item() == 6 * 5

    def test_density_assert(self):
        with pytest.raises(AssertionError):
            generate_random_matrix(n_concepts=4, density=1.5)

    def test_n_concepts_assert(self):
        with pytest.raises(AssertionError):
            generate_random_matrix(n_concepts=1, density=0.5)

    def test_dtype(self):
        W = generate_random_matrix(n_concepts=4, density=0.5, dtype=torch.float64)
        assert W.dtype == torch.float64

    def test_default_device_when_none(self):
        W = generate_random_matrix(n_concepts=4, density=0.5)
        assert W.device == get_default_device()


class TestRounding:
    @pytest.mark.parametrize("value", [0.0, 1.0, 0.5, 3.14159, 1e-5])
    def test_upwards_rounding_increases(self, value):
        x = torch.tensor(value)
        up = upwards_rounding(x)
        assert up > x
        assert up == torch.nextafter(x, torch.tensor(float("inf"), dtype=x.dtype))

    def test_upwards_rounding_preserves_dtype(self):
        x = torch.tensor(0.5, dtype=torch.float64)
        assert upwards_rounding(x).dtype == torch.float64

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.5, 3.14159])
    def test_downwards_rounding_decreases(self, value):
        x = torch.tensor(value)
        down = downwards_rounding(x)
        assert down < x
        assert down == torch.nextafter(x, torch.tensor(float("-inf"), dtype=x.dtype))

    def test_outwards_rounding_shape_assert(self):
        with pytest.raises(AssertionError):
            outwards_rounding(torch.zeros(3, 4))

    def test_outwards_rounding_expands_interval(self):
        x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])
        result = outwards_rounding(x)
        assert torch.all(result[0] <= x[0])
        assert torch.all(result[1] >= x[1])
        assert torch.all(result[0] < x[0])
        assert torch.all(result[1] > x[1])

    def test_outwards_rounding_does_not_mutate_input(self):
        x = torch.tensor([[0.5], [1.0]])
        original = x.clone()
        outwards_rounding(x)
        assert torch.equal(x, original)
