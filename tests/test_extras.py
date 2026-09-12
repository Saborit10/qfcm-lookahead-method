"""
Unit tests para las utilidades adicionales de core.extras.
"""

import torch

from core.extras import rescaled_reasoning


class TestRescaledReasoning:
    def test_shape_and_initial_state(self):
        A = torch.rand(2, 3)
        W = torch.rand(3, 3) - 0.5
        H = rescaled_reasoning(A, W, T=3, phi=1.0)
        assert H.shape == (2, 4, 3)
        assert torch.equal(H[:, 0], A)

    def test_phi_one_matches_transfer(self):
        torch.manual_seed(0)
        A = torch.rand(2, 3)
        W = torch.rand(3, 3) - 0.5
        H = rescaled_reasoning(A, W, T=2, phi=1.0, transfer_function=torch.sigmoid)
        assert torch.allclose(H[:, 1], torch.sigmoid(A @ W))
        assert torch.allclose(H[:, 2], torch.sigmoid(H[:, 1] @ W))

    def test_phi_zero_keeps_initial_state(self):
        A = torch.rand(2, 3)
        W = torch.rand(3, 3) - 0.5
        H = rescaled_reasoning(A, W, T=3, phi=0.0)
        assert torch.allclose(H[:, 1:], A.unsqueeze(1).expand(-1, 3, -1))

    def test_mixing_phi(self):
        torch.manual_seed(1)
        A = torch.rand(2, 3)
        W = torch.rand(3, 3) - 0.5
        phi = 0.5
        H = rescaled_reasoning(A, W, T=1, phi=phi, transfer_function=torch.sigmoid)
        expected = phi * torch.sigmoid(A @ W) + (1 - phi) * A
        assert torch.allclose(H[:, 1], expected)

    def test_preserves_dtype(self):
        A = torch.rand(2, 3, dtype=torch.float64)
        W = torch.rand(3, 3, dtype=torch.float64)
        H = rescaled_reasoning(A, W, T=2, phi=1.0)
        assert H.dtype == torch.float64
