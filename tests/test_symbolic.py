"""
Unit tests para las operaciones de espacio de estados simbólico de core.symbolic.
"""

import pytest
import torch

from core.symbolic import (
    concretize_state_space,
    induced_state_space,
    propagate_state_space,
    symbolic_iteration,
)


class TestInducedStateSpace:
    def test_shape(self):
        state = induced_state_space(n_concepts=3)
        assert state.shape == (2, 3, 7)

    def test_identity_residual_block(self):
        n_concepts = 3
        state = induced_state_space(n_concepts)
        base = state[0]
        assert torch.equal(state[0], state[1])
        assert torch.all(base[:, 0] == 0)
        for i in range(n_concepts):
            assert base[i, i + 1] == 1
        other = base.clone()
        for i in range(n_concepts):
            other[i, i + 1] = 0
        assert torch.all(other == 0)


class TestConcretizeStateSpace:
    def test_identity_state_concretizes_to_inputs(self):
        inputs = torch.tensor([[0.1, -0.2], [0.7, 0.9]])
        state = induced_state_space(n_concepts=2)
        concretized = concretize_state_space(state.unsqueeze(0), inputs)[0]

        assert concretized.shape == (2, 2)
        assert torch.all(concretized[0] <= inputs[0])
        assert torch.all(concretized[1] >= inputs[1])
        assert torch.allclose(concretized, inputs, atol=1e-6)

    def test_shape_asserts(self):
        with pytest.raises(AssertionError):
            concretize_state_space(torch.zeros(1, 2, 2, 5), torch.zeros(2, 3))
        with pytest.raises(AssertionError):
            concretize_state_space(torch.zeros(1, 2, 3, 7), torch.zeros(2, 2))


class TestPropagateStateSpace:
    def test_identity_matrix_keeps_state(self):
        state = induced_state_space(n_concepts=3)
        W = torch.eye(3)
        propagated = propagate_state_space(state, W)
        assert propagated.shape == state.shape
        assert torch.allclose(propagated, state, atol=1e-6)

    def test_zero_matrix_yields_zero(self):
        state = induced_state_space(n_concepts=3)
        W = torch.zeros(3, 3)
        propagated = propagate_state_space(state, W)
        assert torch.allclose(propagated, torch.zeros_like(propagated), atol=1e-30)

    def test_negative_weights_flip_interval(self):
        state = induced_state_space(n_concepts=2)
        W = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
        propagated = propagate_state_space(state, W)
        assert propagated.shape == (2, 2, 5)

        inputs = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        concretized = concretize_state_space(propagated.unsqueeze(0), inputs)[0]
        assert concretized[0, 0].item() <= -2.0
        assert concretized[1, 0].item() >= 0.0
        assert torch.allclose(concretized[:, 1], torch.zeros(2), atol=1e-30)

    def test_shape_asserts(self):
        with pytest.raises(AssertionError):
            propagate_state_space(torch.zeros(2, 3, 7), torch.zeros(2, 2))
        with pytest.raises(AssertionError):
            propagate_state_space(torch.zeros(2, 2, 5), torch.zeros(3, 3))


class TestSymbolicIteration:
    def test_zero_weights_phi_one_sigmoid_constant(self):
        n_concepts = 3
        W = torch.zeros(n_concepts, n_concepts)
        inputs = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        state = symbolic_iteration(
            induced_state_space(n_concepts), W, inputs, 1.0, "sigmoid"
        )

        assert state.shape == (2, n_concepts, 2 * n_concepts + 1)
        concretized = concretize_state_space(state.unsqueeze(0), inputs)[0]
        assert torch.allclose(concretized, torch.full((2, n_concepts), 0.5), atol=1e-4)

    def test_phi_zero_keeps_inputs(self):
        n_concepts = 2
        W = torch.eye(n_concepts)
        inputs = torch.tensor([[0.1, 0.2], [0.7, 0.8]])
        state = symbolic_iteration(
            induced_state_space(n_concepts), W, inputs, 0.0, "sigmoid"
        )
        concretized = concretize_state_space(state.unsqueeze(0), inputs)[0]
        assert torch.all(concretized[0] <= inputs[0])
        assert torch.all(concretized[1] >= inputs[1])
        assert torch.allclose(concretized, inputs, atol=1e-4)
