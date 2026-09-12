"""
Tests de soporte de dispositivos (CPU/CUDA) en los métodos de espacio de estados.

Verifica que los métodos resuelvan `device=None` siguiendo el dispositivo de sus
tensores de entrada (sin errores de device mismatch) y que acepten un `device`
explícito.
"""

import pytest
import torch

from core.methods import interval_state_space, symbolic_state_space
from experimental.methods import lookahead_refined_state_space, lookahead_state_space


def _inputs(n_concepts: int, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [torch.zeros(n_concepts, device=device), torch.ones(n_concepts, device=device)]
    )


class TestMethodsFollowInputDevice:
    @pytest.mark.parametrize(
        "method", ["arithmetic", "symbolic", "lookahead", "lookahead_refined"]
    )
    def test_cpu_inputs_without_device(self, method):
        torch.manual_seed(0)
        W = torch.rand(6, 6, device="cpu")
        inputs = _inputs(6, torch.device("cpu"))

        if method == "arithmetic":
            series = interval_state_space(W, inputs, "sigmoid", 1.0, max_iters=3)
        elif method == "symbolic":
            series = symbolic_state_space(W, inputs, "sigmoid", 1.0, max_iters=3)
        elif method == "lookahead":
            series = lookahead_state_space(
                W, inputs, "sigmoid", 1.0, k=2, max_iters=3
            )
        else:
            series = lookahead_refined_state_space(
                W, inputs, "sigmoid", 1.0, k=2, max_iters=3
            )

        assert series.device == torch.device("cpu")

    def test_explicit_cpu_device(self):
        torch.manual_seed(0)
        W = torch.rand(6, 6, device="cpu")
        inputs = _inputs(6, torch.device("cpu"))
        series = interval_state_space(
            W, inputs, "sigmoid", 1.0, max_iters=3, device="cpu"
        )
        assert series.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_cuda_inputs_without_device(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        W = torch.rand(6, 6, device=device)
        inputs = _inputs(6, device)

        for method in (
            interval_state_space,
            symbolic_state_space,
            lookahead_state_space,
            lookahead_refined_state_space,
        ):
            kwargs = {"W": W, "inputs": inputs, "activation_function": "sigmoid",
                      "phi": 1.0, "max_iters": 3}
            if method is lookahead_state_space:
                kwargs["k"] = 2
            series = method(**kwargs)
            assert series.device == device
