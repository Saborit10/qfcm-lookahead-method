from typing import Any, Optional
import torch

from core.methods import arithmetic_iteration
from core.metrics import states_intersection
from core.symbolic import (
    concretize_state_space,
    induced_state_space,
    symbolic_iteration,
)
from core.utils import outwards_rounding


def lookahead_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    k: Optional[int] = None,
    max_iters: int = 20,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
    refine_forward_arithmetic: bool = False,
) -> torch.Tensor:
    """
    Método con ventana de lookahead (k pasos hacia delante) anclada en i-1.

    En cada iteración i:
      1. best[i] se calcula como la intersección entre un paso aritmético y la
         concretización del estado simbólico, ambos con input best[i-1].
      2. Se propagan k pasos aritméticos y k pasos simbólicos hacia delante
         (posiciones i+1 ... i+k), usando best[i-1] como input de toda la
         cadena, y se refinan los estados futuros con la intersección.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        k: Tamaño de la ventana de lookahead. Si es None, se usa max_iters
            (la ventana llega siempre hasta el final de la serie).
        max_iters: Número máximo de iteraciones.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        device: Dispositivo.
        refine_forward_arithmetic: Si es True, intersecta cada paso aritmético
            hacia delante con su predicción simbólica antes de continuar la
            cadena. El valor False conserva el comportamiento original.

    Returns:
        torch.Tensor: Serie temporal de estados concretizados (max_iters + 1, 2, n_concepts).
    """
    n_concepts = W.shape[0]
    w_shape = (n_concepts, n_concepts)
    inputs_shape = (2, n_concepts)
    dtype = inputs.dtype

    assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"
    assert (
        inputs.shape == inputs_shape
    ), f"inputs must have shape {inputs_shape} but has shape {inputs.shape}"

    if baseline_interval_series is not None:
        assert baseline_interval_series.shape[1:] == (2, n_concepts)

    if device is None:
        device = inputs.device

    W_pos = torch.clamp(W, min=0).to(dtype=dtype)
    W_neg = torch.clamp(W, max=0).to(dtype=dtype)

    if k is None:
        k = max_iters

    concrete_states = torch.empty(
        (max_iters + 1, 2, n_concepts), device=device, dtype=dtype
    )

    concrete_states[0] = inputs
    for i in range(1, max_iters + 1):
        concrete_states[i] = arithmetic_iteration(
            W_pos, W_neg, concrete_states[i - 1], activation_function, phi, inputs
        )

    for i in range(1, max_iters + 1):
        anchor = concrete_states[i - 1]

        arithmetic_i = arithmetic_iteration(
            W_pos, W_neg, anchor, activation_function, phi, inputs
        )

        sym_state = induced_state_space(n_concepts, device=device)
        sym_state = symbolic_iteration(
            sym_state,
            W,
            anchor,
            phi,
            activation_function=activation_function,
            original_inputs=inputs,
            device=device,
        )
        sym_i = concretize_state_space(
            torch.unsqueeze(sym_state, dim=0),
            anchor,
            device=device,
            original_inputs=inputs,
        ).squeeze(dim=0)
        sym_i = outwards_rounding(sym_i)

        best = states_intersection(arithmetic_i, sym_i)
        concrete_states[i] = states_intersection(best, concrete_states[i])

        if baseline_interval_series is not None:
            concrete_states[i] = states_intersection(
                concrete_states[i], baseline_interval_series[i]
            )

        arithmetic_fwd = arithmetic_i.clone()
        sym_fwd = sym_state
        for m in range(1, k + 1):
            target = i + m
            if target > max_iters:
                break

            arithmetic_fwd = arithmetic_iteration(
                W_pos, W_neg, arithmetic_fwd, activation_function, phi, inputs
            )
            sym_fwd = symbolic_iteration(
                sym_fwd,
                W,
                anchor,
                phi,
                activation_function=activation_function,
                original_inputs=inputs,
                device=device,
            )
            sym_fwd_concrete = concretize_state_space(
                torch.unsqueeze(sym_fwd, dim=0),
                anchor,
                device=device,
                original_inputs=inputs,
            ).squeeze(dim=0)
            sym_fwd_concrete = outwards_rounding(sym_fwd_concrete)

            if refine_forward_arithmetic:
                arithmetic_fwd = states_intersection(
                    arithmetic_fwd, sym_fwd_concrete
                )

            concrete_states[target] = states_intersection(
                concrete_states[target],
                states_intersection(arithmetic_fwd, sym_fwd_concrete),
            )

    return concrete_states

