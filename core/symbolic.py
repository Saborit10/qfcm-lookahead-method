"""
Operaciones de espacio de estados simbólico para verificación de FCMs.

Este módulo contiene las funciones para:
- Construir el espacio de estados simbólico inicial
- Concretizar estados simbólicos a intervalos concretos
- Propagar estados simbólicos a través de la matriz de pesos
- Realizar iteraciones simbólicas completas
"""

from typing import Any, Optional

import torch

from core.utils import (
    downwards_rounding,
    get_default_device,
    outwards_rounding,
    upwards_rounding,
)


def induced_state_space(
    n_concepts: int,
    device: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Construye el espacio de estados simbólico inicial.

    Cada concepto depende inicialmente solo de sí mismo.
    Esta versión soporta phi != 1.0 con estado residual.

    Args:
        n_concepts: Número de conceptos en la FCM.
        device: Dispositivo para el tensor. Si es None, se usa
            `get_default_device`.
        dtype: Tipo de dato para el tensor.

    Returns:
        torch.Tensor: Tensor de forma (2, n_concepts, 2*n_concepts + 1) con
            la representación simbólica inicial.
    """
    if device is None:
        device = get_default_device()

    indices = torch.arange(n_concepts, device=device)
    base = torch.zeros((n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype)
    base[indices, indices + 1] = 1

    return torch.stack([base, base])


def concretize_state_space(
    states: torch.Tensor,
    inputs: torch.Tensor,
    original_inputs: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Concretiza el estado simbólico evaluando con los valores de entrada.

    Args:
        states: Tensor de forma (n_iters, 2, n_concepts, 2*n_concepts + 1).
        inputs: Tensor de forma (2, n_concepts) con los intervalos de entrada.
        original_inputs: Tensor opcional con los intervalos originales (para phi != 1).
        device: Dispositivo para tensores temporales. Si es None, se usa el de `states`.

    Returns:
        torch.Tensor: Tensor de forma (n_iters, 2, n_concepts) con los
            intervalos concretizados.
    """
    n_concepts = states.shape[2]
    state_shape = (2, n_concepts, 2 * n_concepts + 1)
    input_shape = (2, n_concepts)

    assert (
        states.shape[1:] == state_shape
    ), f"states must have shape {state_shape}, got {states.shape[1:]}"
    assert (
        inputs.shape == input_shape
    ), f"inputs must have shape {input_shape}, got {inputs.shape}"

    if device is None:
        device = states.device

    if original_inputs is None:
        original_inputs = inputs

    inputs = outwards_rounding(torch.cat([inputs, original_inputs], dim=1))

    input_coeffs = torch.zeros_like(states, device=device)
    input_coeffs[:, :, :, 0] = 1.0

    positive_elements_mask = states[:, :, :, 1:] > 0

    inputs_i = inputs[None, :, None, :]
    inputs_j = inputs.flip(dims=[0])[None, :, None, :]

    input_coeffs[:, :, :, 1:] = torch.where(positive_elements_mask, inputs_i, inputs_j)

    input_coeffs[:, 0, :, :] = downwards_rounding(input_coeffs[:, 0, :, :])
    input_coeffs[:, 1, :, :] = upwards_rounding(input_coeffs[:, 1, :, :])

    result = torch.sum(states * input_coeffs, dim=3)
    result[:, 0, :] = downwards_rounding(result[:, 0, :])
    result[:, 1, :] = upwards_rounding(result[:, 1, :])

    return result


def propagate_state_space(
    state: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """
    Propaga el estado simbólico a través de la matriz de pesos W.

    Args:
        state: Tensor de forma (2, n_concepts, ...) con el estado simbólico.
        W: Tensor de forma (n_concepts, n_concepts) con los pesos de la FCM.

    Returns:
        torch.Tensor: Estado propagado con redondeo hacia afuera.

    Note:
        Separa los pesos positivos y negativos para manejar correctamente
        las desigualdades en aritmética de intervalos.
    """
    n_concepts = W.shape[0]
    w_shape = (n_concepts, n_concepts)
    state_shape_prefix = (2, n_concepts)

    assert (
        state.shape[:2] == state_shape_prefix
    ), f"state shape must have shape {state_shape_prefix} in the first two dimensions, but has shape {state.shape}"

    assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"

    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)

    result = outwards_rounding(
        torch.matmul(W_pos, state) + torch.matmul(W_neg, state.flip(dims=[0]))
    )

    return result


def symbolic_iteration(
    state: torch.Tensor,
    W: torch.Tensor,
    inputs: torch.Tensor,
    phi: float,
    activation_function: str,
    original_inputs: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Realiza una iteración simbólica completa (propagación + activación).

    Args:
        state: Estado simbólico actual.
        W: Matriz de pesos.
        inputs: Intervalos de entrada concretos.
        phi: Parámetro de mezcla.
        activation_function: Función de activación.
        original_inputs: Intervalos originales (para phi != 1).
        device: Dispositivo. Si es None, se usa el de `inputs`.

    Returns:
        torch.Tensor: Nuevo estado simbólico.
    """
    from core.relaxation import apply_relaxed_activation

    if device is None:
        device = inputs.device

    new_state = propagate_state_space(state, W)
    new_state = outwards_rounding(new_state)

    concrete_state = concretize_state_space(
        torch.unsqueeze(new_state, dim=0),
        inputs,
        original_inputs=original_inputs,
        device=device,
    )
    concrete_state = outwards_rounding(concrete_state[0])

    new_state = apply_relaxed_activation(
        new_state, concrete_state, activation_function, phi=phi
    )[0]
    return outwards_rounding(new_state)
