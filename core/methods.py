"""
Métodos principales de verificación de FCMs.

Este módulo implementa los tres métodos principales:
1. Arithmetic: Aritmética de intervalos simple
2. Rectified: Método combinado simbólico-aritmético
3. Symbolic: Método simbólico puro
"""

from typing import Any, Optional

import torch

from core.metrics import is_inside_state, states_intersection
from core.relaxation import (
    apply_relaxed_activation,
    get_function_data,
    overapproximation_from_relaxations,
)
from core.symbolic import (
    concretize_state_space,
    induced_state_space,
    propagate_state_space,
    symbolic_iteration,
)
from core.utils import outwards_rounding


def arithmetic_iteration(
    W_pos: torch.Tensor,
    W_neg: torch.Tensor,
    state: torch.Tensor,
    activation_function: str,
    phi: float,
    inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Realiza una iteración aritmética (intervalica simple) de la FCM.

    Args:
        W_pos: Pesos positivos (W clampado a >= 0).
        W_neg: Pesos negativos (W clampado a <= 0).
        state: Estado actual (2, n_concepts).
        activation_function: Función de activación.
        phi: Parámetro de mezcla.
        inputs: Intervalos de entrada (default: state).

    Returns:
        torch.Tensor: Nuevo estado después de la iteración.
    """
    if inputs is None:
        inputs = state

    transfer_function = (
        torch.sigmoid if activation_function == "sigmoid" else torch.tanh
    )

    new_state = torch.stack(
        [
            (
                torch.matmul(W_pos.to(state.dtype), state[i])
                + torch.matmul(W_neg.to(state.dtype), state[1 - i])
            )
            for i in range(2)
        ],
        dim=0,
    )
    new_state = outwards_rounding(new_state)

    new_state = phi * outwards_rounding(transfer_function(new_state))
    new_state = outwards_rounding(new_state)
    new_state = outwards_rounding(new_state + (1 - phi) * inputs)

    return new_state


def interval_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    max_iters: int = 20,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta el método aritmético (intervalico simple) de Householder-Weng.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        max_iters: Número máximo de iteraciones.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados (max_iters + 1, 2, n_concepts).
    """
    dtype = inputs.dtype
    if device is None:
        device = inputs.device

    W_pos = torch.clamp(W, min=0).to(dtype=dtype)
    W_neg = torch.clamp(W, max=0).to(dtype=dtype)

    all_states = torch.empty((max_iters + 1, *inputs.shape), device=device, dtype=dtype)
    all_states[0] = inputs.clone()

    for t in range(1, max_iters + 1):
        all_states[t] = arithmetic_iteration(
            W_pos, W_neg, all_states[t - 1], activation_function, phi, inputs
        )

    return all_states


def rectified_intersection_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    max_iters: int = 20,
    baseline_interval_series: Optional[torch.Tensor] = None,
    minimal_warming_iters: int = 5,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta el método rectificado con intercepción iterativa para mejorar la precisión.

    Este método ejecuta múltiples pasadas del método rectificado, refinando los
    intervalos en cada iteración mediante intersección con los resultados anteriores.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        max_iters: Número máximo de iteraciones.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        minimal_warming_iters: Iteraciones simbólicas mínimas iniciales.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados concretizados (max_iters + 1, 2, n_concepts).
    """
    n_concepts = W.shape[0]
    dtype = inputs.dtype
    if device is None:
        device = inputs.device

    default_state = torch.stack(
        [
            torch.full((n_concepts,), -torch.inf, dtype=dtype, device=device),
            torch.full((n_concepts,), torch.inf, dtype=dtype, device=device),
        ]
    )

    if baseline_interval_series is not None:
        rectified_intersection = baseline_interval_series.clone()
    else:
        rectified_intersection = default_state.unsqueeze(0).repeat(max_iters + 1, 1, 1)

    for arithmetic_rectifications in range(1, 11):
        rectified_states = containment_guided_state_space(
            W,
            inputs,
            activation_function,
            phi=phi,
            max_iters=max_iters,
            device=device,
            minimal_symbolic_warming_iters=minimal_warming_iters,
            arithmetic_rectifications=arithmetic_rectifications,
            baseline_interval_series=rectified_intersection,
        )

        for i in range(max_iters + 1):
            rectified_intersection[i] = states_intersection(
                rectified_intersection[i], rectified_states[i]
            )
    return rectified_intersection


def containment_guided_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    minimal_symbolic_warming_iters: int = 5,
    arithmetic_rectifications: int = 5,
    max_iters: int = 20,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta el método rectificado combinando enfoques simbólico y aritmético.

    El método alterna entre fases de ejecución simbólica (mientras se mantenga
    la contención) y fases de rectificación aritmética (cuando se pierde contención).

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        minimal_symbolic_warming_iters: Iteraciones simbólicas mínimas iniciales.
        arithmetic_rectifications: Iteraciones de rectificación aritmética.
        max_iters: Número máximo de iteraciones.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados concretizados (max_iters + 1, 2, n_concepts).
    """
    n_concepts = W.shape[0]
    w_shape = (n_concepts, n_concepts)
    inputs_shape = (2, n_concepts)
    dtype = inputs.dtype

    if device is None:
        device = inputs.device

    W_pos = torch.clamp(W, min=0).to(dtype=dtype)
    W_neg = torch.clamp(W, max=0).to(dtype=dtype)

    assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"
    assert (
        inputs.shape == inputs_shape
    ), f"inputs must have shape {inputs_shape} but has shape {inputs.shape}"

    if baseline_interval_series is not None:
        assert baseline_interval_series.shape[1:] == (2, n_concepts)

    all_states = torch.empty(
        (max_iters + 1, 2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
    )
    concrete_states = torch.empty(
        (max_iters + 1, 2, n_concepts), device=device, dtype=dtype
    )

    all_states[0] = induced_state_space(n_concepts, device=device)
    concrete_states[0] = inputs

    index = 1
    current_inputs_index = 0
    pending_symbolic_iters = minimal_symbolic_warming_iters

    while index <= max_iters:
        state = symbolic_iteration(
            all_states[index - 1],
            W,
            concrete_states[current_inputs_index],
            phi,
            activation_function=activation_function,
            original_inputs=inputs,
            device=device,
        )
        concrete_state = concretize_state_space(
            torch.unsqueeze(state, dim=0),
            concrete_states[current_inputs_index],
            device=device,
            original_inputs=inputs,
        ).squeeze(dim=0)
        concrete_state = outwards_rounding(concrete_state)

        concrete_states[index] = arithmetic_iteration(
            W_pos, W_neg, concrete_states[index - 1], activation_function, phi, inputs
        )

        if pending_symbolic_iters > 0 or is_inside_state(
            concrete_state, concrete_states[index]
        ):
            all_states[index] = state
            concrete_states[index] = states_intersection(
                concrete_states[index], concrete_state
            )

            if baseline_interval_series is not None:
                concrete_states[index] = states_intersection(
                    concrete_states[index], baseline_interval_series[index]
                )

            index += 1
            pending_symbolic_iters -= 1

        else:
            for _ in range(min(arithmetic_rectifications, max_iters - index + 1)):
                all_states[index] = induced_state_space(n_concepts, device=device)
                concrete_states[index] = arithmetic_iteration(
                    W_pos,
                    W_neg,
                    concrete_states[index - 1],
                    activation_function,
                    phi,
                    inputs,
                )
                if baseline_interval_series is not None:
                    concrete_states[index] = states_intersection(
                        concrete_states[index], baseline_interval_series[index]
                    )
                current_inputs_index = index
                index += 1

            pending_symbolic_iters = minimal_symbolic_warming_iters

    return concrete_states


def symbolic_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    max_iters: int = 20,
    overapprox_threshold: Optional[float] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta el método simbólico puro de Householder-Weng.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        max_iters: Número máximo de iteraciones.
        overapprox_threshold: Umbral para corrección de sobreaproximación.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados simbólicos.
    """
    from core.relaxation import get_function_data

    n_concepts = W.shape[0]
    w_shape = (n_concepts, n_concepts)
    inputs_shape = (2, n_concepts)
    dtype = inputs.dtype

    assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"
    assert (
        inputs.shape == inputs_shape
    ), f"inputs must have shape {inputs_shape} but has shape {inputs.shape}"

    if overapprox_threshold is None:
        overapprox_threshold = float("inf")

    if device is None:
        device = inputs.device

    func = get_function_data(activation_function, phi)[0]

    all_states = torch.empty(
        (max_iters + 1, 2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
    )
    all_states[0] = induced_state_space(n_concepts, device=device, dtype=dtype)

    for i in range(1, max_iters + 1):
        all_states[i] = propagate_state_space(all_states[i - 1], W)
        concrete_state = concretize_state_space(
            torch.unsqueeze(all_states[i], dim=0), inputs, device=device
        )
        concrete_state = concrete_state[0]

        all_states[i], lower_m, lower_b, upper_m, upper_b = apply_relaxed_activation(
            all_states[i], concrete_state, activation_function, phi=phi, device=device
        )

        # Corregir sobreaproximaciones
        overapprox = overapproximation_from_relaxations(
            inputs, lower_m, lower_b, upper_m, upper_b
        )
        is_overapproximated = overapprox > overapprox_threshold

        symbolic_state_with_concrete_coeffs = torch.zeros_like(
            all_states[i], device=device
        )
        symbolic_state_with_concrete_coeffs[:, :, 0] = func(concrete_state)
        symbolic_state_with_concrete_coeffs[:, :, 0] += (1 - phi) * inputs

        all_states[i] = torch.where(
            is_overapproximated.view(1, -1, 1),
            symbolic_state_with_concrete_coeffs,
            all_states[i],
        )
        all_states[i] = outwards_rounding(all_states[i])

    return all_states


def warm_start_symbolic_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    arithmetic_iters: int,
    phi: float,
    max_iters: int = 20,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta primero iteraciones aritméticas y luego cambia a simbólicas.

    Args:
        W: Matriz de pesos.
        inputs: Intervalos iniciales.
        activation_function: Función de activación.
        arithmetic_iters: Número de iteraciones aritméticas iniciales.
        phi: Parámetro de mezcla.
        max_iters: Número máximo total de iteraciones.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados simbólicos.
    """
    from core.relaxation import get_function_data

    n_concepts = W.shape[0]
    w_shape = (n_concepts, n_concepts)
    inputs_shape = (2, n_concepts)
    dtype = inputs.dtype

    assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"
    assert (
        inputs.shape == inputs_shape
    ), f"inputs must have shape {inputs_shape} but has shape {inputs.shape}"

    if device is None:
        device = inputs.device

    func = get_function_data(activation_function, phi)[0]

    all_states = torch.empty(
        (max_iters + 1, 2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
    )

    all_states[0] = torch.zeros(
        (2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
    )
    all_states[0][:, :, 0] = inputs

    for i in range(1, arithmetic_iters + 1):
        all_states[i] = propagate_state_space(all_states[i - 1], W)
        all_states[i, :, :, 0] = func(all_states[i, :, :, 0]) + (1 - phi) * inputs

    symbolic_inputs = concretize_state_space(
        all_states[arithmetic_iters].unsqueeze(0), inputs, device=device
    )[0]

    all_states[arithmetic_iters + 1 : max_iters + 1, :, :, :] = symbolic_state_space(
        W,
        symbolic_inputs,
        activation_function,
        phi,
        max_iters - arithmetic_iters,
        device=device,
    )[1:]

    return all_states
