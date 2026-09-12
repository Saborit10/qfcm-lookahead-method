"""
Funciones de relajación lineal para funciones de activación.

Estas funciones calculan las cotas lineales (rectas) que envuelven
las funciones de activación no lineales (sigmoid, tanh) en un intervalo dado,
esencial para la verificación simbólica de FCMs.
"""

from typing import Any, Callable, Optional, Tuple

import torch


def get_function_data(
    activation_function: str, phi: float
) -> Tuple[Callable, Callable, Callable, Callable]:
    """
    Obtiene las funciones de activación, derivada y pasos de relajación.

    Args:
        activation_function: Nombre de la función ("sigmoid" o "tanh").
        phi: Parámetro de la función cuasi-no-lineal.

    Returns:
        Tuple[Callable, Callable, Callable, Callable]: Tupla con:
            - Función de activación
            - Derivada de la función
            - Función de paso para relajación inferior
            - Función de paso para relajación superior
    """
    from experimental.tensor_functions import (
        lower_quasi_nonlinear_sigmoid_step,
        lower_quasi_nonlinear_tanh_step,
        quasi_nonlinear_sigmoid,
        quasi_nonlinear_sigmoid_derivative,
        quasi_nonlinear_tanh,
        quasi_nonlinear_tanh_derivative,
        upper_quasi_nonlinear_sigmoid_step,
        upper_quasi_nonlinear_tanh_step,
    )

    func, derivative, lower_step, upper_step = {
        "sigmoid": (
            quasi_nonlinear_sigmoid(phi),
            quasi_nonlinear_sigmoid_derivative(phi),
            lower_quasi_nonlinear_sigmoid_step(phi),
            upper_quasi_nonlinear_sigmoid_step(phi),
        ),
        "tanh": (
            quasi_nonlinear_tanh(phi),
            quasi_nonlinear_tanh_derivative(phi),
            lower_quasi_nonlinear_tanh_step(phi),
            upper_quasi_nonlinear_tanh_step(phi),
        ),
    }[activation_function]

    return func, derivative, lower_step, upper_step


def lower_relaxation(
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    n_iter: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula los coeficientes de relajación lineal inferior para una función de activación.

    Args:
        inputs: Tensor de forma (2, n_concepts) con los límites del intervalo.
        activation_function: Nombre de la función ("sigmoid" o "tanh").
        phi: Parámetro de la función cuasi-no-lineal.
        n_iter: Número de iteraciones para refinamiento numérico.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Coeficientes (m, b) de la recta y = mx + b.

    Note:
        El algoritmo selecciona entre:
        1. Recta secante entre los puntos extremos
        2. Recta tangente en el punto medio
        3. Recta tangente óptima encontrada iterativamente
    """
    from experimental.tensor_functions import (
        line_from_points_vectorized,
        tangent_to_function_vectorized,
    )

    assert inputs.ndim == 2 and inputs.shape[0] == 2, (
        f"inputs must have shape (2, n_concepts), but has shape {inputs.shape}"
    )
    n_concepts = inputs.shape[1]

    func, derivative, lower_step, _ = get_function_data(activation_function, phi)

    func_1 = func(inputs[1])
    derivative_0 = derivative(inputs[0])

    m1, b1 = line_from_points_vectorized(
        torch.stack([inputs[0], func(inputs[0])], dim=0),
        torch.stack([inputs[1], func(inputs[1])], dim=0),
    )
    m2, b2 = tangent_to_function_vectorized(
        (inputs[0] + inputs[1]) / 2, func, derivative
    )

    mask1 = m1 <= derivative_0
    m = torch.where(mask1, m1, torch.zeros_like(m1))
    b = torch.where(mask1, b1, torch.zeros_like(b1))

    mask2 = (~mask1) & (m2 * inputs[1] + b2 <= func_1)
    m = torch.where(mask2, m2, m1)
    b = torch.where(mask2, b2, b1)

    mask3 = (~mask1) & (~mask2)

    if mask3.any():
        z_iter = inputs[0][mask3].clone()
        for _ in range(n_iter):
            z_iter = lower_step(z_iter, inputs[1][mask3])

        m[mask3], b[mask3] = tangent_to_function_vectorized(z_iter, func, derivative)

    return m, b


def upper_relaxation(
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    n_iter: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula los coeficientes de relajación lineal superior para una función de activación.

    Args:
        inputs: Tensor de forma (2, n_concepts) con los límites del intervalo.
        activation_function: Nombre de la función ("sigmoid" o "tanh").
        phi: Parámetro de la función cuasi-no-lineal.
        n_iter: Número de iteraciones para refinamiento numérico.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Coeficientes (m, b) de la recta y = mx + b.
    """
    from experimental.tensor_functions import (
        line_from_points_vectorized,
        tangent_to_function_vectorized,
    )

    assert inputs.ndim == 2 and inputs.shape[0] == 2, (
        f"inputs must have shape (2, n_concepts), but has shape {inputs.shape}"
    )
    n_concepts = inputs.shape[1]

    func, derivative, _, upper_step = get_function_data(activation_function, phi)

    func_0 = func(inputs[0])
    derivative_1 = derivative(inputs[1])

    m1, b1 = line_from_points_vectorized(
        torch.stack([inputs[0], func_0], dim=0),
        torch.stack([inputs[1], func(inputs[1])], dim=0),
    )
    m2, b2 = tangent_to_function_vectorized(
        (inputs[0] + inputs[1]) / 2, func, derivative
    )

    mask1 = m1 <= derivative_1
    m = torch.where(mask1, m1, torch.zeros_like(m1))
    b = torch.where(mask1, b1, torch.zeros_like(b1))

    mask2 = (~mask1) & (m2 * inputs[0] + b2 >= func_0)
    m = torch.where(mask2, m2, m1)
    b = torch.where(mask2, b2, b1)

    mask3 = (~mask1) & (~mask2)

    if mask3.any():
        z_iter = inputs[1][mask3].clone()
        for _ in range(n_iter):
            z_iter = upper_step(z_iter, inputs[0][mask3])

        m[mask3], b[mask3] = tangent_to_function_vectorized(z_iter, func, derivative)

    return m, b


def apply_relaxed_activation(
    state: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float = 1.0,
    device: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aplica la función de activación relajada al estado simbólico.

    Args:
        state: Tensor de forma (2, n_concepts, 2*n_concepts + 1).
        inputs: Tensor de forma (2, n_concepts) con los intervalos concretos.
        activation_function: Nombre de la función ("sigmoid" o "tanh").
        phi: Parámetro de mezcla (1.0 = solo activación, 0.0 = estado residual).
        device: Dispositivo para tensores temporales. Si es None, se usa el de `inputs`.

    Returns:
        Tuple[torch.Tensor, ...]: Tupla con:
            - Nuevo estado simbólico
            - Coeficientes m inferiores
            - Coeficientes b inferiores
            - Coeficientes m superiores
            - Coeficientes b superiores
    """
    from core.utils import downwards_rounding, outwards_rounding, upwards_rounding

    if device is None:
        device = inputs.device

    n_concepts = state.shape[1]
    state_shape = (2, n_concepts, 2 * n_concepts + 1)
    intervals_shape = (2, n_concepts)

    assert (
        state.shape == state_shape
    ), f"state must have shape {state_shape}, but has shape {state.shape}"

    assert (
        inputs.shape == intervals_shape
    ), f"inputs must have shape {intervals_shape} but has shape {inputs.shape}"

    lower_m, lower_b = lower_relaxation(inputs, activation_function, phi)
    upper_m, upper_b = upper_relaxation(inputs, activation_function, phi)

    lower_m = downwards_rounding(lower_m)
    lower_b = downwards_rounding(lower_b)
    upper_m = upwards_rounding(upper_m)
    upper_b = upwards_rounding(upper_b)

    m_coeffs = torch.stack([lower_m, upper_m], dim=0).unsqueeze(-1)
    b_coeffs = torch.stack([lower_b, upper_b], dim=0)

    result = outwards_rounding(m_coeffs * state)
    result[:, :, 0] += b_coeffs
    result = outwards_rounding(result)

    concept_indexes = torch.arange(n_concepts, device=device)
    result[:, concept_indexes, concept_indexes + n_concepts + 1] += 1.0 - phi
    result = outwards_rounding(result)

    return result, lower_m, lower_b, upper_m, upper_b


def overapproximation_from_relaxations(
    inputs: torch.Tensor,
    lower_m: torch.Tensor,
    lower_b: torch.Tensor,
    upper_m: torch.Tensor,
    upper_b: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula la sobreaproximación introducida por las relajaciones lineales.

    Args:
        inputs: Tensor de forma (2, n_concepts) con los límites del intervalo.
        lower_m: Coeficientes de pendiente inferiores.
        lower_b: Coeficientes de intercepto inferiores.
        upper_m: Coeficientes de pendiente superiores.
        upper_b: Coeficientes de intercepto superiores.

    Returns:
        torch.Tensor: Medida de la sobreaproximación por concepto.
    """
    return (upper_m - lower_m) * (inputs[1] * inputs[1] - inputs[0] * inputs[0]) + (
        upper_b - lower_b
    ) * (inputs[1] - inputs[0])
