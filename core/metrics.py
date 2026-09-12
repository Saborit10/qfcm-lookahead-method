"""
Funciones de comparación y métricas de calidad para verificación de FCMs.

Incluye funciones para:
- Verificar contención entre estados y series temporales
- Calcular covering (cobertura) de intervalos
- Calcular gaps entre predicciones y simulaciones
- Calcular intersecciones de estados y series
"""

from typing import Any, Optional

import torch


def is_inside_state(
    state_1: torch.Tensor, state_2: torch.Tensor, tolerance: float = 0.0
) -> bool:
    """
    Verifica si el estado state_1 está completamente contenido en state_2.

    Args:
        state_1: Tensor de forma (2, n_concepts) representando el estado a verificar.
        state_2: Tensor de forma (2, n_concepts) representando el estado contenedor.
        tolerance: Tolerancia numérica para la comparación (default: 0.0).

    Returns:
        bool: True si state_1 ⊆ state_2, False en caso contrario.

    Raises:
        AssertionError: Si los estados tienen diferente número de conceptos.
    """
    assert (
        state_1.shape[1] == state_2.shape[1]
    ), f"State series has different number of concepts ({state_1.shape[1]} != {state_2.shape[1]})"

    is_contained = (state_2[0, :] - tolerance <= state_1[0, :]) & (
        state_1[1, :] <= state_2[1, :] + tolerance
    )

    return bool(is_contained.all().item())


def is_inside_state_series(
    state_series_1: torch.Tensor,
    state_series_2: torch.Tensor,
    tolerance: float = 0.0,
) -> bool:
    """
    Verifica si una serie temporal de estados state_series_1 está contenida en state_series_2.

    Args:
        state_series_1: Tensor de forma (n_iters, 2, n_concepts).
        state_series_2: Tensor de forma (n_iters, 2, n_concepts).
        tolerance: Tolerancia numérica para la comparación.

    Returns:
        bool: True si state_series_1 ⊆ state_series_2 para todas las iteraciones.

    Raises:
        AssertionError: Si las series tienen diferente longitud o número de conceptos.
    """
    assert (
        state_series_1.shape[0] == state_series_2.shape[0]
    ), f"State series has different lengths ({state_series_1.shape[0]} != {state_series_2.shape[0]})"

    assert (
        state_series_1.shape[2] == state_series_2.shape[2]
    ), f"State series has different number of concepts ({state_series_1.shape[2]} != {state_series_2.shape[2]})"

    is_contained = (state_series_2[:, 0, :] - tolerance <= state_series_1[:, 0, :]) & (
        state_series_1[:, 1, :] <= state_series_2[:, 1, :] + tolerance
    )

    return bool(is_contained.all().item())


def state_series_from_samples(
    sampled_state_series: torch.Tensor, device: Optional[Any] = None
) -> torch.Tensor:
    """
    Construye una serie temporal de estados a partir de múltiples muestras simuladas.

    Args:
        sampled_state_series: Tensor de forma (n_samples, n_iters, n_concepts).
        device: Dispositivo para el tensor resultante. Si es None, se usa el de
            `sampled_state_series`.

    Returns:
        torch.Tensor: Tensor de forma (n_iters, 2, n_concepts) que envuelve
            todas las trayectorias muestrales (mínimo y máximo por iteración).
    """
    if device is None:
        device = sampled_state_series.device

    min_values = torch.min(sampled_state_series, dim=0).values.to(device=device)
    max_values = torch.max(sampled_state_series, dim=0).values.to(device=device)

    return torch.stack([min_values, max_values], dim=1)


def covering(state_series: torch.Tensor, iteration: int = -1) -> float:
    """
    Calcula el covering (cobertura) del estado en una iteración dada.

    El covering mide el cambio promedio en el tamaño de los intervalos respecto
    al estado inicial.

    Args:
        state_series: Tensor de forma (n_iters, 2, n_concepts).
        iteration: Índice de iteración para calcular el covering (default: -1, última).

    Returns:
        float: Valor de covering promedio sobre todos los conceptos.
            - covering = 1.0: Intervalos mantienen tamaño original
            - covering < 1.0: Intervalos se redujeron (mejor precisión)
            - covering > 1.0: Intervalos se expandieron (explosión de intervalos)
    """
    n_concepts = state_series[iteration].shape[1]

    covering_value = (
        torch.sum(
            (state_series[iteration, 1, :] - state_series[iteration, 0, :])
            / (state_series[0, 1, :] - state_series[0, 0, :])
        ).item()
        / n_concepts
    )

    return covering_value


def average_gap(
    state_series: torch.Tensor,
    simulations: torch.Tensor,
    iteration: int = -1,
) -> float:
    """
    Calcula el gap promedio entre los intervalos predichos y las simulaciones concretas.

    Args:
        state_series: Tensor de forma (n_iters, 2, n_concepts) con los intervalos predichos.
        simulations: Tensor de forma (n_samples, n_iters, n_concepts) con las simulaciones.
        iteration: Índice de iteración para calcular el gap (default: -1).

    Returns:
        float: Gap promedio entre los límites predichos y los valores simulados.
    """
    min_neuron_pred = state_series[iteration][0]
    max_neuron_pred = state_series[iteration][1]

    min_neuron_val, _ = simulations[:, iteration, :].min(dim=0)
    max_neuron_val, _ = simulations[:, iteration, :].max(dim=0)

    avg_gap = (min_neuron_val - min_neuron_pred + max_neuron_pred - max_neuron_val) / 2

    return torch.mean(avg_gap).item()


def states_intersection(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """
    Calcula la intersección de dos estados (intervalos).

    Args:
        state1: Tensor de forma (2, n_concepts).
        state2: Tensor de forma (2, n_concepts).

    Returns:
        torch.Tensor: Intersección de los estados con redondeo hacia afuera.
    """
    from core.utils import outwards_rounding

    return outwards_rounding(
        torch.stack(
            [
                torch.maximum(state1[0], state2[0]),
                torch.minimum(state1[1], state2[1]),
            ],
            dim=0,
        )
    )


def series_intersection(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """
    Calcula la intersección de dos series temporales de estados.

    Args:
        state1: Tensor de forma (..., n_iters, 2, n_concepts).
        state2: Tensor de forma (..., n_iters, 2, n_concepts).

    Returns:
        torch.Tensor: Intersección de las series con redondeo direccional.
    """
    from core.utils import downwards_rounding, upwards_rounding

    return torch.stack(
        [
            downwards_rounding(torch.maximum(state1[..., 0, :], state2[..., 0, :])),
            upwards_rounding(torch.minimum(state1[..., 1, :], state2[..., 1, :])),
        ],
        dim=-2,
    )


def state_intersection(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """
    Calcula la intersección de dos estados.

    Args:
        state1: Tensor de forma (2, n_concepts).
        state2: Tensor de forma (2, n_concepts).

    Returns:
        torch.Tensor: Intersección de llos estados con redondeo direccional.
    """
    from core.utils import downwards_rounding, upwards_rounding

    return torch.stack(
        [
            downwards_rounding(torch.maximum(state1[0, :], state2[0, :])),
            upwards_rounding(torch.minimum(state1[1, :], state2[1, :])),
        ],
        dim=-2,
    )


def margin(state_series_1: torch.Tensor, state_series_2: torch.Tensor) -> float:
    """
    Calcula el margen mínimo entre dos series de estados.

    Args:
        state_series_1: Tensor de forma (n_iters, 2, n_concepts).
        state_series_2: Tensor de forma (n_iters, 2, n_concepts).

    Returns:
        float: Margen mínimo (positivo si state_series_1 ⊆ state_series_2).
    """
    assert (
        state_series_1.shape[0] == state_series_2.shape[0]
    ), f"State series has different lengths ({state_series_1.shape[0]} != {state_series_2.shape[0]})"

    assert (
        state_series_1.shape[2] == state_series_2.shape[2]
    ), f"State series has different number of concepts ({state_series_1.shape[2]} != {state_series_2.shape[2]})"

    lo = state_series_1[:, 0, :] - state_series_2[:, 0, :]
    up = state_series_2[:, 1, :] - state_series_1[:, 1, :]

    return min(torch.min(lo).item(), torch.min(up).item())


def convergence_iteration(state_series: torch.Tensor, epsilon: float) -> Optional[int]:
    """
    Encuentra la primera iteración donde el estado converge (width < epsilon).

    Args:
        state_series: Tensor de forma (n_iters, 2, n_concepts).
        epsilon: Umbral de convergencia.

    Returns:
        Optional[int]: Índice de la primera iteración convergente, o None si no converge.
    """
    cond = (state_series[:, 1, :] - state_series[:, 0, :] < epsilon).all(dim=1)

    if cond.any():
        return int(torch.nonzero(cond, as_tuple=True)[0][0].item())
    else:
        return None


def series_is_nested(state_series: torch.Tensor, tolerance: float = 0.0) -> bool:
    """
    Verifica si una serie de estados es anidada: cada estado está contenido en el anterior.

    Args:
        state_series: Tensor de forma (n_iters, 2, n_concepts).
        tolerance: Tolerancia numérica para la comparación (default: 0.0).

    Returns:
        bool: True si state_series[i] ⊆ state_series[i-1] para todo i >= 1.
            Una serie de una sola iteración es trivialmente anidada.
    """
    for i in range(1, state_series.shape[0]):
        if not is_inside_state(state_series[i], state_series[i - 1], tolerance=tolerance):
            return False

    return True


def series_is_strictly_contracting(
    state_series: torch.Tensor, epsilon: float = 0.0
) -> bool:
    """
    Verifica si una serie es estrictamente contrayente: la anchura del intervalo
    disminuye en cada iteración.

    Args:
        state_series: Tensor de forma (n_iters, 2, n_concepts).
        epsilon: Disminución mínima exigida por iteración (default: 0.0).

    Returns:
        bool: True si width[i] < width[i-1] - epsilon para todo i >= 1.
            False si la serie tiene menos de dos iteraciones.
    """
    if state_series.shape[0] < 2:
        return False

    widths = (state_series[:, 1, :] - state_series[:, 0, :]).sum(dim=1)
    for i in range(1, widths.shape[0]):
        if not bool((widths[i] < widths[i - 1] - epsilon).item()):
            return False

    return True
