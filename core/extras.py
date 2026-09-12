"""
Funciones utilitarias adicionales para el método rectificado.
"""

from typing import Callable, Optional

import torch


def rescaled_reasoning(
    A: torch.Tensor,
    W: torch.Tensor,
    T: int,
    phi: float,
    transfer_function: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Implementa la regla de razonamiento sigmoid-rescaled para FCMs.

    Args:
        A: Matriz de estado inicial (n_samples, n_concepts).
        W: Matriz de pesos (n_concepts, n_concepts).
        T: Número de pasos de razonamiento.
        phi: Parámetro de mezcla.
        transfer_function: Función de transferencia.
        device: Dispositivo. Si es None, se usa el de `A`.

    Returns:
        torch.Tensor: Tensor de forma (n_samples, T+1, n_concepts) con todas las iteraciones.
    """
    n_samples, n_concepts = A.shape
    dtype = A.dtype
    if device is None:
        device = A.device

    H = torch.empty((n_samples, T + 1, n_concepts), device=device, dtype=dtype)
    H[:, 0] = A

    for t in range(1, T + 1):
        H[:, t] = phi * transfer_function(H[:, t - 1] @ W.to(dtype=dtype)) + (1 - phi) * H[:, 0]

    return H
