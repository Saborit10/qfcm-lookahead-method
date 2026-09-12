"""
Funciones utilitarias para generación de datos y redondeo direccional.

Estas funciones son fundamentales para la aritmética de intervalos,
garantizando la solidez de las operaciones mediante redondeo direccional.
"""

from typing import Any, Optional

import torch


def get_default_device() -> torch.device:
    """
    Devuelve el dispositivo por defecto: CUDA si está disponible, CPU en caso contrario.

    Returns:
        torch.device: Dispositivo por defecto.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_inputs(
    n_concepts: int,
    l: float,
    u: float,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Crea un tensor de intervalo inicial para todos los conceptos.

    Args:
        n_concepts: Número de conceptos en la FCM.
        l: Límite inferior del intervalo.
        u: Límite superior del intervalo.
        device: Dispositivo para el tensor (CPU/GPU). Si es None, se usa el
            dispositivo por defecto (`get_default_device`).

    Returns:
        torch.Tensor: Tensor de forma (2, n_concepts) donde:
            - Fila 0: límite inferior (l) para todos los conceptos
            - Fila 1: límite superior (u) para todos los conceptos
    """
    if device is None:
        device = get_default_device()

    return torch.stack(
        [
            torch.full((n_concepts,), l, device=device),
            torch.full((n_concepts,), u, device=device),
        ],
        dim=0,
    )


def generate_input_samples(
    n_samples: int,
    n_concepts: int,
    l: float,
    u: float,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Genera muestras aleatorias uniformemente distribuidas dentro de un intervalo.

    Args:
        n_samples: Número de muestras a generar.
        n_concepts: Número de conceptos por muestra.
        l: Límite inferior del intervalo.
        u: Límite superior del intervalo.
        device: Dispositivo para el tensor. Si es None, se usa
            `get_default_device`.

    Returns:
        torch.Tensor: Tensor de forma (n_samples, n_concepts) con valores
            aleatorios en [l, u].
    """
    if device is None:
        device = get_default_device()

    return torch.rand(n_samples, n_concepts, device=device) * (u - l) + l


def generate_random_matrix(
    n_concepts: int,
    density: float,
    device: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Genera una matriz aleatoria para un Mapa Cognitivo Difuso (FCM).

    Crea una matriz cuadrada donde los elementos no nulos se distribuyen
    aleatoriamente con la densidad especificada. La diagonal siempre es cero
    (sin auto-bucles) y la matriz garantiza que ningún nodo queda completamente
    desconectado (cada nodo tiene al menos una conexión entrante y una saliente).

    Args:
        n_concepts: Número de conceptos en la FCM. Debe ser al menos 2.
        density: Fracción de elementos no nulos de la matriz, excluyendo la
            diagonal. Debe estar entre 0 y 1.
        device: Dispositivo para el tensor (CPU/GPU). Si es None, se usa
            `get_default_device`.
        dtype: Tipo de dato para el tensor (por defecto `torch.get_default_dtype()`).

    Returns:
        torch.Tensor: Matriz cuadrada de forma (n_concepts, n_concepts) donde:
            - La diagonal siempre es 0
            - Los elementos fuera de la diagonal son 0 o un valor aleatorio en [-1, 1]
            - Ninguna fila ni columna es completamente cero

    Raises:
        AssertionError: Si density no está en [0, 1] o n_concepts < 2.
    """
    assert 0 <= density <= 1, "Density must be between 0 and 1"
    assert (
        isinstance(n_concepts, int) and n_concepts >= 2
    ), "We need at least two n_concepts"

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = get_default_device()

    matrix = torch.zeros(n_concepts, n_concepts, device=device, dtype=dtype)

    # Todos los pares dirigidos (i, j) con i != j, en orden aleatorio.
    undirected = torch.combinations(torch.arange(n_concepts, device=device), r=2)
    directed = torch.cat([undirected, undirected.flip(1)], dim=0)
    perm = torch.randperm(directed.shape[0], device=device)
    selected = directed[perm[: int(n_concepts * n_concepts * density)]]

    if selected.shape[0] > 0:
        matrix[selected[:, 0], selected[:, 1]] = (
            torch.rand(selected.shape[0], device=device, dtype=dtype) * 2 - 1
        )

    # Garantizar que ningún nodo quede completamente desconectado.
    for i in range(n_concepts):
        if torch.all(matrix[i, :] == 0) and torch.all(matrix[:, i] == 0):
            j = int(
                torch.randint(0, n_concepts - 1, (1,), device=device).item()
            )
            if j >= i:
                j += 1
            matrix[i, j] = torch.rand(1, device=device, dtype=dtype) * 2 - 1
            matrix[j, i] = torch.rand(1, device=device, dtype=dtype) * 2 - 1

    return matrix


def upwards_rounding(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica redondeo hacia arriba (hacia +infinito).

    Args:
        x: Tensor de entrada.

    Returns:
        torch.Tensor: Tensor con valores redondeados hacia +infinito.

    Note:
        Usa `torch.nextafter` para obtener el siguiente valor representable
        en punto flotante hacia +infinito, garantizando sobreaproximación.
    """
    return torch.nextafter(
        x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    )


def downwards_rounding(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica redondeo hacia abajo (hacia -infinito).

    Args:
        x: Tensor de entrada.

    Returns:
        torch.Tensor: Tensor con valores redondeados hacia -infinito.

    Note:
        Usa `torch.nextafter` para obtener el siguiente valor representable
        en punto flotante hacia -infinito, garantizando subaproximación.
    """
    return torch.nextafter(
        x, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    )


def outwards_rounding(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica redondeo hacia afuera en ambas direcciones de un intervalo.

    Args:
        x: Tensor de forma (2, ...) donde:
            - x[0]: límite inferior (se redondea hacia -infinito)
            - x[1]: límite superior (se redondea hacia +infinito)

    Returns:
        torch.Tensor: Tensor con redondeo direccional aplicado.

    Raises:
        AssertionError: Si la primera dimensión no tiene tamaño 2.

    Note:
        Esta función es fundamental para mantener la solidez en aritmética
        de intervalos, compensando errores de representación en punto flotante.
    """
    assert x.shape[0] == 2, "The first dimension of the tensor must have size 2"

    result = x.clone()
    result[0] = torch.nextafter(
        result[0], torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    )
    result[1] = torch.nextafter(
        result[1], torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    )

    return result
