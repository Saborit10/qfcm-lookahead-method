from typing import Any, Optional
import torch

from core.methods import arithmetic_iteration
from core.metrics import is_inside_state, state_intersection, states_intersection
from core.symbolic import (
    concretize_state_space,
    induced_state_space,
    symbolic_iteration,
)
from core.utils import outwards_rounding


def periodic_reset_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    t: int = -1,
    max_iters: int = 20,
    arithmetic_warming_iters: int = 0,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta un método con reset fijo periódico y opcionalmente iteraciones aritméticas iniciales.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        t: Período de reset fijo.
        max_iters: Número máximo de iteraciones.
        arithmetic_warming_iters: Número de iteraciones aritméticas iniciales antes del método simbólico.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        device: Dispositivo.

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

    if device is None:
        device = inputs.device

    W_pos = torch.clamp(W, min=0).to(dtype=dtype)
    W_neg = torch.clamp(W, max=0).to(dtype=dtype)

    all_states = torch.empty(
        (max_iters + 1, 2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
    )
    concrete_states = torch.empty(
        (max_iters + 1, 2, n_concepts), device=device, dtype=dtype
    )

    all_states[0] = induced_state_space(n_concepts, device=device)
    concrete_states[0] = inputs

    # Fase de aritmético warming (iteraciones aritméticas iniciales)
    for i in range(1, min(arithmetic_warming_iters + 1, max_iters + 1)):
        concrete_states[i] = arithmetic_iteration(
            W_pos, W_neg, concrete_states[i - 1], activation_function, phi, inputs
        )
        if baseline_interval_series is not None:
            concrete_states[i] = state_intersection(
                concrete_states[i], baseline_interval_series[i]
            )
        all_states[i] = induced_state_space(n_concepts, device=device)

    index = arithmetic_warming_iters + 1
    current_inputs_index = arithmetic_warming_iters

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

        if baseline_interval_series is not None:
            concrete_state = state_intersection(
                concrete_state, baseline_interval_series[index]
            )

        init_iter = max(0, index - 10)
        cs = concrete_states[init_iter]
        for i in range(init_iter + 1, index + 1):
            cs = arithmetic_iteration(
                W_pos,
                W_neg,
                cs,
                activation_function,
                phi,
                inputs,
            )

        concrete_state = state_intersection(concrete_state, cs)

        concrete_states[index] = concrete_state
        all_states[index] = state

        if t > 0 and index % t == 0:
            all_states[index] = induced_state_space(n_concepts, device=device)
            current_inputs_index = index

        index += 1

    return concrete_states


def iterative_periodic_reset_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    max_iters: int = 20,
    arithmetic_warming_iters: int = 0,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ejecuta múltiples pasadas de periodic_reset_state_space con diferentes períodos t.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        max_iters: Número máximo de iteraciones.
        arithmetic_warming_iters: Número de iteraciones aritméticas iniciales.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        device: Dispositivo.

    Returns:
        torch.Tensor: Serie temporal de estados concretizados.
    """

    n_concepts = W.shape[0]
    dtype = inputs.dtype

    if device is None:
        device = inputs.device

    if baseline_interval_series is not None:
        states = baseline_interval_series
    else:
        states = torch.empty((max_iters + 1, 2, n_concepts), device=device, dtype=dtype)
        states[:, 0, :] = -torch.inf
        states[:, 1, :] = torch.inf

    for i in [-1] + list(range(1, 16)):
        states = periodic_reset_state_space(
            W,
            inputs,
            activation_function,
            phi,
            t=i,
            max_iters=max_iters,
            arithmetic_warming_iters=arithmetic_warming_iters,
            baseline_interval_series=states,
            device=device,
        )

    return states


# def containment_guided_state_space(
#     W: torch.Tensor,
#     inputs: torch.Tensor,
#     activation_function: str,
#     phi: float,
#     minimal_symbolic_warming_iters: int = 5,
#     arithmetic_rectifications: int = 5,
#     max_iters: int = 20,
#     baseline_interval_series: Optional[torch.Tensor] = None,
#     device: Optional[Any] = None,
# ) -> torch.Tensor:
#     n_concepts = W.shape[0]
#     w_shape = (n_concepts, n_concepts)
#     inputs_shape = (2, n_concepts)
#     dtype = inputs.dtype

#     W_pos = torch.clamp(W, min=0).to(dtype=dtype)
#     W_neg = torch.clamp(W, max=0).to(dtype=dtype)

#     assert W.shape == w_shape, f"W must have shape {w_shape}, but has shape {W.shape}"
#     assert (
#         inputs.shape == inputs_shape
#     ), f"inputs must have shape {inputs_shape} but has shape {inputs.shape}"

#     if baseline_interval_series is not None:
#         assert baseline_interval_series.shape[1:] == (2, n_concepts)

#     all_states = torch.empty(
#         (max_iters + 1, 2, n_concepts, 2 * n_concepts + 1), device=device, dtype=dtype
#     )
#     concrete_states = torch.empty(
#         (max_iters + 1, 2, n_concepts), device=device, dtype=dtype
#     )

#     all_states[0] = induced_state_space(n_concepts, device=device)
#     concrete_states[0] = inputs

#     index = 1
#     current_inputs_index = 0
#     pending_symbolic_iters = minimal_symbolic_warming_iters

#     last_type = None

#     while index <= max_iters:
#         state = symbolic_iteration(
#             all_states[index - 1],
#             W,
#             concrete_states[current_inputs_index],
#             phi,
#             activation_function=activation_function,
#             original_inputs=inputs,
#             device=device,
#         )
#         concrete_state = concretize_state_space(
#             torch.unsqueeze(state, dim=0),
#             concrete_states[current_inputs_index],
#             device=device,
#             original_inputs=inputs,
#         ).squeeze(dim=0)
#         concrete_state = outwards_rounding(concrete_state)

#         concrete_states[index] = arithmetic_iteration(
#             W_pos, W_neg, concrete_states[index - 1], activation_function, phi, inputs
#         )

#         if pending_symbolic_iters > 0 or is_inside_state(
#             concrete_state, concrete_states[index]
#         ):
#             if last_type != "symbolic":
#                 print(f"SYMBOLIC at {index}")
#             last_type = "symbolic"

#             all_states[index] = state
#             concrete_states[index] = states_intersection(
#                 concrete_states[index], concrete_state
#             )

#             if baseline_interval_series is not None:
#                 concrete_states[index] = states_intersection(
#                     concrete_states[index], baseline_interval_series[index]
#                 )

#             index += 1
#             pending_symbolic_iters -= 1

#         else:
#             if last_type != "arithmetic":
#                 print(f"ARITHMETIC at {index}")
#             last_type = "arithmetic"

#             for _ in range(min(arithmetic_rectifications, max_iters - index + 1)):
#                 all_states[index] = induced_state_space(n_concepts, device=device)
#                 concrete_states[index] = arithmetic_iteration(
#                     W_pos,
#                     W_neg,
#                     concrete_states[index - 1],
#                     activation_function,
#                     phi,
#                     inputs,
#                 )
#                 if baseline_interval_series is not None:
#                     concrete_states[index] = states_intersection(
#                         concrete_states[index], baseline_interval_series[index]
#                     )
#                 current_inputs_index = index
#                 index += 1

#             pending_symbolic_iters = minimal_symbolic_warming_iters

#     return concrete_states


def containment_guided_reset_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    minimal_symbolic_warming_iters: int = 5,
    arithmetic_rectifications: int = 5,
    reset_period: int = -1,
    max_iters: int = 20,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """
    Método guiado por contención con reinicio simbólico periódico.

    Combina la decisión por contención de `containment_guided_state_space`
    con el reinicio periódico de `periodic_reset_state_space`: en la rama
    simbólica, cada `reset_period` pasos el estado simbólico se re-ancla al
    intervalo concreto vigente (induced_state_space + current_inputs_index),
    limitando la acumulación de sobreaproximación simbólica.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        minimal_symbolic_warming_iters: Iteraciones simbólicas mínimas iniciales.
        arithmetic_rectifications: Iteraciones de rectificación aritmética.
        reset_period: Período de reinicio simbólico (<= 0 desactiva).
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

            if reset_period > 0 and index % reset_period == 0:
                all_states[index] = induced_state_space(n_concepts, device=device)
                current_inputs_index = index

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


def lookahead_refined_state_space(
    W: torch.Tensor,
    inputs: torch.Tensor,
    activation_function: str,
    phi: float,
    k: Optional[int] = None,
    max_iters: int = 20,
    baseline_interval_series: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """Ejecuta lookahead refinando la cadena aritmética hacia delante.

    Este método es equivalente a :func:`lookahead_state_space`, pero
    intersecta ``arithmetic_fwd`` con ``sym_fwd_concrete`` en cada paso antes
    de propagarlo a la siguiente iteración.

    Args:
        W: Matriz de pesos (n_concepts, n_concepts).
        inputs: Intervalos iniciales (2, n_concepts).
        activation_function: Función de activación ("sigmoid" o "tanh").
        phi: Parámetro de mezcla.
        k: Tamaño de la ventana. Si es None, se usa max_iters.
        max_iters: Número máximo de iteraciones.
        baseline_interval_series: Serie de intervalos de referencia opcional.
        device: Dispositivo.

    Returns:
        Serie temporal de estados concretizados.
    """
    return lookahead_state_space(
        W,
        inputs,
        activation_function,
        phi,
        k=k,
        max_iters=max_iters,
        baseline_interval_series=baseline_interval_series,
        device=device,
        refine_forward_arithmetic=True,
    )
