import torch
from typing import Callable, Literal


def machine_epsilon(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).eps


ActivationFunction = Literal["sigmoid", "tanh"]


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid function
    :param x:
    :return:
    """
    return 1 / (1 + torch.exp(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def quasi_nonlinear_sigmoid(phi: float):
    def _quasi_nonlinear_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return phi / (1 + torch.exp(-x))

    return _quasi_nonlinear_sigmoid


def quasi_nonlinear_sigmoid_derivative(phi: float):
    def _quasi_nonlinear_sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
        return phi * sigmoid(x) * (1 - sigmoid(x))

    return _quasi_nonlinear_sigmoid_derivative


def quasi_nonlinear_tanh(phi: float):
    def _quasi_nonlinear_tanh(x: torch.Tensor) -> torch.Tensor:
        return phi * (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    return _quasi_nonlinear_tanh


def quasi_nonlinear_tanh_derivative(phi: float):
    def _quasi_nonlinear_tanh_derivative(x: torch.Tensor) -> torch.Tensor:
        return phi * (1 - tanh(x) * tanh(x))

    return _quasi_nonlinear_tanh_derivative


def line_from_points_vectorized(p1: torch.Tensor, p2: torch.Tensor):
    """
    Vectorized version of line_from_points.
    p1, p2: tensors of shape (2, n_concepts)
        p1[0], p2[0] -> x coordinates
        p1[1], p2[1] -> y coordinates
    Returns:
        m, b tensors of shape (n_concepts,)
    """

    assert (
        p1.shape[1] == p2.shape[1]
    ), f"The length of the points arrays is  different: {p1.shape[1]} != {p2.shape[1]}"

    n_concepts = p1.shape[1]

    points_shape = (2, n_concepts)

    assert (
        p1.shape == points_shape
    ), f"p1 must have shape {points_shape} but has shape {p1.shape}"

    assert (
        p2.shape == points_shape
    ), f"p2 must have shape {points_shape} but has shape {p2.shape}"

    denom = p2[0] - p1[0]
    mask = torch.abs(denom) < machine_epsilon(denom.dtype)

    m = torch.where(mask, torch.zeros_like(denom), (p2[1] - p1[1]) / denom)
    b = torch.where(mask, p1[1], p1[1] - m * p1[0])

    return m, b


def tangent_to_function_vectorized(
    x: torch.Tensor, func: Callable, derivative: Callable
):
    m = derivative(x)
    b = func(x) - m * x

    return m, b


def lower_quasi_nonlinear_sigmoid_step(phi: float):
    def lower_sigmoid_step(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        denom = u - z
        m = torch.where(denom == 0, torch.zeros_like(denom), (sigmoid(u) - sigmoid(z)) / denom)
        m = torch.clamp(m, max=0.25)
        f = (1 - torch.sqrt(1 - 4 * m)) / 2
        z = torch.log(torch.clamp(f / (1 - f), min=torch.finfo(z.dtype).eps))

        return z

    return lower_sigmoid_step


def upper_quasi_nonlinear_sigmoid_step(phi: float):
    def upper_sigmoid_step(z: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        denom = z - l
        m = torch.where(denom == 0, torch.zeros_like(denom), (sigmoid(z) - sigmoid(l)) / denom)
        m = torch.clamp(m, max=0.25)
        f = (1 + torch.sqrt(1 - 4 * m)) / 2
        z = torch.log(torch.clamp(f / (1 - f), min=torch.finfo(z.dtype).eps))

        return z

    return upper_sigmoid_step


def lower_quasi_nonlinear_tanh_step(phi: float):
    def _lower_quasi_nonlinear_tanh_step(
        x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        denom = u - x
        m = torch.where(denom == 0, torch.zeros_like(denom), (tanh(u) - tanh(x)) / denom)
        m = torch.clamp(m, max=1.0)
        f = -torch.sqrt(1 - m)
        z = torch.log((1 + f) / (1 - f)) / 2

        return z

    return _lower_quasi_nonlinear_tanh_step


def upper_quasi_nonlinear_tanh_step(phi: float):
    def _upper_quasi_nonlinear_tanh_step(
        x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        denom = x - l
        m = torch.where(denom == 0, torch.zeros_like(denom), (tanh(x) - tanh(l)) / denom)
        m = torch.clamp(m, max=1.0)
        f = torch.sqrt(1 - m)
        z = torch.log((1 + f) / (1 - f)) / 2

        return z

    return _upper_quasi_nonlinear_tanh_step
