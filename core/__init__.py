"""
Core Package for Rectified Method - Fuzzy Cognitive Maps Verification

This package contains the core functionality for FCM verification using
the rectified limit state space method.

Submodules:
    - utils: Utility functions for rounding and data generation
    - metrics: Comparison and quality metrics
    - relaxation: Linear relaxation functions for activation functions
    - symbolic: Symbolic state space operations
    - methods: Main verification methods (arithmetic, symbolic, rectified)
"""

from core.metrics import (
    average_gap,
    convergence_iteration,
    covering,
    is_inside_state,
    is_inside_state_series,
    margin,
    series_intersection,
    series_is_nested,
    series_is_strictly_contracting,
    state_series_from_samples,
    states_intersection,
)
from core.methods import (
    arithmetic_iteration,
    interval_state_space,
    warm_start_symbolic_state_space,
    rectified_intersection_state_space,
    containment_guided_state_space,
    symbolic_state_space,
)
from core.extras import rescaled_reasoning
from core.relaxation import (
    apply_relaxed_activation,
    get_function_data,
    lower_relaxation,
    overapproximation_from_relaxations,
    upper_relaxation,
)
from core.symbolic import (
    concretize_state_space,
    induced_state_space,
    propagate_state_space,
    symbolic_iteration,
)
from core.utils import (
    downwards_rounding,
    generate_input_samples,
    generate_inputs,
    get_default_device,
    outwards_rounding,
    upwards_rounding,
)

__all__ = [
    # Utils
    "get_default_device",
    "generate_inputs",
    "generate_input_samples",
    "upwards_rounding",
    "downwards_rounding",
    "outwards_rounding",
    # Metrics
    "is_inside_state",
    "is_inside_state_series",
    "state_series_from_samples",
    "covering",
    "average_gap",
    "states_intersection",
    "series_intersection",
    "margin",
    "convergence_iteration",
    "series_is_nested",
    "series_is_strictly_contracting",
    # Relaxation
    "get_function_data",
    "lower_relaxation",
    "upper_relaxation",
    "apply_relaxed_activation",
    "overapproximation_from_relaxations",
    # Symbolic
    "induced_state_space",
    "concretize_state_space",
    "propagate_state_space",
    "symbolic_iteration",
    # Methods
    "arithmetic_iteration",
    "interval_state_space",
    "warm_start_symbolic_state_space",
    "containment_guided_state_space",
    "rectified_intersection_state_space",
    "symbolic_state_space",
    # Extras
    "rescaled_reasoning",
]

__version__ = "1.0.0"
