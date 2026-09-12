"""
Compara iterative_periodic_reset_state_space vs rectified_intersection_state_space.

La métrica de comparación es el covering (menor = intervalos más ajustados).
Se barre un grid de configuración: n_concepts, density, phi y activación.

Uso:
    python compare_iterative_vs_rectified.py
    python compare_iterative_vs_rectified.py --n-concepts 6 12 20 --density 0.3 0.7
"""

import argparse
import csv
import itertools
import time

import numpy as np
import torch
from tqdm import tqdm

from core.extras import rescaled_reasoning
from core.methods import (
    interval_state_space,
    rectified_intersection_state_space,
)
from core.metrics import covering, is_inside_state_series, state_series_from_samples
from core.utils import generate_input_samples, generate_inputs, generate_random_matrix
from experimental.methods import (
    containment_guided_reset_state_space,
    iterative_periodic_reset_state_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-concepts", type=int, nargs="+", default=[6, 12, 20])
    parser.add_argument("--density", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--phi", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    parser.add_argument(
        "--activation", choices=["sigmoid"], nargs="+", default=["sigmoid"]
    )
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--l-0", type=float, default=0.0)
    parser.add_argument("--u-0", type=float, default=1.0)
    parser.add_argument("--minimal-warming-iters", type=int, default=5)
    parser.add_argument("--arithmetic-warming-iters", type=int, default=5)
    parser.add_argument("--reset-period", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--csv", type=str, default=None, help="Ruta del .csv de salida")
    return parser.parse_args()


def outside_margin(
    sampled: torch.Tensor, interval: torch.Tensor
) -> tuple[float, float]:
    """
    Cuánto exceden las muestras los límites del intervalo.

    sampled: serie de muestras (n_iters, 2, n_concepts) con [0]=min, [1]=max.
    interval: serie de intervalos (n_iters, 2, n_concepts).

    Returns:
        (violación media, violación máxima) sobre conceptos e iteraciones.
        0.0 significa que todo está dentro (aunque sea por poca diferencia).
    """
    lower_violation = torch.clamp(interval[:, 0] - sampled[:, 0], min=0)
    upper_violation = torch.clamp(sampled[:, 1] - interval[:, 1], min=0)
    violation = torch.maximum(lower_violation, upper_violation)
    return float(violation.mean().item()), float(violation.max().item())


def run_configuration(
    args: argparse.Namespace,
    seed: int,
    n_concepts: int,
    density: float,
    phi: float,
    activation: str,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    W = generate_random_matrix(n_concepts, density, device=args.device, dtype=args.dtype)
    Wt = W.T
    inputs = generate_inputs(n_concepts, args.l_0, args.u_0, device=args.device).to(
        dtype=args.dtype
    )

    arithmetic_states = interval_state_space(
        Wt,
        inputs,
        max_iters=args.max_iters,
        phi=phi,
        activation_function=activation,
        device=args.device,
    )

    A = generate_input_samples(
        args.n_samples, n_concepts, args.l_0, args.u_0, device=args.device
    )
    simulated_states = rescaled_reasoning(
        A,
        W,
        T=args.max_iters,
        phi=phi,
        transfer_function=torch.sigmoid if activation == "sigmoid" else torch.tanh,
        device=args.device,
    )
    sampled_states = state_series_from_samples(simulated_states)

    t0 = time.perf_counter()
    iterative_states = iterative_periodic_reset_state_space(
        Wt,
        inputs,
        activation,
        phi=phi,
        max_iters=args.max_iters,
        arithmetic_warming_iters=args.arithmetic_warming_iters,
        baseline_interval_series=arithmetic_states,
        device=args.device,
    )
    iterative_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rectified_states = rectified_intersection_state_space(
        Wt,
        inputs,
        activation,
        phi,
        args.max_iters,
        minimal_warming_iters=args.minimal_warming_iters,
        baseline_interval_series=arithmetic_states,
    )
    rectified_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reset_states = containment_guided_reset_state_space(
        Wt,
        inputs,
        activation,
        phi,
        reset_period=args.reset_period,
        max_iters=args.max_iters,
        minimal_symbolic_warming_iters=args.minimal_warming_iters,
        baseline_interval_series=arithmetic_states,
        device=args.device,
    )
    reset_time = time.perf_counter() - t0

    return {
        "n_concepts": n_concepts,
        "density": density,
        "phi": phi,
        "activation": activation,
        "seed": seed,
        "arithmetic_covering": covering(arithmetic_states),
        "iterative_covering": covering(iterative_states),
        "rectified_covering": covering(rectified_states),
        "reset_covering": covering(reset_states),
        "iterative_sound": is_inside_state_series(sampled_states, iterative_states),
        "rectified_sound": is_inside_state_series(sampled_states, rectified_states),
        "reset_sound": is_inside_state_series(sampled_states, reset_states),
        "iterative_violation_mean": outside_margin(sampled_states, iterative_states)[0],
        "iterative_violation_max": outside_margin(sampled_states, iterative_states)[1],
        "rectified_violation_mean": outside_margin(sampled_states, rectified_states)[0],
        "rectified_violation_max": outside_margin(sampled_states, rectified_states)[1],
        "reset_violation_mean": outside_margin(sampled_states, reset_states)[0],
        "reset_violation_max": outside_margin(sampled_states, reset_states)[1],
        "iterative_time_s": iterative_time,
        "rectified_time_s": rectified_time,
        "reset_time_s": reset_time,
    }


def write_csv(rows: list[dict], path: str) -> None:
    """
    Escribe las filas de resultados en un archivo CSV.

    Args:
        rows: Lista de dicts (uno por configuración).
        path: Ruta del archivo de salida.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados guardados en: {path}")


def main() -> None:
    args = parse_args()
    args.dtype = torch.float64
    torch.set_grad_enabled(False)
    torch.set_default_dtype(args.dtype)

    print(f"Device: {args.device}")
    print(
        f"Grid: {len(args.n_concepts)}x{len(args.density)}x{len(args.phi)}x{len(args.activation)} "
        f"configuraciones x {len(args.seeds)} seeds\n"
    )

    combos = list(
        itertools.product(args.n_concepts, args.density, args.phi, args.activation, args.seeds)
    )

    rows = []
    iterative_wins = 0
    rectified_wins = 0
    reset_wins = 0
    soundness_issues = []

    for n_concepts, density, phi, activation, seed in tqdm(
        combos, desc="Comparando métodos", unit="config"
    ):
        result = run_configuration(args, seed, n_concepts, density, phi, activation)

        if result["iterative_covering"] < result["rectified_covering"]:
            iterative_wins += 1
        else:
            rectified_wins += 1

        if result["reset_covering"] < result["rectified_covering"]:
            reset_wins += 1

        for method, prefix in [
            ("iterative", "iterative_"),
            ("rectified", "rectified_"),
            ("reset", "reset_"),
        ]:
            if not result[f"{prefix}sound"]:
                soundness_issues.append(
                    (
                        result,
                        method,
                        result[f"{prefix}violation_mean"],
                        result[f"{prefix}violation_max"],
                    )
                )

        rows.append(result)

    header = (
        f"{'n':>4} {'density':>7} {'phi':>4} {'act':>7} {'seed':>4} | "
        f"{'arith':>8} {'iterat':>8} {'rectif':>8} {'reset':>8} | {'win':>6} | "
        f"{'iterat_t':>9} {'rectif_t':>9} {'reset_t':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in rows:
        winner = "iterat" if result["iterative_covering"] < result["rectified_covering"] else "rectif"
        if result["reset_covering"] < result["rectified_covering"]:
            winner = "reset"
        print(
            f"{result['n_concepts']:>4} {result['density']:>7.1f} {result['phi']:>4.1f} "
            f"{result['activation']:>7} {result['seed']:>4} | "
            f"{result['arithmetic_covering']:>8.3f} {result['iterative_covering']:>8.3f} "
            f"{result['rectified_covering']:>8.3f} {result['reset_covering']:>8.3f} | "
            f"{winner:>6} | "
            f"{result['iterative_time_s']:>9.2f} {result['rectified_time_s']:>9.2f} "
            f"{result['reset_time_s']:>9.2f}"
        )
    print("-" * len(header))
    total = len(rows)
    print(f"Victorias iterative: {iterative_wins}/{total}")
    print(f"Victorias rectified: {rectified_wins}/{total}")
    print(f"Victorias reset:     {reset_wins}/{total}")

    mean_iterative = np.mean([r["iterative_covering"] for r in rows])
    mean_rectified = np.mean([r["rectified_covering"] for r in rows])
    mean_reset = np.mean([r["reset_covering"] for r in rows])
    mean_arithmetic = np.mean([r["arithmetic_covering"] for r in rows])
    print(
        f"Covering medio — aritmético: {mean_arithmetic:.3f}, iterative: {mean_iterative:.3f}, "
        f"rectified: {mean_rectified:.3f}, reset: {mean_reset:.3f}"
    )
    print(
        f"Mejora media iterative vs rectified: "
        f"{(mean_iterative - mean_rectified) / mean_rectified * 100:+.1f}%"
    )
    print(
        f"Mejora media reset vs rectified: "
        f"{(mean_reset - mean_rectified) / mean_rectified * 100:+.1f}%"
    )

    if args.csv is not None:
        write_csv(rows, args.csv)

    if soundness_issues:
        print("\nAdvertencia de sonido (muestras fuera del intervalo):")
        print(
            f"{'n':>4} {'density':>7} {'phi':>4} {'seed':>4} {'metodo':>9} | "
            f"{'viol. media':>11} {'viol. max':>10}"
        )
        for result, method, mean_viol, max_viol in soundness_issues:
            print(
                f"{result['n_concepts']:>4} {result['density']:>7.1f} {result['phi']:>4.1f} "
                f"{result['seed']:>4} {method:>9} | {mean_viol:>11.2e} {max_viol:>10.2e}"
            )
        max_violation = max(issue[3] for issue in soundness_issues)
        if max_violation < 1e-12:
            print(
                "\nLa violación es del orden de la precisión de máquina "
                "(< 1e-12): probablemente acumulación de errores de redondeo, no un bug."
            )
        else:
            print(
                f"\nLa violación máxima es {max_violation:.2e}: revisar el código "
                "(posible bug, no es solo redondeo)."
            )
    else:
        print("\nSonido: todas las series contienen las muestras simuladas.")


if __name__ == "__main__":
    main()
