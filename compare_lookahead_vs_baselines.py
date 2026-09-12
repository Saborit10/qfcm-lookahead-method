"""
Compara lookahead_state_space contra baselines: arithmetic, containment y rectified.

La métrica de comparación es el covering (menor = intervalos más ajustados).
Se barre un grid de configuración: n_concepts, density, phi y activación.

Uso:
    python compare_lookahead_vs_baselines.py
    python compare_lookahead_vs_baselines.py --n-concepts 6 12 --density 0.5 0.7
    python compare_lookahead_vs_baselines.py --lookahead-k 1 3 5
"""

import argparse
import csv
import itertools
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from core.extras import rescaled_reasoning
from core.methods import (
    containment_guided_state_space,
    interval_state_space,
    rectified_intersection_state_space,
)
from core.metrics import covering, is_inside_state_series, state_series_from_samples
from core.utils import generate_input_samples, generate_inputs, generate_random_matrix
from experimental.methods import lookahead_state_space


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
    parser.add_argument("--lookahead-k", type=int, nargs="+", default=[5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--csv", type=str, default=None, help="Ruta del .csv de salida")
    parser.add_argument(
        "--maps-json", type=str, default="maps.json", help="Ruta del .json con los mapas"
    )
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
    lookahead_k: int,
) -> tuple[dict, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    map_id = f"n{n_concepts}_d{density:g}_s{seed}"

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
    containment_states = containment_guided_state_space(
        Wt,
        inputs,
        activation,
        phi,
        minimal_symbolic_warming_iters=args.minimal_warming_iters,
        max_iters=args.max_iters,
        baseline_interval_series=arithmetic_states,
        device=args.device,
    )
    containment_time = time.perf_counter() - t0

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
    lookahead_states = lookahead_state_space(
        Wt,
        inputs,
        activation,
        phi,
        k=lookahead_k,
        max_iters=args.max_iters,
        baseline_interval_series=arithmetic_states,
        device=args.device,
    )
    lookahead_time = time.perf_counter() - t0

    map_data = {
        "map_id": map_id,
        "n_concepts": n_concepts,
        "density": density,
        "seed": seed,
        "W": W.tolist(),
        "inputs": inputs.tolist(),
    }

    return {
        "map_id": map_id,
        "n_concepts": n_concepts,
        "density": density,
        "phi": phi,
        "activation": activation,
        "seed": seed,
        "lookahead_k": lookahead_k,
        "arithmetic_covering": covering(arithmetic_states),
        "containment_covering": covering(containment_states),
        "rectified_covering": covering(rectified_states),
        "lookahead_covering": covering(lookahead_states),
        "containment_sound": is_inside_state_series(sampled_states, containment_states),
        "rectified_sound": is_inside_state_series(sampled_states, rectified_states),
        "lookahead_sound": is_inside_state_series(sampled_states, lookahead_states),
        "containment_violation_mean": outside_margin(
            sampled_states, containment_states
        )[0],
        "containment_violation_max": outside_margin(sampled_states, containment_states)[
            1
        ],
        "rectified_violation_mean": outside_margin(sampled_states, rectified_states)[0],
        "rectified_violation_max": outside_margin(sampled_states, rectified_states)[1],
        "lookahead_violation_mean": outside_margin(sampled_states, lookahead_states)[0],
        "lookahead_violation_max": outside_margin(sampled_states, lookahead_states)[1],
        "containment_time_s": containment_time,
        "rectified_time_s": rectified_time,
        "lookahead_time_s": lookahead_time,
    }, map_data


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
        f"configuraciones x {len(args.seeds)} seeds x {len(args.lookahead_k)} k\n"
    )

    combos = list(
        itertools.product(
            args.n_concepts, args.density, args.phi, args.activation, args.seeds, args.lookahead_k
        )
    )

    rows = []
    maps: dict[str, dict] = {}
    containment_wins = 0
    rectified_wins = 0
    lookahead_vs_rectified_wins = 0
    lookahead_vs_containment_wins = 0
    soundness_issues = []

    for n_concepts, density, phi, activation, seed, lookahead_k in tqdm(
        combos, desc="Comparando métodos", unit="config"
    ):
        result, map_data = run_configuration(
            args, seed, n_concepts, density, phi, activation, lookahead_k
        )

        maps.setdefault(map_data["map_id"], map_data)

        if result["containment_covering"] < result["rectified_covering"]:
            containment_wins += 1
        else:
            rectified_wins += 1

        if result["lookahead_covering"] < result["rectified_covering"]:
            lookahead_vs_rectified_wins += 1
        if result["lookahead_covering"] < result["containment_covering"]:
            lookahead_vs_containment_wins += 1

        for method, prefix in [
            ("containment", "containment_"),
            ("rectified", "rectified_"),
            ("lookahead", "lookahead_"),
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
        f"{'map_id':>16} {'n':>4} {'density':>7} {'phi':>4} {'act':>7} {'seed':>4} {'k':>2} | "
        f"{'arith':>8} {'contain':>8} {'rectif':>8} {'lookahd':>8} | "
        f"{'contain_t':>9} {'rectif_t':>9} {'lookahd_t':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in rows:
        print(
            f"{result['map_id']:>16} {result['n_concepts']:>4} {result['density']:>7.1f} "
            f"{result['phi']:>4.1f} "
            f"{result['activation']:>7} {result['seed']:>4} {result['lookahead_k']:>2} | "
            f"{result['arithmetic_covering']:>8.3f} {result['containment_covering']:>8.3f} "
            f"{result['rectified_covering']:>8.3f} {result['lookahead_covering']:>8.3f} | "
            f"{result['containment_time_s']:>9.2f} {result['rectified_time_s']:>9.2f} "
            f"{result['lookahead_time_s']:>9.2f}"
        )
    print("-" * len(header))
    total = len(rows)
    print(f"Victorias containment: {containment_wins}/{total}")
    print(f"Victorias rectified:   {rectified_wins}/{total}")
    print(f"Victorias lookahead vs rectified:    {lookahead_vs_rectified_wins}/{total}")
    print(f"Victorias lookahead vs containment:  {lookahead_vs_containment_wins}/{total}")

    mean_arithmetic = np.mean([r["arithmetic_covering"] for r in rows])
    mean_containment = np.mean([r["containment_covering"] for r in rows])
    mean_rectified = np.mean([r["rectified_covering"] for r in rows])
    mean_lookahead = np.mean([r["lookahead_covering"] for r in rows])
    print(
        f"Covering medio — aritmético: {mean_arithmetic:.3f}, containment: {mean_containment:.3f}, "
        f"rectified: {mean_rectified:.3f}, lookahead: {mean_lookahead:.3f}"
    )
    print(
        f"Mejora media lookahead vs rectified: "
        f"{(mean_lookahead - mean_rectified) / mean_rectified * 100:+.1f}%"
    )
    print(
        f"Mejora media lookahead vs containment: "
        f"{(mean_lookahead - mean_containment) / mean_containment * 100:+.1f}%"
    )

    if args.csv is not None:
        write_csv(rows, args.csv)

    if args.maps_json is not None and maps:
        with open(args.maps_json, "w") as f:
            json.dump(maps, f, indent=2)
        print(f"Mapas guardados en: {args.maps_json}")

    if soundness_issues:
        print("\nAdvertencia de sonido (muestras fuera del intervalo):")
        print(
            f"{'n':>4} {'density':>7} {'phi':>4} {'seed':>4} {'k':>2} {'metodo':>11} | "
            f"{'viol. media':>11} {'viol. max':>10}"
        )
        for result, method, mean_viol, max_viol in soundness_issues:
            print(
                f"{result['n_concepts']:>4} {result['density']:>7.1f} {result['phi']:>4.1f} "
                f"{result['seed']:>4} {result['lookahead_k']:>2} {method:>11} | "
                f"{mean_viol:>11.2e} {max_viol:>10.2e}"
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
