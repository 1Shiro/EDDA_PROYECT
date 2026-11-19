"""
Script de evaluación experimental (Benchmark) para los 3 algoritmos de Convex Hull 2D:
- Jarvis March
- Andrew's Monotone Chain
- Quickhull
"""

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

from algorithms import (
    Point,
    jarvis_march,
    andrew_monotone_chain,
    quickhull,
)
from datasets import DATASET_GENERATORS

# Aumentamos el límite de recursión de Python por seguridad para Quickhull
# cuando se evalúa en curvas estrictamente convexas (h = n) con miles de puntos.
sys.setrecursionlimit(20000)

ALGORITHMS: Dict[str, Callable[[List[Point]], List[Point]]] = {
    "Jarvis March": jarvis_march,
    "Andrew Monotone Chain": andrew_monotone_chain,
    "Quickhull": quickhull,
}

# Tamaños de n por defecto para el experimento.
# Nota: En hull_grande_circulo y curva_parabola (donde h = n), Jarvis March y Quickhull
# toman tiempo cuadrático O(n^2); con n <= 5000 las 30 repeticiones terminan en pocos minutos
# mostrando claramente la curva cuadrática vs O(n log n).
DEFAULT_SIZES = [100, 500, 1000, 2500, 5000]
DEFAULT_REPETITIONS = 30


def measure_single_run(
    algo_fn: Callable[[List[Point]], List[Point]],
    points: List[Point]
) -> Tuple[float, int]:
    
    #Mide exclusivamente el tiempo de ejecución de algo_fn(points_copy) en (ms)
    
    pts_copy = list(points)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        hull = algo_fn(pts_copy)
        t1 = time.perf_counter()
    finally:
        if gc_was_enabled:
            gc.enable()

    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms, len(hull)


def run_benchmarks(
    sizes: List[int],
    repetitions: int = DEFAULT_REPETITIONS,
    output_csv: str = os.path.join("resultados", "benchmark_results.csv"),
    base_seed: int = 1000,
) -> List[Dict[str, object]]:

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows: List[Dict[str, object]] = []

    print("=" * 95)
    print(f"inciando prueba en 2d ({repetitions} repeticiones por configuración)")
    print(f"Tamaños de muestra (n): {sizes}")
    print(f"Datasets a evaluar    : {list(DATASET_GENERATORS.keys())}")
    print("=" * 95)

    for dataset_name, gen_fn in DATASET_GENERATORS.items():
        print(f"\n>>> Dataset: {dataset_name}")
        print(
            f"{'n':>7} | {'Algoritmo':<23} | {'h (media)':>9} | "
            f"{'Media (ms)':>12} | {'Varianza (ms²)':>16} | {'Desv.Est (ms)':>13}"
        )
        print("-" * 95)

        for n in sizes:
            # Pre-generamos las `repetitions` instancias de puntos (fuera del cronómetro)
            # para que los 3 algoritmos sean evaluados exactamente sobre los mismos puntos
            instances: List[List[Point]] = [
                gen_fn(n, base_seed + r) for r in range(repetitions)
            ]

            for algo_name, algo_fn in ALGORITHMS.items():
                times_ms: List[float] = []
                hull_sizes: List[int] = []

                for r in range(repetitions):
                    elapsed_ms, h_len = measure_single_run(algo_fn, instances[r])
                    times_ms.append(elapsed_ms)
                    hull_sizes.append(h_len)

                mean_ms = statistics.mean(times_ms)
                var_ms2 = statistics.variance(times_ms) if repetitions > 1 else 0.0
                stdev_ms = statistics.stdev(times_ms) if repetitions > 1 else 0.0
                min_ms = min(times_ms)
                max_ms = max(times_ms)
                mean_h = statistics.mean(hull_sizes)

                print(
                    f"{n:7d} | {algo_name:<23} | {mean_h:9.1f} | "
                    f"{mean_ms:12.4f} | {var_ms2:16.6f} | {stdev_ms:13.4f}"
                )

                rows.append({
                    "dataset": dataset_name,
                    "n": n,
                    "algorithm": algo_name,
                    "repetitions": repetitions,
                    "mean_h": round(mean_h, 2),
                    "mean_ms": round(mean_ms, 6),
                    "variance_ms2": round(var_ms2, 8),
                    "std_ms": round(stdev_ms, 6),
                    "min_ms": round(min_ms, 6),
                    "max_ms": round(max_ms, 6),
                })

    # Guardar resultados en CSV
    fieldnames = [
        "dataset",
        "n",
        "algorithm",
        "repetitions",
        "mean_h",
        "mean_ms",
        "variance_ms2",
        "std_ms",
        "min_ms",
        "max_ms",
    ]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 95)
    print(f"Benchmark finalizado. Resultados guardados en: {output_csv}")
    print("=" * 95)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark experimental para los 3 algoritmos de Convex Hull 2D."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Lista de tamaños n a evaluar (por defecto: 100 500 1000 2500 5000).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Número de repeticiones por configuración.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("resultados", "benchmark_results.csv"),
        help="Ruta del archivo CSV de salida.",
    )
    args = parser.parse_args()

    run_benchmarks(sizes=args.sizes, repetitions=args.reps, output_csv=args.output)


if __name__ == "__main__":
    main()
