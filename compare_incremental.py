# Comparación: inserción incremental vs recalcular desde cero
#
# Partiendo de n0 puntos con su hull ya calculado, se mide cuánto cuesta agregar
# k puntos nuevos teniendo el hull actualizado después de cada uno:
#       - incremental: k llamadas a AndrewModificado.insertar
#       - desde cero: k llamadas a andrew_monotone_chain sobre todos los puntos acumulados
#
# Solo se mide la inserción (la construcción inicial queda fuera del cronómetro)
# Cada configuración se repite REPS veces con datos distintos y se reporta
# media y varianza muestral
#
# Los resultados se imprimen y se guardan en resultados/incremental_results.csv
#
# Ejecutar con: python compare_incremental.py


import csv
import math
import os
import random
import statistics
import time
from typing import Callable, Dict, List

from algorithms import Point, andrew_monotone_chain
from andrew_modificado import AndrewModificado

REPS = 30   # repeticiones por configuración (>= 30)
K = 100     # puntos insertados por repetición
OUTPUT_CSV = os.path.join("resultados", "incremental_results.csv")


# Uniforme en un cuadrado: hull pequeño, casi todo es interior
def gen_uniform(rng: random.Random, n: int) -> List[Point]:
    return [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(n)]


# Sobre una circunferencia: peor caso para el incremental (h cercano a n)
def gen_circle(rng: random.Random, n: int) -> List[Point]:
    angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
    return [(math.cos(a) * 100, math.sin(a) * 100) for a in angles]


# time_incremental
# segundos que toma insertar los puntos nuevos uno a uno en un hull ya construido
def time_incremental(base: List[Point], new: List[Point]) -> float:
    ih = AndrewModificado(base)  # fuera del cronómetro
    t0 = time.perf_counter()
    for p in new:
        ih.insertar(p)
    return time.perf_counter() - t0


# time_from_scratch
# segundos que toma recalcular el hull completo después de cada punto nuevo
def time_from_scratch(base: List[Point], new: List[Point]) -> float:
    pts = list(base)  # fuera del cronómetro
    t0 = time.perf_counter()
    for p in new:
        pts.append(p)
        andrew_monotone_chain(pts)
    return time.perf_counter() - t0


# run_config
# repite la medición REPS veces, imprime media, varianza y la razón entre ambos tiempos,
# y devuelve una fila con esos valores para el CSV
def run_config(name: str, gen: Callable, n0: int) -> Dict[str, object]:
    inc_ms: List[float] = []
    scr_ms: List[float] = []
    for rep in range(REPS):
        rng = random.Random(rep)
        pts = gen(rng, n0 + K)
        base, new = pts[:n0], pts[n0:]
        inc_ms.append(time_incremental(base, new) * 1000)
        scr_ms.append(time_from_scratch(base, new) * 1000)

    m_inc, v_inc = statistics.mean(inc_ms), statistics.variance(inc_ms)
    m_scr, v_scr = statistics.mean(scr_ms), statistics.variance(scr_ms)
    print(f"{name:<15} n0={n0:<6} | incremental: media={m_inc:10.4f} ms  var={v_inc:10.6f} ms^2"
          f" | desde cero: media={m_scr:10.3f} ms  var={v_scr:10.3f} ms^2"
          f" | razón={m_scr / m_inc:8.1f}x")
    return {
        "dataset": name,
        "n0": n0,
        "k": K,
        "reps": REPS,
        "incremental_mean_ms": m_inc,
        "incremental_variance_ms2": v_inc,
        "scratch_mean_ms": m_scr,
        "scratch_variance_ms2": v_scr,
        "speedup": m_scr / m_inc,
    }


# save_csv
# guarda una fila por configuración en OUTPUT_CSV
def save_csv(rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados guardados en: {OUTPUT_CSV}")


if __name__ == "__main__":
    print(f"Tiempo total de insertar K={K} puntos (REPS={REPS} repeticiones por fila)")
    rows = []
    for n0 in (1000, 5000):
        rows.append(run_config("uniforme", gen_uniform, n0))
    for n0 in (1000, 5000):
        rows.append(run_config("circunferencia", gen_circle, n0))
    save_csv(rows)
