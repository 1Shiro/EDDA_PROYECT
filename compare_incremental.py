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
# Los resultados se imprimen, se guardan en resultados/incremental_results.csv
# y se grafican en resultados/plots/ (si matplotlib está instalado): un gráfico por
# conjunto (incremental_<conjunto>.png) y uno con ambos lado a lado (incremental.png)
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
SIZES = (100, 500, 1000, 2500, 5000)  # tamaños del hull inicial (n0), los mismos del benchmark
OUTPUT_CSV = os.path.join("resultados", "incremental_results.csv")
PLOTS_DIR = os.path.join("resultados", "plots")
OUTPUT_PNG = os.path.join(PLOTS_DIR, "incremental.png")


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
    print(f"  {name:<15} n0={n0:<6} | incremental: media={m_inc:10.4f} ms  var={v_inc:10.6f} ms^2"
          f" | desde cero: media={m_scr:10.3f} ms  var={v_scr:10.3f} ms^2"
          f" | razón={m_scr / m_inc:8.1f}x")
    return {
        "dataset": name,
        "n0": n0,
        "k": K,
        "reps": REPS,
        "incremental_mean_ms": m_inc,
        "incremental_variance_ms2": v_inc,
        "incremental_std_ms": math.sqrt(v_inc),
        "scratch_mean_ms": m_scr,
        "scratch_variance_ms2": v_scr,
        "scratch_std_ms": math.sqrt(v_scr),
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
    print(f"  resultados en {OUTPUT_CSV}")


# plot_dataset
# dibuja en `ax` el tiempo de agregar K puntos según n0 para ambos métodos de un conjunto,
# con barras de desviación estándar y escala logarítmica en el tiempo
def plot_dataset(ax, rows: List[Dict[str, object]], ds: str, title: str) -> None:
    data = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["n0"])
    ns = [r["n0"] for r in data]
    ax.errorbar(ns, [r["scratch_mean_ms"] for r in data],
                yerr=[r["scratch_std_ms"] for r in data],
                marker="o", capsize=4, linewidth=1.8, label="Andrew desde cero")
    ax.errorbar(ns, [r["incremental_mean_ms"] for r in data],
                yerr=[r["incremental_std_ms"] for r in data],
                marker="s", capsize=4, linewidth=1.8, label="Andrew Modificado")
    ax.set_title(title)
    ax.set_xlabel("Puntos del hull inicial ($n_0$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()


# save_plot
# guarda un gráfico por conjunto y uno con todos los conjuntos lado a lado
def save_plot(rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib no está instalado, no se genera el gráfico")
        return

    titles = {"uniforme": "Uniforme en cuadrado", "circunferencia": "Circunferencia"}
    ylabel = f"Tiempo de agregar {K} puntos (ms, escala log)"
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Un gráfico por conjunto
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_dataset(ax, rows, ds, f"Inserción vs recalcular — {titles.get(ds, ds)}")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"incremental_{ds}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"  gráfico en {out}")

    # 2. Todos los conjuntos lado a lado
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.5), sharey=True)
    for ax, ds in zip(axes, datasets):
        plot_dataset(ax, rows, ds, titles.get(ds, ds))
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()
    print(f"  gráfico en {OUTPUT_PNG}")


if __name__ == "__main__":
    print(f"  tiempo total de insertar K={K} puntos (REPS={REPS} repeticiones por fila)")
    rows = []
    for n0 in SIZES:
        rows.append(run_config("uniforme", gen_uniform, n0))
    for n0 in SIZES:
        rows.append(run_config("circunferencia", gen_circle, n0))
    save_csv(rows)
    save_plot(rows)
