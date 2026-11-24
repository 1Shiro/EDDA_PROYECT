import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

DATASET_TITLES: Dict[str, str] = {
    "uniforme_cuadrado": "Distribución Uniforme en Cuadrado ($E[h] = O(\\log n)$)",
    "uniforme_disco": "Distribución Uniforme en Disco ($E[h] = O(n^{1/3})$)",
    "hull_grande_circulo": "Hull Grande — Circunferencia ($h = n$)",
    "hull_chico_cuadrado": "Hull Chico — Cuadrado con Interior ($h = 4$)",
    "colineales_bordes": "Puntos Colineales en Bordes y Diagonales ($h = 4$)",
    "curva_parabola": "Curva Parabólica ($h = n$)",
}

# Títulos cortos para la figura resumen (con tildes).
SHORT_TITLES: Dict[str, str] = {
    "uniforme_cuadrado": "Uniforme en cuadrado",
    "uniforme_disco": "Uniforme en disco",
    "hull_grande_circulo": "Hull grande (circunferencia)",
    "hull_chico_cuadrado": "Hull chico (cuadrado)",
    "colineales_bordes": "Colineales",
    "curva_parabola": "Curva parabólica",
}

ALGO_ORDER = [
    "Jarvis March",
    "Andrew Monotone Chain",
    "Quickhull",
]


def load_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el archivo {csv_path}. Ejecuta primero: python benchmark.py"
        )
    with open(csv_path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_latex_tables(rows: List[Dict[str, str]], out_tex_path: str) -> None:
    """
    Genera tablas en formato LaTeX (usando booktabs) para cada dataset,
    reportando h promedio, tiempo medio (ms) y varianza (ms^2) para los 3 algoritmos.
    """
    # Agrupar por dataset 
    grouped: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        ds = r["dataset"]
        n = int(r["n"])
        algo = r["algorithm"]
        grouped[ds][n][algo] = {
            "mean_h": float(r["mean_h"]),
            "mean_ms": float(r["mean_ms"]),
            "variance_ms2": float(r["variance_ms2"]),
            "std_ms": float(r["std_ms"]),
        }

    lines: List[str] = [
        "% Tablas generadas automáticamente por plot_results.py",
        "% Requiere \\usepackage{booktabs} en el preámbulo de LaTeX\n",
    ]

    for ds, by_n in grouped.items():
        title = DATASET_TITLES.get(ds, ds)
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Resultados experimentales para {title} (30 repeticiones).}}")
        lines.append(f"\\label{{tab:{ds}}}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{rrcccccc}")
        lines.append("\\toprule")
        lines.append(
            " & & \\multicolumn{2}{c}{\\textbf{Jarvis March}} & "
            "\\multicolumn{2}{c}{\\textbf{Andrew Monotone Chain}} & "
            "\\multicolumn{2}{c}{\\textbf{Quickhull}} \\\\"
        )
        lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")
        lines.append(
            "$n$ & $\\bar{h}$ & Media (ms) & Varianza ($\\text{ms}^2$) & "
            "Media (ms) & Varianza ($\\text{ms}^2$) & Media (ms) & Varianza ($\\text{ms}^2$) \\\\"
        )
        lines.append("\\midrule")

        for n in sorted(by_n.keys()):
            algos_data = by_n[n]
            first_algo = next(iter(algos_data.values()))
            mean_h = first_algo["mean_h"]
            row_cells = [f"{n}", f"{mean_h:.1f}"]
            for algo in ALGO_ORDER:
                if algo in algos_data:
                    m = algos_data[algo]["mean_ms"]
                    v = algos_data[algo]["variance_ms2"]
                    row_cells.append(f"{m:.4f}")
                    row_cells.append(f"{v:.5f}")
                else:
                    row_cells.extend(["-", "-"])
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}\n")

    os.makedirs(os.path.dirname(out_tex_path), exist_ok=True)
    with open(out_tex_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  tablas LaTeX en {out_tex_path}")


def generate_plots(rows: List[Dict[str, str]], plots_dir: str) -> None:
    """
    Genera gráficos comparativos por dataset y una figura resumen 2x3 si matplotlib está disponible.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib no está instalado, no se generan los gráficos")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Agrupar por dataset -> algorithm -> lista de (n, mean_ms, std_ms)
    grouped: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        ds = r["dataset"]
        algo = r["algorithm"]
        grouped[ds][algo].append(
            (int(r["n"]), float(r["mean_ms"]), float(r["std_ms"]))
        )

    markers = {
        "Jarvis March": "o",
        "Andrew Monotone Chain": "s",
        "Quickhull": "^",
    }

    # 1. Gráfico individual por cada dataset
    for ds, by_algo in grouped.items():
        plt.figure(figsize=(7, 4.5))
        for algo in ALGO_ORDER:
            if algo not in by_algo:
                continue
            pts = sorted(by_algo[algo], key=lambda x: x[0])
            ns = [p[0] for p in pts]
            means = [p[1] for p in pts]
            stds = [p[2] for p in pts]
            plt.errorbar(
                ns,
                means,
                yerr=stds,
                marker=markers.get(algo, "o"),
                capsize=4,
                linewidth=1.8,
                label=algo,
            )

        plt.title(f"Tiempo de ejecución vs $n$ — {DATASET_TITLES.get(ds, ds)}", fontsize=10)
        plt.xlabel("Número de puntos ($n$)")
        plt.ylabel("Tiempo medio de ejecución (ms, escala log)")
        # Escala logarítmica: Jarvis puede ser ~1000 veces más lento que los
        # otros dos y, en escala lineal, Andrew y Quickhull quedan pegados al 0.
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(plots_dir, f"benchmark_{ds}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"  gráfico en {out_png}")

    # 2. Figura resumen con todos los datasets en una grilla 2x3
    datasets_list = list(grouped.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, ds in enumerate(datasets_list[:6]):
        ax = axes_flat[idx]
        by_algo = grouped[ds]
        for algo in ALGO_ORDER:
            if algo not in by_algo:
                continue
            pts = sorted(by_algo[algo], key=lambda x: x[0])
            ns = [p[0] for p in pts]
            means = [p[1] for p in pts]
            stds = [p[2] for p in pts]
            ax.errorbar(
                ns,
                means,
                yerr=stds,
                marker=markers.get(algo, "o"),
                capsize=3,
                linewidth=1.6,
                label=algo,
            )
        ax.set_title(SHORT_TITLES.get(ds, ds.replace("_", " ")))
        ax.set_xlabel("$n$")
        ax.set_ylabel("Tiempo (ms, escala log)")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    summary_png = os.path.join(plots_dir, "benchmark_resumen_todos.png")
    plt.savefig(summary_png, dpi=200)
    plt.close()
    print(f"  resumen en {summary_png}")


if __name__ == "__main__":
    csv_file = os.path.join("resultados", "benchmark_results.csv")
    tex_file = os.path.join("resultados", "tables_latex.tex")
    plots_folder = os.path.join("resultados", "plots")

    data_rows = load_csv(csv_file)
    generate_latex_tables(data_rows, tex_file)
    generate_plots(data_rows, plots_folder)
