# Convex Hull 2D — Proyecto semestral EDAA

Implementación y comparación experimental de tres algoritmos para calcular el
convex hull en dos dimensiones (**Jarvis March**, **Andrew's Monotone Chain** y
**Quickhull**), una versión de Andrew que permite **insertar puntos** sin
recalcular todo, y una **herramienta visual** que muestra paso a paso cómo
trabaja cada algoritmo.

## Requisitos

- Python 3.8 o superior (probado con Python 3.14.9)
- Matplotlib: `python3 -m pip install -r requirements.txt`
- [fish shell](https://fishshell.com/) para correr el script

## Reproducir todo

```bash
./correr_todo.fish
```

El script corre los tests y ejecuta los dos experimentos. Tarda unos 15
minutos, casi todo en Jarvis March sobre los conjuntos donde todos los puntos
están en el hull.

Si el script no tiene permiso de ejecución: `chmod +x correr_todo.fish`.

### Resultados que se generan

| Archivo | Contenido |
|---|---|
| `resultados/benchmark_results.csv` | Experimento 1: media, varianza y desviación de los 3 algoritmos |
| `resultados/tables_latex.tex` | Tablas del anexo del informe |
| `resultados/plots/benchmark_*.png` | Gráficos del experimento 1 |
| `resultados/incremental_results.csv` | Experimento 2: inserción vs recalcular |
| `resultados/plots/incremental*.png` | Gráficos del experimento 2 |

Las semillas son fijas, así que los puntos generados son siempre los mismos.
Los tiempos absolutos dependen de la máquina, pero las tendencias (qué
algoritmo es más rápido y cómo crece cada uno) se mantienen.

## Herramienta visual

```bash
python3 convexHullVisual.py
```

Opciones: `-n 50` (cantidad de puntos), `--seed 7`, `--algorithm jarvis|andrew|quickhull`,
`--animate` (parte con la animación paso a paso) y `--points archivo.txt`
(una línea `x,y` por punto).

Desde la ventana se pueden generar puntos, elegir el algoritmo, avanzar paso a
paso e insertar puntos haciendo click (con Andrew seleccionado). El botón de
guardar deja la imagen en `figuras/`.

## Ejecutar cada parte por separado

```bash
python3 test_algorithms.py      # tests de los 3 algoritmos
python3 test_incremental.py     # tests de Andrew modificado
python3 benchmark.py            # experimento 1 (acepta --sizes, --reps, --output)
python3 plot_results.py         # tablas y gráficos del experimento 1
python3 compare_incremental.py  # experimento 2
```

Todos los comandos se ejecutan desde la carpeta raíz del proyecto.

## Estructura

| Archivo | Contenido |
|---|---|
| `algorithms.py` | Jarvis March, Andrew's Monotone Chain y Quickhull |
| `andrew_modificado.py` | Andrew con inserción de puntos |
| `datasets.py` | Los 6 generadores de conjuntos de puntos |
| `benchmark.py`, `plot_results.py` | Experimento 1 |
| `compare_incremental.py` | Experimento 2 |
| `pasoapaso.py`, `convexHullVisual.py` | Herramienta visual |
| `test_algorithms.py`, `test_incremental.py` | Tests |
| `correr_todo.fish` | Reproduce todo |
