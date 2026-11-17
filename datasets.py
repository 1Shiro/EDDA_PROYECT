"""
Módulo generador de datasets de puntos en 2D para evaluar algoritmos de Convex Hull.

Incluye 6 distribuciones (se piden al menos 5 en el enunciado), diseñadas para
poner a prueba los casos favorables, promedio y peores casos de Jarvis March,
Andrew's Monotone Chain y Quickhull:

1. uniform_square : Puntos distribuidos uniformemente en un cuadrado -> E[h] = O(log n)
2. uniform_disk   : Puntos distribuidos uniformemente dentro de un disco -> E[h] = O(n^(1/3))
3. large_hull     : Puntos sobre una circunferencia (todos en el borde) -> h = n (Peor caso Jarvis y Quickhull)
4. small_hull     : 4 vértices extremos de un cuadrado + (n-4) puntos interiores -> h = 4 (Mejor caso Jarvis)
5. collinear_grid : Puntos con alta colinealidad en rectas y grilla -> Prueba robustez ante degeneraciones
6. parabola_curve : Puntos sobre la parábola y = x^2 -> h = n con particiones desbalanceadas en Quickhull
"""

import math
import random
from typing import Callable, Dict, List, Optional
from algorithms import Point


def generate_uniform_square(
    n: int,
    seed: Optional[int] = None,
    scale: float = 1000.0
) -> List[Point]:
    """
    1. Distribución Uniforme en un Cuadrado [-scale, scale]^2.
    Propiedad teórica: El número esperado de vértices en el Convex Hull es E[h] = O(log n).
    Representa el caso promedio estándar con pocos vértices en la frontera.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    return [(rng.uniform(-scale, scale), rng.uniform(-scale, scale)) for _ in range(n)]


def generate_uniform_disk(
    n: int,
    seed: Optional[int] = None,
    radius: float = 1000.0
) -> List[Point]:
    """
    2. Distribución Uniforme dentro de un Disco de radio `radius`.
    Para muestrear uniformemente en área polar, se usa r = radius * sqrt(U(0, 1))
    y theta = U(0, 2*pi).
    Propiedad teórica: El número esperado de vértices en el Convex Hull es E[h] = O(n^(1/3)).
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pts: List[Point] = []
    for _ in range(n):
        r = radius * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


def generate_large_hull_circle(
    n: int,
    seed: Optional[int] = None,
    radius: float = 1000.0
) -> List[Point]:
    """
    3. Hull Grande (Puntos sobre la circunferencia de un círculo).
    Todos los puntos generados son extremos y pertenecen estrictamente al Convex Hull.
    Propiedad teórica: h = n.
    - Es el PEOR CASO para Jarvis March: O(n * h) = O(n^2).
    - También degrada a Quickhull porque ningún punto queda en el interior de los triángulos
      para ser descartado tempranamente.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    # Generamos ángulos distintos para evitar duplicados y mezclamos el orden
    pts: List[Point] = []
    for i in range(n):
        # Ángulo base con perturbación aleatoria menor que el paso para garantizar n puntos distintos
        angle = (2.0 * math.pi * (i + rng.uniform(0.1, 0.9))) / n
        pts.append((radius * math.cos(angle), radius * math.sin(angle)))
    rng.shuffle(pts)
    return pts


def generate_small_hull(
    n: int,
    seed: Optional[int] = None,
    scale: float = 1000.0
) -> List[Point]:
    """
    4. Hull Chico (Polígono exterior fijo de h = 4 vértices con n - 4 puntos interiores).
    Si n >= 4, coloca las 4 esquinas del cuadrado [-scale, scale]^2 y el resto de los
    puntos estrictamente en el interior [-0.85*scale, 0.85*scale]^2.
    Propiedad teórica: h = 4 (constante, independiente de n).
    - Es el MEJOR CASO para Jarvis March: O(n * h) = O(4n) = O(n) tiempo lineal.
    - En Quickhull, la primera pasada de los 2 triángulos descarta casi todos los puntos interiores.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    corners: List[Point] = [
        (-scale, -scale),
        (scale, -scale),
        (scale, scale),
        (-scale, scale),
    ]
    if n <= 4:
        return corners[:n]

    inner_limit = scale * 0.85
    interior = [
        (rng.uniform(-inner_limit, inner_limit), rng.uniform(-inner_limit, inner_limit))
        for _ in range(n - 4)
    ]
    pts = corners + interior
    rng.shuffle(pts)
    return pts


def generate_collinear_and_border(
    n: int,
    seed: Optional[int] = None,
    scale: float = 1000.0
) -> List[Point]:
    """
    5. Puntos Colineales en Bordes y Diagonales.
    Distribuye los puntos exactamente sobre las 4 aristas y las 2 diagonales de un cuadrado
    [-scale, scale]^2.
    Propiedad teórica: h = 4, pero existen cientos/miles de puntos colineales sobre el mismo
    borde del Convex Hull y en las diagonales interiores.
    Evalúa la estabilidad numérica y el descarte correcto de vértices colineales intermedios.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    corners: List[Point] = [
        (-scale, -scale),
        (scale, -scale),
        (scale, scale),
        (-scale, scale),
    ]
    if n <= 4:
        return corners[:n]

    pts: List[Point] = list(corners)
    for i in range(n - 4):
        t = rng.uniform(-scale, scale)
        mode = i % 6
        if mode == 0:
            pts.append((t, -scale))      # Arista inferior
        elif mode == 1:
            pts.append((scale, t))       # Arista derecha
        elif mode == 2:
            pts.append((t, scale))       # Arista superior
        elif mode == 3:
            pts.append((-scale, t))      # Arista izquierda
        elif mode == 4:
            pts.append((t, t))           # Diagonal principal
        else:
            pts.append((t, -t))          # Diagonal secundaria

    rng.shuffle(pts)
    return pts


def generate_parabola_curve(
    n: int,
    seed: Optional[int] = None,
    scale: float = 100.0
) -> List[Point]:
    """
    6. Curva Parabólica (Bonus: Peor caso de partición asimétrica para Quickhull).
    Puntos sobre una curva estrictamente convexa con espaciamiento geométrico/progresivo,
    donde h = n y las particiones de Quickhull son altamente desbalanceadas.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pts: List[Point] = []
    for i in range(n):
        x = (i + rng.uniform(0.1, 0.9)) / n * scale
        y = (x * x) / scale
        pts.append((x, y))
    rng.shuffle(pts)
    return pts


# Diccionario catálogo con todos los generadores disponibles (útil para benchmark y visualizador)
DATASET_GENERATORS: Dict[str, Callable[[int, Optional[int]], List[Point]]] = {
    "uniforme_cuadrado": generate_uniform_square,
    "uniforme_disco": generate_uniform_disk,
    "hull_grande_circulo": generate_large_hull_circle,
    "hull_chico_cuadrado": generate_small_hull,
    "colineales_bordes": generate_collinear_and_border,
    "curva_parabola": generate_parabola_curve,
}


if __name__ == "__main__":
    # Pequeña prueba de verificación de los 6 datasets con los 3 algoritmos
    from algorithms import jarvis_march, andrew_monotone_chain, quickhull

    print("=== Verificación de los generadores de Datasets (n = 200) ===")
    n_test = 200
    for name, gen_fn in DATASET_GENERATORS.items():
        pts = gen_fn(n_test, 42)
        h_j = jarvis_march(pts)
        h_a = andrew_monotone_chain(pts)
        h_q = quickhull(pts)

        assert h_j == h_a == h_q, (
            f"Error de consistencia en dataset '{name}': "
            f"Jarvis={len(h_j)}, Andrew={len(h_a)}, Quickhull={len(h_q)}"
        )
        print(f"  - {name:<22} | n = {len(pts):4d} | vértices en el hull (h) = {len(h_a):4d} | [OK: 3 algoritmos coinciden]")
