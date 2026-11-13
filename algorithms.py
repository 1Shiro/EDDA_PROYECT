"""
3 algoritmos de Convex Hull 2D:
1. Jarvis March (Gift Wrapping) - O(n * h)
2. Andrew's Monotone Chain - O(n log n)
3. Quickhull - O(n log n) promedio, O(n^2) peor caso

Convención de salida para todos los algoritmos:
- Cada punto es una tupla (x, y) de floats/ints.
- La salida es una lista de puntos [(x0, y0), (x1, y1), ...] que forman los vértices
  estrictos del Convex Hull (sin puntos colineales intermedios en las aristas).
- Los vértices se entregan ordenados en sentido antihorario (CCW), comenzando
  por el punto lexicográficamente menor (menor x, y en caso de empate menor y).
- El punto inicial no se repite al final de la lista.
"""

from typing import List, Tuple

Point = Tuple[float, float]
EPS = 1e-12


def cross_product(o: Point, a: Point, b: Point) -> float:
    """
    Calcula el producto cruz 2D de los vectores OA y OB: (a - o) x (b - o).
    - > 0: giro antihorario (a la izquierda) de OA a OB.
    - < 0: giro horario (a la derecha) de OA a OB.
    - == 0: los puntos o, a, b son colineales.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def squared_distance(a: Point, b: Point) -> float:
    #Calcula la distancia euclidiana al cuadrado entre 2 puntos a y b."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def normalize_hull(hull: List[Point]) -> List[Point]:
    """
    Asegura que el polígono convexo comience en el punto lexicográficamente menor
    y esté en sentido antihorario (CCW), sin puntos colineales redundantes.
    """
    if len(hull) <= 1:
        return list(hull)

    # Eliminar posibles colineales en el cierre
    cleaned: List[Point] = []
    n = len(hull)
    for i in range(n):
        prev_pt = hull[(i - 1) % n]
        curr_pt = hull[i]
        next_pt = hull[(i + 1) % n]
        if abs(cross_product(prev_pt, curr_pt, next_pt)) > EPS:
            cleaned.append(curr_pt)

    if not cleaned:
        # Todos los puntos eran colineales entonces devuelve extremos
        min_pt = min(hull)
        max_pt = max(hull)
        return [min_pt, max_pt] if min_pt != max_pt else [min_pt]

    # Rotar la lista para empezar desde el punto lexicográficamente menor
    min_idx = min(range(len(cleaned)), key=lambda i: cleaned[i])
    return cleaned[min_idx:] + cleaned[:min_idx]


# 1. JARVIS MARCH 

def jarvis_march(points: List[Point]) -> List[Point]:
    """
    Calcula el Convex Hull 2D usando el algoritmo de Jarvis March.
    Complejidad temporal: O(n * h), donde n es el número de puntos y h el número de vértices en el hull..
    """
    unique_pts = list(set(points))
    n = len(unique_pts)
    if n <= 1:
        return unique_pts

    # El punto lexicográficamente menor siempre pertenece al Convex Hull
    start_pt = min(unique_pts)
    hull: List[Point] = []
    current = start_pt

    while True:
        hull.append(current)
        # Elegir un candidato inicial
        candidate = unique_pts[0] if unique_pts[0] != current else unique_pts[1]

        for pt in unique_pts:
            if pt == current or pt == candidate:
                continue
            cp = cross_product(current, candidate, pt)
            # En sentido CCW, buscamos el punto que esté más a la derecha (cp < 0)
            # de todos los demás si son colineales (|cp| <= EPS), nos quedamoscon el más lejano para descartar vértices colineales intermedios.
            if cp < -EPS or (abs(cp) <= EPS and squared_distance(current, pt) > squared_distance(current, candidate)):
                candidate = pt

        current = candidate
        if current == start_pt:
            break

    return normalize_hull(hull)


# 2. ANDREWS MONOTONE CHAIN

def andrew_monotone_chain_chains(points: List[Point]) -> Tuple[List[Point], List[Point]]:
    """
    Variante auxiliar de Andrew's Monotone Chain que devuelve por separado la cadena
    inferior y la cadena superior (upper hull), incluyendo ambos extremos (min_pt y max_pt) en cada cadena.
    """
    sorted_pts = sorted(set(points))
    n = len(sorted_pts)
    if n <= 1:
        return list(sorted_pts), list(sorted_pts)

    # Construcción de la cadena inferior (lower hull) de izquierda a derecha
    lower: List[Point] = []
    for pt in sorted_pts:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], pt) <= EPS:
            lower.pop()
        lower.append(pt)

    # Construcción de la cadena superior de derecha a izquierda
    upper: List[Point] = []
    for pt in reversed(sorted_pts):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], pt) <= EPS:
            upper.pop()
        upper.append(pt)

    return lower, upper


def andrew_monotone_chain(points: List[Point]) -> List[Point]:
    """
    Calcula el Convex Hull 2D usando el algoritmo de Andrew Monotone .
    """
    lower, upper = andrew_monotone_chain_chains(points)
    if len(lower) <= 1:
        return lower

    # Caso degenerado: todos los puntos son colineales (lower y upper tienen solo los 2 extremos)
    if len(lower) == 2 and len(upper) == 2 and lower[0] == upper[1] and lower[1] == upper[0]:
        return lower

    # Concatenar quitando el último punto de cada lista porque se duplica con el inicio de la otra
    return lower[:-1] + upper[:-1]


# 3. QUICKHULL

def _find_hull_quickhull(pts: List[Point], p1: Point, p2: Point) -> List[Point]:
    """
    Subrutina recursiva de Quickhull.
    Encuentra los vértices del hull que están estrictamente a la izquierda
    del segmento dirigido p1 a p2, ordenados en sentido CCW desde p1 hasta p2
    """
    if not pts:
        return []

    # la idea es encontrar el punto con mayor distancia perpendicular a la recta p1 -> p2.
    # Como la base |p2 - p1| es constante, equivale a maximizar cross_product(p1, p2, pt).
    # En caso de empate, desempatamos maximizando el ángulo o distancia a p1 para determinismo.
    farthest = max(
        pts,
        key=lambda pt: (cross_product(p1, p2, pt), squared_distance(p1, pt))
    )

    # Filtrar puntos que quedan estrictamente afuera del triángulo.
    left_of_p1_farthest = [
        pt for pt in pts
        if pt != farthest and cross_product(p1, farthest, pt) > EPS
    ]
    left_of_farthest_p2 = [
        pt for pt in pts
        if pt != farthest and cross_product(farthest, p2, pt) > EPS
    ]

    # Para recorrido CCW de p1 a p2 pasando por el semiplano izquierdo de p1->p2,
    # primero visitamos los puntos entre p1 y farthest
    part1 = _find_hull_quickhull(left_of_p1_farthest, p1, farthest)
    part2 = _find_hull_quickhull(left_of_farthest_p2, farthest, p2)

    return part1 + [farthest] + part2


def quickhull(points: List[Point]) -> List[Point]:
    """
    Calcula el Convex Hull 2D usando el algoritmo Quickhull (divide y conquistarass).
    """
    unique_pts = list(set(points))
    n = len(unique_pts)
    if n <= 1:
        return unique_pts

    min_pt = min(unique_pts)
    max_pt = max(unique_pts)

    if min_pt == max_pt:
        return [min_pt]

    # Dividir los puntos en dos conjuntos respecto a la línea dirigida min_pt -> max_pt:
    # - upper_set: puntos estrictamente a la izquierda de min_pt -> max_pt (cp > EPS)
    # - lower_set: puntos estrictamente a la izquierda de max_pt -> min_pt (es decir, a la derecha de min_pt -> max_pt)
    upper_set: List[Point] = []
    lower_set: List[Point] = []

    for pt in unique_pts:
        if pt == min_pt or pt == max_pt:
            continue
        cp = cross_product(min_pt, max_pt, pt)
        if cp > EPS:
            upper_set.append(pt)
        elif cp < -EPS:
            lower_set.append(pt)

    # _find_hull_quickhull(upper_set, min_pt, max_pt) recorre el arco superior (sentido horario)
    # _find_hull_quickhull(lower_set, max_pt, min_pt) recorre el arco inferior (sentido horario)
    upper_hull_arc = _find_hull_quickhull(upper_set, min_pt, max_pt)
    lower_hull_arc = _find_hull_quickhull(lower_set, max_pt, min_pt)

    # Si todos los puntos eran colineales, ambos arcos son vacíos
    if not upper_hull_arc and not lower_hull_arc:
        return [min_pt, max_pt]

    # [min_pt] + reversed(lower_hull_arc) + [max_pt] + reversed(upper_hull_arc) probar dsp
    ccw_hull = [min_pt] + list(reversed(lower_hull_arc)) + [max_pt] + list(reversed(upper_hull_arc))

    return normalize_hull(ccw_hull)
