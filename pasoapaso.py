"""
Versiones paso a paso de los 3 algoritmos de Convex Hull, para animarlos

Aquí hay una copia de cada algoritmo con la misma lógica, el mismo EPS y el mismo orden de recorrido,
escrita como generador que entrega un Step

Los puntos se recorren en el mismo orden que en algorithms.py:

    1. Jarvis y Quickhull recorren list(set(points)), sin ordenar
    2. Andrew los ordena, porque el algoritmo mismo parte ordenándolos

El último Step de cada generador trae el hull final, que debe ser igual al que
entrega algorithms.py

"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from algorithms import EPS, Point, cross_product, normalize_hull, squared_distance

Segment = Tuple[Point, Point]


# Un cuadro de la animación, o sea qué dibujar en un instante del algoritmo
@dataclass
class Step:
    text: str                                                       #  que está pasando
    done_edges: List[Segment] = field(default_factory=list)         # aristas ya aceptadas (EN línea continua)
    pending_edges: List[Segment] = field(default_factory=list)      # aristas en prueba (en línea punteada)
    highlight_edges: List[Segment] = field(default_factory=list)    # aristas destacadas en este paso
    active: List[Point] = field(default_factory=list)               # puntos que se están examinando
    removed: List[Point] = field(default_factory=list)              # vertices descartados justo en este paso
    discarded: List[Point] = field(default_factory=list)            # puntos descartados 
    minor: bool = False                                             # paso de detalle o no tan impotante
    hull: Optional[List[Point]] = None                              # hull final 


def _fmt(p: Point) -> str:
    """
    Función _fmt
    Recibe: un punto p
    Devuelve: el punto como texto con un decimal, para los mensajes de cada paso
    """
    return f"({p[0]:.1f}, {p[1]:.1f})"


def _polyline(chain: List[Point]) -> List[Segment]:
    """
    Función _polyline
    Recibe: una cadena abierta de puntos
    Devuelve: las aristas entre puntos consecutivos (sin cerrar la cadena)
    """
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


def _closed(hull: List[Point]) -> List[Segment]:
    """
    Función _closed
    Recibe: los vertices de un polígono
    Devuelve: sus aristas
    """
    if len(hull) < 2:
        return []

    return [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]


def _final_step(name: str, hull: List[Point]) -> Step:
    """
    Función _final_step
    Recibe: el nombre del algoritmo y el hull que calculó
    Devuelve: el último Step de la animación, con el hull completo
    """
    return Step(f"{name}: hull completo, {len(hull)} vértice(s).", done_edges=_closed(hull), hull=hull)


# 1. jarvis march

def jarvis_steps(points: List[Point]) -> Iterator[Step]:
    """
    Función jarvis_steps
    Recibe: una lista de puntos
    Devuelve: un generador con los pasos de Jarvis March

    Parte del punto lexicográficamente menor, que siempre está en el hull, y
    desde el vertice actual revisa todos los puntos quedándose con el más
    externo, que pasa a ser el siguiente vertice, repitiendo esto hasta volver
    al punto inicial

    Para cada punto revisado hay dos casos:

        1. Queda más a la derecha que el candidato, o es colineal y más lejano:
           pasa a ser el nuevo candidato
        2. Si no, se sigue con el siguiente punto

    Los puntos se recorren en el orden de list(set(points)), igual que en
    algorithms.py, por lo que el candidato inicial y los empates se resuelven
    igual

    O(n·h), igual que Jarvis, con h la cantidad de vertices del hull
    """
    pts = list(set(points))
    n = len(pts)

    if n <= 1:
        yield _final_step("Jarvis", pts)
        return

    start = min(pts)                    # el punto lexicográficamente menor siempre está en el hull
    hull: List[Point] = []
    current = start

    while True:
        hull.append(current)
        done = _polyline(hull)
        candidate = pts[0] if pts[0] != current else pts[1]
        yield Step(f"Vértice {len(hull)} en {_fmt(current)}: se busca el siguiente.",
                   done_edges=done, highlight_edges=[(current, candidate)], active=[current])

        for pt in pts:
            if pt == current or pt == candidate:
                continue

            cp = cross_product(current, candidate, pt)
            yield Step(f"Probando {_fmt(pt)} contra el candidato {_fmt(candidate)}.",
                       done_edges=done, pending_edges=[(current, pt)],
                       highlight_edges=[(current, candidate)], active=[pt], minor=True)

            # más a la derecha, o colineal y más lejano (mismo criterio que algorithms.py)
            if cp < -EPS or (abs(cp) <= EPS and squared_distance(current, pt) > squared_distance(current, candidate)):
                candidate = pt
                yield Step(f"{_fmt(pt)} queda más afuera: es el nuevo candidato.",
                           done_edges=done, highlight_edges=[(current, candidate)], active=[pt], minor=True)

        yield Step(f"Ningún punto queda más afuera que {_fmt(candidate)}: se agrega la arista.",
                   done_edges=done + [(current, candidate)], active=[candidate])
        current = candidate

        if current == start:
            break

    yield _final_step("Jarvis", normalize_hull(hull))


# 2. andrew's monotone chain

def andrew_steps(points: List[Point]) -> Iterator[Step]:
    """
    Función andrew_steps
    Recibe: una lista de puntos
    Devuelve: un generador con los pasos de Andrew's Monotone Chain

    Ordena los puntos (el algoritmo lo hace por sí mismo), arma la cadena
    inferior de izquierda a derecha y luego la superior de derecha a izquierda,
    y al final une las dos cadenas igual que en andrew_monotone_chain

    El algoritmo es O(n log n), pero cada paso guarda una copia de las aristas
    de la cadena para dibujarlas, asi que generar todos los pasos cuesta O(n²)
    en el peor caso
    """
    pts = sorted(set(points))
    n = len(pts)

    if n <= 1:
        yield _final_step("Andrew", pts)
        return

    yield Step(f"Andrew: {n} puntos ordenados de izquierda a derecha.", active=[pts[0], pts[-1]])

    def build(order: List[Point], name: str, fixed: List[Segment]) -> Iterator[Step]:
        """
        Función build
        Recibe: los puntos en el orden en que se recorren, el nombre de la cadena
        y las aristas fixed de la otra cadena, que ya están terminadas
        Devuelve: la cadena armada (como valor de retorno del generador)

        Al llegar un punto se sacan los vertices del final mientras el giro no
        sea convexo, y después se agrega el punto al final de la cadena

        O(n) amortizado en operaciones de la cadena, ya que cada punto entra y
        sale a lo más una vez
        """
        chain: List[Point] = []

        for pt in order:
            if chain:
                yield Step(f"Cadena {name}: llega {_fmt(pt)}.",
                           done_edges=fixed + _polyline(chain), pending_edges=[(chain[-1], pt)],
                           active=[pt], minor=True)

            while len(chain) >= 2 and cross_product(chain[-2], chain[-1], pt) <= EPS:
                popped = chain.pop()
                yield Step(f"Cadena {name}: {_fmt(popped)} no forma un giro convexo; se descarta.",
                           done_edges=fixed + _polyline(chain), pending_edges=[(chain[-1], pt)],
                           active=[pt], removed=[popped])

            chain.append(pt)
            yield Step(f"Cadena {name}: se agrega {_fmt(pt)}.",
                       done_edges=fixed + _polyline(chain), active=[pt])

        return chain

    lower = yield from build(pts, "inferior", [])
    upper = yield from build(list(reversed(pts)), "superior", _polyline(lower))

    # misma unión final que andrew_monotone_chain
    if len(lower) == 2 and len(upper) == 2 and lower[0] == upper[1] and lower[1] == upper[0]:
        hull = lower                        # todos colineales
    else:
        hull = lower[:-1] + upper[:-1]

    yield _final_step("Andrew", hull)


# 3. quickhull

def quickhull_steps(points: List[Point]) -> Iterator[Step]:
    """
    Función quickhull_steps
    Recibe: una lista de puntos
    Devuelve: un generador con los pasos de Quickhull

    Parte con la recta entre el punto menor y el mayor, que divide los puntos
    en dos lados (los que quedan sobre la recta se descartan), y después, para
    cada arista pendiente:

        1. Si no quedan puntos fuera de ella, es arista del hull
        2. Si quedan, el más lejano es vertice del hull y el triángulo que forma
           con la arista descarta los puntos de su interior, asi que la arista se
           reemplaza por las dos nuevas

    Los puntos se recorren en el orden de list(set(points)), igual que en
    algorithms.py, asi que los empates del punto más lejano se resuelven igual

    El algoritmo es O(n log n) en promedio y O(n²) en el peor caso, pero cada
    paso guarda copias de las aristas y los descartados para dibujarlos, asi
    que generar todos los pasos cuesta O(n²)
    """
    pts = list(set(points))
    n = len(pts)

    if n <= 1:
        yield _final_step("Quickhull", pts)
        return

    min_pt, max_pt = min(pts), max(pts)
    upper_set: List[Point] = []
    lower_set: List[Point] = []
    discarded: List[Point] = []

    for pt in pts:
        if pt == min_pt or pt == max_pt:
            continue

        cp = cross_product(min_pt, max_pt, pt)

        if cp > EPS:
            upper_set.append(pt)
        elif cp < -EPS:
            lower_set.append(pt)
        else:
            discarded.append(pt)            # sobre la recta, no puede ser vertice

    done: List[Segment] = []
    pending: List[Segment] = [(min_pt, max_pt)]
    yield Step("Quickhull: la recta entre los extremos divide los puntos en dos lados.",
               pending_edges=list(pending), active=[min_pt, max_pt], discarded=list(discarded))
    pending = [(min_pt, max_pt), (max_pt, min_pt)]      # un lado por cada sentido de la recta

    def find_hull(subset: List[Point], p1: Point, p2: Point) -> Iterator[Step]:
        """
        Función find_hull
        Recibe: los puntos subset que quedan a la izquierda de la arista p1 -> p2
        Devuelve: los vertices del hull entre p1 y p2 (como valor de retorno del generador)

        Es la misma recursión que _find_hull_quickhull de algorithms.py, pero
        además va moviendo las aristas de pending a done y acumulando los
        puntos descartados, para poder dibujarlos

        O(n²) en el peor caso, igual que Quickhull
        """
        if not subset:
            pending.remove((p1, p2))
            done.append((p1, p2))
            yield Step("No quedan puntos fuera de esta arista: es arista del hull.",
                       done_edges=list(done), pending_edges=list(pending),
                       highlight_edges=[(p1, p2)], discarded=list(discarded), minor=True)
            return []

        farthest = max(subset, key=lambda pt: (cross_product(p1, p2, pt), squared_distance(p1, pt)))
        yield Step(f"El punto más lejano de la arista es {_fmt(farthest)}: es vértice del hull.",
                   done_edges=list(done), pending_edges=list(pending),
                   highlight_edges=[(p1, p2)], active=[farthest], discarded=list(discarded))

        left1 = [pt for pt in subset if pt != farthest and cross_product(p1, farthest, pt) > EPS]
        left2 = [pt for pt in subset if pt != farthest and cross_product(farthest, p2, pt) > EPS]

        # lo que no quedó a la izquierda de ninguna de las dos aristas nuevas está dentro del triángulo
        afuera = set(left1) | set(left2)
        inside = [pt for pt in subset if pt != farthest and pt not in afuera]
        discarded.extend(inside)

        pending.remove((p1, p2))
        pending.extend([(p1, farthest), (farthest, p2)])
        yield Step(f"El triángulo descarta {len(inside)} punto(s) interior(es); quedan dos aristas por revisar.",
                   done_edges=list(done), pending_edges=list(pending),
                   highlight_edges=[(p1, farthest), (farthest, p2)], active=[farthest],
                   removed=inside, discarded=list(discarded))

        part1 = yield from find_hull(left1, p1, farthest)
        part2 = yield from find_hull(left2, farthest, p2)
        return part1 + [farthest] + part2

    upper_arc = yield from find_hull(upper_set, min_pt, max_pt)
    lower_arc = yield from find_hull(lower_set, max_pt, min_pt)

    # misma unión final que quickhull
    if not upper_arc and not lower_arc:
        hull = [min_pt, max_pt]             # todos colineales
    else:
        hull = normalize_hull([min_pt] + list(reversed(lower_arc)) + [max_pt] + list(reversed(upper_arc)))

    yield _final_step("Quickhull", hull)


# Mismas claves que ALGORITHMS en convexHullVisual.py
STEP_GENERATORS: Dict[str, Callable[[List[Point]], Iterator[Step]]] = {
    "Jarvis": jarvis_steps,
    "Andrew": andrew_steps,
    "Quickhull": quickhull_steps,
}


def select_frames(steps: List[Step], limit: int) -> List[Step]:
    """
    Función select_frames
    Recibe: la lista de pasos y la cantidad máxima de cuadros limit
    Devuelve: a lo más limit pasos para mostrar

    Si la animación quedaría demasiado larga se reducen los cuadros:

        1. Si los pasos principales (no minor) caben, se conservan todos y el
           espacio que sobra se llena con una muestra pareja de los de detalle
        2. Si ni siquiera caben los principales, se toma una muestra pareja de ellos

    El último paso (hull completo) siempre se conserva
    """
    if len(steps) <= limit:
        return steps

    def sample(indices: List[int], k: int) -> List[int]:
        """
        Función sample
        Recibe: una lista de índices y una cantidad k
        Devuelve: k índices repartidos de forma pareja a lo largo de la lista
        """
        stride = len(indices) / k
        return [indices[int(j * stride)] for j in range(k)]

    last = len(steps) - 1
    major = [i for i in range(last) if not steps[i].minor]
    minor = [i for i in range(last) if steps[i].minor]

    if len(major) + 1 >= limit:
        keep = sample(major, limit - 1)
    else:
        keep = sorted(major + sample(minor, limit - 1 - len(major)))

    return [steps[i] for i in keep] + [steps[last]]
