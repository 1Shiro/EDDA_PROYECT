import math
import random
from algorithms import (
    jarvis_march,
    andrew_monotone_chain,
    quickhull,
    Point,
)


def run_known_small_case() -> None:
    print("1. Caso pequeño conocido")
    # Definimos un hexágono convexo exterior cuyos vértices deben ser el hull exacto en CCW:
    expected_hull = [
        (0.0, 2.0),   # Punto lexicográficamente menor (menor x, menor y)
        (2.0, 0.0),
        (6.0, 0.0),
        (8.0, 3.0),
        (5.0, 7.0),
        (1.0, 6.0),
    ]
    # Agregamos puntos interiores y puntos colineales sobre las aristas
    interior_and_collinear = [
        (3.0, 3.0),   
        (4.0, 2.0),   
        (2.5, 4.5),   
        (5.0, 4.0),   
        (4.0, 0.0),   
        (1.0, 1.0),   
        (3.0, 3.0),   
        (0.0, 2.0),   
    ]

    points = expected_hull + interior_and_collinear
    random.seed(42)
    random.shuffle(points)

    h_jarvis = jarvis_march(points)
    h_andrew = andrew_monotone_chain(points)
    h_quick = quickhull(points)

    print(f"Total de puntos de entrada : {len(points)}")
    print(f"Hull esperado (CCW)        : {expected_hull}")
    print(f"Jarvis March               : {h_jarvis}")
    print(f"Andrews Monotone Chain    : {h_andrew}")
    print(f"Quickhull                  : {h_quick}")

    assert h_jarvis == expected_hull, f"Fallo en Jarvis March: {h_jarvis} != {expected_hull}"
    assert h_andrew == expected_hull, f"Fallo en Andrew: {h_andrew} != {expected_hull}"
    assert h_quick == expected_hull, f"Fallo en Quickhull: {h_quick} != {expected_hull}"
    print("Los 3 algoritmos coinciden exactamente con el hull.\n")


def run_edge_cases() -> None:
    print("2. Casos bordes (Edge Cases)")

    edge_cases = {
        "Vacío": ([], []),
        "Un solo punto": ([(5.0, 5.0)], [(5.0, 5.0)]),
        "Puntos todos idénticos": ([(2.0, 3.0)] * 10, [(2.0, 3.0)]),
        "Dos puntos distintos": ([(5.0, 5.0), (1.0, 2.0)], [(1.0, 2.0), (5.0, 5.0)]),
        "Cinco puntos colineales diagonales": (
            [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0), (1.0, 1.0), (3.0, 3.0)],
            [(0.0, 0.0), (4.0, 4.0)],
        ),
        "Cinco puntos colineales verticales": (
            [(2.0, 5.0), (2.0, 1.0), (2.0, 9.0), (2.0, 3.0)],
            [(2.0, 1.0), (2.0, 9.0)],
        ),
        "Cuadrado con puntos en bordes e interior": (
            [
                (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
                (5.0, 0.0), (10.0, 5.0), (5.0, 10.0), (0.0, 5.0),
                (5.0, 5.0), (2.0, 8.0),
            ],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        ),
    }

    for name, (pts, expected) in edge_cases.items():
        hj = jarvis_march(pts)
        ha = andrew_monotone_chain(pts)
        hq = quickhull(pts)
        assert hj == expected, f"[{name}] Jarvis dio {hj}, esperaba {expected}"
        assert ha == expected, f"[{name}] Andrew dio {ha}, esperaba {expected}"
        assert hq == expected, f"[{name}] Quickhull dio {hq}, esperaba {expected}"
        print(f"  - {name}: OK ({len(hj)} vértices)")

    print("Todos los casos bordes superados con éxito.\n")


def run_random_cross_validation() -> None:
    print("Validación cruzada en datasets aleatorios")
    rng = random.Random(12345)

    # Probamos varias distribuciones para verificar que los 3 algoritmos coinciden siempre
    for trial in range(1, 21):
        n = rng.randint(25, 500)
        # Alternamos entre puntos uniformes en cuadrado, en círculo y en grilla entera
        mode = trial % 3
        if mode == 0:
            pts = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(n)]
        elif mode == 1:
            # Puntos sobre una circunferencia (todos en el hull)
            angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
            pts = [(round(math.cos(a) * 100, 6), round(math.sin(a) * 100, 6)) for a in angles]
        else:
            # Grilla entera
            pts = [(float(rng.randint(-20, 20)), float(rng.randint(-20, 20))) for _ in range(n)]

        hj = jarvis_march(pts)
        ha = andrew_monotone_chain(pts)
        hq = quickhull(pts)

        assert hj == ha == hq, (
            f"Discrepancia en trial {trial} (modo={mode}):\n"
            f"  Jarvis   ({len(hj)}): {hj}\n"
            f"  Andrew   ({len(ha)}): {ha}\n"
            f"  Quickhull({len(hq)}): {hq}"
        )

    print("20 pruebas aleatorias validadas:")
    print("Jarvis March == Andrew's Monotone Chain == Quickhull en el 100% de los casos.")


if __name__ == "__main__":
    run_known_small_case()
    run_edge_cases()
    run_random_cross_validation()
