# Tests de AndrewModificado (andrew_modificado.py)
#
# Después de cada inserción, el hull incremental debe ser igual al de
# andrew_monotone_chain calculado desde cero con todos los puntos insertados
# hasta ese momento
#
# También se revisa que el resultado de insertar (changed, added, removed)
# describa la diferencia entre el hull de antes y el de después
#
# Ejecutar con: python test_incremental.py


import math
import random
from typing import List

from algorithms import Point, andrew_monotone_chain
from andrew_modificado import AndrewModificado


# check_sequence
# inserta los puntos uno a uno y valida después de cada inserción
# los primeros `initial` puntos se entregan al constructor y el resto se inserta de a uno
# devuelve cuántas inserciones se validaron
def check_sequence(name: str, points: List[Point], initial: int = 0) -> int:
    ih = AndrewModificado(points[:initial])
    seen = list(points[:initial])
    assert ih.hull() == andrew_monotone_chain(seen), f"[{name}] constructor incorrecto"

    for step, p in enumerate(points[initial:]):
        before = ih.hull()
        res = ih.insertar(p)
        seen.append(p)
        after = ih.hull()
        expected = andrew_monotone_chain(seen)

        # 1. El hull incremental coincide con Andrew desde cero
        assert after == expected, (
            f"[{name}] paso {step}, insertar{p}:\n"
            f"  incremental: {after}\n  desde cero : {expected}"
        )
        # 2. El resultado de insertar() describe bien el cambio
        set_before, set_after = set(before), set(after)
        assert res.changed == (before != after), f"[{name}] paso {step}: changed incorrecto"
        assert set(res.removed) == set_before - set_after, f"[{name}] paso {step}: removed incorrecto"
        assert len(res.removed) == len(set(res.removed)), f"[{name}] paso {step}: removed con repetidos"
        if res.changed:
            assert res.added == p and p in set_after, f"[{name}] paso {step}: added incorrecto"
        else:
            assert res.added is None and res.removed == [], f"[{name}] paso {step}: no debía cambiar"

    return len(points) - initial


# Generadores de puntos (propios; no dependen de datasets.py)

# Puntos uniformes en un cuadrado (la mayoría termina siendo interior)
def gen_uniform(rng: random.Random, n: int) -> List[Point]:
    return [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(n)]


# Puntos sobre una circunferencia (casi todos son vértices del hull)
def gen_circle(rng: random.Random, n: int) -> List[Point]:
    angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
    return [(round(math.cos(a) * 100, 6), round(math.sin(a) * 100, 6)) for a in angles]


# Grilla entera pequeña (muchos duplicados, colineales y x repetidas)
def gen_grid(rng: random.Random, n: int) -> List[Point]:
    return [(float(rng.randint(-8, 8)), float(rng.randint(-8, 8))) for _ in range(n)]


# Tests

# Casos borde
def run_edge_cases() -> None:
    print("1. Casos borde explícitos")
    cases = {
        "Hull vacío y luego un punto": [(1.0, 1.0)],
        "Mismo punto repetido": [(2.0, 3.0)] * 5,
        "Dos puntos y duplicados": [(0.0, 0.0), (3.0, 1.0), (0.0, 0.0), (3.0, 1.0)],
        "Colineales horizontales (desordenados)": [(2.0, 0.0), (0.0, 0.0), (4.0, 0.0), (1.0, 0.0), (-3.0, 0.0), (9.0, 0.0)],
        "Colineales verticales (misma x)": [(2.0, 5.0), (2.0, 1.0), (2.0, 9.0), (2.0, 3.0), (2.0, -4.0)],
        "Colineales diagonales": [(1.0, 1.0), (3.0, 3.0), (0.0, 0.0), (2.0, 2.0), (5.0, 5.0)],
        "Colineales y luego uno fuera (arriba)": [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0), (1.0, 3.0)],
        "Colineales y luego uno fuera (abajo)": [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0), (3.0, 1.0)],
        "Verticales y luego uno a la izquierda": [(2.0, 0.0), (2.0, 4.0), (2.0, 2.0), (0.0, 1.0)],
        "Verticales y luego uno a la derecha": [(2.0, 0.0), (2.0, 4.0), (2.0, 2.0), (5.0, 1.0)],
        "x menor que todos (extiende a la izquierda)": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (-5.0, 1.0)],
        "x mayor que todos (extiende a la derecha)": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (9.0, 1.0)],
        "Misma x que el extremo izquierdo (abajo y arriba)": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (0.0, -2.0), (0.0, 5.0), (0.0, 1.0)],
        "Misma x que el extremo derecho (abajo y arriba)": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (4.0, -2.0), (4.0, 5.0), (4.0, 1.0)],
        "Punto interior": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 2.0)],
        "Punto sobre una arista": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 2.0)],
        "Duplicado de un vértice": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (4.0, 4.0), (0.0, 0.0)],
        "Un punto que elimina muchos vértices": [(0.0, 0.0), (1.0, -1.0), (2.0, -1.5), (3.0, -1.0), (4.0, 0.0), (2.0, 3.0), (2.0, -50.0)],
        "Punto que envuelve todo el hull": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-10.0, -10.0), (10.0, -10.0), (0.0, 20.0)],
    }
    for name, pts in cases.items():
        # Se prueba con todos los tamaños posibles de construcción inicial
        for initial in range(len(pts) + 1):
            check_sequence(name, pts, initial)
        print(f"  - {name}: OK")

    # Chequeo directo del resultado de insertar() en un caso conocido
    ih = AndrewModificado([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    r = ih.insertar((2.0, 2.0))
    assert (r.changed, r.added, r.removed) == (False, None, [])
    r = ih.insertar((6.0, 2.0))
    assert (r.changed, r.added, r.removed) == (True, (6.0, 2.0), [])
    r = ih.insertar((10.0, 2.0))
    assert r.changed and r.added == (10.0, 2.0) and r.removed == [(6.0, 2.0)]
    r = ih.insertar((-1.0, -1.0))
    assert r.changed and r.removed == [(0.0, 0.0)]
    print("  - Valores devueltos por insertar() en caso conocido: OK")
    print("Casos borde superados.\n")


# Secuencias aleatorias
def run_random_sequences() -> None:
    print("2. Secuencias aleatorias (validación tras CADA inserción)")
    generators = {
        "uniforme": (gen_uniform, 300),
        "circunferencia": (gen_circle, 300),
        "grilla con duplicados": (gen_grid, 300),
    }
    seeds = range(10)
    for name, (gen, n) in generators.items():
        total = 0
        for seed in seeds:
            rng = random.Random(seed)
            pts = gen(rng, n)
            total += check_sequence(f"{name}, seed={seed}", pts)
            # Misma secuencia, pero con la mitad construida en bloque
            total += check_sequence(f"{name}, seed={seed}, inicial=n/2", pts, initial=n // 2)
        print(f"  - {name}: OK ({len(seeds)} seeds, {total} inserciones validadas)")
    print("Secuencias aleatorias superadas.\n")


# Órdenes de inserción adversos
def run_ordered_sequences() -> None:
    print("3. Órdenes de inserción adversos")
    rng = random.Random(2026)
    base = gen_uniform(rng, 300) + gen_circle(rng, 300)
    orders = {
        "x creciente (siempre nuevo máximo)": sorted(base),
        "x decreciente (siempre nuevo mínimo)": sorted(base, reverse=True),
        "de adentro hacia afuera": sorted(base, key=lambda q: q[0] ** 2 + q[1] ** 2),
        "de afuera hacia adentro": sorted(base, key=lambda q: -(q[0] ** 2 + q[1] ** 2)),
    }
    for name, pts in orders.items():
        total = check_sequence(name, pts)
        print(f"  - {name}: OK ({total} inserciones validadas)")
    print("Órdenes adversos superados.\n")


if __name__ == "__main__":
    run_edge_cases()
    run_random_sequences()
    run_ordered_sequences()
    print("TODOS LOS TESTS DE INSERCIÓN INCREMENTAL PASARON.")
