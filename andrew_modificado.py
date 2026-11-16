# Andrew modificado: Convex Hull 2D con inserción incremental de puntos
#
# La idea es no recalcular el hull completo cada vez que llega un punto nuevo,
# para eso se guarda el hull dividido en dos cadenas (la inferior y la
# superior), ambas ordenadas de izquierda a derecha. 
# 
# Al insertar un punto p se busca su posición con búsqueda binaria, se revisa si queda fuera del hull:
#       - SI queda fuera del hull: se agrega y se eliminan los vértices vecinos que p deja cubiertos
#       - SI no queda fuera del hull (cuando p cae dentro o sobre el borde) no sé hace nada
#
# Los puntos interiores no se guardan,  porque el hull solo puede expanderse hacia fuera
# 
# Con h la cantidad de vertices del hull y n todos los puntos :
# cada inserción cuesta O(log h) amortizado en comparaciones y O(h) en el peor caso por modificar la lista.
# Recalcular desde cero costaría O(n log n)


from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Optional

from algorithms import EPS, Point, andrew_monotone_chain_chains, cross_product


# Signo del giro de cada cadena (si es ono vertice)
LOWER_SIGN = 1
UPPER_SIGN = -1


# Resultado de insertar un punto: 
# - changed indica si el hull cambio 
# - added es el punto agregado (None si no cambió) 
# - removed son los vertices que salieron del hull (vacio si nada cambia)
@dataclass
class InsertResult:
    changed: bool
    added: Optional[Point] = None
    removed: List[Point] = field(default_factory=list)


# es_convexo indica si b es un vertice estrictamente convexo entre a y c.
def es_convexo(a: Point, b: Point, c: Point, sign: int) -> bool:
    return sign * cross_product(a, b, c) > EPS


# en_cadena busca p en la cadena con búsqueda binaria
def en_cadena(chain: List[Point], p: Point) -> bool:
    i = bisect_left(chain, p)
    return i < len(chain) and chain[i] == p


class AndrewModificado:

    
    # algorithms.py entrega la cadena superior de derecha a izquierda, se invierte para que ambas
    # cadenas queden de izquierda a derecha.
    def __init__(self, points: Optional[List[Point]] = None) -> None:
        lower, upper = andrew_monotone_chain_chains(list(points) if points else [])
        self._lower: List[Point] = list(lower)
        self._upper: List[Point] = list(reversed(upper))

    # insertar 
    # agrega el punto p y actualiza el hull reparando cada cadena x separado
    def insertar(self, p: Point) -> InsertResult:
        p = (p[0], p[1])

        in_lower, removed_lower = self.insertar_en_cadena(self._lower, p, LOWER_SIGN)
        in_upper, removed_upper = self.insertar_en_cadena(self._upper, p, UPPER_SIGN)

        if not (in_lower or in_upper):
            return InsertResult(changed=False)

        removed: List[Point] = []
        for v in removed_lower + removed_upper:
            if v in removed:
                continue
            if not en_cadena(self._lower, v) and not en_cadena(self._upper, v):
                removed.append(v)

        return InsertResult(changed=True, added=p, removed=removed)

    # insertar_en_cadena 
    # intenta insertar p en una sola cadena y devuelve si
    # quedó insertado junto con los vetices que salieron
    @staticmethod
    def insertar_en_cadena(chain: List[Point], p: Point, sign: int):
        n = len(chain)
        i = bisect_left(chain, p)

        # Duplicado
        if i < n and chain[i] == p:
            return False, []

        # p queda dentro o sobre la cadena
        is_new_extreme = (i == 0 or i == n)
        if not is_new_extreme and not es_convexo(chain[i - 1], p, chain[i], sign):
            return False, []

        # Vecinos a eliminar hacia la izquierda
        lo = i
        while lo >= 2 and not es_convexo(chain[lo - 2], chain[lo - 1], p, sign):
            lo -= 1

        # Vecinos a eliminar hacia la derecha
        hi = i
        while hi + 1 < n and not es_convexo(p, chain[hi], chain[hi + 1], sign):
            hi += 1

        removed = chain[lo:hi]
        chain[lo:hi] = [p]
        return True, removed

    # hull 
    # devuelve los vertices en sentido antihorario, partiendo del punto menor y sin repetir el inicial 
    # recorre la cadena inferior de izq a der y luego la superior de
    # der a izq, sin repetir los extremos que comparten
    def hull(self) -> List[Point]:
        if len(self._lower) <= 1:
            return list(self._lower)

        # Todos los puntos colineales
        if self._lower == self._upper:
            return list(self._lower)

        return self._lower[:-1] + self._upper[:0:-1]

    # Copia de la cadena inferior
    def cadena_inferior(self) -> List[Point]:
        return list(self._lower)

    # Copia de la cadena superior
    def cadena_superior(self) -> List[Point]:
        return list(self._upper)

    # Nº de vértices del hull
    def __len__(self) -> int:
        return len(self.hull())


# Demo
if __name__ == "__main__":
    ih = AndrewModificado([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    print(f"Hull inicial: {ih.hull()}")
    for q in [(2.0, 2.0), (2.0, 0.0), (6.0, 2.0), (-1.0, -1.0), (4.0, 4.0)]:
        r = ih.insertar(q)
        estado = "CAMBIÓ" if r.changed else "no cambió"
        print(f"insertar{q}: {estado}; eliminados={r.removed}; hull={ih.hull()}")
