"""
Visualizador de puntos 2D y su Convex Hull (matplotlib)

Recibe: 
- Un conjunto de puntos que puede venir de un archivo (--points)
- Un conjunto aleatorio de puntos al usoar el botón Generar
- Un conjunto ingresado como texto en la barra inferior
i- UN punto al hacer clicks sobre el gráfico

El hull se calcula desde cero con el algoritmo seleccionado (Jarvis, Andrew o
Quickhull), y hay dos modos:

    1. Animación OFF: se muestra directamente el hull resultante
    2. Animación ON: se muestra paso a paso cómo se arma el hull, con los
       pasos que entrega pasoapaso.py

La inserción de puntos usa AndrewModificado.insertar y es solo con Andrew
seleccionado, asi que con otro algoritmo los botones Agregar puntos e Insertar
punto muestran un aviso y el modo inserción no se puede activar, y al insertar
se resalta el punto nuevo, los vertices eliminados y el hull anterior

Se puede guardar la imagen actual como PNG con el botón Guardar PNG

Uso :
    python convexHullVisual.py                         parte con puntos aleatorios
    python convexHullVisual.py --points puntos.csv     parte con los puntos de un archivo
    python convexHullVisual.py --animate               parte con la animación activa
    python convexHullVisual.py --algorithm jarvis      parte con otro algoritmo (jarvis, andrew o quickhull)

El archivo de puntos lleva un punto por línea, "x,y" o "x y"

"""

import argparse
import os
import random
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from pasoapaso import STEP_GENERATORS, Step, select_frames
from algorithms import Point, andrew_monotone_chain, jarvis_march, quickhull
from andrew_modificado import AndrewModificado, InsertResult

# Algoritmos disponibles para calcular desde cero
ALGORITHMS: Dict[str, Callable[[List[Point]], List[Point]]] = {
    "Jarvis": jarvis_march,
    "Andrew": andrew_monotone_chain,
    "Quickhull": quickhull,
}

COORD_MIN, COORD_MAX = 0.0, 100.0   # rango de los puntos aleatorios
MARGIN = 12.0                       # margen visible alrededor, para poder insertar fuera del hull
DEFAULT_VIEW = (COORD_MIN - MARGIN, COORD_MAX + MARGIN, COORD_MIN - MARGIN, COORD_MAX + MARGIN)
DEFAULT_N = 30                      # cant de puntos aleatorios por defecto
STEP_SECONDS = 0.4                  # duración de cada paso de la animación en pantalla
OUTPUT_DIR = "figuras"              # carpeta por defecto para los PNG
MAX_SCREEN_FRAMES = 600             # sobre esto, en pantalla se omiten pasos de detalle


# Estilo
TEXT_COLOR = "#0B2A5B"       # azul oscuro: todo el texto
BTN_COLOR = "#BFE6FF"        # celeste: botones
BTN_HOVER = "#96D3F7"        # celeste más intenso: botón bajo el mouse
BTN_ACTIVE = "#5BB8F0"       # botón "encendido" (modo inserción ON)
BTN_EDGE = "#7CC4EE"         # borde de botones y cuadros
PANEL_COLOR = "#EEF8FF"      # fondo del panel de controles
HULL_COLOR = "#1976D2"       # borde y vertices del hull
HULL_FILL = "#7CC4EE"        # relleno del hull
POINT_COLOR = "#4A5F85"      # puntos
POINT_FADED = "#C9D6E8"      # puntos ya descartados (animación)
PENDING_COLOR = "#8FA3BF"    # líneas punteadas (hull anterior / en revisión)
ERROR_COLOR = "#C62828"      # mensajes de error y vertices eliminados


def apply_style() -> None:
    """
    Función apply_style

    Deja configurado el estilo :

        1. Pone todo el texto y los ejes en azul oscuro
        2. Quita todos los atajos de teclado de matplotlib (guardar, grilla,
           pantalla completa, cerrar y otros), para que todo se haga con botones
    """
    for key in ("text.color", "axes.labelcolor", "axes.edgecolor", "xtick.color", "ytick.color"):
        plt.rcParams[key] = TEXT_COLOR

    plt.rcParams["grid.color"] = BTN_EDGE

    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = []


def parse_points(text: str) -> List[Point]:
    """
    Función parse_points
    Recibe: un texto con puntos
    Devuelve: la lista de puntos que contiene

    Acepta "x,y" para un punto o "(x,y),(x,y),..." para varios, en tres pasos:

        1. Los paréntesis, comas y punto y coma pasan a ser espacios
        2. Cada parte del texto se convierte a número
        3. Los números se juntan de a dos: (x, y)

    Si a un punto le falta una coordenada lanza ValueError, y si hay algo que
    no es número lo lanza float
    """
    for simbolo in "(),;":
        text = text.replace(simbolo, " ")

    numeros = [float(parte) for parte in text.split()]

    if len(numeros) % 2 != 0:
        raise ValueError("a un punto le falta una coordenada")

    return [(numeros[i], numeros[i + 1]) for i in range(0, len(numeros), 2)]


def clipboard_text() -> str:
    """
    Función clipboard_text
    Devuelve: el texto del portapapeles, o "" si no se puede leer

    Lo usa el botón Pegar e intenta primero con Qt y después con Tk, según el
    backend con que se abrió la ventana
    """
    try:
        from matplotlib.backends.qt_compat import QtWidgets
        app = QtWidgets.QApplication.instance()

        if app is not None:
            return app.clipboard().text() or ""
    except Exception:
        pass

    try:
        import tkinter
        root = tkinter._default_root

        if root is not None:
            return root.clipboard_get() or ""
    except Exception:
        pass

    return ""


def random_points(n: int, rng: random.Random) -> List[Point]:
    """
    Función random_points
    Recibe: la cantidad n de puntos y el generador rng
    Devuelve: n puntos uniformes en el cuadrado [COORD_MIN, COORD_MAX]
    """
    return [(rng.uniform(COORD_MIN, COORD_MAX), rng.uniform(COORD_MIN, COORD_MAX)) for _ in range(n)]


def load_points(path: str) -> List[Point]:
    """
    Función load_points
    Recibe: la ruta de un archivo de texto
    Devuelve: los puntos del archivo

    Lee un punto por línea, "x,y" o "x y", e ignora las líneas vacías, los
    comentarios (lo que va después de #) y las líneas que no son números,
    como un encabezado
    """
    points: List[Point] = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split("#")[0].replace(",", " ").replace(";", " ").split()

            try:
                points.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue

    return points


class HullVisualizer:
    """
    Clase HullVisualizer

    Guarda el estado del visualizador (puntos, hull, inserción y animación) y
    maneja el dibujo y los controles de la ventana
    """

    def __init__(self, n: int = DEFAULT_N, algorithm: str = "Andrew", animate: bool = False) -> None:
        self.n = n

        # Estado
        self.points: List[Point] = []                   # todos los puntos (incluye interiores)
        self.ihull = AndrewModificado()                 
        self.hull: List[Point] = []                     # hull que se está mostrando
        self.algorithm = algorithm                      # algoritmo para calcular desde cero
        self.hull_source = f"{algorithm} desde cero"    # cómo se obtuvo el hull mostrado
        self.view = DEFAULT_VIEW                        # límites visibles 
        self.insert_mode = False                        # click en el gráfico = insertar punto

        # Datos de la última inserción, para resaltarla (None si no aplica)
        self.last_result: Optional[InsertResult] = None
        self.last_point: Optional[Point] = None
        self.prev_hull: List[Point] = []

        # Animación de la construcción
        self.animate = animate                          # true = con animación
        self.steps: List[Step] = []                     # cuadros de la animación en curso
        self.step_idx = 0                               # cuadro que se está mostrando
        self.in_animation = False                       # se está mostrando un paso intermedio
        self.playing = False                            # la animación avanza sola (no está en pausa)
        self.message = "Presiona 'Generar', escribe puntos abajo o activa el modo inserción."
        self.message_color = TEXT_COLOR

        self._build_figure()
        self.redraw()


    # armado de la ventana y sus botones

    def _build_figure(self) -> None:
        """
        Función _build_figure

        Arma la ventana: el gráfico a la derecha, el panel de controles a la
        izquierda (selector de algoritmo, slider y botones) y la barra inferior
        para ingresar puntos como texto, y además crea el temporizador de la
        animación y conecta el click del mouse
        """
        apply_style()

        self.fig = plt.figure(figsize=(12.5, 7.6), facecolor="white")
        self.fig.canvas.manager.set_window_title("Convex Hull 2D - visualizador")
        self.ax = self.fig.add_axes([0.33, 0.27, 0.64, 0.65])

        # Fondo del panel de controles (columna izquierda)
        self.fig.add_artist(FancyBboxPatch(
            (0.015, 0.03), 0.265, 0.94, boxstyle="round,pad=0,rounding_size=0.012",
            transform=self.fig.transFigure, facecolor=PANEL_COLOR, edgecolor=BTN_EDGE,
            linewidth=1.0, zorder=0))
        self.fig.text(0.1475, 0.935, "Convex Hull 2D", ha="center", va="center",
                      fontsize=14, fontweight="bold")

        def frame(ax) -> None:
            """
            Función frame
            Recibe: unos ejes ax

            Les pone el borde celeste del visualizador
            """
            for spine in ax.spines.values():
                spine.set_edgecolor(BTN_EDGE)
                spine.set_linewidth(1.0)

        # Selector de algoritmo
        ax_radio = self.fig.add_axes([0.03, 0.765, 0.235, 0.115], facecolor="white")
        frame(ax_radio)
        ax_radio.set_title("Algoritmo (desde cero)", fontsize=10)
        self.radio = RadioButtons(
            ax_radio, list(ALGORITHMS), active=list(ALGORITHMS).index(self.algorithm),
            activecolor=HULL_COLOR,
            label_props={"color": [TEXT_COLOR] * 3, "fontsize": [10] * 3},
            radio_props={"s": [55] * 3, "edgecolor": [TEXT_COLOR] * 3})
        self.radio.on_clicked(self.set_algorithm)

        def slider(y: float, label: str, vmin: float, vmax: float, value: float) -> Slider:
            """
            Función slider
            Recibe: la altura y, la etiqueta, el rango [vmin, vmax] y el valor inicial
            Devuelve: un Slider de valores enteros
            """
            widget = Slider(self.fig.add_axes([0.105, y, 0.115, 0.022]), label, vmin, vmax,
                            valinit=value, valstep=1, color=BTN_ACTIVE, track_color="white",
                            handle_style={"facecolor": HULL_COLOR, "edgecolor": TEXT_COLOR, "size": 9})
            widget.label.set_fontsize(9.5)
            widget.valtext.set_fontsize(9.5)
            return widget

        self.slider = slider(0.715, "N puntos", 3, 500, self.n)
        self.slider.on_changed(self._on_slider)

        def button(rect: List[float], label: str, action: Callable[[], object]) -> Button:
            """
            Función button
            Recibe: la posición rect, el texto del botón y la acción que ejecuta
            Devuelve: un boton
            """
            ax = self.fig.add_axes(rect)
            frame(ax)
            btn = Button(ax, label, color=BTN_COLOR, hovercolor=BTN_HOVER)
            btn.label.set_color(TEXT_COLOR)
            btn.label.set_fontsize(9.5)
            btn.on_clicked(lambda _event: action())
            return btn

        def textbox(rect: List[float], hint: str = "") -> TextBox:
            """
            Función textbox
            Recibe: la posición rect y un texto de ayuda hint (opcional)
            Devuelve: un cuadro de texto vacío

            El texto de ayuda se ve en gris dentro del cuadro mientras está vacío,
            se borra al hacer click para escribir y vuelve a aparecer si el
            cuadro queda vacío
            """
            ax = self.fig.add_axes(rect)
            frame(ax)
            box = TextBox(ax, "", initial="", color="white", hovercolor="#F4FBFF")
            box.text_disp.set_color(TEXT_COLOR)
            box.text_disp.set_fontsize(10)
            box.text_disp.set_clip_on(True)             # un texto largo no se sale del cuadro
            box.cursor.set_color(TEXT_COLOR)

            if hint:
                ayuda = ax.text(0.02, 0.5, hint, transform=ax.transAxes, va="center",
                                fontsize=9, color=PENDING_COLOR, clip_on=True)

                def al_hacer_click(event) -> None:
                    if event.inaxes is ax:
                        ayuda.set_visible(False)
                        self.fig.canvas.draw_idle()

                self.fig.canvas.mpl_connect("button_press_event", al_hacer_click)
                box.on_submit(lambda texto: ayuda.set_visible(texto == ""))

            return box

        # Botones del panel izquierdo
        x, w, h = 0.03, 0.235, 0.048
        self.btn_generate = button([x, 0.645, w, h], "Generar N puntos", self.generate)
        self.btn_clear = button([x, 0.583, w, h], "Limpiar", self.clear)
        self.btn_anim = button([x, 0.501, w, h], "", lambda: self.set_animate(not self.animate))

        # sin animación se ve "Calcular hull" y con animación se ven "Empezar / Pausar construcción" al lado, "Paso" 
        # para los pasos xd
        self.btn_compute = button([x, 0.439, w, h], "Calcular hull", self.compute)
        self.btn_start = button([x, 0.439, 0.158, h], "", self.start_or_pause)
        self.btn_step = button([x + 0.165, 0.439, w - 0.165, h], "Paso  →", self.step_forward)
        self.btn_start.label.set_fontsize(9)
        self.btn_insert = button([x, 0.357, w, h], "", self.toggle_insert_mode)
        self.btn_save = button([x, 0.275, w, h], "Guardar PNG", self.save_png)
        self.fig.text(0.1475, 0.205,
                      "Insertar puntos: SOLO con Andrew\n(click en el gráfico o barra inferior)",
                      ha="center", va="center", fontsize=8.5, alpha=0.75)

        # Semilla de los puntos aleatorios: Generar usa la que está escrita y
        # Nueva semilla escribe otra al azar y genera con ella
        self.fig.text(0.03, 0.1225, "Semilla", va="center", fontsize=10)
        self.box_seed = textbox([0.085, 0.10, 0.075, 0.045])
        self.btn_seed = button([0.165, 0.10, 0.10, 0.045], "Nueva semilla", self.new_seed)
        self.fig.text(0.1475, 0.06, "Generar usa la semilla escrita,\n'Nueva semilla' elige otra al azar",
                      ha="center", va="center", fontsize=8, alpha=0.75)

        # Barra inferior: ingreso de puntos por texto
        self.fig.text(0.33, 0.1075, "Lista de puntos", va="center", fontsize=10)
        self.box_list = textbox([0.425, 0.085, 0.325, 0.045], "ej: (10,20),(30,40),(50,15)")
        self.btn_paste = button([0.757, 0.085, 0.065, 0.045], "Pegar", self._paste_list)
        self.btn_add = button([0.83, 0.085, 0.14, 0.045], "Agregar puntos", self.add_points_from_text)
        self.fig.text(0.33, 0.0475, "Un punto", va="center", fontsize=10)
        self.box_point = textbox([0.425, 0.025, 0.16, 0.045], "ej: 50,60")
        self.btn_insert_point = button([0.593, 0.025, 0.14, 0.045], "Insertar punto",
                                       self.insert_point_from_text)

        # temporizador que hace avanzar la animación en la ventana
        self._timer = self.fig.canvas.new_timer(interval=int(1000 * STEP_SECONDS))
        self._timer.add_callback(self._tick)

       
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)


    # lo que hace cada acción

    def _clear_highlight(self) -> None:
        """
        Función _clear_highlight

        Borra los datos de la última inserción, para que ya no se resalte
        """
        self.last_result = None
        self.last_point = None
        self.prev_hull = []

    def _stop_animation(self) -> None:
        """
        Función _stop_animation

        Corta la animación en curso y, si quedó a medias, calcula el hull
        completo para no dejar un hull incompleto en pantalla
        """
        if self.in_animation:
            self.hull = ALGORITHMS[self.algorithm](self.points)
            self.hull_source = f"{self.algorithm} desde cero"

        self.in_animation = False
        self.playing = False
        self._timer.stop()

    def _fit_view(self) -> None:
        """
        Función _fit_view

        Ajusta el área visible:

            1. Si todos los puntos caben en el área por defecto, se usa esa
            2. si no, se encuadran todos los puntos dejando un margen del 12%
        """
        x0, x1, y0, y1 = DEFAULT_VIEW

        if all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in self.points):
            self.view = DEFAULT_VIEW
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        pad = 0.12 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        self.view = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def set_points(self, points: List[Point]) -> None:
        """
        Función set_points
        Recibe: una lista de puntos

        Reemplaza el conjunto actual por estos puntos, deja lista la estructura
        incremental (AndrewModificado) y calcula su hull
        """
        self._stop_animation()
        self.points = list(points)
        self.ihull = AndrewModificado(self.points)
        self._fit_view()
        self.compute()

    def generate(self) -> None:
        """
        Función generate

        Botón Generar: genera N puntos aleatorios (N es el valor del slider) con
        la semilla escrita en el cuadro Semilla, asi que la misma semilla da
        siempre los mismos puntos, y si solo se cambia N, los primeros puntos
        se mantienen y se agregan o quitan los del final
        """
        try:
            semilla = int(self.box_seed.text)
        except ValueError:
            self.message = "La semilla debe ser un número entero."
            self.message_color = ERROR_COLOR
            self.redraw()
            return

        self.set_points(random_points(self.n, random.Random(semilla)))

    def new_seed(self) -> None:
        """
        Función new_seed

        Botón Nueva semilla: escribe una semilla al azar en el cuadro Semilla y
        genera los puntos con ella
        """
        self.box_seed.set_val(str(random.randrange(1000000)))
        self.generate()

    def clear(self) -> None:
        """
        Función clear

        Borra todos los puntos y deja el visualizador vacío
        """
        self._stop_animation()
        self.points = []
        self.ihull = AndrewModificado()
        self.hull = []
        self.hull_source = f"{self.algorithm} desde cero"
        self.view = DEFAULT_VIEW
        self._clear_highlight()
        self.message = "Conjunto vacío."
        self.message_color = TEXT_COLOR
        self.redraw()

    def set_algorithm(self, name: str) -> None:
        """
        Función set_algorithm
        Recibe: el nombre del algoritmo

        Cambia el algoritmo y calcula desde cero sobre el conjunto actual, y
        como la inserción solo existe para Andrew, con otro algoritmo se apaga
        el modo inserción
        """
        self.algorithm = name

        if name != "Andrew":
            self.insert_mode = False

        self.compute()

    def compute(self) -> None:
        """
        Función compute

        Calcula el hull desde cero con el algoritmo seleccionado:

            1. Animación OFF: muestra directamente el resultado
            2. Animación ON: genera todos los pasos y deja la construcción en
               el primero, detenida
        """
        self._stop_animation()
        self._clear_highlight()
        self.message_color = TEXT_COLOR

        if self.animate and self.points:
            steps = list(STEP_GENERATORS[self.algorithm](self.points))
            self.steps = select_frames(steps, MAX_SCREEN_FRAMES)
            self.step_idx = 0
            self.in_animation = True
            self.playing = False
        else:
            self.hull = ALGORITHMS[self.algorithm](self.points)
            self.hull_source = f"{self.algorithm} desde cero"
            self.message = (f"{len(self.points)} puntos. Hull calculado desde cero con "
                            f"{self.algorithm}: {len(self.hull)} vértice(s).")

        self.redraw()

    def set_animate(self, value: bool) -> None:
        """
        Función set_animate
        Recibe: True para activar la animación, False para desactivarla

        """
        self.animate = value
        self._stop_animation()
        self.message_color = TEXT_COLOR

        if value:
            self.message = "Animación ON: presiona 'Empezar construcción' o avanza con 'Paso'."

            if self.points:
                self.compute()
                return
        else:
            self.message = "Animación OFF: se muestra directamente el hull resultante."

        self.redraw()

    def start_or_pause(self) -> None:
        """
        Función start_or_pause

        Botón Empezar / Pausar / Reanudar construcción (animacion ON):

            1. Si no hay puntos, avisa y no hace nada
            2. Si la construcción ya termino o se interrumpr, parte de nuevo
            3. Si no, pausa o reanuda la animación
        """
        if not self.points:
            self.message = "No hay puntos: genera o agrega puntos antes de empezar."
            self.message_color = TEXT_COLOR
            self.redraw()
            return

        if not self.in_animation:
            self.compute()

        self.toggle_pause()

    def toggle_pause(self) -> None:
        """
        Función toggle_pause

        Pausa o reanuda la animación en curso, y al reanudar, el primer paso se
        muestra de inmediato y los siguientes cada STEP_SECONDS
        """
        self.playing = not self.playing

        if self.playing:
            self.step_forward()

            if self.in_animation:
                self._timer.start()
        else:
            self._timer.stop()
            self.redraw()

    def step_forward(self) -> None:
        """
        Función step_forward

        Avanza un paso de la construcción:

            1. Si no hay construcción en curso, la deja lista en su primer paso
            2. Si se llegó al penúltimo paso, termina la animación
            3. Si no, pasa al siguiente paso
        """
        if not self.in_animation:
            self.compute()
            return

        if self.step_idx + 1 >= len(self.steps) - 1:
            self._finish_animation()
        else:
            self.step_idx += 1

        self.redraw()

    def _finish_animation(self) -> None:
        """
        Función _finish_animation

        Deja en pantalla el hull final de la animación, que viene en el último paso
        """
        self.in_animation = False
        self.playing = False
        self._timer.stop()
        self.hull = self.steps[-1].hull
        self.hull_source = f"{self.algorithm} desde cero"
        self.message = (f"Animación terminada. {self.algorithm}: {len(self.hull)} vértice(s) "
                        f"sobre {len(self.points)} puntos.")

    def _tick(self) -> None:
        """
        Función _tick

        La llama el temporizador cada STEP_SECONDS y avanza la animación un
        paso, si no está en pausa
        """
        if self.in_animation and self.playing:
            self.step_forward()

    def _insertion_blocked(self) -> bool:
        """
        Función _insertion_blocked
        Devuelve: True si no se puede insertar, False si se puede

        La inserción solo está implementada sobre Andrew, asi que con otro
        algoritmo seleccionado se muestra un aviso en rojo y se devuelve True
        """
        if self.algorithm == "Andrew":
            return False

        self.message = ("La inserción de puntos solo está disponible con Andrew.\n"
                        f"Selecciona Andrew para insertar (ahora: {self.algorithm}).")
        self.message_color = ERROR_COLOR
        self.redraw()
        return True

    def toggle_insert_mode(self) -> None:
        """
        Función toggle_insert_mode

        Activa o desactiva el modo inserción, en el que un click sobre el
        gráfico inserta un punto, y que solo se puede activar con Andrew
        """
        if not self.insert_mode and self._insertion_blocked():
            return

        self.insert_mode = not self.insert_mode
        self.message = ("Modo inserción ACTIVADO: haz click en el gráfico para insertar un punto."
                        if self.insert_mode else "Modo inserción desactivado.")
        self.message_color = TEXT_COLOR
        self.redraw()

    def insert_point(self, p: Point) -> None:
        """
        Función insert_point
        Recibe: un punto p

        Inserta p con AndrewModificado.insertar, sin recalcular desde cero, y
        deja resaltado el resultado: el punto nuevo, los vertices eliminados y
        el hull anterior
        """
        self._stop_animation()
        self.prev_hull = self.ihull.hull()
        result = self.ihull.insertar(p)                 # <-- inserción incremental
        self.points.append(p)
        self.hull = self.ihull.hull()
        self.hull_source = "incremental"
        self.last_result = result
        self.last_point = p

        if result.changed:
            text = f"El hull CAMBIÓ: se agregó el punto y se eliminaron {len(result.removed)} vértice(s)."
        else:
            text = "El hull NO cambió: el punto quedó dentro o sobre el borde."

        self.message = text
        self.message_color = TEXT_COLOR
        self._fit_view()
        self.redraw()

    def add_points(self, new_points: List[Point]) -> None:
        """
        Función add_points
        Recibe: una lista de puntos nuevos

        Inserta los puntos uno por uno con AndrewModificado.insertar y muestra
        cuántos de ellos cambiaron el hull
        """
        self._stop_animation()
        self._clear_highlight()
        changed = 0

        for p in new_points:
            changed += bool(self.ihull.insertar(p).changed)
            self.points.append(p)

        self.hull = self.ihull.hull()
        self.hull_source = "incremental"
        self.message = (f"Se agregaron {len(new_points)} punto(s); {changed} cambiaron el hull. "
                        f"Total: {len(self.points)} puntos, {len(self.hull)} vértice(s).")
        self.message_color = TEXT_COLOR
        self._fit_view()
        self.redraw()

    def _input_error(self, detail: str) -> None:
        """
        Función _input_error
        Recibe: el detalle del error

        Muestra en rojo un error de formato en los cuadros de texto
        """
        self.message = f"Entrada no válida: {detail}.\nFormato: (x,y),(x,y),…  o  x,y para un punto."
        self.message_color = ERROR_COLOR
        self.redraw()

    def add_points_from_text(self) -> None:
        """
        Función add_points_from_text

        Botón Agregar puntos (solo con Andrew): lee "(x,y),(x,y),..." del cuadro
        Lista de puntos y agrega esos puntos con add_points, pero si el formato
        no es válido, muestra el error
        """
        if self._insertion_blocked():
            return

        try:
            new_points = parse_points(self.box_list.text)
        except ValueError as error:
            self._input_error(str(error))
            return

        self.box_list.set_val("")
        self.add_points(new_points)

    def insert_point_from_text(self) -> None:
        """
        Función insert_point_from_text

        Botón Insertar punto (solo con Andrew): lee una coordenada "x,y" del
        cuadro Un punto y la inserta con insert_point, pero si el formato no es
        válido o vienen varios puntos, muestra el error
        """
        if self._insertion_blocked():
            return

        try:
            new_points = parse_points(self.box_point.text)
        except ValueError as error:
            self._input_error(str(error))
            return

        if len(new_points) != 1:
            self._input_error("aquí va un solo punto; para varios usa 'Lista de puntos'")
            return

        self.box_point.set_val("")
        self.insert_point(new_points[0])

    def _paste_list(self) -> None:
        """
        Función _paste_list

        Botón Pegar: copia el portapapeles al cuadro Lista de puntos, juntando
        todo en una sola línea, y si el portapapeles está vacío, lo avisa
        """
        text = " ".join(clipboard_text().split())

        if text:
            self.box_list.set_val(text)
        else:
            self.message = "El portapapeles está vacío o no se pudo leer."
            self.message_color = TEXT_COLOR
            self.redraw()


    # guardar la imagen

    def save_png(self) -> None:
        """
        Función save_png

        Botón Guardar PNG: guarda lo que se está mostrando en la carpeta figuras,
        recortado al gráfico (ejes, título y etiquetas), sin el panel de botones

        Los archivos se llaman CH_<algoritmo>_<numero>.png, por ejemplo
        CH_Andrew_1.png, CH_Andrew_2.png, y se usa el primer numero que no
        exista todavia, asi que no se sobrescribe ninguna imagen anterior
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        numero = 1

        while os.path.exists(os.path.join(OUTPUT_DIR, f"CH_{self.algorithm}_{numero}.png")):
            numero += 1

        path = os.path.join(OUTPUT_DIR, f"CH_{self.algorithm}_{numero}.png")

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        tight = self.ax.get_tightbbox(renderer)         # ejes + título + etiquetas (en píxeles)
        pad = 10
        padded = Bbox.from_extents(tight.x0 - pad, tight.y0 - pad, tight.x1 + pad, tight.y1 + pad)
        bbox = padded.transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(path, dpi=150, bbox_inches=bbox)
        print(f"Figura guardada en {path}")


    # dibujo del gráfico

    def redraw(self) -> None:
        """
        Función redraw

        Redibuja la ventana: el paso actual si hay una animación en curso, o el
        estado normal si no, y además actualiza el texto y el color de los
        botones según el modo, y muestra "Calcular hull" o "Empezar / Paso"
        según si la animación está activa
        """
        if self.in_animation:
            self._draw(self.ax, self.steps[self.step_idx], self.step_idx, len(self.steps))
        else:
            self._draw(self.ax, None)

        self.btn_insert.label.set_text(f"Modo inserción: {'ON' if self.insert_mode else 'OFF'}")
        self.btn_insert.color = BTN_ACTIVE if self.insert_mode else BTN_COLOR
        self.btn_insert.hovercolor = BTN_ACTIVE if self.insert_mode else BTN_HOVER
        self.btn_insert.ax.set_facecolor(self.btn_insert.color)
        self.btn_anim.label.set_text(f"Animación: {'ON' if self.animate else 'OFF'}")
        self.btn_anim.color = BTN_ACTIVE if self.animate else BTN_COLOR
        self.btn_anim.hovercolor = BTN_ACTIVE if self.animate else BTN_HOVER
        self.btn_anim.ax.set_facecolor(self.btn_anim.color)

        if self.playing:
            start_label = "Pausar construcción"
        elif self.in_animation and self.step_idx > 0:
            start_label = "Reanudar construcción"
        else:
            start_label = "Empezar construcción"

        self.btn_start.label.set_text(start_label)

        for btn, visible in ((self.btn_compute, not self.animate),
                             (self.btn_start, self.animate), (self.btn_step, self.animate)):
            btn.ax.set_visible(visible)
            btn.set_active(visible)

        self.fig.canvas.draw_idle()

    def _draw(self, ax, step: Optional[Step], idx: int = 0, total: int = 0) -> None:
        """
        Función _draw
        Recibe: los ejes ax donde dibujar, el paso step (o None), y su posición
        idx dentro de los total pasos

        Limpia los ejes, fija el área visible y dibuja:

            1. Con step=None, el estado normal (puntos, hull y última inserción)
            2. Con un Step, ese cuadro de la animación
        """
        ax.clear()
        ax.set_xlim(self.view[0], self.view[1])
        ax.set_ylim(self.view[2], self.view[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)

        if step is None:
            self._draw_state(ax)
        else:
            self._draw_step(ax, step, idx, total)

    def _draw_state(self, ax) -> None:
        """
        Función _draw_state
        Recibe: los ejes ax donde dibujar

        Dibuja el estado normal: los puntos, el hull y, si hubo una inserción,
        el hull anterior (punteado), los vertices eliminados (x roja) y el
        punto nuevo (estrella verde si cambió el hull, naranja si no)
        """
        # hull anterior a la última inserción (solo si cambió)
        if self.last_result is not None and self.last_result.changed and len(self.prev_hull) >= 2:
            xs = [q[0] for q in self.prev_hull] + [self.prev_hull[0][0]]
            ys = [q[1] for q in self.prev_hull] + [self.prev_hull[0][1]]
            ax.plot(xs, ys, linestyle="--", color=PENDING_COLOR, linewidth=1.2, label="Hull anterior", zorder=2)

        if self.points:
            ax.scatter([q[0] for q in self.points], [q[1] for q in self.points],
                       s=18, color=POINT_COLOR, label="Puntos", zorder=3)

        self._draw_hull(ax, self.hull, label="Hull")

        # resaltado de la última inserción
        if self.last_result is not None:
            if self.last_result.removed:
                ax.scatter([q[0] for q in self.last_result.removed],
                           [q[1] for q in self.last_result.removed],
                           s=140, marker="x", color=ERROR_COLOR, linewidths=2.5,
                           label="Vértices eliminados", zorder=6)

            color = "tab:green" if self.last_result.changed else "tab:orange"
            label = "Punto nuevo (cambia el hull)" if self.last_result.changed else "Punto nuevo (no cambia el hull)"
            ax.scatter([self.last_point[0]], [self.last_point[1]], s=220, marker="*",
                       color=color, edgecolors=TEXT_COLOR, label=label, zorder=7)

        ax.set_title(f"{len(self.points)} puntos | {len(self.hull)} vértices en el hull | hull: {self.hull_source}",
                     fontsize=11)
        ax.set_xlabel(self.message, fontsize=10, color=self.message_color)

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    @staticmethod
    def _draw_hull(ax, hull: List[Point], label: Optional[str] = None) -> None:
        """
        Función _draw_hull
        Recibe: los ejes ax, los vertices del hull y una etiqueta opcional para la leyenda

        Dibuja el hull como polígono cerrado con relleno suave y los vertices destacados
        """
        if len(hull) >= 2:
            xs = [q[0] for q in hull] + [hull[0][0]]
            ys = [q[1] for q in hull] + [hull[0][1]]
            ax.fill(xs, ys, color=HULL_FILL, alpha=0.22, zorder=1)
            ax.plot(xs, ys, color=HULL_COLOR, linewidth=1.8, label=label, zorder=4)

        if hull:
            ax.scatter([q[0] for q in hull], [q[1] for q in hull], s=40, color=HULL_COLOR, zorder=5)

    def _draw_step(self, ax, step: Step, idx: int, total: int) -> None:
        """
        Función _draw_step
        Recibe: los ejes ax, el paso step y su posición idx dentro de los total pasos

        Dibuja un cuadro de la animación con los datos del Step:

            1. Los puntos ya descartados en gris claro y el resto normal
            2. Las aristas aceptadas en azul (o el hull completo en el último paso)
            3. Las aristas en revisión punteadas y las destacadas en naranjo
            4. Los puntos activos en naranjo y los descartados en este paso con x roja

        La leyenda es la misma en todos los pasos para que el gráfico no "salte"
        """
        discarded = set(step.discarded)
        alive = [q for q in self.points if q not in discarded]

        if discarded:
            ax.scatter([q[0] for q in discarded], [q[1] for q in discarded], s=18, color=POINT_FADED, zorder=2)

        if alive:
            ax.scatter([q[0] for q in alive], [q[1] for q in alive], s=18, color=POINT_COLOR, zorder=3)

        if step.hull is not None:
            self._draw_hull(ax, step.hull)              # último cuadro: hull completo
        else:
            for a, b in step.done_edges:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=HULL_COLOR, linewidth=1.8, zorder=4)

            vertices = {q for edge in step.done_edges for q in edge}

            if vertices:
                ax.scatter([q[0] for q in vertices], [q[1] for q in vertices], s=40, color=HULL_COLOR, zorder=5)

        for a, b in step.pending_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=PENDING_COLOR, linewidth=1.2, linestyle="--", zorder=4)

        for a, b in step.highlight_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:orange", linewidth=2.6, zorder=6)

        if step.active:
            ax.scatter([q[0] for q in step.active], [q[1] for q in step.active], s=130,
                       color="tab:orange", edgecolors=TEXT_COLOR, zorder=7)

        if step.removed:
            ax.scatter([q[0] for q in step.removed], [q[1] for q in step.removed], s=140,
                       marker="x", color=ERROR_COLOR, linewidths=2.5, zorder=8)

        handles = [
            Line2D([], [], color=HULL_COLOR, linewidth=1.8, marker="o", label="Aceptado"),
            Line2D([], [], color=PENDING_COLOR, linewidth=1.2, linestyle="--", label="En revisión"),
            Line2D([], [], color="tab:orange", linewidth=2.6, marker="o", markeredgecolor=TEXT_COLOR, label="Actual / candidato"),
            Line2D([], [], color=ERROR_COLOR, linestyle="", marker="x", markeredgewidth=2.5, label="Descartado ahora"),
            Line2D([], [], color=POINT_FADED, linestyle="", marker="o", label="Ya descartado"),
        ]
        # va sobre el gráfico (entre el título y los ejes) para no tapar puntos
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
                  fontsize=7.5, frameon=False, handletextpad=0.4, columnspacing=1.0)
        paused = "" if self.playing else " (en pausa)"
        ax.set_title(f"{self.algorithm}: paso {idx + 1} de {total}{paused} | {len(self.points)} puntos",
                     fontsize=11, pad=26)
        ax.set_xlabel(step.text, fontsize=10)


    # clicks y slider

    def _on_slider(self, value: float) -> None:
        """
        Función _on_slider
        Recibe: el valor del slider

        Guarda la cantidad de puntos que se generarán con el botón Generar
        """
        self.n = int(value)

    def _on_click(self, event) -> None:
        """
        Función _on_click
        Recibe: el evento de click de matplotlib

        Inserta el punto clickeado, solo si se cumple todo esto:

            1. El click fue con el botón izquierdo dentro del gráfico
            2. El modo inserción está activo (y por lo tanto el algoritmo es
               Andrew, porque el modo solo se activa con Andrew)
            3. No está activa la lupa ni el "pan" de la barra de matplotlib
               (en ese caso el click es para esa herramienta)
        """
        if event.inaxes is not self.ax or not self.insert_mode or event.button != 1:
            return

        toolbar = getattr(self.fig.canvas, "toolbar", None)

        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        self.insert_point((float(event.xdata), float(event.ydata)))


def main() -> None:
    """
    Función main

    Lee los argumentos de la línea de comandos y abre la ventana interactiva,
    con los puntos iniciales del archivo --points, o generados al azar si no
    se entregó
    """
    parser = argparse.ArgumentParser(description="Visualizador de Convex Hull 2D con inserción incremental.")
    parser.add_argument("--points", metavar="ARCHIVO", help="archivo con los puntos (una línea 'x,y' o 'x y' por punto)")
    parser.add_argument("--algorithm", choices=["jarvis", "andrew", "quickhull"], default="andrew", help="algoritmo inicial")
    parser.add_argument("--animate", action="store_true", help=" modo animación activo")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help="cantidad de puntos aleatorios, si esque no se usa --points")
    parser.add_argument("--seed", type=int, default=None, help="semilla del generador")
    args = parser.parse_args()

    viz = HullVisualizer(n=args.n, algorithm=args.algorithm.capitalize(), animate=args.animate)

    if args.points:
        points = load_points(args.points)
        print(f"{len(points)} puntos leídos de {args.points}")
        viz.set_points(points)
    elif args.seed is not None:
        viz.box_seed.set_val(str(args.seed))
        viz.generate()
    else:
        viz.new_seed()

    plt.show()


if __name__ == "__main__":
    main()
