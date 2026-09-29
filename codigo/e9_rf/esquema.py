"""
Esquema del Ejercicio 9: como se construye un Random Forest.

Salida: codigo/resultados/e9/esquema_rf.png

Mismo estilo que los esquemas del Ejercicio 1 y del Ejercicio 8.

El dibujo tiene que dejar ver las dos fuentes de azar que definen al metodo,
porque son las que lo separan de bagging y las que gobiernan el compromiso
entre correlacion y fuerza: el bootstrap de filas y el sorteo de m variables
en cada nodo. La rama de abajo es la bolsa, que es lo que hace que el error
de generalizacion salga gratis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e9"
OUT.mkdir(parents=True, exist_ok=True)

GRIS = "#222222"


def _box(ax, x, y, w, h, text, fc="#f4f4f4", size=9.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, facecolor=fc, edgecolor=GRIS))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#111111")


def _arrow(ax, x1, y1, x2, y2, guion=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.0,
                                linestyle="--" if guion else "-"))


def figura() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.05))
    ax.set_xlim(0, 16.3)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # Anchos calculados para que ningun rotulo desborde su caja: la primera
    # version cortaba "N filas" y "N casos con reemplazo".
    y = 4.30
    _box(ax, 0.10, y, 3.00, 1.30, "Datos\n$N$ filas, $M$ variables", "#e8e8e8")
    _box(ax, 3.76, y, 3.60, 1.30,
         "Muestra bootstrap\n$N$ casos con reemplazo", "#d9e2ec")
    _box(ax, 8.02, y, 3.80, 1.30,
         "Árbol $k$ sin poda: en cada\nnodo sortea $m$ de $M$", "#f0e6d2")
    _box(ax, 12.48, y, 3.60, 1.30,
         "Voto mayoritario\nde los $B$ árboles", "#dce8d4")

    _arrow(ax, 3.14, y + 0.65, 3.72, y + 0.65)
    _arrow(ax, 7.40, y + 0.65, 7.98, y + 0.65)
    _arrow(ax, 11.86, y + 0.65, 12.44, y + 0.65)

    # Ciclo: cada arbol repite el bootstrap y el sorteo de variables.
    ax.add_patch(FancyArrowPatch(
        (9.92, y - 0.06), (5.56, y - 0.06), connectionstyle="arc3,rad=-0.28",
        arrowstyle="-|>", mutation_scale=11, color=GRIS, lw=1.0,
        linestyle=(0, (4, 2))))
    ax.text(7.74, 3.26, "repetir $B$ veces", ha="center", fontsize=8.2,
            color="#333333", style="italic")

    # Rama de la bolsa: sale del bootstrap y no vuelve al bosque.
    _box(ax, 3.76, 1.05, 3.60, 1.25,
         "Fuera de la bolsa\n(alrededor de $1/3$)", "#f7d6d6")
    _box(ax, 8.02, 1.05, 3.80, 1.25,
         "Error OOB e importancia\npor permutación", "#d6e8f7")
    _arrow(ax, 4.60, y - 0.06, 4.60, 2.34, guion=True)
    _arrow(ax, 7.40, 1.68, 7.98, 1.68)

    ax.text(8.1, 0.28,
            "Las dos fuentes de azar son el bootstrap de filas y el sorteo de "
            "$m$ variables por nodo;\nbagging solo tiene la primera.",
            ha="center", fontsize=8.2, color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "esquema_rf.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Esquema escrito en", OUT / "esquema_rf.png")


if __name__ == "__main__":
    figura()
