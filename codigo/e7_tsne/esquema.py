"""
Esquema del Ejercicio 7: el flujo de t-SNE.

Salida: codigo/resultados/e7/esquema_tsne.png

Mismo estilo que los esquemas de los Ejercicios 1, 8 y 9.

El dibujo tiene que dejar claro que hay dos espacios con su propia nocion de
vecindad y que lo unico que se optimiza son las posiciones del mapa. Es la
confusion mas habitual con t-SNE: creer que entrena un modelo que despues se
aplica a datos nuevos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e7"
OUT.mkdir(parents=True, exist_ok=True)

GRIS = "#222222"


def _box(ax, x, y, w, h, text, fc="#f4f4f4", size=9.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, facecolor=fc, edgecolor=GRIS))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#111111")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.0))


def figura() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 1.95))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 3.55)
    ax.axis("off")

    # Rotulos cortos: el detalle de que distribucion usa cada espacio va en la
    # nota al pie de la figura, si no el texto desborda las cajas.
    y = 1.62
    cajas = [
        (0.10, 2.70, "Datos en\n$D$ dimensiones", "#e8e8e8"),
        (3.40, 2.90, "Vecindades $P$\nen el original", "#d9e2ec"),
        (6.90, 2.90, "Vecindades $Q$\nen el mapa 2D", "#f0e6d2"),
        (10.40, 3.10, "Divergencia KL\nentre $P$ y $Q$", "#f7d6d6"),
        (14.10, 3.00, "Mover los puntos\ndel mapa", "#dce8d4"),
    ]
    for x, w, t, fc in cajas:
        _box(ax, x, y, w, 1.30, t, fc=fc)
    for i in range(len(cajas) - 1):
        x, w = cajas[i][0], cajas[i][1]
        _arrow(ax, x + w + 0.04, y + 0.65, cajas[i + 1][0] - 0.04, y + 0.65)

    # El ciclo solo toca el mapa: los datos originales no se vuelven a mirar.
    ax.add_patch(FancyArrowPatch(
        (15.60, y - 0.06), (8.35, y - 0.06), connectionstyle="arc3,rad=-0.16",
        arrowstyle="-|>", mutation_scale=11, color=GRIS, lw=1.0,
        linestyle=(0, (4, 2))))
    ax.text(11.97, 0.34, "se repite: lo único que cambia son las posiciones del mapa",
            ha="center", fontsize=8.2, color="#333333", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "esquema_tsne.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Esquema escrito en", OUT / "esquema_tsne.png")


if __name__ == "__main__":
    figura()
