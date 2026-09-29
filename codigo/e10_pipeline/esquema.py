"""
Esquema del Ejercicio 10: el pipeline de comparacion.

Salida: codigo/resultados/e10/esquema_pipeline.png

Mismo estilo que los esquemas de los Ejercicios 1, 7, 8 y 9.

Lo que el dibujo tiene que dejar ver es DONDE se ajusta el escalador. La fila
de abajo hace el zoom sobre un fold: el StandardScaler se ajusta con el fold
de ajuste y recien despues se aplica al de validacion. Si se escala antes del
corte, la media y el desvio se calculan con datos que despues terminan del
otro lado, y eso es fuga de informacion. Es el punto de la practica 2 de la
Clase 1 y es la razon por la que el preprocesamiento va dentro del Pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e10"
OUT.mkdir(parents=True, exist_ok=True)

GRIS = "#222222"


def _box(ax, x, y, w, h, text, fc="#f4f4f4", size=9.0, borde=GRIS, guion=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, facecolor=fc, edgecolor=borde,
        linestyle=(0, (4, 2)) if guion else "solid"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#111111")


def _arrow(ax, x1, y1, x2, y2, guion=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.0,
                                linestyle="--" if guion else "-"))


def figura() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    ax.set_xlim(0, 17.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # ---- flujo principal ---------------------------------------------------
    y = 5.55
    _box(ax, 0.10, y, 2.45, 1.25, "Datos", "#e8e8e8")
    _box(ax, 3.05, y, 2.60, 1.25, "Corte\n75 / 25", "#d9e2ec")
    _box(ax, 6.15, y, 4.05, 1.25,
         "Búsqueda sobre el 75 %:\n20 sorteos × 5 folds", "#f0e6d2")
    _box(ax, 10.70, y, 2.90, 1.25, "Mejor\nconfiguración", "#dce8d4")
    _box(ax, 14.10, y, 3.20, 1.25, "Evaluación en\nel 25 % reservado", "#dce8d4")

    for x0, w0, x1 in [(0.10, 2.45, 3.05), (3.05, 2.60, 6.15),
                       (6.15, 4.05, 10.70), (10.70, 2.90, 14.10)]:
        _arrow(ax, x0 + w0 + 0.04, y + 0.62, x1 - 0.04, y + 0.62)

    # El 25 % no participa de la busqueda: baja por un costado y vuelve al final.
    _box(ax, 3.05, 3.55, 2.60, 1.05, "El 25 % no\nse toca", "#f7d6d6", size=8.5)
    _arrow(ax, 4.35, y - 0.04, 4.35, 4.64, guion=True)
    ax.add_patch(FancyArrowPatch(
        (5.69, 4.08), (15.70, y - 0.04), connectionstyle="arc3,rad=-0.12",
        arrowstyle="-|>", mutation_scale=11, color=GRIS, lw=1.0,
        linestyle=(0, (4, 2))))

    # ---- zoom sobre un fold ------------------------------------------------
    yz = 1.15
    _box(ax, 6.15, yz, 11.15, 1.85, "", "#fbfbfb", borde="#999999", guion=True)
    ax.text(6.40, yz + 1.58, "dentro de cada fold:", fontsize=8.5,
            color="#333333", style="italic", ha="left")

    _box(ax, 6.55, yz + 0.30, 3.05, 0.95, "Fold de ajuste", "#d9e2ec", size=8.5)
    _box(ax, 10.05, yz + 0.30, 3.30, 0.95,
         "Scaler ajustado acá\ny modelo entrenado", "#f0e6d2", size=8.5)
    _box(ax, 13.80, yz + 0.30, 3.10, 0.95, "Fold de validación", "#dce8d4", size=8.5)
    _arrow(ax, 9.64, yz + 0.78, 10.01, yz + 0.78)
    _arrow(ax, 13.39, yz + 0.78, 13.76, yz + 0.78)

    _arrow(ax, 8.20, y - 0.04, 8.20, yz + 1.92, guion=True)

    ax.text(8.7, 0.28,
            "El escalador se ajusta dentro del fold, nunca antes del corte: "
            "si no, la media y el desvío\nse calculan con datos que después "
            "quedan del otro lado.",
            ha="center", fontsize=8.2, color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "esquema_pipeline.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Esquema escrito en", OUT / "esquema_pipeline.png")


if __name__ == "__main__":
    figura()
