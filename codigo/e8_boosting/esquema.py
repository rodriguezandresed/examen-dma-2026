"""
Esquema del Ejercicio 8: el ciclo de boosting en sus dos variantes.

Salida: codigo/resultados/e8/esquema_boosting.png

Mismo estilo que los esquemas del Ejercicio 1 (cajas redondeadas, paleta
apagada, rotulos a 9 pt) para que el documento se lea de una sola mano.

La idea del dibujo es que las dos filas tengan la misma forma y difieran en
una sola caja, que es donde esta la diferencia entre los dos algoritmos: en
AdaBoost cambia el peso de las observaciones, en Gradient Boosting cambia el
target. El resto del ciclo es identico.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e8"
OUT.mkdir(parents=True, exist_ok=True)

GRIS = "#222222"


def _box(ax, x, y, w, h, text, fc="#f4f4f4", size=9.0, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, facecolor=fc, edgecolor=GRIS))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#111111", fontweight=weight)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.0))


def _vuelta(ax, x1, x2, y, dip, rotulo):
    """Flecha de retorno que cierra el ciclo, por debajo de la fila.

    El rad va negativo para que la curva baje: con rad positivo y los puntos
    de derecha a izquierda la flecha pasaba por encima y cortaba las cajas.
    """
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), connectionstyle="arc3,rad=%.2f" % dip,
        arrowstyle="-|>", mutation_scale=11, color=GRIS, lw=1.0,
        linestyle=(0, (4, 2))))
    ax.text((x1 + x2) / 2, y - 0.62, rotulo, ha="center", fontsize=8.2,
            color="#333333", style="italic")


def figura() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # ---- fila de AdaBoost --------------------------------------------------
    ya = 5.05
    ax.text(0.05, 6.95, "AdaBoost: en cada ronda cambia el PESO de las observaciones",
            fontsize=9.5, fontweight="bold", color="#111111")
    cajas_a = [
        (0.10, 2.55, "Datos con pesos\n$D_t(i)$", "#e8e8e8"),
        (3.05, 2.30, "Modelo débil\n$h_t$", "#d9e2ec"),
        (5.75, 2.60, "Error $\\varepsilon_t$\ny voto $\\alpha_t$", "#f0e6d2"),
        (8.75, 3.05, "Sube el peso de\nlas mal clasificadas", "#f7d6d6"),
    ]
    for x, w, t, fc in cajas_a:
        _box(ax, x, ya, w, 1.15, t, fc=fc)
    for i in range(len(cajas_a) - 1):
        x, w = cajas_a[i][0], cajas_a[i][1]
        _arrow(ax, x + w + 0.04, ya + 0.58, cajas_a[i + 1][0] - 0.04, ya + 0.58)
    _vuelta(ax, 10.20, 1.38, ya - 0.06, -0.22, "repetir $T$ rondas")
    _box(ax, 12.45, ya, 2.85, 1.15, "Voto ponderado\n$H(x)$", fc="#dce8d4")
    _arrow(ax, 11.65, ya + 0.58, 12.41, ya + 0.58)

    # ---- fila de Gradient Boosting -----------------------------------------
    yg = 1.40
    ax.text(0.05, 3.30, "Gradient Boosting: en cada iteración cambia el TARGET",
            fontsize=9.5, fontweight="bold", color="#111111")
    cajas_g = [
        (0.10, 2.55, "Datos, target =\nresiduo $r_m$", "#e8e8e8"),
        (3.05, 2.30, "Árbol\n$h_m$", "#d9e2ec"),
        (5.75, 2.60, "Se suma con\npaso $v$", "#f0e6d2"),
        (8.75, 3.05, "Nuevo residuo\n$r_{m+1}$", "#d6e8f7"),
    ]
    for x, w, t, fc in cajas_g:
        _box(ax, x, yg, w, 1.15, t, fc=fc)
    for i in range(len(cajas_g) - 1):
        x, w = cajas_g[i][0], cajas_g[i][1]
        _arrow(ax, x + w + 0.04, yg + 0.58, cajas_g[i + 1][0] - 0.04, yg + 0.58)
    _vuelta(ax, 10.20, 1.38, yg - 0.06, -0.22, "repetir $M$ iteraciones")
    _box(ax, 12.45, yg, 2.85, 1.15, "Suma\n$F_M(x)$", fc="#dce8d4")
    _arrow(ax, 11.65, yg + 0.58, 12.41, yg + 0.58)

    ax.text(7.7, 0.05,
            "El peso de las observaciones no cambia nunca en Gradient Boosting; "
            "el target no cambia nunca en AdaBoost.",
            ha="center", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "esquema_boosting.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Esquema escrito en", OUT / "esquema_boosting.png")


if __name__ == "__main__":
    figura()
