"""
Figuras del Ejercicio 1: capas CNN y celda LSTM.

Salida: codigo/resultados/e1/cnn_capas.png y lstm_celda.png
Correr:  .venv-tf/bin/python codigo/e1_lstm/figuras.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e1"
OUT.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#f4f4f4", ec="#222222", size=8, weight="normal"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#111111",
        fontweight=weight,
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.0),
    )


def figura_cnn() -> None:
    # Lienzo chato y tipografía grande: en el PDF entra al ancho del texto
    # y los rótulos tienen que leerse a unos 9 pt, no como una tira.
    fig, ax = plt.subplots(figsize=(7.4, 1.58))
    ax.set_xlim(0, 17.6)
    ax.set_ylim(0, 2.05)
    ax.axis("off")

    boxes = [
        (0.10, 0.46, 1.9, 1.22, "Imagen\nn × n × 3", "#e8e8e8"),
        (2.22, 0.46, 2.15, 1.22, "CONV\n+ ReLU", "#d9e2ec"),
        (4.59, 0.46, 1.7, 1.22, "POOL\nmáximo", "#f0e6d2"),
        (6.51, 0.46, 2.25, 1.22, "CONV 2\n+ ReLU", "#d9e2ec"),
        (8.98, 0.46, 1.7, 1.22, "POOL\nmáximo", "#f0e6d2"),
        (10.90, 0.46, 1.85, 1.22, "Aplanado\nvector", "#e8e8e8"),
        (12.97, 0.46, 1.85, 1.22, "Capa\ndensa", "#dce8d4"),
        (15.04, 0.46, 2.15, 1.22, "Softmax\nclases", "#dce8d4"),
    ]
    for x, y, w, h, t, fc in boxes:
        _box(ax, x, y, w, h, t, fc=fc, size=9.0)

    xs = [b[0] for b in boxes]
    ws = [b[2] for b in boxes]
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + ws[i] + 0.02, 1.07, xs[i + 1] - 0.02, 1.07)

    ax.text(
        8.8,
        0.08,
        "El bloque CONV-POOL se puede repetir. Cada filtro comparte sus pesos.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUT / "cnn_capas.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_lstm() -> None:
    # Las cuatro puertas van en una fila. Cada flecha baja a su caja y no
    # atraviesa otra: olvido, entrada y candidata entran a c_t; la salida, a h_t.
    fig, ax = plt.subplots(figsize=(7.05, 2.32))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 4.45)
    ax.axis("off")

    ax.text(6.9, 4.18, "Celda LSTM, paso $t$", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 1.85, 2.58, 2.45, 1.12, "Olvido $f_t$\nsigmoidal", fc="#f7d6d6", size=9)
    _box(ax, 4.55, 2.58, 2.45, 1.12, "Entrada $i_t$\nsigmoidal", fc="#d6e8f7", size=9)
    _box(ax, 7.25, 2.58, 2.55, 1.12, "Candidata $\\tilde{c}_t$\ntanh", fc="#e8e0d6", size=9)
    _box(ax, 10.15, 2.58, 2.45, 1.12, "Salida $o_t$\nsigmoidal", fc="#d6f0d6", size=9)

    _box(
        ax,
        1.85,
        0.38,
        7.95,
        1.22,
        r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$",
        fc="#fff6cc",
        size=9,
    )
    _box(ax, 10.15, 0.72, 2.45, 0.88, r"$h_t = o_t \odot \tanh(c_t)$", fc="#e8e8e8", size=8.2)

    # De cada puerta a la caja de abajo. Terminan en el borde superior.
    _arrow(ax, 3.07, 2.58, 3.07, 1.62)
    _arrow(ax, 5.77, 2.58, 5.77, 1.62)
    _arrow(ax, 8.52, 2.58, 8.52, 1.62)
    _arrow(ax, 11.37, 2.58, 11.37, 1.62)

    # c_t alimenta a h_t por el costado, sin cruzar la caja gris.
    _arrow(ax, 9.82, 1.15, 10.13, 1.15)

    ax.text(0.08, 3.05, r"$x_t,\, h_{t-1}$", fontsize=10)
    _arrow(ax, 1.42, 3.14, 1.83, 3.14)
    ax.text(0.15, 0.85, r"$c_{t-1}$", fontsize=11)
    _arrow(ax, 1.15, 0.98, 1.83, 0.98)

    _arrow(ax, 12.62, 1.16, 13.15, 1.16)
    ax.text(13.2, 1.02, r"$h_t$", fontsize=11)
    # c_t sale por debajo de h_t: la caja gris queda arriba de esta flecha.
    _arrow(ax, 9.82, 0.52, 13.15, 0.52)
    ax.text(13.2, 0.38, r"$c_t$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "lstm_celda.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figura_cnn()
    figura_lstm()
    print("Figuras en", OUT)
