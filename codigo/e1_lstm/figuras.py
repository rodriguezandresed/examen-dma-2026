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
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.2, 1.35, 2.2, 1.5, "Imagen\n$n\\times n\\times 3$", "#e8e8e8"),
        (2.9, 1.35, 2.4, 1.5, "CONV + ReLU\nfiltros\nlocales", "#d9e2ec"),
        (5.8, 1.35, 2.2, 1.5, "POOL\nmax pooling", "#f0e6d2"),
        (8.5, 1.35, 2.4, 1.5, "CONV + ReLU\nrasgos más\nabstractos", "#d9e2ec"),
        (11.4, 1.35, 2.2, 1.5, "POOL", "#f0e6d2"),
        (14.1, 1.35, 2.0, 1.5, "FLATTEN\nvector", "#e8e8e8"),
        (16.5, 1.35, 1.6, 1.5, "FC\ndensa", "#dce8d4"),
        (18.4, 1.35, 1.4, 1.5, "Softmax\nclases", "#dce8d4"),
    ]
    for x, y, w, h, t, fc in boxes:
        _box(ax, x, y, w, h, t, fc=fc, size=7.2)

    xs = [0.2, 2.9, 5.8, 8.5, 11.4, 14.1, 16.5, 18.4]
    ws = [2.2, 2.4, 2.2, 2.4, 2.2, 2.0, 1.6, 1.4]
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + ws[i] + 0.02, 2.1, xs[i + 1] - 0.02, 2.1)

    ax.text(
        10,
        0.35,
        "Bloques CONV-POOL repetibles; peso compartido en cada filtro.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUT / "cnn_capas.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_lstm() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Celda grande
    _box(ax, 1.6, 0.7, 8.6, 5.5, "", fc="#fafafa", ec="#444444")
    ax.text(5.9, 5.9, "Celda LSTM (paso $t$)", ha="center", fontsize=10, fontweight="bold")

    # Entradas
    ax.text(0.2, 4.7, r"$x_t$", fontsize=11)
    ax.text(0.2, 3.3, r"$h_{t-1}$", fontsize=11)
    ax.text(0.2, 1.5, r"$c_{t-1}$", fontsize=11)
    _arrow(ax, 0.9, 4.75, 1.58, 4.75)
    _arrow(ax, 0.9, 3.4, 1.58, 3.4)
    _arrow(ax, 0.9, 1.55, 1.58, 1.55)

    # Cuatro transformaciones
    _box(ax, 2.0, 4.35, 2.15, 1.05, "Olvido $f_t$\nsigmoidal", fc="#f7d6d6", size=7.5)
    _box(ax, 4.35, 4.35, 2.15, 1.05, "Entrada $i_t$\nsigmoidal", fc="#d6e8f7", size=7.5)
    _box(ax, 6.7, 4.35, 2.15, 1.05, "Candidata $\\tilde{c}_t$\ntanh", fc="#e8e0d6", size=7.5)
    _box(ax, 2.0, 2.85, 2.15, 1.05, "Salida $o_t$\nsigmoidal", fc="#d6f0d6", size=7.5)

    # Estado de celda
    _box(ax, 4.6, 1.15, 3.4, 1.25, r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$", fc="#fff6cc", size=7.4)
    _box(ax, 8.3, 2.85, 1.6, 1.05, r"$h_t = o_t \odot \tanh(c_t)$", fc="#e8e8e8", size=7.2)

    _arrow(ax, 3.07, 4.35, 3.07, 2.40)  # f -> c
    _arrow(ax, 5.42, 4.35, 5.8, 2.42)  # i -> c
    _arrow(ax, 7.77, 4.35, 7.2, 2.42)  # g -> c
    _arrow(ax, 4.15, 3.35, 8.28, 3.35)  # o -> h
    _arrow(ax, 8.0, 1.77, 9.1, 2.82)  # c -> h

    # Salidas
    _arrow(ax, 10.22, 3.35, 11.3, 3.35)
    _arrow(ax, 8.0, 1.20, 11.3, 1.20)
    ax.text(11.4, 3.25, r"$h_t$", fontsize=11)
    ax.text(11.4, 1.10, r"$c_t$", fontsize=11)

    ax.text(
        5.9,
        0.25,
        r"Cuatro transformaciones: 4 $\times$ [(units $\times$ input_dim) + units$^2$ + units] parámetros.",
        ha="center",
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(OUT / "lstm_celda.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figura_cnn()
    figura_lstm()
    print("Figuras en", OUT)
