"""
Figura del Ejercicio 8, a partir de los CSV que deja adaboost_vs_gb.py.

  mecanismo_y_error.png : una sola figura de cuatro paneles, en dos filas.
    Fila 1, que modifica cada algoritmo en cada iteracion. Los dos paneles estan
      en la misma unidad (valor dividido por su valor inicial, con rangos
      distintos) para que se vea que lo que se mueve en un algoritmo es lo que
      queda quieto en el otro.
    Fila 2, error de train y de test de AdaBoost contra iteraciones, con las
      etiquetas originales y con el 10 % dadas vuelta.

Dos intentos descartados que conviene dejar anotados:

  1. Primera version de la fila 1: los pesos D_t en unidades absolutas contra el
     desvio del residuo en unidades del target. No servia, son escalas distintas
     y el panel de la derecha se veia plano. Se paso a la escala relativa.
  2. Hasta la revision cruzada esto eran dos figuras separadas
     (pesos_vs_residuos.png y curvas_error.png). Cada una se llevaba su titulo,
     su nota y su espaciado de float, y entre las dos empujaban la seccion a una
     quinta hoja. Se fusionaron en una grilla 2x2 mas baja: la evidencia es la
     misma y se recuperan unas diez lineas de texto.

Correr:  python codigo/e8_boosting/figuras.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e8"

GRIS = "#8a8a8a"
AZUL = "#1f4e79"
ROJO = "#a33a3a"


def panel_mecanismo(ax_izq, ax_der):
    """Fila 1: AdaBoost mueve los pesos, Gradient Boosting mueve el target."""
    dp = pd.read_csv(OUT / "adaboost_pesos.csv")
    dg = pd.read_csv(OUT / "gb_residuos.csv")

    base = dp["peso_uniforme"].iloc[0]
    ax_izq.plot(dp["ronda"], dp["peso_mal_r1"] / base, color=ROJO, lw=1.5,
                label="mal clasificadas por $h_1$")
    ax_izq.plot(dp["ronda"], dp["peso_bien_r1"] / base, color=AZUL, lw=1.5,
                label="bien clasificadas por $h_1$")
    ax_izq.axhline(1.0, color=GRIS, ls="--", lw=1.1, label="target $y_i$ (no cambia)")
    ax_izq.set_title("AdaBoost: cambia el peso $D_t(i)$", fontsize=9.5)
    ax_izq.set_xlabel("ronda $t$", fontsize=8.5)
    ax_izq.set_ylabel("valor relativo al inicial", fontsize=8.5)
    ax_izq.legend(fontsize=7, loc="upper right", framealpha=0.9)

    r0 = dg["sd_residuo"].iloc[0]
    ax_der.plot(dg["iter"], dg["sd_residuo"] / r0, color=ROJO, lw=1.5,
                label="desvio del pseudo-residuo $r_m$")
    ax_der.axhline(1.0, color=GRIS, ls="--", lw=1.1, label="peso $1/N$ (no cambia)")
    ax_der.set_title("Gradient Boosting: cambia el target $r_m$", fontsize=9.5)
    ax_der.set_xlabel("iteracion $m$", fontsize=8.5)
    ax_der.set_ylim(0, 1.3)
    ax_der.legend(fontsize=7, loc="upper right", framealpha=0.9)


def panel_curvas(ax_izq, ax_der):
    """Fila 2: la subida del error de test solo aparece con etiquetas ruidosas."""
    d = pd.read_csv(OUT / "curvas_error.csv")

    ax_izq.plot(d["m"], d["err_train_limpio"], color=AZUL, lw=1.3, label="train")
    ax_izq.plot(d["m"], d["err_test_limpio"], color=ROJO, lw=1.3, label="test")
    ax_izq.set_title("AdaBoost: etiquetas originales", fontsize=9.5)
    ax_izq.set_ylabel("error de clasificacion", fontsize=8.5)

    ax_der.plot(d["m"], d["err_train_ruido"], color=AZUL, lw=1.3, label="train")
    ax_der.plot(d["m"], d["err_test_ruido"], color=ROJO, lw=1.3, label="test")
    ax_der.set_title("AdaBoost: 10 % de etiquetas dadas vuelta", fontsize=9.5)

    for a in (ax_izq, ax_der):
        a.set_xlabel("cantidad de stumps $T$", fontsize=8.5)
        a.legend(fontsize=7.5)


def figura():
    # Solo los dos paneles del mecanismo. La fila de curvas de error se saco del
    # informe: la consigna pide la diferencia entre los dos algoritmos y una
    # explicacion intuitiva, no el comportamiento del error contra iteraciones.
    # panel_curvas() queda en el script por si hace falta para el oral.
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.9))
    panel_mecanismo(ax[0], ax[1])
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.6)
        a.tick_params(labelsize=9)
        a.title.set_fontsize(10.5)
        a.xaxis.label.set_fontsize(9.5)
        a.yaxis.label.set_fontsize(9.5)
    fig.tight_layout()
    fig.savefig(OUT / "mecanismo_y_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figura()
    print("Figura escrita en", OUT)
