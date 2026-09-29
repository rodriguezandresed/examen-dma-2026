"""
Ejercicio 9: Random Forest. Cuatro experimentos que ponen a prueba las
afirmaciones del documento de Breiman y Cutler que se repartio en la clase 6.

Datos: Breast Cancer Wisconsin (Diagnostic), 569 filas, 30 variables continuas,
dos clases. Viene en sklearn, asi que el script se corre sin bajar nada.

Que se mide, y por que:
  E1  error OOB vs cantidad de arboles      -> "los RF no se sobreajustan"
  E2  barrido de m (max_features)           -> correlacion vs fuerza, rango optimo
  E3  OOB vs test reservado                 -> la estimacion OOB es insesgada
  E4  importancia Gini vs permutacion       -> el sesgo de Gini por cardinalidad

Correr:
  python codigo/e9_rf/rf_breiman.py

Nota de trabajo: la primera version media la correlacion entre arboles como la
correlacion de Pearson promedio entre los vectores de prediccion de cada par de
arboles. Con 300 arboles son 44.850 pares y la corrida tardaba minutos por cada
valor de m. Se reemplazo por el estimador del propio Breiman (2001),
rho = var(mr) / (E_theta[sd(theta)])^2, que da lo mismo en O(T) y ademas permite
calcular la cota PE* <= rho*(1-s^2)/s^2 con las mismas cantidades.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display: el script corre desde consola
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "resultados" / "e9"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta sobria: el PDF se imprime en blanco y negro con frecuencia.
C_OOB, C_TEST, C_AUX = "#1b3b6f", "#c1440e", "#5a5a5a"


def cargar():
    """Particion estratificada 70/30. El test se reserva para E3: no se toca
    en la eleccion de ningun hiperparametro."""
    d = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        d.data, d.target, test_size=0.30, random_state=SEED, stratify=d.target
    )
    return X_tr, X_te, y_tr, y_te, list(d.feature_names)


# ----------------------------------------------------------------------------
# Fuerza y correlacion, tal como las define Breiman (2001)
# ----------------------------------------------------------------------------
def fuerza_y_correlacion(bosque: RandomForestClassifier, X, y):
    """Devuelve (s, rho, cota).

    Para dos clases la funcion de margen crudo de un arbol vale +1 si acierta y
    -1 si se equivoca, porque la unica clase incorrecta es la otra. Entonces:
      mr(x,y) = promedio sobre arboles del margen crudo = 2*p_acierto - 1
      s       = E[mr]                       (fuerza del bosque)
      rho     = var(mr) / (E_theta sd)^2    (correlacion media entre arboles)
      cota    = rho*(1-s^2)/s^2             (Teorema 2.3 de Breiman)
    """
    # matriz arboles x observaciones con +1 / -1
    margenes = np.array(
        [np.where(arbol.predict(X).astype(int) == y, 1.0, -1.0) for arbol in bosque.estimators_]
    )
    mr = margenes.mean(axis=0)
    s = float(mr.mean())
    sd_por_arbol = margenes.std(axis=1, ddof=0)
    denom = float(sd_por_arbol.mean()) ** 2
    rho = float(mr.var(ddof=0) / denom) if denom > 0 else np.nan
    cota = float(rho * (1 - s**2) / s**2) if s > 0 else np.nan
    # acierto promedio de un arbol suelto: la lectura intuitiva de la fuerza
    acierto_medio = float(((margenes + 1) / 2).mean())
    return s, rho, cota, acierto_medio


# ----------------------------------------------------------------------------
# E1: error OOB contra cantidad de arboles
# ----------------------------------------------------------------------------
def exp_curva_oob(X_tr, y_tr, X_te, y_te):
    """warm_start acumula arboles sobre el mismo bosque, asi la curva es de un
    unico bosque que crece y no de bosques independientes. Se mide en paralelo
    el error sobre el test reservado para ver si en algun punto empeora."""
    grilla = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
    rf = RandomForestClassifier(
        n_estimators=grilla[0],
        max_features="sqrt",
        oob_score=True,
        warm_start=True,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )
    filas = []
    for t in grilla:
        rf.set_params(n_estimators=t)
        rf.fit(X_tr, y_tr)
        filas.append(
            {
                "n_arboles": t,
                "err_oob": 1 - rf.oob_score_,
                "err_test": 1 - rf.score(X_te, y_te),
            }
        )
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "oob_vs_arboles.csv", index=False)

    return df


# ----------------------------------------------------------------------------
# E2 y E3: barrido de m, con OOB, test, fuerza y correlacion
# ----------------------------------------------------------------------------
def exp_barrido_m(X_tr, y_tr, X_te, y_te):
    """m recorre 1..30 sobre las 30 variables. 300 arboles por punto: alcanza
    para que el OOB se estabilice (ver E1) y la corrida entera queda en pocos
    minutos. La fuerza y la correlacion se miden sobre el test, que es donde
    tienen la interpretacion poblacional que les da Breiman."""
    ms = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
    filas = []
    for m in ms:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_features=m,
            oob_score=True,
            bootstrap=True,
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        s, rho, cota, acierto = fuerza_y_correlacion(rf, X_te, y_te)
        filas.append(
            {
                "m": m,
                "err_oob": 1 - rf.oob_score_,
                "err_test": 1 - rf.score(X_te, y_te),
                "fuerza": s,
                "correlacion": rho,
                "cota_breiman": cota,
                "acierto_arbol_solo": acierto,
            }
        )
    df = pd.DataFrame(filas)
    df["brecha_oob_test"] = df.err_oob - df.err_test
    df.to_csv(OUT / "barrido_m.csv", index=False)

    return df


def figura_bosque(curva: pd.DataFrame, barrido: pd.DataFrame) -> None:
    """Los dos primeros experimentos en una figura de dos paneles.

    En el panel derecho se descarto superponer el error OOB en un segundo eje:
    varia 0,013 en todo el barrido y el eje comprimido convertia ese ruido en
    picos enormes que se leian como una senal. En su lugar va la cota de
    Breiman, que combina fuerza y correlacion y si tiene un minimo interior."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 2.8))

    a1.plot(curva.n_arboles, curva.err_oob, "o-", color=C_OOB, ms=4, lw=1.6,
            label="Error OOB")
    a1.plot(curva.n_arboles, curva.err_test, "s--", color=C_TEST, ms=4, lw=1.4,
            label="Error de test")
    a1.set_xscale("log")
    a1.set_xlabel("Cantidad de árboles (escala logarítmica)")
    a1.set_ylabel("Tasa de error")
    a1.grid(alpha=0.25, lw=0.6)
    a1.legend(frameon=False, fontsize=8.5)

    a2.plot(barrido.m, barrido.correlacion, "o-", color=C_OOB, ms=4, lw=1.6,
            label=r"Correlación $\bar{\rho}$")
    a2.plot(barrido.m, barrido.fuerza, "s-", color=C_TEST, ms=4, lw=1.6,
            label="Fuerza $s$")
    a2.set_xlabel("$m$ (variables sorteadas por nodo)")
    a2.set_ylabel(r"$\bar{\rho}$   y   $s$")
    a2.set_ylim(0.20, 0.95)
    a2.grid(alpha=0.25, lw=0.6)
    a3 = a2.twinx()
    a3.plot(barrido.m, barrido.cota_breiman, "^:", color=C_AUX, ms=4, lw=1.2,
            label=r"Cota $\bar{\rho}(1-s^2)/s^2$")
    a3.set_ylabel("Cota de Breiman", color=C_AUX)
    a3.tick_params(axis="y", colors=C_AUX)
    a3.set_ylim(0.150, 0.195)
    lineas = a2.get_lines() + a3.get_lines()
    a2.legend(lineas, [l.get_label() for l in lineas], frameon=False,
              fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "bosque.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# E4: Gini contra permutacion, con variables ruido plantadas
# ----------------------------------------------------------------------------
def exp_importancias(X_tr, y_tr, X_te, y_te, nombres):
    """Se agregan tres variables sin ninguna relacion con el diagnostico, que
    difieren solo en cuantos cortes distintos ofrecen: continua, cuatro niveles
    y binaria. Si el sesgo de Gini existe, la importancia Gini tiene que
    ordenarlas por cardinalidad y la de permutacion tiene que dejarlas en cero."""
    rng = np.random.default_rng(SEED)
    n_tr, n_te = len(X_tr), len(X_te)

    def ruidos(n):
        return np.column_stack(
            [
                rng.normal(size=n),  # continua
                rng.integers(0, 4, size=n).astype(float),  # 4 niveles
                rng.integers(0, 2, size=n).astype(float),  # binaria
            ]
        )

    nombres_ruido = ["ruido continuo", "ruido 4 niveles", "ruido binario"]
    Xtr2 = np.hstack([X_tr, ruidos(n_tr)])
    Xte2 = np.hstack([X_te, ruidos(n_te)])
    cols = nombres + nombres_ruido

    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf.fit(Xtr2, y_tr)

    gini = rf.feature_importances_
    # La permutacion se calcula sobre el test: sobre entrenamiento un arbol sin
    # podar memoriza y la medida pierde sentido.
    perm = permutation_importance(
        rf, Xte2, y_te, n_repeats=30, random_state=SEED, n_jobs=-1
    )
    df = pd.DataFrame(
        {
            "variable": cols,
            "gini": gini,
            "permutacion": perm.importances_mean,
            "permutacion_sd": perm.importances_std,
            "es_ruido": [c in nombres_ruido for c in cols],
        }
    )
    df["rank_gini"] = df.gini.rank(ascending=False).astype(int)
    df["rank_perm"] = df.permutacion.rank(ascending=False).astype(int)
    df.to_csv(OUT / "importancias.csv", index=False)

    # Primera version de la figura: barras Gini y permutacion lado a lado para
    # las ocho variables principales. Quedo ilegible, porque la Gini reparte una
    # unidad entre 33 variables y la permutacion mide caida de exactitud: las
    # barras de permutacion desaparecian contra las de Gini. Se cambio por dos
    # paneles que comparan lo unico comparable, el ordenamiento, y por el detalle
    # de las tres variables de ruido en su propia escala.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for _a in (a1, a2):
        _a.tick_params(labelsize=11)

    a1.scatter(df[~df.es_ruido].rank_gini, df[~df.es_ruido].rank_perm,
               s=22, color=C_OOB, zorder=3, label="Variables del conjunto")
    a1.scatter(df[df.es_ruido].rank_gini, df[df.es_ruido].rank_perm,
               s=40, color=C_TEST, marker="D", zorder=4, label="Ruido plantado")
    a1.plot([1, len(df)], [1, len(df)], color=C_AUX, lw=0.9, ls="--", zorder=2)
    for v in ["worst texture", "mean texture"]:
        f = df[df.variable == v].iloc[0]
        a1.annotate(v, (f.rank_gini, f.rank_perm), fontsize=10,
                    xytext=(6, -1), textcoords="offset points")
    a1.set_xlabel("Puesto por importancia Gini", fontsize=11)
    a1.set_ylabel("Puesto por permutación", fontsize=11)
    a1.grid(alpha=0.25, lw=0.6)
    a1.legend(frameon=False, fontsize=10, loc="upper left")

    ruido = df[df.es_ruido].iloc[::-1]
    y = np.arange(len(ruido))
    a2.barh(y - 0.19, ruido.gini, height=0.36, color=C_OOB, label="Gini")
    a2.barh(y + 0.19, ruido.permutacion, height=0.36, color=C_TEST, label="Permutación")
    for yi, p in zip(y, ruido.permutacion):
        a2.annotate(f"{p:.3f}".replace(".", ","), (0, yi + 0.19), fontsize=10,
                    xytext=(3, -2), textcoords="offset points", color=C_TEST)
    a2.set_yticks(y)
    a2.set_yticklabels(ruido.variable, fontsize=11)
    a2.set_xlabel("Importancia", fontsize=11)
    a2.axvline(0, color="black", lw=0.8)
    a2.grid(axis="x", alpha=0.25, lw=0.6)
    a2.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "importancias.png", dpi=200)
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E3 bis: OOB contra test sobre varias particiones
# ----------------------------------------------------------------------------
def exp_oob_insesgado():
    """El barrido de m deja el error OOB por debajo del de test en los 15 puntos,
    siempre con el mismo signo. Eso puede ser un sesgo del estimador OOB o puede
    ser que la particion 70/30 con semilla 42 haya dejado un test dificil. Se
    distingue con 20 particiones distintas del mismo conjunto: si el promedio de
    la brecha se va a cero, el signo del barrido era de la particion."""
    d = load_breast_cancer()
    filas = []
    for k in range(20):
        X_tr, X_te, y_tr, y_te = train_test_split(
            d.data, d.target, test_size=0.30, random_state=SEED + k, stratify=d.target
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", oob_score=True,
            bootstrap=True, random_state=SEED, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        filas.append(
            {
                "particion": k,
                "err_oob": 1 - rf.oob_score_,
                "err_test": 1 - rf.score(X_te, y_te),
            }
        )
    df = pd.DataFrame(filas)
    df["brecha"] = df.err_oob - df.err_test
    df.to_csv(OUT / "oob_particiones.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# E5: contraste con bagging puro
# ----------------------------------------------------------------------------
def exp_bagging(X_tr, y_tr, X_te, y_te):
    """max_features=None hace que cada nodo evalue las 30 variables: eso es
    bagging de arboles, sin el subespacio aleatorio. Unica diferencia con la
    corrida de al lado, que usa sqrt(30) = 5."""
    filas = []
    for etiqueta, mf in [("Bagging de arboles", None), ("Random Forest", "sqrt")]:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_features=mf,
            oob_score=True,
            bootstrap=True,
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        s, rho, cota, acierto = fuerza_y_correlacion(rf, X_te, y_te)
        filas.append(
            {
                "metodo": etiqueta,
                "m": X_tr.shape[1] if mf is None else int(np.sqrt(X_tr.shape[1])),
                "err_oob": 1 - rf.oob_score_,
                "err_test": 1 - rf.score(X_te, y_te),
                "fuerza": s,
                "correlacion": rho,
                "acierto_arbol_solo": acierto,
            }
        )
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "bagging_vs_rf.csv", index=False)
    return df


def main() -> None:
    X_tr, X_te, y_tr, y_te, nombres = cargar()
    print(f"train {X_tr.shape}  test {X_te.shape}")

    curva = exp_curva_oob(X_tr, y_tr, X_te, y_te)
    barrido = exp_barrido_m(X_tr, y_tr, X_te, y_te)
    figura_bosque(curva, barrido)
    imp = exp_importancias(X_tr, y_tr, X_te, y_te, nombres)
    part = exp_oob_insesgado()
    bag = exp_bagging(X_tr, y_tr, X_te, y_te)

    mejor = barrido.loc[barrido.err_oob.idxmin()]
    # Error estandar binomial del error OOB en el minimo: la vara con la que se
    # juzga si la variacion a lo largo del barrido de m es real o es ruido.
    n_oob = len(X_tr)
    ee = float(np.sqrt(mejor.err_oob * (1 - mejor.err_oob) / n_oob))
    plano = barrido[barrido.err_oob <= mejor.err_oob + ee]
    # coincidencia de ordenamientos entre las dos importancias
    rho_rank = float(imp.gini.rank().corr(imp.permutacion.rank(), method="spearman"))
    # prueba t pareada sobre la brecha OOB - test en las 20 particiones
    from scipy import stats

    t_par = stats.ttest_1samp(part.brecha, 0.0)

    meta = {
        "status": "ok",
        "semilla": SEED,
        "datos": "Breast Cancer Wisconsin (Diagnostic), sklearn",
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_variables": int(X_tr.shape[1]),
        "curva_oob": {
            "err_oob_10": float(curva.err_oob.iloc[0]),
            "err_oob_1000": float(curva.err_oob.iloc[-1]),
            "err_oob_min": float(curva.err_oob.min()),
            "arboles_desde_100": int(curva[curva.n_arboles >= 100].n_arboles.min()),
            "amplitud_desde_100": float(
                curva[curva.n_arboles >= 100].err_oob.max()
                - curva[curva.n_arboles >= 100].err_oob.min()
            ),
            "err_test_1000": float(curva.err_test.iloc[-1]),
        },
        "barrido_m": {
            "m_optimo": int(mejor.m),
            "err_oob_optimo": float(mejor.err_oob),
            "err_oob_max": float(barrido.err_oob.max()),
            "amplitud_err_oob": float(barrido.err_oob.max() - barrido.err_oob.min()),
            "ee_binomial": ee,
            "rango_plano": [int(plano.m.min()), int(plano.m.max())],
            "n_plano": int(len(plano)),
            "n_puntos": int(len(barrido)),
            "corr_m1": float(barrido.correlacion.iloc[0]),
            "corr_m30": float(barrido.correlacion.iloc[-1]),
            "fuerza_m1": float(barrido.fuerza.iloc[0]),
            "fuerza_m30": float(barrido.fuerza.iloc[-1]),
            "cota_min": float(barrido.cota_breiman.min()),
        },
        "oob_vs_test": {
            "brecha_media_barrido": float(barrido.brecha_oob_test.mean()),
            "n_brechas_negativas_barrido": int((barrido.brecha_oob_test < 0).sum()),
            "particiones": {
                "n": int(len(part)),
                "err_oob_medio": float(part.err_oob.mean()),
                "err_test_medio": float(part.err_test.mean()),
                "brecha_media": float(part.brecha.mean()),
                "brecha_sd": float(part.brecha.std(ddof=1)),
                "brecha_abs_media": float(part.brecha.abs().mean()),
                "t": float(t_par.statistic),
                "p": float(t_par.pvalue),
                "gl": int(len(part) - 1),
            },
        },
        "importancias": {
            "rank_gini_ruido_continuo": int(
                imp.loc[imp.variable == "ruido continuo", "rank_gini"].iloc[0]
            ),
            "rank_perm_ruido_continuo": int(
                imp.loc[imp.variable == "ruido continuo", "rank_perm"].iloc[0]
            ),
            "gini_ruido_continuo": float(
                imp.loc[imp.variable == "ruido continuo", "gini"].iloc[0]
            ),
            "gini_ruido_binario": float(
                imp.loc[imp.variable == "ruido binario", "gini"].iloc[0]
            ),
            "gini_cat4": float(imp.loc[imp.variable == "ruido 4 niveles", "gini"].iloc[0]),
            "razon_gini_continuo_binario": float(
                imp.loc[imp.variable == "ruido continuo", "gini"].iloc[0]
                / imp.loc[imp.variable == "ruido binario", "gini"].iloc[0]
            ),
            "perm_max_ruido": float(imp[imp.es_ruido].permutacion.max()),
            "n_variables_gini_menor_que_ruido": int(
                (imp[~imp.es_ruido].gini
                 < imp.loc[imp.variable == "ruido continuo", "gini"].iloc[0]).sum()
            ),
            "spearman_gini_perm": rho_rank,
        },
        "bagging": bag.to_dict(orient="records"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
