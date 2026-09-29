"""
Ejercicio 8: diferencias entre AdaBoost y Gradient Boosting.

La idea del script NO es comparar accuracy (con estos datos los dos empatan y no
se aprende nada). Es mostrar la diferencia de mecanismo:

  - AdaBoost cambia la DISTRIBUCION D_t sobre las observaciones y deja el target fijo.
  - Gradient Boosting deja los pesos uniformes y cambia el TARGET (pseudo-residuo).

Lo que calcula, en este orden:
  1. AdaBoost desde cero sobre stumps, con la notacion del pseudocodigo de las
     slides (D_1(i)=1/m, eps_t, alpha_t = 1/2 log((1-eps)/eps), Z_t), y cotejo
     contra AdaBoostClassifier de sklearn.
  2. El ejemplo de Schapire (1998) de las slides: se le aplica la formula de alpha
     a los eps impresos. Da distinto en las rondas 2 y 3 (ver nota mas abajo).
  3. Evolucion de D_t (AdaBoost) contra encogimiento de los residuos (GB).
  4. Verificacion numerica de que el gradiente de la MSE es el vector residual,
     y el de la MAE el vector de signos.
  5. Curvas de error de train y test contra iteraciones, con y sin ruido en las
     etiquetas, para ver si aparece el aumento del error de test que avisan las
     slides.
  6. Grilla de tasa de aprendizaje v contra n_estimators en Gradient Boosting.

Correr:  python codigo/e8_boosting/adaboost_vs_gb.py
Salida:  codigo/resultados/e8/*.csv  (las figuras las arma figuras.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import approx_fprime
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42
M_ADA = 50           # rondas de AdaBoost para el cotejo contra sklearn
M_CURVA = 400        # rondas para las curvas de error (hace falta pasarse de largo)
M_GB = 200           # iteraciones de Gradient Boosting
V_GB = 0.1           # tasa de aprendizaje de referencia (el v de las slides)
RUIDO = 0.10         # proporcion de etiquetas de train que se dan vuelta

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e8"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. AdaBoost desde cero, con la notacion del pseudocodigo de las slides
# ---------------------------------------------------------------------------

def adaboost_scratch(X, y_pm1, M, seed=SEED, convencion="slides"):
    """AdaBoost discreto sobre decision stumps.

    y_pm1 viene en {-1, +1}, como en el pseudocodigo de la catedra.

    convencion="slides"  -> alpha = 1/2 log((1-eps)/eps),  D <- D*exp(-alpha*y*h)/Z
    convencion="sklearn" -> alpha =     log((1-eps)/eps),  D <- D*exp(alpha*[h!=y])/Z

    Las dos se corren para ver que dan la MISMA distribucion D_t. sklearn usa
    SAMME con alpha = lr*log((1-eps)/eps) + log(K-1), y para K=2 el log(K-1) se
    anula: queda exactamente el doble del alpha de las slides. Como el factor 2
    es global, el signo del voto ponderado no cambia.
    """
    m = len(y_pm1)
    D = np.full(m, 1.0 / m)          # D_1(i) = 1/m
    stumps, alphas, historia = [], [], []
    for t in range(M):
        h = DecisionTreeClassifier(max_depth=1, random_state=seed)
        h.fit(X, y_pm1, sample_weight=D)
        pred = h.predict(X)
        mal = pred != y_pm1
        eps = float(D[mal].sum())
        # Guarda: con eps=0 el alpha explota. No paso con estos datos, pero el
        # pseudocodigo de las slides tampoco lo contempla y conviene cortar.
        if eps <= 0 or eps >= 0.5:
            break
        if convencion == "slides":
            alpha = 0.5 * np.log((1 - eps) / eps)
            D = D * np.exp(-alpha * y_pm1 * pred)
        else:
            alpha = np.log((1 - eps) / eps)
            D = D * np.exp(alpha * mal)
        D = D / D.sum()               # Z_t, el factor de normalizacion
        stumps.append(h)
        alphas.append(alpha)
        historia.append({"ronda": t + 1, "eps": eps, "alpha": alpha, "D": D.copy(),
                         "mal": mal.copy()})
    return stumps, np.array(alphas), historia


def adaboost_predict(stumps, alphas, X):
    """H(x) = sign(sum_t alpha_t h_t(x))."""
    voto = np.sum([a * h.predict(X) for a, h in zip(alphas, stumps)], axis=0)
    return np.where(voto >= 0, 1, -1)


def bloque_adaboost(Xtr, ytr, Xte, yte, verif):
    ytr_pm, yte_pm = np.where(ytr == 1, 1, -1), np.where(yte == 1, 1, -1)

    st_sl, al_sl, hist_sl = adaboost_scratch(Xtr, ytr_pm, M_ADA, convencion="slides")
    st_sk, al_sk, hist_sk = adaboost_scratch(Xtr, ytr_pm, M_ADA, convencion="sklearn")

    # Las dos convenciones tienen que dar la misma D_t ronda a ronda.
    dif_D = max(np.abs(a["D"] - b["D"]).max() for a, b in zip(hist_sl, hist_sk))
    razon_alpha = float(np.max(np.abs(al_sk / al_sl)))

    # sklearn: en 1.7 el parametro algorithm esta deprecado y el default ya es
    # SAMME, asi que no se pasa (pasarlo tira FutureWarning).
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=M_ADA,
        learning_rate=1.0,
        random_state=SEED,
    ).fit(Xtr, ytr_pm)

    p_scratch = adaboost_predict(st_sl, al_sl, Xte)
    p_sklearn = ada.predict(Xte)
    coincidencia = float((p_scratch == p_sklearn).mean())

    acc_scratch = accuracy_score(yte_pm, p_scratch)
    acc_sklearn = accuracy_score(yte_pm, p_sklearn)

    verif.append(("D_t con la convención de las diapositivas y con la de sklearn",
                  "0", f"{dif_D:.2e}", dif_D < 1e-12))
    verif.append(("alpha de sklearn sobre alpha de las diapositivas",
                  "2", f"{razon_alpha:.4f}", abs(razon_alpha - 2) < 1e-9))
    verif.append(("Predicciones de test iguales a las de AdaBoostClassifier",
                  "100 %", f"{100 * coincidencia:.0f} %", coincidencia == 1.0))
    verif.append(("Accuracy de test de la implementación propia",
                  f"{acc_sklearn:.4f}", f"{acc_scratch:.4f}", acc_scratch == acc_sklearn))

    return st_sl, al_sl, hist_sl, ada, acc_scratch, acc_sklearn


# ---------------------------------------------------------------------------
# 2. El ejemplo de Schapire (1998) tal como aparece en las slides
# ---------------------------------------------------------------------------

def bloque_schapire():
    """Las slides dan (eps, alpha) = (0,30 / 0,42), (0,21 / 0,65), (0,14 / 0,92).

    No se reconstruyen las coordenadas de los diez puntos: las slides muestran el
    dibujo pero no los valores, y una reconstruccion a ojo no prueba nada. Lo que
    si se puede chequear es la formula: alpha = 1/2 log((1-eps)/eps) sobre los eps
    impresos. Rondas 2 y 3 no cierran, y el despeje inverso muestra por que.
    """
    eps_slide = np.array([0.30, 0.21, 0.14])
    alpha_slide = np.array([0.42, 0.65, 0.92])
    alpha_formula = 0.5 * np.log((1 - eps_slide) / eps_slide)
    # Despeje: eps = 1 / (1 + exp(2*alpha))
    eps_implicito = 1.0 / (1.0 + np.exp(2 * alpha_slide))
    return pd.DataFrame({
        "ronda": [1, 2, 3],
        "eps_slide": eps_slide,
        "alpha_slide": alpha_slide,
        "alpha_formula": alpha_formula,
        "eps_implicito": eps_implicito,
        "brecha_alpha": alpha_formula - alpha_slide,
    })


# ---------------------------------------------------------------------------
# 3. Gradient Boosting desde cero (perdida cuadratica) y encogimiento del residuo
# ---------------------------------------------------------------------------

def gb_scratch(Xtr, ytr, Xte, yte, M=M_GB, v=V_GB, prof=2, seed=SEED):
    """F_m(x) = F_{m-1}(x) + v * beta_m * h(x; a_m), con h ajustado por minimos
    cuadrados sobre el pseudo-residuo r_m = y - F_{m-1}(x).

    Con perdida cuadratica beta_m = 1 (el arbol ya predice el residuo medio de la
    hoja), asi que el unico coeficiente que queda es v.
    """
    F0 = float(np.mean(ytr))          # el f_0 del apunte: la media minimiza la MSE
    Ftr = np.full(len(ytr), F0)
    Fte = np.full(len(yte), F0)
    filas = []
    for m in range(1, M + 1):
        r = ytr - Ftr                 # pseudo-residuo
        h = DecisionTreeRegressor(max_depth=prof, random_state=seed).fit(Xtr, r)
        Ftr = Ftr + v * h.predict(Xtr)
        Fte = Fte + v * h.predict(Xte)
        filas.append({
            "iter": m,
            "sd_residuo": float(np.std(r)),
            "mae_residuo": float(np.mean(np.abs(r))),
            "peso_obs": 1.0 / len(ytr),    # nunca cambia: es la diferencia con AdaBoost
            "mse_train": float(mean_squared_error(ytr, Ftr)),
            "mse_test": float(mean_squared_error(yte, Fte)),
        })
    return F0, pd.DataFrame(filas)


def bloque_apartamentos(verif):
    """Los cinco alquileres de Parr y Howard, que usa el apunte de la catedra.

    Sirve para dejar anclado el arranque: F_0 = media = 1418 con perdida
    cuadratica, y F_0 = mediana = 1280 con perdida absoluta.
    """
    sup = np.array([750, 800, 850, 900, 950], dtype=float)
    renta = np.array([1160, 1200, 1280, 1450, 2000], dtype=float)
    media, mediana = float(np.mean(renta)), float(np.median(renta))
    verif.append(("F_0 con pérdida cuadrática (media de los alquileres)",
                  "1418", f"{media:.0f}", media == 1418))
    verif.append(("F_0 con pérdida absoluta (mediana de los alquileres)",
                  "1280", f"{mediana:.0f}", mediana == 1280))
    r1 = renta - media
    return pd.DataFrame({"sup": sup, "renta": renta, "F0": media, "residuo_1": r1})


# ---------------------------------------------------------------------------
# 4. El gradiente de la MSE es el vector residual
# ---------------------------------------------------------------------------

def bloque_gradiente(verif, n=40, seed=SEED):
    """Deriva numericamente L(y_hat) y compara contra el residuo y contra el signo.

    Con L_MSE = 1/2 * sum (y - y_hat)^2 el gradiente es -(y - y_hat); con
    L_MAE = sum |y - y_hat| es -sign(y - y_hat). El apunte se saltea el 1/2 y el
    N porque no cambian la direccion; aca se ponen para que el cotejo de exacto.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    y_hat = rng.normal(size=n)

    L_mse = lambda p: 0.5 * np.sum((y - p) ** 2)
    L_mae = lambda p: np.sum(np.abs(y - p))

    paso = np.full(n, 1e-6)
    g_mse = approx_fprime(y_hat, L_mse, paso)
    g_mae = approx_fprime(y_hat, L_mae, paso)

    err_mse = float(np.max(np.abs(-g_mse - (y - y_hat))))
    err_mae = float(np.max(np.abs(-g_mae - np.sign(y - y_hat))))

    verif.append(("Gradiente de la MSE contra el vector residual (máx. dif.)",
                  "0", f"{err_mse:.2e}", err_mse < 1e-5))
    verif.append(("Gradiente de la MAE contra el vector de signos (máx. dif.)",
                  "0", f"{err_mae:.2e}", err_mae < 1e-5))
    return err_mse, err_mae


# ---------------------------------------------------------------------------
# 5. Curvas de error contra iteraciones, con y sin ruido en las etiquetas
# ---------------------------------------------------------------------------

def curvas_error(Xtr, ytr, Xte, yte, M=M_CURVA, seed=SEED):
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=M,
        learning_rate=1.0,
        random_state=seed,
    ).fit(Xtr, ytr)
    err_tr = np.array([1 - s for s in ada.staged_score(Xtr, ytr)])
    err_te = np.array([1 - s for s in ada.staged_score(Xte, yte)])
    return err_tr, err_te


def bloque_curvas(Xtr, ytr, Xte, yte, seed=SEED):
    tr_l, te_l = curvas_error(Xtr, ytr, Xte, yte)

    # Con las etiquetas limpias el error de test no sube; para que el aviso de las
    # slides se vea hay que ensuciar el train. Se dan vuelta el 10 % de las y.
    rng = np.random.default_rng(seed)
    y_ruido = ytr.copy()
    idx = rng.choice(len(ytr), size=int(RUIDO * len(ytr)), replace=False)
    y_ruido[idx] = 1 - y_ruido[idx]
    tr_r, te_r = curvas_error(Xtr, y_ruido, Xte, yte)

    n = min(len(tr_l), len(tr_r))
    return pd.DataFrame({
        "m": np.arange(1, n + 1),
        "err_train_limpio": tr_l[:n], "err_test_limpio": te_l[:n],
        "err_train_ruido": tr_r[:n], "err_test_ruido": te_r[:n],
    })


# ---------------------------------------------------------------------------
# 6. Tasa de aprendizaje v contra cantidad de iteraciones
# ---------------------------------------------------------------------------

def bloque_lr(Xtr, ytr, Xte, yte, seed=SEED):
    filas = []
    for v in (1.0, 0.5, 0.1, 0.05, 0.01):
        gb = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=v, max_depth=2, random_state=seed
        ).fit(Xtr, ytr)
        mses = [mean_squared_error(yte, p) for p in gb.staged_predict(Xte)]
        for M in (50, 100, 250, 500, 1000):
            filas.append({"v": v, "n_estimators": M, "mse_test": mses[M - 1]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------

def main():
    verif = []

    # --- clasificacion: Breast Cancer Wisconsin, el mismo dataset de la clase 4
    bc = load_breast_cancer()
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        bc.data, bc.target, test_size=0.30, random_state=SEED, stratify=bc.target
    )

    st, al, hist, ada, acc_scratch, acc_sklearn = bloque_adaboost(
        Xtr_c, ytr_c, Xte_c, yte_c, verif
    )

    # Evolucion de D_t. Se separan las observaciones segun si el PRIMER stump las
    # erro o no: son las "dificiles" que el algoritmo va a empezar a mirar.
    mal_r1 = hist[0]["mal"]
    k = max(1, int(0.10 * len(ytr_c)))
    n_tr = len(ytr_c)
    # Fila 0: el punto de partida D_1(i) = 1/m, antes de la primera reponderacion.
    # En las filas siguientes, la ronda t trae eps_t y alpha_t junto con D_{t+1},
    # que es la distribucion que ya sale actualizada hacia la ronda que viene.
    filas_D = [{
        "ronda": 0, "eps": np.nan, "alpha": np.nan,
        "peso_mal_r1": 1.0 / n_tr, "peso_bien_r1": 1.0 / n_tr,
        "peso_max": 1.0 / n_tr, "share_top10": k / n_tr, "peso_uniforme": 1.0 / n_tr,
    }]
    for h in hist:
        D = h["D"]
        filas_D.append({
            "ronda": h["ronda"],
            "eps": h["eps"],
            "alpha": h["alpha"],
            "peso_mal_r1": float(D[mal_r1].mean()),
            "peso_bien_r1": float(D[~mal_r1].mean()),
            "peso_max": float(D.max()),
            "share_top10": float(np.sort(D)[-k:].sum()),
            "peso_uniforme": 1.0 / n_tr,
        })
    df_D = pd.DataFrame(filas_D)

    df_schapire = bloque_schapire()
    df_apto = bloque_apartamentos(verif)
    err_mse, err_mae = bloque_gradiente(verif)

    # --- regresion: diabetes, para que el pseudo-residuo se lea directo
    db = load_diabetes()
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        db.data, db.target, test_size=0.30, random_state=SEED
    )
    F0, df_gb = gb_scratch(Xtr_r, ytr_r, Xte_r, yte_r)

    # Cotejo de la implementacion propia contra GradientBoostingRegressor.
    gb_sk = GradientBoostingRegressor(
        n_estimators=M_GB, learning_rate=V_GB, max_depth=2,
        random_state=SEED, subsample=1.0, criterion="squared_error",
    ).fit(Xtr_r, ytr_r)
    mse_sk = mean_squared_error(yte_r, gb_sk.predict(Xte_r))
    mse_propio = float(df_gb["mse_test"].iloc[-1])
    rel = abs(mse_propio - mse_sk) / mse_sk
    verif.append(("MSE de test propio contra GradientBoostingRegressor (dif. rel.)",
                  "< 5 %", f"{100 * rel:.2f} %", rel < 0.05))

    df_curvas = bloque_curvas(Xtr_c, ytr_c, Xte_c, yte_c)
    df_lr = bloque_lr(Xtr_r, ytr_r, Xte_r, yte_r)

    df_verif = pd.DataFrame(verif, columns=["chequeo", "esperado", "obtenido", "ok"])

    df_verif.to_csv(OUT / "verificaciones.csv", index=False)
    df_D.to_csv(OUT / "adaboost_pesos.csv", index=False)
    df_schapire.to_csv(OUT / "schapire.csv", index=False)
    df_apto.to_csv(OUT / "apartamentos.csv", index=False)
    df_gb.to_csv(OUT / "gb_residuos.csv", index=False)
    df_curvas.to_csv(OUT / "curvas_error.csv", index=False)
    df_lr.to_csv(OUT / "lr_grilla.csv", index=False)

    mejor = df_lr.loc[df_lr["mse_test"].idxmin()]
    meta = {
        "status": "ok",
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "seed": SEED,
        "dataset_clasificacion": "breast_cancer (569 x 30), split 70/30 estratificado",
        "dataset_regresion": "diabetes (442 x 10), split 70/30",
        "adaboost": {"M": M_ADA, "base": "stump max_depth=1",
                     "acc_test_propio": acc_scratch, "acc_test_sklearn": acc_sklearn,
                     # iloc[1]: la fila 0 es D_1 uniforme y no tiene eps ni alpha,
                     # asi que iloc[0] guardaba NaN y rompia el JSON estricto.
                     "eps_ronda1": float(df_D["eps"].iloc[1]),
                     "alpha_ronda1": float(df_D["alpha"].iloc[1])},
        "gb": {"M": M_GB, "v": V_GB, "max_depth": 2, "F0": F0,
               "mse_test_propio": mse_propio, "mse_test_sklearn": float(mse_sk)},
        "grilla_lr_mejor": {"v": float(mejor["v"]),
                            "n_estimators": int(mejor["n_estimators"]),
                            "mse_test": float(mejor["mse_test"])},
        "gradiente": {"max_dif_mse": err_mse, "max_dif_mae": err_mae},
        "nota_ruido": (
            f"Las curvas con ruido dan vuelta el {int(100 * RUIDO)} % de las etiquetas "
            "de train. Sin ruido el error de test de AdaBoost no sube, que es lo "
            "contrario de lo que avisan las slides."
        ),
        "nota_schapire": (
            "No se reconstruyeron las coordenadas de los diez puntos del ejemplo de "
            "Schapire: las slides no las dan. Solo se chequeo la formula de alpha "
            "sobre los eps impresos."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(df_verif.to_string(index=False))
    print("\nSchapire:\n", df_schapire.to_string(index=False))
    print("\nEscrito en", OUT)


if __name__ == "__main__":
    main()
