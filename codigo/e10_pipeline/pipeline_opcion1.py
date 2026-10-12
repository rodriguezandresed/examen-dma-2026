"""
Ejercicio 10, Opción 1: pipeline comparativo de SVM, Random Forest, Gradient
Boosting y MLP sobre una tarea de clasificación y una de regresión.

Esquema de validación:
  1. Hold-out externo del 20 % (estratificado en clasificación), que no se toca
     durante la búsqueda de hiperparámetros.
  2. Sobre el 80 % de entrenamiento, RandomizedSearchCV con k-fold de 5 bloques.
  3. La configuración elegida se reentrena sobre todo el 80 % y se evalúa una
     sola vez sobre el hold-out.
El preproceso (StandardScaler) vive dentro del Pipeline. Semilla 42.

Correr desde la raíz del repositorio:
  python codigo/e10_pipeline/pipeline_opcion1.py

Salidas en codigo/resultados/e10/: comparacion_modelos.csv, espacio_busqueda.csv,
hiperparametros_elegidos.csv, comparacion_modelos.png y meta.json.
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from sklearn.datasets import fetch_california_housing, load_breast_cancer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (  # noqa: E402
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC, SVR  # noqa: E402

SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
# 30 combinaciones por modelo: con grillas de 24 a 150 puntos alcanza para que
# cada valor de cada hiperparámetro aparezca varias veces (Bergstra y Bengio,
# 2012). Cuando la grilla tiene 30 puntos o menos, la búsqueda aleatoria la
# recorre completa y equivale a un GridSearchCV; eso se registra en el CSV.
N_ITER = 30
N_JOBS = -1  # 8 núcleos; cada fit es independiente y lleva su propia semilla.
# Tiempos observados en la corrida que generó los resultados del informe (Apple
# Silicon, 8 núcleos): clasificación SVM 2 s, RF 7 s, GB 6 s, MLP 5 s; regresión
# SVM 39 s, RF 66 s, GB 172 s, MLP 31 s; total 331 s. Una corrida previa, con la
# máquina ocupada por otros procesos, tardó 637 s (GB de regresión 313 s): el
# tiempo depende de la carga.
# Los valores de cada corrida quedan en meta.json.
# California Housing tiene 20 640 filas. En una prueba exploratoria aparte, un
# SVR con C = 1000 y gamma = 1 tardó 46,5 s por fit sobre un bloque de
# entrenamiento de una submuestra de 8000 filas (5120 filas) y 80,7 s con una de
# 12 000. Se submuestrea a 8000 antes de la partición para que el SVR entre en
# el presupuesto de tiempo y los cuatro modelos compitan sobre los mismos datos.
# Esa prueba no se repite en cada corrida (cuesta unos 2 min); sus tiempos se
# copian a meta.json para que el informe los lea de ahí.
N_SUB_REG = 8000
TIEMPO_SVR_PRUEBA = {
    "C": 1000, "gamma": 1.0, "filas_submuestra": [8000, 12000],
    "segundos_por_fit": [46.5, 80.7],
}

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e10"

ETIQUETAS = {
    "SVM": "SVM",
    "RF": "Random Forest",
    "GB": "Gradient Boosting",
    "MLP": "MLP",
}


def _cv(tarea: str):
    if tarea == "clasificacion":
        return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    return KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)


def specs_clasificacion(p: int) -> list[dict]:
    """Modelos y grillas para Breast Cancer (p = 30 predictores)."""
    return [
        {
            "modelo": "SVM",
            # SVC sin probability=True no usa random_state: no se fija.
            "pipe": Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
            # Grilla logarítmica en C y gamma, como sugieren Hsu et al. (2003).
            # En la corrida anterior el óptimo fue C = 10 y gamma = 0,01 sobre
            # una grilla de C hasta 100: se agregan 0,01 y 1000 en C y 0,1 y 1
            # en gamma para que ese óptimo quede en el interior.
            "grilla": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "model__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            },
            "fijos": {"model__kernel": "rbf"},
            "limites": {},
        },
        {
            "modelo": "RF",
            "pipe": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300, random_state=SEED, n_jobs=1
                        ),
                    )
                ]
            ),
            # Más árboles no sobreajustan, solo cuestan tiempo (Breiman, 2001):
            # se fija n_estimators = 300 y se busca sobre m (max_features) y el
            # tamaño mínimo de hoja. El max_depth = 20 de la corrida anterior
            # quedó en el borde; se reemplaza por min_samples_leaf, que controla
            # la misma complejidad sin un tope arbitrario de profundidad.
            # m = 30 es bagging puro; m = 2 casi no deja elegir variable.
            "grilla": {
                "model__max_features": [2, 4, 6, 10, 15, p],
                "model__min_samples_leaf": [1, 2, 4, 8],
            },
            "fijos": {"model__n_estimators": 300},
            # Bordes que no se pueden extender: m = p y hoja de 1 observación.
            "limites": {
                "model__max_features": [p],
                "model__min_samples_leaf": [1],
            },
        },
        {
            "modelo": "GB",
            "pipe": Pipeline(
                [("model", GradientBoostingClassifier(random_state=SEED))]
            ),
            # Antes ganó lr 0,1 / profundidad 4 / 50 etapas, con 0,1 y 4 como
            # máximos de la grilla. Se amplía la profundidad a 6 y se agregan 25
            # y 400 etapas; profundidad 1 son stumps (Ejercicio 8). Con lr hasta
            # 0,3 una corrida previa eligió 0,3, otra vez en el borde: se agregó
            # 1,0, el tope natural de nu (paso completo, sin contracción). En la
            # corrida final ganó de nuevo 0,3, ahora en el interior.
            "grilla": {
                "model__n_estimators": [25, 50, 100, 200, 400],
                "model__learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0],
                "model__max_depth": [1, 2, 3, 4, 6],
            },
            "fijos": {},
            "limites": {"model__max_depth": [1], "model__learning_rate": [1.0]},
        },
        {
            "modelo": "MLP",
            # Prueba de convergencia (función prueba_parada_temprana, 5 folds,
            # una capa de 64, sin búsqueda): con early_stopping el F1 medio cae
            # fuerte con lr 0,0003; sin early_stopping sube y no hay avisos de
            # no convergencia con max_iter = 1000. Con 364 filas de entrenamiento en
            # cada vuelta, el 10 % interno de validación son unas 36 filas y corta pronto.
            # Se apaga early_stopping y se sube max_iter a 2000 por margen. La
            # configuración elegida terminó en 407 épocas, lejos del tope.
            "pipe": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            max_iter=2000, early_stopping=False, random_state=SEED
                        ),
                    ),
                ]
            ),
            # El lr 0,01 elegido antes era el máximo; ahora llega a 0,03. Una
            # corrida previa eligió 0,0003, el mínimo de entonces: se agregó
            # 0,0001 y en la corrida final 0,0003 ya queda en el interior.
            # alpha recorre seis órdenes de magnitud, de casi nula a fuerte.
            "grilla": {
                "model__hidden_layer_sizes": [(16,), (32,), (64,), (128,), (64, 32)],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "model__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
            },
            "fijos": {"model__max_iter": 2000, "model__early_stopping": False},
            "limites": {},
        },
    ]


def specs_regresion(p: int) -> list[dict]:
    """Modelos y grillas para California Housing (p = 8 predictores)."""
    return [
        {
            "modelo": "SVM",
            "pipe": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
            # Antes ganó C = 10 y epsilon = 0,2, ambos en el máximo. Se sube C a
            # 100 y no a 1000: C = 1000 con gamma = 1 tardó 46 s por fit y la
            # búsqueda completa no entraba en el presupuesto. epsilon está en
            # unidades de la respuesta (100 000 USD): 0,8 es un tubo muy ancho.
            # gamma = 1/8 (la regla "scale" con datos estandarizados) cae entre
            # 0,1 y 0,3 de esta grilla.
            "grilla": {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": [0.01, 0.03, 0.1, 0.3, 1.0],
                "model__epsilon": [0.05, 0.1, 0.2, 0.4, 0.8],
            },
            "fijos": {"model__kernel": "rbf"},
            "limites": {},
        },
        {
            "modelo": "RF",
            "pipe": Pipeline(
                [
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300, random_state=SEED, n_jobs=1
                        ),
                    )
                ]
            ),
            # Con p = 8, m recorre todos los valores útiles de 1 a 8 (m = 8 es
            # bagging). La hoja llega a 16 para dar margen de suavizado.
            "grilla": {
                "model__max_features": [1, 2, 3, 4, 6, p],
                "model__min_samples_leaf": [1, 2, 4, 8, 16],
            },
            "fijos": {"model__n_estimators": 300},
            "limites": {
                "model__max_features": [1, p],
                "model__min_samples_leaf": [1],
            },
        },
        {
            "modelo": "GB",
            "pipe": Pipeline(
                [("model", GradientBoostingRegressor(random_state=SEED))]
            ),
            # Antes ganó lr 0,1 / profundidad 3 / 200 etapas, los tres en el
            # máximo. Se agregan lr 0,3, profundidades 4 a 8 y más etapas.
            # Con tope en 800 etapas ganó 800 (borde): se agregó 1600 y ganó 1600
            # con lr 0,03, otra vez en el borde. Es el canje entre nu y la
            # cantidad de etapas (Ejercicio 8). No se agregó 3200: la búsqueda de
            # GB ya es la más cara de la corrida (de 3 a 5 min según la carga) y
            # con nu chico más etapas suelen mejorar poco. Se informa como
            # limitación.
            "grilla": {
                "model__n_estimators": [100, 200, 400, 800, 1600],
                "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
                "model__max_depth": [2, 3, 4, 6, 8],
            },
            "fijos": {},
            "limites": {},
        },
        {
            "modelo": "MLP",
            # Con 5120 filas por fold el 10 % de validación interna son 512
            # filas: ahí early_stopping sí es estable y acota el tiempo. La
            # configuración elegida se detuvo en la época 62.
            "pipe": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            max_iter=2000,
                            early_stopping=True,
                            n_iter_no_change=20,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "grilla": {
                "model__hidden_layer_sizes": [
                    (32,), (64,), (128,), (64, 32), (128, 64)
                ],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "model__learning_rate_init": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
            },
            "fijos": {
                "model__max_iter": 2000,
                "model__early_stopping": True,
                "model__n_iter_no_change": 20,
            },
            "limites": {},
        },
    ]


def _fmt(v) -> str:
    """Valor legible para las tablas del informe: coma decimal y punto de miles,
    la misma convención que usa el texto del .Rmd."""
    if isinstance(v, tuple):
        return "(" + ", ".join(str(x) for x in v) + ")"
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, float) and not (v.is_integer() and v >= 1000):
        return np.format_float_positional(v, trim="-").replace(".", ",")
    if isinstance(v, (int, float)) and v >= 1000:
        return f"{int(v):,}".replace(",", ".")
    return str(v)


def _bordes(spec: dict, best: dict, tarea: str) -> list[dict]:
    """Marca qué hiperparámetros elegidos quedaron en un extremo de su grilla."""
    filas = []
    for hp, valores in spec["grilla"].items():
        elegido = best[hp]
        ordinal = all(isinstance(v, (int, float)) for v in valores)
        if ordinal:
            orden = sorted(valores)
            en_borde = elegido in (orden[0], orden[-1])
            extensible = en_borde and elegido not in spec["limites"].get(hp, [])
        else:
            # hidden_layer_sizes no tiene un orden natural único: no se evalúa.
            en_borde, extensible = None, None
        filas.append(
            {
                "tarea": tarea,
                "modelo": ETIQUETAS[spec["modelo"]],
                "hiperparametro": hp.replace("model__", ""),
                "valor": _fmt(elegido),
                "en_borde": en_borde,
                "borde_extensible": extensible,
            }
        )
    return filas


def correr_tarea(tarea, X, y, specs, scoring, meta):
    strat = y if tarea == "clasificacion" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=strat, random_state=SEED
    )
    meta["particion"][tarea] = {"train": int(len(ytr)), "test": int(len(yte))}
    if tarea == "clasificacion":
        meta["particion"][tarea]["malignos_test"] = int(yte.sum())

    filas, grillas, elegidos, mejores = [], [], [], {}
    for spec in specs:
        n_grilla = int(np.prod([len(v) for v in spec["grilla"].values()]))
        n_eval = min(N_ITER, n_grilla)
        search = RandomizedSearchCV(
            spec["pipe"],
            spec["grilla"],
            n_iter=n_eval,
            scoring=scoring,
            refit=list(scoring)[0],
            cv=_cv(tarea),
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        t0 = time.perf_counter()
        search.fit(Xtr, ytr)
        seg = time.perf_counter() - t0
        best = search.best_estimator_
        res, i = search.cv_results_, search.best_index_
        clave = list(scoring)[0]
        signo = -1.0 if tarea == "regresion" else 1.0
        fila = {
            "tarea": tarea,
            "modelo": ETIQUETAS[spec["modelo"]],
            "cv_media": signo * float(res[f"mean_test_{clave}"][i]),
            "cv_sd": float(res[f"std_test_{clave}"][i]),
            "combinaciones_grilla": n_grilla,
            "combinaciones_evaluadas": n_eval,
            "cubre_grilla": n_eval == n_grilla,
            "segundos": round(seg, 1),
            "best_params": json.dumps(
                {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                sort_keys=True,
            ),
        }
        if tarea == "clasificacion":
            pred = best.predict(Xte)
            # roc_auc acepta decision_function (SVC) o predict_proba (resto).
            if hasattr(best, "predict_proba"):
                score = best.predict_proba(Xte)[:, 1]
            else:
                score = best.decision_function(Xte)
            tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
            fila.update(
                {
                    "test": f1_score(yte, pred),
                    "cv_accuracy": float(res["mean_test_accuracy"][i]),
                    "test_accuracy": accuracy_score(yte, pred),
                    "test_auc": roc_auc_score(yte, score),
                    "test_fn": int(fn),
                    "test_fp": int(fp),
                }
            )
        else:
            pred = best.predict(Xte)
            fila.update(
                {
                    "test": root_mean_squared_error(yte, pred),
                    "cv_r2": float(res["mean_test_r2"][i]),
                    "test_r2": r2_score(yte, pred),
                }
            )
        filas.append(fila)
        mejores[spec["modelo"]] = search.best_params_
        elegidos += _bordes(spec, search.best_params_, tarea)
        for hp, valores in {**spec["grilla"], **spec["fijos"]}.items():
            fijo = hp in spec["fijos"]
            grillas.append(
                {
                    "tarea": tarea,
                    "modelo": ETIQUETAS[spec["modelo"]],
                    "hiperparametro": hp.replace("model__", ""),
                    "valores": _fmt(valores)
                    if fijo
                    else "; ".join(_fmt(v) for v in valores),
                    "fijo": fijo,
                }
            )
        if spec["modelo"] == "MLP":
            meta["mlp_epocas_mejor"][tarea] = int(best.named_steps["model"].n_iter_)
        print(
            f"[{tarea}] {spec['modelo']}: CV {fila['cv_media']:.4f} "
            f"± {fila['cv_sd']:.4f} | test {fila['test']:.4f} | {seg:.0f} s"
        )
    return filas, grillas, elegidos, mejores, (Xtr, ytr)


def prueba_escalador_arboles(Xtr, ytr, mejores, meta) -> None:
    """RF y GB con y sin StandardScaler, con la configuración elegida.

    Los árboles cortan por umbrales sobre una variable por vez; un cambio de
    escala monótono mueve los umbrales pero no el orden de las observaciones.
    Se espera el mismo F1. Corrida registrada (clasificación, 5 folds): RF
    0,9552 con y sin escalador; GB 0,9703 con y sin escalador. Los números son
    idénticos: el escalador solo agrega un paso, por eso queda fuera de los
    Pipeline de árboles.
    """
    base = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
        "GB": GradientBoostingClassifier(random_state=SEED),
    }
    salida = {}
    for nombre, est in base.items():
        params = {k.replace("model__", ""): v for k, v in mejores[nombre].items()}
        est.set_params(**params)
        res = {}
        for con in (False, True):
            pasos = [("scaler", StandardScaler())] if con else []
            pipe = Pipeline(pasos + [("model", est)])
            s = cross_val_score(
                pipe, Xtr, ytr, cv=_cv("clasificacion"), scoring="f1", n_jobs=N_JOBS
            )
            res["con_escalador" if con else "sin_escalador"] = round(float(s.mean()), 4)
        salida[ETIQUETAS[nombre]] = res
        print(f"[escalador] {nombre}: {res}")
    meta["prueba_escalador_arboles_f1_cv"] = salida


def prueba_parada_temprana(Xtr, ytr, meta) -> None:
    """MLP (64,) con y sin early_stopping, dos tasas, CV de 5 bloques.

    Corrida registrada: con parada, F1 0,8628 (lr 0,0003) y 0,9620 (lr 0,03);
    sin parada, 0,9644 y 0,9646; 0 avisos de no convergencia. Se corre con
    n_jobs = 1 para poder contar los ConvergenceWarning en este proceso.
    """
    tasas = [3e-4, 3e-2]
    # validation_fraction por defecto (0,1): la fracción interna que usa la parada.
    salida = {"tasas": tasas, "fraccion_validacion": MLPClassifier().validation_fraction}
    avisos = 0
    for es in (True, False):
        f1s = []
        for lr in tasas:
            mlp = MLPClassifier(
                (64,), learning_rate_init=lr, max_iter=1000, early_stopping=es,
                n_iter_no_change=20, random_state=SEED,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("model", mlp)])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                s = cross_val_score(pipe, Xtr, ytr, cv=_cv("clasificacion"), scoring="f1")
            avisos += sum("converge" in str(x.message).lower() for x in w)
            f1s.append(round(float(s.mean()), 4))
        salida["con_parada" if es else "sin_parada"] = f1s
    salida["avisos_no_convergencia"] = avisos
    meta["prueba_parada_temprana_mlp_f1_cv"] = salida
    print(f"[parada temprana] {salida}")


def figura(df: pd.DataFrame, outfile: Path) -> None:
    # Con 2 decimales fijos, ticks cada 0,025 salían redondeados (0,575 -> "0,57").
    # Se fija el paso a 0,01 / 0,02 para que cada rótulo sea exacto.
    coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    paso = {"clasificacion": 0.01, "regresion": 0.02}
    paneles = [
        ("clasificacion", "F1 de la clase maligna", "Clasificación: Breast Cancer"),
        ("regresion", "RMSE (100 000 USD)", "Regresión: California Housing"),
    ]
    # Primera versión a 9 x 3,6 pulgadas: en el PDF, reducida al ancho de texto,
    # las etiquetas quedaban ilegibles. Con un lienzo menor y las mismas fuentes,
    # el texto queda más grande en proporción a la figura.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    for ax, (tarea, ylab, titulo) in zip(axes, paneles):
        sub = df[df["tarea"] == tarea].reset_index(drop=True)
        x = np.arange(len(sub))
        ax.errorbar(
            x - 0.1, sub["cv_media"], yerr=sub["cv_sd"], fmt="o", color="#1f4e79",
            capsize=4, label="CV interna (media ± DE)",
        )
        ax.plot(x + 0.1, sub["test"], "D", color="#c0504d", label="Hold-out (test)")
        ax.set_xticks(x, [m.replace(" ", "\n") for m in sub["modelo"]], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(titulo, fontsize=10)
        ax.yaxis.set_major_locator(MultipleLocator(paso[tarea]))
        ax.yaxis.set_major_formatter(coma)
        ax.grid(axis="y", alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> None:
    t_total = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": SEED,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "n_iter": N_ITER,
        "particion": {},
        "mlp_epocas_mejor": {},
    }

    # Clasificación. En load_breast_cancer el 0 es maligno y el 1 benigno; se
    # recodifica para que la clase positiva (1) sea maligno y el F1 se calcule
    # sobre la clase que interesa detectar.
    bc = load_breast_cancer()
    Xc, yc = bc.data, (bc.target == 0).astype(int)
    meta["breast_cancer"] = {
        "n": int(len(yc)), "p": int(Xc.shape[1]), "malignos": int(yc.sum())
    }
    sc_clf = {"f1": "f1", "accuracy": "accuracy"}
    f_c, g_c, e_c, mej_c, (Xtr_c, ytr_c) = correr_tarea(
        "clasificacion", Xc, yc, specs_clasificacion(Xc.shape[1]), sc_clf, meta
    )
    prueba_escalador_arboles(Xtr_c, ytr_c, mej_c, meta)
    prueba_parada_temprana(Xtr_c, ytr_c, meta)
    meta["tiempo_svr_prueba_previa_s"] = TIEMPO_SVR_PRUEBA

    # Regresión. La respuesta es el valor mediano de la vivienda por grupo
    # censal, en unidades de 100 000 USD (tope en 5,00001 en los datos).
    Xr, yr = fetch_california_housing(return_X_y=True)
    meta["california_housing"] = {"n": int(len(yr)), "p": int(Xr.shape[1])}
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(yr), size=N_SUB_REG, replace=False)
    Xr, yr = Xr[idx], yr[idx]
    meta["california_housing"]["n_submuestra"] = N_SUB_REG
    sc_reg = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
    f_r, g_r, e_r, _, _ = correr_tarea(
        "regresion", Xr, yr, specs_regresion(Xr.shape[1]), sc_reg, meta
    )

    comp = pd.DataFrame(f_c + f_r)
    comp.to_csv(OUT / "comparacion_modelos.csv", index=False)
    pd.DataFrame(g_c + g_r).to_csv(OUT / "espacio_busqueda.csv", index=False)
    pd.DataFrame(e_c + e_r).to_csv(OUT / "hiperparametros_elegidos.csv", index=False)
    figura(comp, OUT / "comparacion_modelos.png")

    meta["segundos_por_busqueda"] = {
        f"{r['tarea']}/{r['modelo']}": r["segundos"] for r in f_c + f_r
    }
    meta["segundos_total"] = round(time.perf_counter() - t_total, 1)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Total: {meta['segundos_total']} s. Salidas en {OUT}")


if __name__ == "__main__":
    main()
