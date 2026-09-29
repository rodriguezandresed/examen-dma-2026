"""
Ejercicio 10 - Opcion 1
Pipeline comparativo de SVM, Random Forest, Gradient Boosting y MLP
sobre un problema de clasificacion y uno de regresion.

Examen Data Mining Avanzado 2026 - Universidad Austral

Notas de desarrollo (por que esta armado asi):

- Todo el preprocesamiento va DENTRO del Pipeline. Esto no es un detalle de
  prolijidad: si se escala antes del split, la media y el desvio se calculan
  con datos que despues terminan en el fold de validacion, y eso es fuga de
  informacion. Es el punto que se machaco en la Clase 1 (practica 2 de 2).

- Los arboles (RF y GB) no necesitan escalado, pero el Pipeline se deja igual
  para los cuatro modelos. Cuesta unos milisegundos y evita tener dos caminos
  de codigo distintos, que es donde se cuelan los errores.

- California Housing se submuestrea a 5000 observaciones. Motivo concreto: SVR
  resuelve un problema cuadratico y escala mal con n (entre O(n^2) y O(n^3)).
  Con las 20640 filas completas la busqueda de hiperparametros no terminaba en
  un tiempo razonable en esta maquina. El costo de la decision es que las
  metricas de regresion son algo mas pesimistas y mas variables de lo que
  serian con el dataset completo, pero la COMPARACION entre modelos sigue
  siendo valida porque los cuatro ven exactamente las mismas filas.

- Se usa RandomizedSearchCV y no GridSearchCV. Con 4 modelos y espacios de
  entre 3 y 5 hiperparametros, la grilla completa era de varios miles de
  combinaciones. RandomizedSearch con 20 sorteos cubre el espacio bastante
  bien a una fraccion del costo.
"""

import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, KFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e10"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_ITER = 20      # sorteos de RandomizedSearchCV
N_FOLDS = 5      # k de la validacion cruzada
N_SUB = 5000     # submuestra de California Housing (ver nota de arriba)


# =====================================================================
# 1. CLASIFICACION - Breast Cancer Wisconsin
# =====================================================================
# Dataset elegido porque es el que se uso en la Clase 4 y en el script
# ejemploVSMC.py del profesor, asi que los ordenes de magnitud de las
# metricas son comparables con lo visto en clase.

def correr_clasificacion():
    datos = load_breast_cancer()
    X, y = datos.data, datos.target

    # Hold-out estratificado. El test NO se toca hasta el final: la seleccion
    # de hiperparametros se hace solo con CV sobre el train.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Espacio de busqueda EXPLICITO por modelo (lo pide la consigna).
    configs = {
        "SVM": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
            {"clf__C": np.logspace(-2, 3, 12),
             "clf__gamma": np.logspace(-4, 0, 10),
             "clf__kernel": ["rbf", "poly", "linear"]},
        ),
        "Random Forest": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", RandomForestClassifier(random_state=RANDOM_STATE,
                                                     n_jobs=-1))]),
            {"clf__n_estimators": [200, 400, 800],
             "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
             "clf__min_samples_leaf": [1, 2, 4, 8],
             "clf__max_depth": [None, 8, 16]},
        ),
        "Gradient Boosting": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 300, 600],
             "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
             "clf__max_depth": [2, 3, 4],
             "clf__subsample": [0.6, 0.8, 1.0]},
        ),
        "MLP": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", MLPClassifier(max_iter=2000,
                                            random_state=RANDOM_STATE))]),
            {"clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
             "clf__alpha": np.logspace(-5, 0, 10),
             "clf__learning_rate_init": [0.0005, 0.001, 0.01],
             "clf__activation": ["relu", "tanh"]},
        ),
    }

    filas = []
    mejores = {}
    for nombre, (pipe, espacio) in configs.items():
        t0 = time.time()
        buscador = RandomizedSearchCV(
            pipe, espacio, n_iter=N_ITER, cv=cv, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        buscador.fit(X_tr, y_tr)
        t = time.time() - t0

        # Metricas sobre el hold-out, que es lo unico que no vio el buscador.
        y_pred = buscador.predict(X_te)
        y_prob = buscador.predict_proba(X_te)[:, 1]

        filas.append({
            "Modelo": nombre,
            "AUC_CV": buscador.best_score_,
            "AUC_test": roc_auc_score(y_te, y_prob),
            "Accuracy_test": accuracy_score(y_te, y_pred),
            "F1_test": f1_score(y_te, y_pred),
            "Tiempo_seg": t,
        })
        mejores[nombre] = {k: str(v) for k, v in buscador.best_params_.items()}
        print(f"[clasif] {nombre:18s} AUC_cv={buscador.best_score_:.4f} "
              f"AUC_test={roc_auc_score(y_te, y_prob):.4f} ({t:.1f}s)")

    return pd.DataFrame(filas), mejores, X.shape, len(y_te)


# =====================================================================
# 2. REGRESION - California Housing
# =====================================================================

def correr_regresion():
    datos = fetch_california_housing()
    X, y = datos.data, datos.target

    # Submuestreo (ver nota de encabezado). Se sortea una sola vez y con
    # semilla fija para que los cuatro modelos compitan sobre las mismas filas.
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(y), size=N_SUB, replace=False)
    X, y = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    configs = {
        "SVM": (
            Pipeline([("sc", StandardScaler()), ("reg", SVR())]),
            {"reg__C": np.logspace(-1, 3, 10),
             "reg__gamma": np.logspace(-4, 0, 10),
             "reg__epsilon": [0.01, 0.05, 0.1, 0.2],
             "reg__kernel": ["rbf", "linear"]},
        ),
        "Random Forest": (
            Pipeline([("sc", StandardScaler()),
                      ("reg", RandomForestRegressor(random_state=RANDOM_STATE,
                                                    n_jobs=-1))]),
            {"reg__n_estimators": [200, 400, 800],
             "reg__max_features": ["sqrt", "log2", 0.5, 1.0],
             "reg__min_samples_leaf": [1, 2, 4, 8],
             "reg__max_depth": [None, 12, 20]},
        ),
        "Gradient Boosting": (
            Pipeline([("sc", StandardScaler()),
                      ("reg", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {"reg__n_estimators": [200, 500, 900],
             "reg__learning_rate": [0.01, 0.05, 0.1],
             "reg__max_depth": [2, 3, 4, 5],
             "reg__subsample": [0.6, 0.8, 1.0]},
        ),
        "MLP": (
            Pipeline([("sc", StandardScaler()),
                      ("reg", MLPRegressor(max_iter=2000,
                                           early_stopping=True,
                                           random_state=RANDOM_STATE))]),
            {"reg__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
             "reg__alpha": np.logspace(-5, 0, 10),
             "reg__learning_rate_init": [0.0005, 0.001, 0.01],
             "reg__activation": ["relu", "tanh"]},
        ),
    }

    filas = []
    mejores = {}
    for nombre, (pipe, espacio) in configs.items():
        t0 = time.time()
        buscador = RandomizedSearchCV(
            pipe, espacio, n_iter=N_ITER, cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        buscador.fit(X_tr, y_tr)
        t = time.time() - t0

        y_pred = buscador.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

        filas.append({
            "Modelo": nombre,
            "RMSE_CV": -buscador.best_score_,
            "RMSE_test": rmse,
            "MAE_test": mean_absolute_error(y_te, y_pred),
            "R2_test": r2_score(y_te, y_pred),
            "Tiempo_seg": t,
        })
        mejores[nombre] = {k: str(v) for k, v in buscador.best_params_.items()}
        print(f"[regres] {nombre:18s} RMSE_cv={-buscador.best_score_:.4f} "
              f"RMSE_test={rmse:.4f} R2={r2_score(y_te, y_pred):.4f} ({t:.1f}s)")

    return pd.DataFrame(filas), mejores, X.shape, len(y_te)


if __name__ == "__main__":
    print("=" * 70)
    print("EJERCICIO 10 - OPCION 1")
    print("=" * 70)

    df_clf, mej_clf, shape_clf, n_te_clf = correr_clasificacion()
    print()
    df_reg, mej_reg, shape_reg, n_te_reg = correr_regresion()

    df_clf.to_csv(OUT / "clasificacion.csv", index=False)
    df_reg.to_csv(OUT / "regresion.csv", index=False)
    with io.open(OUT / "hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({"clasificacion": mej_clf, "regresion": mej_reg},
                  f, indent=2, ensure_ascii=False)

    print()
    print("--- CLASIFICACION (Breast Cancer, %d x %d, test n=%d) ---"
          % (shape_clf[0], shape_clf[1], n_te_clf))
    print(df_clf.to_string(index=False))
    print()
    print("--- REGRESION (California Housing submuestreado, %d x %d, test n=%d) ---"
          % (shape_reg[0], shape_reg[1], n_te_reg))
    print(df_reg.to_string(index=False))
