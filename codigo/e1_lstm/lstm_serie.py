"""
Ejercicio 1.VI: LSTM sobre una serie sintética (TensorFlow/Keras).

Ventana de 20, corte temporal 80/20, shuffle=False, cuatro configuraciones,
tres semillas por configuración. La elección se hace por validación; el test
se mira una sola vez al final.

Correr desde la raíz del repo:
  .venv-tf/bin/python codigo/e1_lstm/lstm_serie.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SEED = 42
# Con una sola semilla, lstm_32 y lstm_64 quedaban a menos de 0,001 de MAE y el
# orden podía ser azar de la inicialización. Tres semillas dan al menos una
# idea de esa variación sin multiplicar demasiado el tiempo de corrida.
SEEDS = [42, 43, 44]
LOOKBACK = 20  # una ventana de 20 cubre casi un ciclo del seno rápido (2*pi/0,12 ≈ 52)
EPOCHS = 20
BATCH = 32
NOISE_SD = 0.1
OUT = Path(__file__).resolve().parents[1] / "resultados" / "e1"
OUT.mkdir(parents=True, exist_ok=True)


def lstm_param_count(units: int, input_dim: int) -> int:
    # Fórmula del apunte: 4 bloques (3 puertas + candidata), cada uno con
    # pesos de entrada, pesos recurrentes y sesgo.
    return 4 * ((units * input_dim) + units * units + units)


def params_stack(layers: list[int], input_dim: int = 1) -> int:
    # En una LSTM apilada, la entrada de la segunda capa es la salida
    # de la primera: input_dim pasa a ser las units de la capa anterior.
    total, prev = 0, input_dim
    for units in layers:
        total += lstm_param_count(units, prev)
        prev = units
    return total


def clean_signal(n: int) -> np.ndarray:
    t = np.arange(n)
    return np.sin(0.05 * t) + 0.3 * np.sin(0.12 * t)


def make_series(n: int = 1200) -> np.ndarray:
    # Serie arbitraria: dos senos de distinta frecuencia más ruido gaussiano.
    # Al conocer la señal limpia se puede calcular el piso de error que
    # ningún modelo puede bajar (el ruido no es predecible).
    rng = np.random.default_rng(SEED)
    return clean_signal(n) + NOISE_SD * rng.normal(size=n)


def windowing(series: np.ndarray, lookback: int = LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    return np.asarray(X)[..., np.newaxis], np.asarray(y)


def build_model(layers_spec: list[int], lookback: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    model.add(layers.Input(shape=(lookback, 1)))
    for i, units in enumerate(layers_spec):
        # Solo la última LSTM devuelve un vector; las anteriores pasan la
        # secuencia completa a la siguiente.
        model.add(layers.LSTM(units, return_sequences=(i < len(layers_spec) - 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main() -> None:
    import tensorflow as tf
    from tensorflow import keras

    t0 = time.time()
    series = make_series()
    # No se escala: la serie ya vive en [-1,5; 1,5], el rango donde tanh y la
    # sigmoidal de las puertas no saturan.
    X, y = windowing(series)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # 10 % final del train como validación temporal: es el tramo que se usa
    # para elegir configuración. El test queda para el número final.
    v = int(0.9 * len(X_train))
    X_tr, X_val = X_train[:v], X_train[v:]
    y_tr, y_val = y_train[:v], y_train[v:]

    # Dos referencias para leer el MAE de la LSTM:
    # - persistencia: predecir y_t con y_{t-1}, el modelo más tonto posible;
    # - piso de ruido: MAE contra la señal limpia, lo mínimo alcanzable.
    persist_mae = float(np.mean(np.abs(y_test - X_test[:, -1, 0])))
    clean_test = clean_signal(len(series))[LOOKBACK:][split:]
    floor_mae = float(np.mean(np.abs(y_test - clean_test)))

    configs = [
        ("lstm_32", [32]),
        ("lstm_64", [64]),
        ("lstm_32_16", [32, 16]),
        ("lstm_64_32", [64, 32]),
    ]

    runs = []
    models = {}
    for name, layers_spec in configs:
        for seed in SEEDS:
            tf.keras.utils.set_random_seed(seed)
            model = build_model(layers_spec, LOOKBACK)
            # shuffle=False: el orden temporal de los lotes se respeta.
            hist = model.fit(
                X_tr,
                y_tr,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_data=(X_val, y_val),
                shuffle=False,
                verbose=0,
            )
            test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
            runs.append(
                {
                    "config": name,
                    "seed": seed,
                    "val_mae": float(hist.history["val_mae"][-1]),
                    "test_mse": float(test_mse),
                    "test_mae": float(test_mae),
                }
            )
            print(runs[-1])
            if seed == SEED:
                models[name] = model

    df_runs = pd.DataFrame(runs)
    df_runs.to_csv(OUT / "lstm_runs.csv", index=False)

    rows = []
    for name, layers_spec in configs:
        r = df_runs[df_runs["config"] == name]
        rows.append(
            {
                "config": name,
                "layers": "-".join(str(u) for u in layers_spec),
                "params_formula_lstm": params_stack(layers_spec),
                "params_keras": int(models[name].count_params()),
                "val_mae_mean": r["val_mae"].mean(),
                "test_mae_mean": r["test_mae"].mean(),
                "test_mae_sd": r["test_mae"].std(ddof=1),
                "test_mse_mean": r["test_mse"].mean(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lstm_configs.csv", index=False)

    # Se elige por validación. Elegir por test sería usar el test para decidir
    # y el número final dejaría de ser una estimación honesta.
    best = df.loc[df["val_mae_mean"].idxmin()]
    best_name = best["config"]
    best_model = models[best_name]

    preds = best_model.predict(X_test, verbose=0).ravel()
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(y_test, label="real", lw=1.0)
    ax.plot(preds, label=f"predicción {best_name}", lw=1.1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("índice temporal (test)")
    ax.set_ylabel("valor")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(OUT / "lstm_pred.png", dpi=150)
    plt.close(fig)

    with open(OUT / "keras_summary.txt", "w", encoding="utf-8") as f:
        best_model.summary(print_fn=lambda line, **_: f.write(line + "\n"))

    n_fit = len(X_tr)
    meta = {
        "status": "ok",
        "backend": "tensorflow-cpu",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "seeds": SEEDS,
        "lookback": LOOKBACK,
        "epochs": EPOCHS,
        "split": "temporal_80_20",
        "shuffle": False,
        "n_fit": n_fit,
        "n_val": len(X_val),
        "n_test": len(X_test),
        "noise_sd": NOISE_SD,
        "persist_mae": persist_mae,
        "floor_mae": floor_mae,
        "best_by_val": best_name,
        "d_over_l_best": float(best["params_keras"]) / n_fit,
        "runtime_s": round(time.time() - t0, 1),
        "formula": "4*((units*input_dim)+units^2+units)",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Escrito en", OUT)


if __name__ == "__main__":
    main()
