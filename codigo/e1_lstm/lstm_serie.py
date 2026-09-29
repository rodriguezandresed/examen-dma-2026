"""
Ejercicio 1.VI: LSTM sobre serie sintética (TensorFlow/Keras).

Intento 1 (descartado para métricas): NumPy + BPTT en resultados/e1_lstm/.
Intento 2 (reportado): venv .venv-tf (CPython 3.12) + TensorFlow.
Lookback 20, split temporal 80/20, shuffle=False, 4 configs.

Correr:
  .venv-tf/bin/python codigo/e1_lstm/lstm_serie.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
LOOKBACK = 20
EPOCHS = 20
BATCH = 32
OUT = Path(__file__).resolve().parents[1] / "resultados" / "e1"
OUT.mkdir(parents=True, exist_ok=True)


def lstm_param_count(units: int, input_dim: int) -> int:
    """Parametros de una capa LSTM, segun la formula del apunte de catedra.

    El 4 son las tres puertas mas la candidata. El termino units*units es el
    que hace que el costo crezca de forma cuadratica con la cantidad de
    neuronas: duplicar units no duplica los parametros, los cuadruplica casi.
    Se compara despues contra model.count_params() para verificar que la
    formula del apunte y lo que arma Keras dan lo mismo.
    """
    return 4 * ((units * input_dim) + units * units + units)


def params_stack(layers: list[int], input_dim: int = 1) -> int:
    total, prev = 0, input_dim
    for units in layers:
        total += lstm_param_count(units, prev)
        prev = units
    return total


def make_series(n: int = 1200) -> np.ndarray:
    """Serie con dos estacionalidades de periodo distinto mas ruido.

    Se usa una serie generada y no una real a proposito: como se conoce la
    estructura verdadera, se puede distinguir si el modelo aprendio la
    estacionalidad o si solo esta copiando el ultimo valor observado, que es
    el modo tipico en que una red parece andar bien en series sin haber
    aprendido nada.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(n)
    return np.sin(0.05 * t) + 0.3 * np.sin(0.12 * t) + 0.1 * rng.normal(size=n)


def windowing(series: np.ndarray, lookback: int = LOOKBACK):
    """Pasa la serie a la forma (batch, time_steps, features) que pide Keras.

    Con una serie univariada, features = 1, y se pierden `lookback`
    observaciones al armar las ventanas. El lookback de 20 queda por debajo de
    los dos periodos de la serie, 125.7 pasos para el termino de 0.05*t y 52.4
    para el de 0.12*t, asi que la ventana no ve un ciclo completo: la red tiene
    que inferir la periodicidad de la pendiente local. Es una eleccion
    deliberada, porque con ventanas que cubran un ciclo entero la tarea se
    vuelve casi trivial.
    """
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
        # return_sequences va en True en todas las capas menos la ultima: una
        # LSTM apilada necesita recibir la secuencia completa, no el ultimo
        # estado. Sin esto Keras corta con un error de dimensiones, que fue el
        # unico problema real al armar las dos configuraciones apiladas.
        model.add(layers.LSTM(units, return_sequences=(i < len(layers_spec) - 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main() -> None:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(SEED)
    series = make_series()
    X, y = windowing(series)
    # PARTICION TEMPORAL, NO ALEATORIA. Si se sortean las ventanas, ventanas
    # del futuro entran al entrenamiento y el modelo predice el pasado con
    # informacion del porvenir. Es la fuga de informacion propia de las series
    # de tiempo, y por el mismo motivo va shuffle=False en el fit.
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # 10 % final del train como validación temporal
    v = int(0.9 * len(X_train))
    X_tr, X_val = X_train[:v], X_train[v:]
    y_tr, y_val = y_train[:v], y_train[v:]

    configs = [
        ("lstm_32", [32]),
        ("lstm_64", [64]),
        ("lstm_32_16", [32, 16]),
        ("lstm_64_32", [64, 32]),
    ]

    rows = []
    best_name, best_mae, best_model = None, float("inf"), None

    for name, layers_spec in configs:
        model = build_model(layers_spec, LOOKBACK)
        hist = model.fit(
            X_tr,
            y_tr,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_data=(X_val, y_val),
            shuffle=False,
            verbose=0,
        )
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        # Las dos cuentas tienen que coincidir salvo por la capa densa final,
        # que la formula del apunte no incluye: params_keras - formula debe dar
        # units_ultima + 1.
        formula = params_stack(layers_spec)
        keras_params = int(model.count_params())
        row = {
            "config": name,
            "layers": "-".join(str(u) for u in layers_spec),
            "params_formula_lstm": formula,
            "params_keras": keras_params,
            "test_mse": float(loss),
            "test_mae": float(mae),
            "val_mae_last": float(hist.history["val_mae"][-1]),
        }
        rows.append(row)
        print(row)
        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lstm_configs.csv", index=False)

    preds = best_model.predict(X_test, verbose=0).ravel()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(y_test[:250], label="real", lw=1.1)
    ax.plot(preds[:250], label="predicción", lw=1.1)
    ax.legend()
    mae_txt = f"{best_mae:.4f}".replace(".", ",")
    ax.set_title(f"LSTM {best_name} · EAM = {mae_txt}")
    ax.set_xlabel("índice temporal (test)")
    ax.set_ylabel("valor")
    fig.tight_layout()
    fig.savefig(OUT / "lstm_pred.png", dpi=120)
    plt.close(fig)

    with open(OUT / "keras_summary.txt", "w", encoding="utf-8") as f:
        best_model.summary(print_fn=lambda l: f.write(l + "\n"))

    best_row = df[df["config"] == best_name].iloc[0].to_dict()
    meta = {
        "status": "ok",
        "backend": "tensorflow-cpu",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "seed": SEED,
        "lookback": LOOKBACK,
        "epochs": EPOCHS,
        "split": "temporal_80_20",
        "shuffle": False,
        "best": best_row,
        "formula": "4*((units*input_dim)+units^2+units)",
        "nota_numpy_previo": (
            "Corrida NumPy en resultados/e1_lstm/ documentada como limitante; "
            "métricas del informe salen de esta corrida TF."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Escrito en", OUT)


if __name__ == "__main__":
    main()
