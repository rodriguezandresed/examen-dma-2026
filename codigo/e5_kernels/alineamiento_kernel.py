"""
Ejercicio 5 - Alineamiento kernel-objetivo sobre un data set real.

La pregunta del enunciado es si existen metodos sencillos para elegir el kernel
apropiado en un determinado data set. El Ejercicio 5 contesta con bibliografia,
y hasta aca era el unico ejercicio del examen sin una medicion propia. Esto lo
corrige: se calcula el alineamiento de Cristianini et al. (2002) sobre los
mismos datos y la misma particion del Ejercicio 10, y se compara el orden que
produce con el que produjo la validacion cruzada.

Eso es lo que de verdad se le pide al metodo. El alineamiento se vende como un
atajo: elegir kernel sin entrenar un modelo por candidato. Si el atajo sirve,
tiene que coincidir con lo que dio entrenar. Si no coincide, tambien es un
resultado, y es el que hay que informar.

    A = <K, yy'>_F / (||K||_F * ||yy'||_F)

con y en {-1, +1}. Se usa la version SIN centrar, que es la del paper original
y la que esta escrita en el cuerpo del examen. La variante centrada es
posterior; no se mezcla para que el numero del informe sea el de la formula
que el informe muestra.

Salida: codigo/resultados/e5/alineamiento.csv
Correr:  python codigo/e5_kernels/alineamiento_kernel.py
"""

import csv
import json
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AQUI = os.path.dirname(os.path.abspath(__file__))
SAL = os.path.join(AQUI, "..", "resultados", "e5")
os.makedirs(SAL, exist_ok=True)

# Los mismos del Ejercicio 10. Si cambian alla, cambian aca, porque el valor de
# esta comparacion es que las dos miren exactamente el mismo conjunto.
SEED = 42
TEST_SIZE = 0.2

datos = load_breast_cancer()
X, y01 = datos.data, datos.target

# El Ejercicio 10 recodifica: maligno es la clase positiva. scikit-learn la trae
# como 0, asi que se da vuelta antes de armar el vector de etiquetas.
y = np.where(y01 == 0, 1, -1)

Xtr, _, ytr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED,
                                  stratify=y)

# Se escala con el mismo StandardScaler del pipeline y ajustado SOLO con el
# entrenamiento. Sin esto el alineamiento del RBF no significa nada: gamma
# opera sobre distancias euclideas y las 30 variables tienen escalas muy
# distintas (el area va a miles, la dimension fractal a centesimas).
Xtr = StandardScaler().fit_transform(Xtr)
n = Xtr.shape[0]

ideal = np.outer(ytr, ytr)
norma_ideal = np.linalg.norm(ideal)  # con y en {-1,+1} da exactamente n


def alineamiento(K):
    return float(np.sum(K * ideal) / (np.linalg.norm(K) * norma_ideal))


# Los candidatos son los del enunciado del Ejercicio 5 (lineal, polinomico,
# RBF) con los gamma de la grilla que el Ejercicio 10 recorrio de verdad, para
# que las dos listas sean comparables fila por fila.
candidatos = [("lineal", "", linear_kernel(Xtr))]
for q in (2, 3):
    candidatos.append(("polinomico", "grado %d" % q,
                       polynomial_kernel(Xtr, degree=q, gamma=1.0, coef0=1.0)))
for g in (0.0001, 0.001, 0.01, 0.1, 1.0):
    candidatos.append(("RBF", "gamma %g" % g, rbf_kernel(Xtr, gamma=g)))

filas = [(fam, par, alineamiento(K)) for fam, par, K in candidatos]
mejor = max(filas, key=lambda f: f[2])

with open(os.path.join(SAL, "alineamiento.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["familia", "parametro", "alineamiento"])
    for fam, par, a in filas:
        w.writerow([fam, par, "%.4f" % a])

# El RBF elegido por validacion cruzada en el Ejercicio 10, para poder decir en
# el texto si el atajo coincide o no con el resultado de entrenar.
gamma_cv = 0.01
a_cv = [a for fam, par, a in filas if par == "gamma %g" % gamma_cv][0]
meta = {
    "n_entrenamiento": n,
    "mejor_familia": mejor[0],
    "mejor_parametro": mejor[1],
    "mejor_alineamiento": round(mejor[2], 4),
    "gamma_elegido_por_cv": gamma_cv,
    "alineamiento_del_gamma_cv": round(a_cv, 4),
    "coincide_con_cv": mejor[1] == "gamma %g" % gamma_cv,
}
with open(os.path.join(SAL, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("alineamiento kernel-objetivo sobre %d casos de entrenamiento" % n)
for fam, par, a in filas:
    marca = "  <- mejor" if (fam, par) == (mejor[0], mejor[1]) else ""
    marca += "  <- elegido por CV en E10" if par == "gamma %g" % gamma_cv else ""
    print("  %-12s %-12s A = %.4f%s" % (fam, par, a, marca))
print("coincide con la validacion cruzada:", meta["coincide_con_cv"])
print("escrito en resultados/e5: alineamiento.csv, meta.json")
