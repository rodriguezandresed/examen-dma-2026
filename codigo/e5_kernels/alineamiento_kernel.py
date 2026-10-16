"""
Ejercicio 5 - Alineamiento kernel-objetivo sobre un data set real.

La pregunta del enunciado es si existen metodos sencillos para elegir el kernel
apropiado en un determinado data set. El alineamiento se presenta como un atajo:
elegir kernel sin entrenar un modelo por candidato. Si el atajo sirve, tiene que
coincidir con lo que dio entrenar, asi que se calcula sobre los mismos datos y la
misma particion del Ejercicio 10 y se compara con la gamma que eligio la CV.

    A = <K, yy'>_F / (||K||_F * ||yy'||_F),   y en {-1, +1}

La primera version usaba solo la formula sin centrar de Cristianini et al.
(2002). Dio que el lineal ganaba lejos y que el atajo no coincidia con la CV,
pero al revisarla aparecio el problema: sin centrar, A premia o castiga la parte
constante de K (el polinomico con coef0 = 1 suma un termino enorme igual para
todos los pares, y el RBF con gamma chica es casi todo unos), y eso no tiene que
ver con las clases. Cortes, Mohri y Rostamizadeh (2012) proponen centrar K y yy'
en el espacio de caracteristicas, HKH con H = I - 11'/n, antes de comparar. Se
informan las dos versiones; la conclusion del informe sale de la centrada.

Ademas se verifica de forma directa lo que el Ejercicio 10 afirma sobre la
separabilidad: un SVM lineal de margen duro (C muy grande) sobre el entrenamiento
escalado. Si clasifica bien todos los casos, el entrenamiento es linealmente
separable.

Salida: codigo/resultados/e5/alineamiento.csv y meta.json
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
from sklearn.svm import SVC

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
H = np.eye(n) - np.ones((n, n)) / n  # centrado en el espacio de caracteristicas


def alineamiento(K, centrar):
    Y = ideal
    if centrar:
        K, Y = H @ K @ H, H @ ideal @ H
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


# Los candidatos son los del enunciado del Ejercicio 5 (lineal, polinomico,
# RBF) con los gamma de la grilla que el Ejercicio 10 recorrio de verdad, para
# que las dos listas sean comparables fila por fila.
candidatos = [("lineal", "", linear_kernel(Xtr))]
for q in (2, 3):
    candidatos.append(("polinomico", "grado %d" % q,
                       polynomial_kernel(Xtr, degree=q, gamma=1.0, coef0=1.0)))
for g in (0.0001, 0.001, 0.01, 0.1, 1.0):
    candidatos.append(("RBF", "gamma %g" % g, rbf_kernel(Xtr, gamma=g)))

filas = [(fam, par, alineamiento(K, False), alineamiento(K, True))
         for fam, par, K in candidatos]

with open(os.path.join(SAL, "alineamiento.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["familia", "parametro", "alineamiento", "alineamiento_centrado"])
    for fam, par, a, ac in filas:
        w.writerow([fam, par, "%.4f" % a, "%.4f" % ac])

# La gamma que eligio la CV se lee del Ejercicio 10, no se tipea: si el
# pipeline cambia, esta comparacion lo acompana.
hp = os.path.join(AQUI, "..", "resultados", "e10", "hiperparametros_elegidos.csv")
with open(hp, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if (r["tarea"], r["modelo"], r["hiperparametro"]) == ("clasificacion", "SVM", "gamma"):
            gamma_cv = float(r["valor"].replace(",", "."))

mejor_sc = max(filas, key=lambda f: f[2])
mejor_c = max(filas, key=lambda f: f[3])
fila_cv = [f for f in filas if f[1] == "gamma %g" % gamma_cv][0]

# Separabilidad lineal del entrenamiento: margen duro aproximado con C enorme.
svm_duro = SVC(kernel="linear", C=1e6).fit(Xtr, ytr)
acierto_duro = float(svm_duro.score(Xtr, ytr))

meta = {
    "n_entrenamiento": n,
    "gamma_elegido_por_cv": gamma_cv,
    "sin_centrar_mejor": "%s %s" % (mejor_sc[0], mejor_sc[1]),
    "centrado_mejor_familia": mejor_c[0],
    "centrado_mejor_parametro": mejor_c[1],
    "centrado_mejor": round(mejor_c[3], 4),
    "alineamiento_del_gamma_cv": round(fila_cv[2], 4),
    "alineamiento_centrado_del_gamma_cv": round(fila_cv[3], 4),
    "coincide_con_cv_centrado": mejor_c[1] == "gamma %g" % gamma_cv,
    "acierto_entrenamiento_svm_lineal_margen_duro": acierto_duro,
}
with open(os.path.join(SAL, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("alineamiento kernel-objetivo sobre %d casos de entrenamiento" % n)
print("  %-12s %-12s %8s %9s" % ("familia", "parametro", "A", "A centr."))
for fam, par, a, ac in filas:
    marca = "  <- elegido por CV en E10" if par == "gamma %g" % gamma_cv else ""
    print("  %-12s %-12s %8.4f %9.4f%s" % (fam, par, a, ac, marca))
print("mejor sin centrar:", meta["sin_centrar_mejor"])
print("mejor centrado:", mejor_c[0], mejor_c[1], "| coincide con CV:",
      meta["coincide_con_cv_centrado"])
print("SVM lineal de margen duro, acierto en entrenamiento:", acierto_duro)
