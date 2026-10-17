"""Ejercicio 4: AG binario contra AG real sobre la función de Rastrigin, con DEAP.

Se eligió Rastrigin porque tiene un mínimo global en el origen rodeado de una grilla
de mínimos locales: con la esfera x1^2 + x2^2 cualquier método de gradiente llega solo,
y el AG no mostraba nada. Los dos AG comparten población, generaciones, selección y
probabilidades; solo cambian la representación y los operadores, que es la diferencia
que pide el enunciado.

Salidas en codigo/resultados/e4/: ag_corridas.csv (una fila por semilla y codificación)
y meta.json. Se armó también una figura de convergencia, pero las bandas
intercuartílicas de las dos codificaciones se pisaban de punta a punta; la tabla del
informe (óptimo, cuenca, valle vecino) dice más en menos lugar, y la figura se sacó.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

SEED = 42
SEMILLAS = list(range(SEED, SEED + 20))  # 20 corridas: con una sola semilla el azar decide
N_VAR = 2
LIM = 5.12  # dominio clásico de Rastrigin
BITS = 16  # por variable: paso de 10,24 / (2^16 - 1), cerca de 1,6e-4
POP, NGEN, CXPB, MUTPB = 50, 100, 0.8, 0.2
UMBRAL = 1e-2  # se cuenta como "llegó al óptimo global" si f < 0,01

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e4"
OUT.mkdir(parents=True, exist_ok=True)


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def decodificar(bits):
    # x = a + decimal(cadena) * (b - a) / (2^m - 1), la fórmula del apunte
    xs = []
    for k in range(N_VAR):
        trozo = bits[k * BITS:(k + 1) * BITS]
        entero = int("".join(str(b) for b in trozo), 2)
        xs.append(-LIM + entero * (2 * LIM) / (2**BITS - 1))
    return xs


# weights=(-1.0,) porque se minimiza
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)


def toolbox_binario():
    tb = base.Toolbox()
    tb.register("gen", random.randint, 0, 1)
    tb.register("individuo", tools.initRepeat, creator.Individuo, tb.gen, N_VAR * BITS)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", lambda ind: (rastrigin(decodificar(ind)),))
    tb.register("mate", tools.cxOnePoint)  # corta y cambia colas, como en las diapositivas
    tb.register("mutate", tools.mutFlipBit, indpb=1.0 / (N_VAR * BITS))
    tb.register("select", tools.selTournament, tournsize=3)
    return tb


def recortar(ind):
    # cxBlend y la mutación gaussiana pueden sacar el gen del dominio: se lo lleva al borde
    for i, v in enumerate(ind):
        ind[i] = min(max(v, -LIM), LIM)
    return ind


def toolbox_real():
    tb = base.Toolbox()
    tb.register("gen", random.uniform, -LIM, LIM)
    tb.register("individuo", tools.initRepeat, creator.Individuo, tb.gen, N_VAR)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", lambda ind: (rastrigin(ind),))

    def cruce(a, b):
        tools.cxBlend(a, b, alpha=0.5)
        return recortar(a), recortar(b)

    def mutacion(ind):
        # sigma 0,3: primero se probó 1,0 y el AG saltaba entre valles sin afinar
        tools.mutGaussian(ind, mu=0.0, sigma=0.3, indpb=0.5)
        return (recortar(ind),)

    tb.register("mate", cruce)
    tb.register("mutate", mutacion)
    tb.register("select", tools.selTournament, tournsize=3)
    return tb


def correr(nombre, tb, semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    pop = tb.poblacion(n=POP)
    hof = tools.HallOfFame(1)  # guarda el mejor visto, no reinyecta (sin elitismo)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    _, log = algorithms.eaSimple(pop, tb, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                 stats=stats, halloffame=hof, verbose=False)
    # mejor hasta la generación g (acumulado): sin elitismo el mínimo de P(t) puede subir
    mejor_acum = np.minimum.accumulate(log.select("min"))
    mejor = hof[0]
    x = decodificar(mejor) if nombre == "binario" else list(mejor)
    llego = np.where(mejor_acum < UMBRAL)[0]
    return {
        "codificacion": nombre,
        "semilla": semilla,
        "mejor_f": float(mejor.fitness.values[0]),
        "x1": x[0],
        "x2": x[1],
        "gen_umbral": int(llego[0]) if len(llego) else None,
        # eaSimple solo evalua a los individuos que el cruce o la mutacion tocaron,
        # asi que POP * (NGEN + 1) es un tope, no lo que se gasto.
        "evaluaciones": int(sum(log.select("nevals"))),
        "cromosoma": "".join(str(b) for b in mejor) if nombre == "binario" else "",
    }


filas = []
for nombre, fabrica in [("binario", toolbox_binario), ("real", toolbox_real)]:
    for s in SEMILLAS:
        filas.append(correr(nombre, fabrica(), s))

corridas = pd.DataFrame(filas)
corridas.to_csv(OUT / "ag_corridas.csv", index=False)


def resumen(nombre):
    d = corridas[corridas.codificacion == nombre]
    return {
        "mediana_final": float(d.mejor_f.median()),
        "n_umbral": int((d.mejor_f < UMBRAL).sum()),
        "gen_umbral_mediana": float(d.gen_umbral.dropna().median()),
        # cuenca global: |x_i| < 0,5 en las dos variables, aunque falte el ajuste fino.
        # Se agregó al ver que varias corridas reales terminaban a pocas centésimas del
        # origen con f > 0,01, mientras las binarias fallidas quedaban clavadas en x = ±1.
        "n_cuenca": int(((d.x1.abs() < 0.5) & (d.x2.abs() < 0.5)).sum()),
        "n_vecino": int(((d.x1.abs() - 1).abs().lt(0.1)
                         | (d.x2.abs() - 1).abs().lt(0.1)).sum()),
    }

# Por que las binarias fallidas no salen del valle vecino. La primera version de este
# calculo comparaba el entero de x = 1 con los de x = 0 (6 bits), pero los cromosomas
# clavados no estan exactamente en ese entero. Se mide sobre los cromosomas reales:
# la menor cantidad de bits que hay que invertir A LA VEZ para bajar f, y cuantas
# combinaciones de ese tamano lo logran.
from itertools import combinations


def salida(crom):
    bits = [int(c) for c in crom]
    f0 = rastrigin(decodificar(bits))
    for k in range(1, 5):
        n_mejora = 0
        for idx in combinations(range(len(bits)), k):
            b = bits[:]
            for i in idx:
                b[i] = 1 - b[i]
            if rastrigin(decodificar(b)) < f0:
                n_mejora += 1
        if n_mejora:
            return k, n_mejora
    return None, 0


d_bin = corridas[corridas.codificacion == "binario"]
clavadas = d_bin[~((d_bin.x1.abs() < 0.5) & (d_bin.x2.abs() < 0.5))]
sal = [salida(c) for c in clavadas.cromosoma]
k_min = int(np.median([k for k, _ in sal]))
p_bit = 1.0 / (N_VAR * BITS)
# probabilidad de que una mutacion acierte alguna de las combinaciones que mejoran,
# tomando la corrida tipica (mediana de combinaciones utiles con k_min bits)
n_utiles = float(np.median([n for k, n in sal if k == k_min]))
p_salida = n_utiles * p_bit**k_min * (1 - p_bit)**(N_VAR * BITS - k_min)


meta = {
    "seed": SEED, "n_semillas": len(SEMILLAS), "pop": POP, "ngen": NGEN,
    "cxpb": CXPB, "mutpb": MUTPB, "bits": BITS, "lim": LIM, "umbral": UMBRAL,
    "paso_binario": 2 * LIM / (2**BITS - 1),
    "evaluaciones_media": {c: float(corridas[corridas.codificacion == c].evaluaciones.mean())
                           for c in ("binario", "real")},
    "binario": resumen("binario"), "real": resumen("real"),
    "p_mut_bit": p_bit, "bits_salida": k_min,
    "bits_salida_max": int(max(k for k, _ in sal)),
    "mutaciones_por_salida": 1.0 / p_salida,
    "mutaciones_por_corrida": POP * MUTPB * NGEN,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(json.dumps(meta, indent=2, ensure_ascii=False))
