"""
Ejercicio 7 - t-SNE sobre digitos manuscritos

Aplica el pipeline de tres pasos que recomienda el apunte de la Clase 8:
  (1) explorar con t-SNE sobre datos estandarizados,
  (2) hacer el clustering sobre el espacio original (aca, sobre el PCA previo),
  (3) validar superponiendo las etiquetas sobre la proyeccion.

Nota personal sobre el paso 2: la tentacion es correr DBSCAN directamente sobre
las dos coordenadas del embedding, porque ahi los grupos se ven separados y
"funciona". El apunte advierte que eso esta mal, y el motivo es que las
distancias del mapa t-SNE no son confiables: el algoritmo preserva vecindad
local y distorsiona todo lo demas, asi que un eps ajustado sobre el embedding
no significa nada en el espacio de los datos.
"""

import os
import csv

AQUI = os.path.dirname(os.path.abspath(__file__))
SAL = os.path.join(AQUI, "..", "resultados", "e7")
os.makedirs(SAL, exist_ok=True)

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import f_oneway

RANDOM_STATE = 42

datos = load_digits()
X, y = datos.data, datos.target
M = {}
M["n"] = X.shape[0]; M["p"] = X.shape[1]; M["clases"] = len(np.unique(y))
print("dataset:", X.shape, "clases:", M["clases"])

X_std = StandardScaler().fit_transform(X)

# PCA previo a 30 componentes. El apunte lo recomienda para inicializar y
# acelerar; ademas filtra ruido antes de calcular las distancias que alimentan
# las probabilidades p_ij.
pca = PCA(n_components=30, random_state=RANDOM_STATE)
X30 = pca.fit_transform(X_std)
M["var_pca30"] = float(pca.explained_variance_ratio_.sum())
print("varianza explicada por 30 componentes: %.3f" % M["var_pca30"])

# --- Sensibilidad a la perplejidad -----------------------------------------
# Se prueban tres valores dentro del rango 5-50 que indica el apunte. La idea
# es verificar en carne propia la limitante: si la cantidad de grupos que se
# "ven" cambia con la perplejidad, el mapa no puede leerse como una particion.
embeddings = {}
for perp in [5, 30, 50]:
    emb = TSNE(n_components=2, perplexity=perp, init="pca",
               learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X30)
    embeddings[perp] = emb
    sil = silhouette_score(emb, y)
    M["sil_p%d" % perp] = float(sil)
    print("perplexity=%2d -> silhouette de las clases reales en el mapa: %.3f"
          % (perp, sil))

# --- Clustering sobre el espacio original (no sobre el embedding) ----------
# Primer intento con DBSCAN. La primera grilla de eps arrancaba en 5 y en todo
# ese rango devolvia un unico cluster con ARI = 0; se lo leyo como prueba de que
# DBSCAN no separaba nada. Estaba mal ubicada: la distancia de cada punto a su
# 8.o vecino (el grafico de distancia k de Ester et al.) tiene la mediana cerca
# de 3,8, asi que con eps >= 5 casi todo queda conectado con todo. Se la extiende
# hacia abajo, con paso fino entre 2 y 4, que es donde DBSCAN encuentra grupos.
vecinos = NearestNeighbors(n_neighbors=8).fit(X30)
dist_k = vecinos.kneighbors(X30)[0][:, -1]  # incluye al propio punto, como min_samples
for q in [10, 50, 90]:
    M["kdist_q%d" % q] = float(np.percentile(dist_k, q))
print("distancia al 8.o vecino, cuantiles 10/50/90: %.2f %.2f %.2f"
      % (M["kdist_q10"], M["kdist_q50"], M["kdist_q90"]))
# El informe atribuye el fracaso de un eps unico a que la densidad cambia de un
# digito a otro. Se mide en vez de suponerlo: mediana de esa distancia por digito.
kd_dig = [float(np.median(dist_k[y == d])) for d in range(10)]
M["kdist_dig_min"] = min(kd_dig); M["kdist_dig_min_d"] = int(np.argmin(kd_dig))
M["kdist_dig_max"] = max(kd_dig); M["kdist_dig_max_d"] = int(np.argmax(kd_dig))

barrido = []
print("eps  clusters  ruido   ARI  ARI_sin_ruido")
for eps in [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5, 7, 9, 12,
            16, 20]:
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(X30)
    n = len(set(lab)) - (1 if -1 in lab else 0)
    nr = lab != -1
    # ARI sobre todos los puntos (el ruido cuenta como un grupo mas) y aparte
    # sobre los que no son ruido: con eps chico el segundo sale alto solo porque
    # DBSCAN se queda con los nucleos faciles y tira el resto.
    a_nr = adjusted_rand_score(y[nr], lab[nr]) if nr.sum() > 0 else float("nan")
    barrido.append((eps, n, int((~nr).sum()), adjusted_rand_score(y, lab), a_nr))
    print("%4.2f %6d %8d  %.3f  %.3f" % barrido[-1])

# El eps que se informa es el de mayor ARI sobre todos los puntos. Se recalcula
# aparte para que el numero que cita el texto salga de un print y no de la tabla.
eps_mejor = max(barrido, key=lambda f: f[3])[0]
db = DBSCAN(eps=eps_mejor, min_samples=8).fit_predict(X30)
n_db = len(set(db)) - (1 if -1 in db else 0)
M["db_eps"] = eps_mejor
M["db_clusters"] = n_db
M["db_ruido"] = int((db == -1).sum())
M["ari_dbscan"] = float(adjusted_rand_score(y, db))
M["ari_dbscan_sin_ruido"] = float(adjusted_rand_score(y[db != -1], db[db != -1]))
print("DBSCAN sobre PCA30 (eps=%.2f): %d clusters, %d puntos de ruido, ARI=%.3f"
      % (eps_mejor, n_db, M["db_ruido"], M["ari_dbscan"]))

km = KMeans(n_clusters=10, n_init=10, random_state=RANDOM_STATE).fit_predict(X30)
M["ari_kmeans_pca"] = float(adjusted_rand_score(y, km))
print("KMeans k=10 sobre PCA30: ARI=%.3f" % M["ari_kmeans_pca"])

# --- Perfiles de los grupos (capitulo 5 de la guia, Assessing Clustering Results) ---
# El capitulo propone comparar primero las medias de las variables de entrada entre
# grupos y recien despues ir al test. Con 64 pixeles la tabla es ilegible, asi que se
# perfila sobre las primeras cinco componentes, que el PCA ya dejo ordenadas por
# varianza. La aclaracion que hace el propio capitulo y que conviene no perder: los
# p-valores de abajo NO son una prueba de hipotesis valida, porque los grupos se
# construyeron a partir de estos mismos datos. Sirven para ordenar cual componente
# separa mas, y para nada mas que eso.
print("\nperfiles de los 10 grupos de k-means, medias sobre PC1..PC5")
print("grupo     n" + "".join("%9s" % ("PC%d" % (j + 1)) for j in range(5)))
for c in range(10):
    m = km == c
    print("%5d %5d" % (c, m.sum())
          + "".join("%9.2f" % X30[m, j].mean() for j in range(5)))

print("\nANOVA de una via por componente sobre los grupos de k-means")
for j in range(5):
    F, p = f_oneway(*[X30[km == c, j] for c in range(10)])
    print("PC%d  F=%8.1f  p=%.3g" % (j + 1, F, p))

# --- Contraejemplo deliberado: clustering sobre el embedding ---------------
# Se deja en el codigo porque el contraste es el argumento: el ARI sobre el
# embedding sale MEJOR, y justamente por eso es una trampa. El mapa ya separo
# los grupos, asi que agrupar sobre el es medir el trabajo de t-SNE, no
# descubrir estructura en los datos.
km_emb = KMeans(n_clusters=10, n_init=10,
                random_state=RANDOM_STATE).fit_predict(embeddings[30])
M["ari_kmeans_emb"] = float(adjusted_rand_score(y, km_emb))
print("KMeans k=10 sobre el embedding t-SNE: ARI=%.3f (ver comentario)"
      % M["ari_kmeans_emb"])



# --- Paso 3 del pipeline: validacion visual ---------------------------------
# Superponer las etiquetas sobre la proyeccion es el paso 3 que el apunte recomienda y
# el unico que el informe describia sin ejecutar. Se grafican los dos paneles juntos a
# proposito: a la izquierda el digito verdadero y a la derecha el grupo de k-means, que
# es la unica forma honesta de mostrar donde el clustering acierta y donde parte un
# digito en dos.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

emb = embeddings[30]
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=6, linewidths=0)
ax[0].set_title("mapa t-SNE (perplejidad 30), color = digito real", fontsize=9)
ax[1].scatter(emb[:, 0], emb[:, 1], c=km, cmap="tab10", s=6, linewidths=0)
ax[1].set_title("mismo mapa, color = grupo de k-means sobre PCA30", fontsize=9)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(SAL, "tsne_mapa.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print("\nescrito tsne_mapa.png")


# --- Exportacion para el informe ------------------------------------------
# El .Rmd no ejecuta Python: lee estos CSV con read.csv(), asi que ningun
# numero del texto se tipea a mano.
M["ruido_eps_min"] = barrido[0][2]
M["ruido_eps_max"] = barrido[-1][2]

with open(os.path.join(SAL, "metricas.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metrica", "valor"])
    for k, v in M.items():
        w.writerow([k, v])

with open(os.path.join(SAL, "barrido_eps.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["eps", "clusters", "ruido", "ari", "ari_sin_ruido"])
    for fila in barrido:
        w.writerow(["%.2f" % fila[0], fila[1], fila[2], "%.3f" % fila[3],
                    "%.3f" % fila[4]])

with open(os.path.join(SAL, "perfiles_kmeans.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["grupo", "n"] + ["PC%d" % (j + 1) for j in range(5)])
    for c in range(10):
        m = km == c
        w.writerow([c, int(m.sum())] + ["%.2f" % X30[m, j].mean() for j in range(5)])

with open(os.path.join(SAL, "anova_componentes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["componente", "F"])
    for j in range(5):
        F, _ = f_oneway(*[X30[km == c, j] for c in range(10)])
        w.writerow(["PC%d" % (j + 1), "%.1f" % F])

# Paso 2 del capitulo, el que faltaba. El MANOVA va ANTES del ANOVA por
# variable, porque pregunta si los grupos difieren en el VECTOR completo y asi
# no infla el error de tipo I con una prueba por componente. Se calcula a mano
# con las matrices de dispersion para no sumar statsmodels: Lambda de Wilks es
# det(W)/det(T), con W la dispersion dentro de los grupos y T la total, y la
# aproximacion de Bartlett la lleva a una chi cuadrado. Lambda cerca de 0 es
# separacion total y cerca de 1 es ninguna. La advertencia del ANOVA vale
# igual: el p esta sesgado hacia la significacion porque los grupos salieron de
# estos mismos datos, asi que esto se lee como tamano de la diferencia.
P5 = X30[:, :5]
n_obs, p_var = P5.shape
k_gr = 10
W = np.zeros((p_var, p_var))
for c in range(k_gr):
    Z = P5[km == c] - P5[km == c].mean(axis=0)
    W += Z.T @ Z
Zt = P5 - P5.mean(axis=0)
T = Zt.T @ Zt
wilks = float(np.linalg.det(W) / np.linalg.det(T))
gl_manova = p_var * (k_gr - 1)
chi2 = float(-(n_obs - 1 - (p_var + k_gr) / 2) * np.log(wilks))

# CANDISC, el otro paso que el capitulo pide junto al ANOVA. Los ejes canonicos
# maximizan la separacion ENTRE grupos, que NO es lo mismo que la varianza
# total que ordena las componentes principales. Primera version: comparaba el
# % de separacion de 2 ejes canonicos (sobre las 5 componentes) con el % de
# varianza de PC1+PC2 sobre los 64 pixeles: dos bases distintas, la comparacion
# no decia nada. Ahora las dos se miden sobre las MISMAS 5 componentes.
cand = LinearDiscriminantAnalysis().fit(P5, km)
sep_can2 = float(cand.explained_variance_ratio_[:2].sum())
var_pca2 = float(P5[:, :2].var(axis=0).sum() / P5.var(axis=0).sum())
# Que componentes arman cada eje canonico: correlacion de estructura dentro de
# los grupos (la que el capitulo usa para decir que variables producen la
# diferencia).
Zc = cand.transform(P5)[:, :2]
Pw = P5 - np.array([P5[km == c].mean(axis=0) for c in range(k_gr)])[km]
Zw = Zc - np.array([Zc[km == c].mean(axis=0) for c in range(k_gr)])[km]
estructura = np.array([[np.corrcoef(Pw[:, j], Zw[:, e])[0, 1] for e in range(2)]
                       for j in range(p_var)])

with open(os.path.join(SAL, "manova_candisc.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metrica", "valor"])
    w.writerow(["wilks_lambda", "%.4f" % wilks])
    w.writerow(["chi2_bartlett", "%.0f" % chi2])
    w.writerow(["gl", gl_manova])
    w.writerow(["sep_canonica_2ejes", "%.3f" % sep_can2])
    w.writerow(["var_pca_2comp", "%.3f" % var_pca2])
    for j in range(p_var):
        for e in range(2):
            w.writerow(["estructura_pc%d_eje%d" % (j + 1, e + 1), "%.2f" % estructura[j, e]])

# Segunda forma de perfilado del capitulo (Assessing Clustering Results 5.2):
# respecto de una clase dada. Aca la clase existe, el digito real, asi que se
# cruza en lugar de inferir la mezcla solo por los tamanos de los grupos.
import pandas as pd
cruce = pd.crosstab(pd.Series(km, name="grupo"), pd.Series(y, name="digito"))
cruce.to_csv(os.path.join(SAL, "cruce_grupo_digito.csv"))

print("\nMANOVA sobre PC1..PC5: Lambda de Wilks=%.4f, chi2 Bartlett=%.0f (gl=%d)"
      % (wilks, chi2, gl_manova))
print("CANDISC: los 2 primeros ejes canonicos concentran %.1f%% de la separacion; "
      "las 2 primeras componentes explican %.1f%% de la varianza total"
      % (100 * sep_can2, 100 * var_pca2))

# Paso 3 del perfilado, con un agregado propio. El capitulo cierra con el ANOVA
# por variable, cuyos p quedan sesgados porque los grupos se construyeron para
# maximizar la separacion. Un clasificador contra la etiqueta de grupo no tiene
# ese problema: no asume nada sobre la distribucion, y su importancia de
# variables dice lo mismo que buscaba el ANOVA sin pedir prestado un valor p.
# oob_score da ademas una medida de cuan separables son los grupos.
#
# Se entrena sobre las MISMAS 5 componentes de la tabla de perfiles, no sobre
# las 30, para que la importancia sea comparable con los F de arriba. Primero
# lo probe sobre las 30 y la importancia se reparte entre tantas componentes
# que no se puede contrastar con nada.
rf = RandomForestClassifier(n_estimators=500, oob_score=True,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X30[:, :5], km)
imp = rf.feature_importances_  # caida media de impureza de Gini
# La de Gini tiene sesgos conocidos (Ejercicio 9), asi que se controla el orden
# con la importancia por permutacion.
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X30[:, :5], km, n_repeats=10,
                              random_state=RANDOM_STATE, n_jobs=-1).importances_mean
M["orden_perm_igual"] = int(list(np.argsort(-perm)) == list(np.argsort(-imp)))

with open(os.path.join(SAL, "importancia_rf.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["componente", "importancia"])
    for j in range(5):
        w.writerow(["PC%d" % (j + 1), "%.3f" % imp[j]])

M["rf_oob"] = "%.3f" % rf.oob_score_
with open(os.path.join(SAL, "metricas.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metrica", "valor"])
    for k, v in M.items():
        w.writerow([k, v])

# Estabilidad entre semillas. Hasta aca todo salia de UNA corrida con semilla
# fija, y en el informe se afirma un ORDEN de importancia y una forma de la
# curva de siluetas. Afirmar un orden a partir de una corrida es leer ruido si
# las brechas son del tamano de la dispersion entre semillas, asi que hay que
# medirla en vez de suponerla. Se rehace el bosque con 8 semillas; el t-SNE no
# se rehace aca porque cuesta minutos y se verifico aparte (4 semillas, desvio
# de silueta 0,001).
ordenes, oobs = [], []
for sem in range(8):
    r = RandomForestClassifier(n_estimators=500, oob_score=True,
                               random_state=sem, n_jobs=-1).fit(X30[:, :5], km)
    oobs.append(r.oob_score_)
    ordenes.append(tuple(np.argsort(-r.feature_importances_)))
estable = len(set(ordenes)) == 1

with open(os.path.join(SAL, "estabilidad_rf.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metrica", "valor"])
    w.writerow(["semillas", len(oobs)])
    w.writerow(["orden_identico", int(estable)])
    w.writerow(["oob_media", "%.4f" % (sum(oobs) / len(oobs))])
    w.writerow(["oob_min", "%.4f" % min(oobs)])
    w.writerow(["oob_max", "%.4f" % max(oobs)])

print("estabilidad: orden identico en %d/%d semillas, oob %.4f-%.4f"
      % (len(oobs) if estable else 0, len(oobs), min(oobs), max(oobs)))

print("escritos en resultados/e7: metricas.csv, barrido_eps.csv, "
      "perfiles_kmeans.csv, anova_componentes.csv, manova_candisc.csv, "
      "importancia_rf.csv, estabilidad_rf.csv, "
      "tsne_mapa.png")
