# -*- coding: utf-8 -*-
"""
Ejercicio 7 - Preproceso visual, t-SNE y perfilado de clusters.

Que hace este script, en el orden del pipeline que recomienda el apunte de
Volpacchio (explorar / modelar / validar):

  0. Carga digits (1797 x 64) y estandariza. Descarta los pixeles de varianza
     cero, que en digits son los del borde: siempre valen 0 y StandardScaler
     los dejaria en 0 igual, pero ensucian el ANOVA (F indefinido).
  1. Ayudas graficas del capitulo 2 del curso de SAS: PCA, MDS y CDA. Son las
     tres proyecciones que el capitulo propone para "ver" si hay grupos antes
     de clusterizar. Se agrega t-SNE como cuarta.
  2. Barrido de perplejidad (5, 30, 50). Es la limitante principal de t-SNE:
     el mapa cambia con el parametro.
  3. Clustering sobre el ESPACIO ORIGINAL (k-means). k por pseudo-F
     (calinski_harabasz_score en sklearn es exactamente el PSF del capitulo 5).
  4. Contraste: el mismo k-means sobre el embedding 2D. Se mide la trampa
     (silhouette infladisimo) y la inestabilidad frente a la perplejidad.
  5. DBSCAN con barrido de eps, en el espacio original, en PCA y en el
     embedding. Spoiler: en 61 dimensiones no encuentra nada. Se deja el
     fracaso documentado con la evidencia del barrido.
  6. Perfilado: ANOVA por variable + CDA, y dos controles que muestran por que
     el capitulo 5 avisa que estas pruebas se leen con cuidado.

Semilla fija en SEED. Salidas a codigo/resultados/e7/.

Notas de lo que se probo y no quedo:
  - Primero corri t-SNE sin estandarizar. El mapa sale parecido porque todos
    los pixeles comparten la escala 0-16, pero DBSCAN cambiaba mucho, asi que
    deje la estandarizacion para todo el pipeline y que sea una sola decision.
  - Probe MDS sobre las 1797 observaciones: tarda varios minutos y la figura
    queda igual de ilegible. Quedo con una submuestra de 400, declarada.
  - Probe MinPts = 2*d (122) como dice la regla de dedo para DBSCAN. Con 1797
    puntos pide vecindarios enormes y etiqueta todo como ruido para cualquier
    eps razonable. Quedaron 5 y 10, que son mas favorables al algoritmo.
"""

import json
import os
import platform

import matplotlib
matplotlib.use("Agg")  # sin display: esto corre desde consola
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import f_oneway
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(AQUI, "..", "resultados", "e7"))
os.makedirs(OUT, exist_ok=True)


def ruta(nombre):
    return os.path.join(OUT, nombre)


# ---------------------------------------------------------------- 0. datos
digits = load_digits()
X_crudo, y = digits.data, digits.target
assert not np.isnan(X_crudo).any(), "digits no deberia traer NA; si aparecen, revisar"

var_col = X_crudo.var(axis=0)
mask = var_col > 0
pix_idx = np.where(mask)[0]          # indice original del pixel, para nombrarlo
X_util = X_crudo[:, mask]
n_desc = int((~mask).sum())

X = StandardScaler().fit_transform(X_util)
n, d = X.shape
print(f"digits: {X_crudo.shape} -> {X.shape} ({n_desc} pixeles de varianza cero descartados)")


# ------------------------------------- 1. ayudas graficas del capitulo 2
pca = PCA(n_components=2, random_state=SEED)
Z_pca = pca.fit_transform(X)
var_pca2 = float(pca.explained_variance_ratio_[:2].sum())

# MDS sobre submuestra: el algoritmo es O(n^2) por iteracion y con 1797 puntos
# tardaba demasiado para lo poco que aporta la figura.
idx_mds = np.random.RandomState(SEED).choice(n, 400, replace=False)
Z_mds = MDS(n_components=2, random_state=SEED, n_init=2, max_iter=150,
            normalized_stress="auto").fit_transform(X[idx_mds])


# ------------------------------------------- 2. barrido de perplejidad
perplejidades = [5, 30, 50]
embeddings = {}
filas_perp = []

for p in perplejidades:
    ts = TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto",
              max_iter=1000, random_state=SEED)
    Z = ts.fit_transform(X)
    embeddings[p] = Z
    # trustworthiness mide cuanta vecindad local sobrevive a la proyeccion:
    # 1 es perfecto. Es el numero honesto para comparar mapas entre si.
    tw = trustworthiness(X, Z, n_neighbors=12)
    km_emb = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit(Z)
    filas_perp.append(dict(
        perplejidad=p,
        kl_final=float(ts.kl_divergence_),
        iteraciones=int(ts.n_iter_),
        trustworthiness=float(tw),
        silhouette_emb=float(silhouette_score(Z, km_emb.labels_)),
        ari_emb_vs_digito=float(adjusted_rand_score(y, km_emb.labels_)),
        etiquetas=km_emb.labels_,
    ))
    print(f"perp {p:>2}: KL={ts.kl_divergence_:.3f}  TW={tw:.3f}")

perp_df = pd.DataFrame([{k: v for k, v in f.items() if k != "etiquetas"}
                        for f in filas_perp])
perp_df.to_csv(ruta("tsne_perplejidad.csv"), index=False)


# ------------------------- 3. k por pseudo-F sobre el espacio original
# calinski_harabasz_score = PSF del capitulo 5. Se busca el pico, igual que en
# la Cluster History de PROC CLUSTER.
filas_k = []
for k in range(2, 16):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    filas_k.append(dict(
        k=k,
        pseudo_f=float(calinski_harabasz_score(X, km.labels_)),
        silhouette=float(silhouette_score(X, km.labels_)),
        inercia=float(km.inertia_),
        ari_vs_digito=float(adjusted_rand_score(y, km.labels_)),
    ))
k_df = pd.DataFrame(filas_k)
k_df.to_csv(ruta("pseudo_f.csv"), index=False)
k_psf = int(k_df.loc[k_df["pseudo_f"].idxmax(), "k"])
print("pico de pseudo-F en k =", k_psf)

K = 10  # se fija 10 por la exploracion t-SNE y por las 10 clases de digits
km_orig = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
lab_orig = km_orig.labels_


# ----------------------------------- 4. contraste original vs embedding
lab_emb30 = filas_perp[1]["etiquetas"]
lab_emb5 = filas_perp[0]["etiquetas"]
lab_emb50 = filas_perp[2]["etiquetas"]

# estabilidad del espacio original: mismo k-means con otra semilla
lab_orig_b = KMeans(n_clusters=K, n_init=10, random_state=7).fit(X).labels_

comp = pd.DataFrame([
    dict(espacio="Original (61 var.)",
         silhouette_propio=float(silhouette_score(X, lab_orig)),
         ari_vs_digito=float(adjusted_rand_score(y, lab_orig)),
         estabilidad=float(adjusted_rand_score(lab_orig, lab_orig_b)),
         detalle_estabilidad="semillas 42 vs 7"),
    dict(espacio="Embedding t-SNE 2D",
         silhouette_propio=float(silhouette_score(embeddings[30], lab_emb30)),
         ari_vs_digito=float(adjusted_rand_score(y, lab_emb30)),
         estabilidad=float(adjusted_rand_score(lab_emb5, lab_emb50)),
         detalle_estabilidad="perplejidad 5 vs 50"),
])
comp.to_csv(ruta("comparacion_espacios.csv"), index=False)

# El silhouette del embedding se mide sobre coordenadas que t-SNE optimizo
# para verse separadas: por eso no es comparable con el del espacio original.
# Se agrega el silhouette del particionamiento del embedding medido en el
# espacio original, que es la comparacion que si es justa.
sil_cruzado = float(silhouette_score(X, lab_emb30))
print("silhouette de la particion del embedding, medida en el espacio original:",
      round(sil_cruzado, 4))


# ------------------------------------------------- 5. DBSCAN, el fracaso
def barrer_dbscan(datos, nombre, eps_grid, min_samples_list=(5, 10)):
    filas = []
    for ms in min_samples_list:
        for eps in eps_grid:
            et = DBSCAN(eps=eps, min_samples=ms).fit_predict(datos)
            ncl = len(set(et)) - (1 if -1 in et else 0)
            nr = et != -1
            # silhouette solo sobre los puntos no-ruido: es el criterio sin
            # etiquetas que se usa despues para elegir la mejor corrida.
            sil = (float(silhouette_score(datos[nr], et[nr]))
                   if ncl >= 2 and nr.sum() > ncl else float("nan"))
            tam = np.array([(et == c).sum() for c in np.unique(et) if c != -1])
            filas.append(dict(espacio=nombre, eps=round(float(eps), 3),
                              min_samples=ms, n_clusters=ncl,
                              pct_ruido=100.0 * float((et == -1).mean()),
                              pct_mayor_cluster=(100.0 * float(tam.max()) / len(datos)
                                                 if len(tam) else float("nan")),
                              silhouette=sil))
    return filas


def grilla_eps(datos, k=10, m=18):
    """Rejilla de eps entre los percentiles 1 y 99 de la distancia al k-esimo
    vecino. Es la heuristica del codo de Ester et al. (1996), pero barrida
    entera para poder decir que no fue por falta de busqueda."""
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(datos).kneighbors(datos)
    dk = np.sort(dist[:, k])
    return np.linspace(np.percentile(dk, 1), np.percentile(dk, 99), m)


X_pca10 = PCA(n_components=10, random_state=SEED).fit_transform(X)

filas_db = []
filas_db += barrer_dbscan(X, "Original (61 var.)", grilla_eps(X))
filas_db += barrer_dbscan(X_pca10, "PCA 10 componentes", grilla_eps(X_pca10))
filas_db += barrer_dbscan(embeddings[30], "Embedding t-SNE 2D",
                          grilla_eps(embeddings[30]))
db = pd.DataFrame(filas_db)
db.to_csv(ruta("dbscan_barrido.csv"), index=False)

# Resumen por espacio. Primera version: me quedaba con la corrida de mas
# clusters y menos de 30 % de ruido, y las tres daban "exito". Esa regla es
# tramposa contra el embedding, donde el eps mas chico fabrica 83 pedazos.
# Contar clusters tampoco alcanza: hay que ver si son grupos o un bloque con
# migas. Regla final: entre las corridas con 2 a 30 clusters y menos de 30 %
# de ruido, la de mayor silhouette (criterio sin etiquetas). Recien despues se
# mira el ARI contra el digito, que es el juez externo y no se usa para elegir.
datos_por_espacio = {"Original (61 var.)": X, "PCA 10 componentes": X_pca10,
                     "Embedding t-SNE 2D": embeddings[30]}
res_db = []
for esp, g in db.groupby("espacio", sort=False):
    ok = g[(g["n_clusters"] >= 2) & (g["n_clusters"] <= 30) &
           (g["pct_ruido"] <= 30) & g["silhouette"].notna()]
    if len(ok):
        mejor = ok.loc[ok["silhouette"].idxmax()]
        et = DBSCAN(eps=float(mejor["eps"]),
                    min_samples=int(mejor["min_samples"])).fit_predict(
                        datos_por_espacio[esp])
        res_db.append(dict(espacio=esp, eps_probados=int(len(g)),
                           n_clusters=int(mejor["n_clusters"]),
                           eps_mejor=float(mejor["eps"]),
                           min_samples_mejor=int(mejor["min_samples"]),
                           pct_ruido=float(mejor["pct_ruido"]),
                           pct_mayor_cluster=float(mejor["pct_mayor_cluster"]),
                           silhouette=float(mejor["silhouette"]),
                           ari_vs_digito=float(adjusted_rand_score(y, et))))
    else:
        res_db.append(dict(espacio=esp, eps_probados=int(len(g)),
                           n_clusters=int(g["n_clusters"].max()),
                           eps_mejor=float("nan"), min_samples_mejor=-1,
                           pct_ruido=float("nan"),
                           pct_mayor_cluster=float("nan"),
                           silhouette=float("nan"),
                           ari_vs_digito=float("nan")))
res_db = pd.DataFrame(res_db)
res_db.to_csv(ruta("dbscan_resumen.csv"), index=False)
print(res_db)

# Evidencia de la concentracion de distancias: cociente entre el desvio y la
# media de las distancias entre pares. Si tiende a 0, todos los puntos estan
# a la misma distancia y no existe un eps que discrimine.
rs = np.random.RandomState(SEED)
sub = rs.choice(n, 600, replace=False)


def concentracion(datos):
    from scipy.spatial.distance import pdist
    dd = pdist(datos)
    return float(dd.std() / dd.mean()), float(dd.min()), float(dd.max())


cv_o, dmin_o, dmax_o = concentracion(X[sub])
cv_p, dmin_p, dmax_p = concentracion(X_pca10[sub])
cv_e, dmin_e, dmax_e = concentracion(embeddings[30][sub])
conc = pd.DataFrame([
    dict(espacio="Original (61 var.)", dim=d, cv_distancias=cv_o,
         razon_max_min=dmax_o / dmin_o),
    dict(espacio="PCA 10 componentes", dim=10, cv_distancias=cv_p,
         razon_max_min=dmax_p / dmin_p),
    dict(espacio="Embedding t-SNE 2D", dim=2, cv_distancias=cv_e,
         razon_max_min=dmax_e / dmin_e),
])
conc.to_csv(ruta("concentracion_distancias.csv"), index=False)


# ------------------------------------------------------- 6. perfilado
def anova_por_variable(datos, etiquetas, nombres):
    filas = []
    for j, nom in enumerate(nombres):
        grupos = [datos[etiquetas == c, j] for c in np.unique(etiquetas)]
        F, p = f_oneway(*grupos)
        # eta cuadrado: proporcion de varianza de la variable explicada por la
        # particion. Es la medida de tamano de efecto que falta en la salida
        # de PROC ANOVA y que evita leer una F grande como si fuera grande.
        gm = datos[:, j].mean()
        ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in grupos)
        sst = ((datos[:, j] - gm) ** 2).sum()
        filas.append(dict(variable=nom, F=float(F), p=float(p),
                          eta2=float(ssb / sst)))
    return pd.DataFrame(filas)


nombres = [f"p{r},{c}" for r, c in
           ((int(i) // 8, int(i) % 8) for i in pix_idx)]
alfa = 0.05
bonf = alfa / d

perfil = anova_por_variable(X, lab_orig, nombres).sort_values("F", ascending=False)
perfil["significativa_bonf"] = (perfil["p"] < bonf).astype(int)
perfil.to_csv(ruta("anova_perfil.csv"), index=False)

# medias por grupo de las 6 variables de mayor F, en la escala original 0-16
top = perfil.head(6)["variable"].tolist()
pos = [nombres.index(t) for t in top]
medias = pd.DataFrame(
    {t: [X_util[lab_orig == c, j].mean() for c in range(K)]
     for t, j in zip(top, pos)},
    index=[f"G{c}" for c in range(K)],
).T.reset_index().rename(columns={"index": "variable"})
medias = medias.merge(perfil[["variable", "F", "eta2"]], on="variable")
medias.to_csv(ruta("perfil_medias.csv"), index=False)

# --- control A: particion al azar de los mismos datos
lab_azar = rs.randint(0, K, size=n)
perfil_azar = anova_por_variable(X, lab_azar, nombres)

# --- control B: k-means sobre datos uniformes en un hiper-rectangulo.
# Es exactamente la H0 del CCC de Sarle (1983): no hay estructura. Si el ANOVA
# igual da significativo, la prueba no sirve para decidir si los grupos existen.
X_unif = rs.uniform(X.min(axis=0), X.max(axis=0), size=(n, d))
lab_unif = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_unif)
perfil_unif = anova_por_variable(X_unif, lab_unif, nombres)

controles = pd.DataFrame([
    dict(escenario="k-means sobre digits",
         n_signif=int((perfil["p"] < bonf).sum()), F_max=float(perfil["F"].max()),
         F_mediana=float(perfil["F"].median()), eta2_max=float(perfil["eta2"].max())),
    dict(escenario="Particion al azar de digits",
         n_signif=int((perfil_azar["p"] < bonf).sum()),
         F_max=float(perfil_azar["F"].max()),
         F_mediana=float(perfil_azar["F"].median()),
         eta2_max=float(perfil_azar["eta2"].max())),
    dict(escenario="k-means sobre datos uniformes",
         n_signif=int((perfil_unif["p"] < bonf).sum()),
         F_max=float(perfil_unif["F"].max()),
         F_mediana=float(perfil_unif["F"].median()),
         eta2_max=float(perfil_unif["eta2"].max())),
])
controles.to_csv(ruta("anova_controles.csv"), index=False)
print(controles)

# --- CDA sobre la solucion de clusters (equivalente a PROC CANDISC)
cda = LinearDiscriminantAnalysis(solver="svd", n_components=2)
Z_cda = cda.fit_transform(X, lab_orig)
evr = getattr(cda, "explained_variance_ratio_", np.array([np.nan, np.nan]))

# estructura canonica intra-grupo combinada (pooled within): se centra cada
# variable y cada funcion canonica por la media de su grupo y se correlaciona.
Xc = X.copy().astype(float)
Zc = Z_cda.copy().astype(float)
for c in np.unique(lab_orig):
    m = lab_orig == c
    Xc[m] -= Xc[m].mean(axis=0)
    Zc[m] -= Zc[m].mean(axis=0)
estr = pd.DataFrame({
    "variable": nombres,
    "Can1": [float(np.corrcoef(Xc[:, j], Zc[:, 0])[0, 1]) for j in range(d)],
    "Can2": [float(np.corrcoef(Xc[:, j], Zc[:, 1])[0, 1]) for j in range(d)],
})
estr["abs1"] = estr["Can1"].abs()
estr.sort_values("abs1", ascending=False).drop(columns="abs1").to_csv(
    ruta("cda_estructura.csv"), index=False)


# ------------------------------------------------------------- figuras
def panel(ax, Z, color, titulo, cmap="tab10"):
    # s=7 y no 5: en el PDF los paneles quedan de menos de 4 cm de ancho y con
    # s=5 la nube se ve como una mancha gris. Titulo de una sola linea, que dos
    # lineas se comen medio panel.
    ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=cmap, s=7, alpha=0.8,
               linewidths=0)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# Figura unica, 2 x 4. Historia de esto: primero eran tres figuras sueltas
# (ayudas graficas, barrido de perplejidad, validacion). Entre titulo, nota y
# espaciado de floats, tres figuras se comian casi una hoja de las cuatro que
# permite la consigna, y ademas los paneles en fila de a cinco quedaban
# ilegibles. En una grilla de dos filas cada panel gana casi un centimetro de
# ancho y se paga una sola vez el costo de titulo y nota.
#   Fila 1: las tres ayudas graficas del capitulo 2 de SAS, mas la curva de
#           pseudo-F, que es el criterio de cantidad de clusters del capitulo 5.
#   Fila 2: el barrido de perplejidad (5, 30, 50) y la validacion, o sea las
#           etiquetas de k-means del espacio original sobre el mapa.
fig, axs = plt.subplots(2, 4, figsize=(14.0, 7.0))

panel(axs[0, 0], Z_pca, y, f"PCA ({var_pca2*100:.1f} % de varianza)")
panel(axs[0, 1], Z_mds, y[idx_mds], "MDS metrico (n = 400)")
panel(axs[0, 2], Z_cda, lab_orig, "CDA sobre los grupos k-means")

ax = axs[0, 3]
ax.plot(k_df["k"], k_df["pseudo_f"], marker="o", ms=3, color="#1f4e79")
ax.axvline(K, color="#c00000", ls="--", lw=1)
ax.set_xlabel("cantidad de clusters (k)", fontsize=8)
ax.set_ylabel("pseudo-F", fontsize=8)
ax.set_title("Pseudo-F sobre el espacio original", fontsize=9)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.3)

for col, f in enumerate(filas_perp):
    panel(axs[1, col], embeddings[f["perplejidad"]], y,
          f"t-SNE, perplejidad {f['perplejidad']} (TW = {f['trustworthiness']:.3f})")
panel(axs[1, 3], embeddings[30], lab_orig,
      "Validacion: grupos k-means sobre el mapa")

fig.tight_layout()
fig.savefig(ruta("ayudas_graficas.png"), dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- meta
meta = dict(
    semilla=SEED,
    var_pca2=var_pca2,
    n_eps_por_espacio=int(len(grilla_eps(X))),
    python=platform.python_version(),
    sklearn=sklearn.__version__,
    numpy=np.__version__,
    scipy=scipy.__version__,
    n_obs=int(n),
    n_var_original=int(X_crudo.shape[1]),
    n_var_usadas=int(d),
    n_var_descartadas=n_desc,
    k_elegido=K,
    k_pico_pseudo_f=k_psf,
    alfa=alfa,
    alfa_bonferroni=bonf,
    silhouette_particion_embedding_en_espacio_original=sil_cruzado,
    cda_var_can1=float(evr[0]) if np.isfinite(evr[0]) else None,
    cda_var_can2=float(evr[1]) if np.isfinite(evr[1]) else None,
    perplejidades=perplejidades,
    # t-SNE no expone transform(): no hay forma de proyectar una observacion
    # nueva sobre el mapa ya entrenado. Se chequea en vez de afirmarlo.
    tsne_tiene_transform=hasattr(TSNE, "transform"),
    pseudo_f_pico_interior=bool(1 < int(k_df["pseudo_f"].idxmax()) < len(k_df) - 1),
    silhouette_k_max=int(k_df.loc[k_df["silhouette"].idxmax(), "k"]),
)
with open(ruta("meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# El .Rmd lee esto con read.csv() y no con jsonlite: el informe se arma entre
# varios y no quiero agregarle una libreria de R al documento compartido.
pd.DataFrame([{k: v for k, v in meta.items() if not isinstance(v, list)}]).to_csv(
    ruta("meta.csv"), index=False)

print("listo ->", OUT)
