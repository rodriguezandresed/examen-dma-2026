"""Ejercicio 6: por qué la regresión común falla con censura, sobre los datos Rossi.

Datos Rossi (Rossi et al., 1980), los mismos de la clase 7: 432 liberados seguidos 52
semanas; `week` es la semana de re-arresto o de fin del seguimiento y `arrest` vale 1
si hubo re-arresto. Se comparan dos atajos de regresión con Kaplan-Meier y Cox.

Salida: codigo/resultados/e6/meta.json (el .Rmd lee todos los números de ahí).
"""

import json
from pathlib import Path

import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time
from sklearn.linear_model import LinearRegression

SEED = 42  # nada es aleatorio acá; se fija igual por regla del trabajo
np.random.seed(SEED)

OUT = Path(__file__).resolve().parents[1] / "resultados" / "e6"
OUT.mkdir(parents=True, exist_ok=True)

df = load_rossi()
T, E = df["week"], df["arrest"]
COVS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
horizonte = int(T.max())

# Atajo 1: tirar los censurados y promediar solo los re-arrestados
atajo_descarta = float(T[E == 1].mean())
# Atajo 2: tratar el fin del seguimiento como si fuera el re-arresto
atajo_imputa = float(T.mean())
# Toda la censura de Rossi cae en la semana 52 (fin administrativo del estudio)
censura_unica = bool((T[E == 0] == horizonte).all())

km = KaplanMeierFitter().fit(T, E)
s_horizonte = float(km.survival_function_at_times(horizonte).iloc[0])
s_26 = float(km.survival_function_at_times(26).iloc[0])
mediana_km = km.median_survival_time_  # inf si S(t) nunca baja de 0,5
# Media restringida a 52 semanas (área bajo S(t)). Primero se la puso como "la media
# correcta" contra el atajo 2 y dio idéntica: con censura toda en t = 52, el atajo 2
# estima justamente esta media truncada, no el tiempo medio hasta el re-arresto.
rmst = float(restricted_mean_survival_time(km, t=horizonte))

# Efecto de la ayuda financiera (fin): regresión sobre re-arrestados contra Cox
ols = LinearRegression().fit(df.loc[E == 1, COVS], T[E == 1])
coef_ols_fin = float(ols.coef_[COVS.index("fin")])
# Probabilidad de re-arresto según fin, ignorando el tiempo (la "logística sin tiempo")
tasa_fin = df.groupby("fin")["arrest"].mean()

cph = CoxPHFitter().fit(df, duration_col="week", event_col="arrest")
res = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]

# El informe explica la prueba de Grambsch y Therneau y al principio no la corria
# sobre el Cox que interpreta. Se corre aca, con transformacion por rangos como en el
# notebook de la clase: si fin la pasa, su hazard ratio se puede leer aunque otras
# covariables no la pasen.
from lifelines.statistics import proportional_hazard_test
ph = proportional_hazard_test(cph, df, time_transform="rank").summary["p"]
ph_rechaza = sorted(c for c in COVS if ph[c] < 0.05)

lr = logrank_test(T[df.fin == 1], T[df.fin == 0], E[df.fin == 1], E[df.fin == 0])
km1 = KaplanMeierFitter().fit(T[df.fin == 1], E[df.fin == 1])
km0 = KaplanMeierFitter().fit(T[df.fin == 0], E[df.fin == 0])
meta = {
    "n": int(len(df)), "eventos": int(E.sum()), "censurados": int((E == 0).sum()),
    "pct_censura": float((E == 0).mean() * 100), "horizonte": horizonte,
    "atajo_descarta": atajo_descarta, "atajo_imputa": atajo_imputa, "rmst": rmst,
    "censura_unica": censura_unica,
    "s_horizonte": s_horizonte, "s_26": s_26,
    "mediana_km_finita": bool(np.isfinite(mediana_km)),
    "coef_ols_fin": coef_ols_fin,
    "tasa_arresto_fin1": float(tasa_fin[1]), "tasa_arresto_fin0": float(tasa_fin[0]),
    "hr_fin": float(res.loc["fin", "exp(coef)"]),
    "hr_fin_lo": float(res.loc["fin", "exp(coef) lower 95%"]),
    "hr_fin_hi": float(res.loc["fin", "exp(coef) upper 95%"]),
    "p_fin": float(res.loc["fin", "p"]),
    "hr_prio": float(res.loc["prio", "exp(coef)"]),
    "hr_age": float(res.loc["age", "exp(coef)"]),
    "c_index": float(cph.concordance_index_),
    "logrank_p": float(lr.p_value),
    "s_horizonte_fin1": float(km1.survival_function_at_times(horizonte).iloc[0]),
    "s_horizonte_fin0": float(km0.survival_function_at_times(horizonte).iloc[0]),
    "ph_p_fin": float(ph["fin"]),
    "ph_rechaza": ph_rechaza,
    "ph_p_rechaza": [float(ph[c]) for c in ph_rechaza],
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(json.dumps(meta, indent=2, ensure_ascii=False))
