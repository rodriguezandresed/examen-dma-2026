# -*- coding: utf-8 -*-
"""Auditoria del entregable contra REQUISITOS.md.

Un chequeo por renglon del checklist de la seccion 7, mas los cinco puntos de
la seccion 2b. Imprime PASA / FALLA / REVISAR y, cuando falla, el detalle.

REVISAR marca lo que no se puede decidir de forma automatica y necesita ojo
humano (por ejemplo, si una cita es pertinente, o si el oral esta preparado).

Correr:  python auditoria.py
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RMD = RAIZ / "examen-dma-2026.Rmd"
PDF = RAIZ / "examen-dma-2026.pdf"

EJ = [1, 2, 3, 7, 8, 9, 10]          # ejercicios resueltos hasta ahora
CONCEPTUALES = [7, 8, 9, 10]          # los que escribimos nosotros

resultados = []


def chequeo(nombre, estado, detalle=""):
    resultados.append((nombre, estado, detalle))


def seccion(texto, ej):
    """Devuelve el cuerpo del ejercicio `ej`, sin el enunciado en cursiva."""
    try:
        s = texto.split("# Ejercicio %d ·" % ej)[1]
    except IndexError:
        return ""
    s = re.split(r"# (Ejercicio|Apéndice)", s)[0]
    return s


def prosa(s):
    """Saca chunks, comentarios HTML y enunciados citados."""
    s = re.sub(r"```\{r.*?```", " ", s, flags=re.S)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"^\*[^\n]*\*$", " ", s, flags=re.M)
    return s


texto = io.open(RMD, encoding="utf-8").read()

# ---------------------------------------------------------------- formales --
nombres = re.search(r"apptocmd\{\\maketitle\}.*?\n", texto)
tiene_nombres = bool(nombres and "Rodriguez" in nombres.group(0)
                     and "Maceo" in nombres.group(0))
chequeo("Portada con los nombres de los integrantes",
        "PASA" if tiene_nombres else "FALLA",
        "" if tiene_nombres else "no encuentro los dos nombres en el maketitle")

sin_fuentes = [e for e in EJ if "Fuentes del Ejercicio %d" % e not in texto]
chequeo("Cada respuesta tiene su fuente detallada",
        "PASA" if not sin_fuentes else "FALLA",
        "" if not sin_fuentes else "faltan en: %s" % sin_fuentes)

# ---------------------------------------------------------------- paginas ---
if PDF.exists():
    import fitz
    d = fitz.open(PDF)
    toc = [t for t in d.get_toc() if t[0] == 1]
    excedidos = []
    for j, (l, t, pg) in enumerate(toc):
        fin = toc[j + 1][2] if j + 1 < len(toc) else len(d) + 1
        if t.startswith("Ejercicio") and fin - pg > 4:
            excedidos.append("%s: %d hojas" % (t.split("·")[0].strip(), fin - pg))
    chequeo("Ningun ejercicio supera las 4 hojas",
            "PASA" if not excedidos else "FALLA", "; ".join(excedidos))

    # tabla partida entre hojas: caption en una hoja y filas en la siguiente
    partidas = []
    for i in range(len(d) - 1):
        t1 = d[i].get_text()
        caps = re.findall(r"Tabla (\d+):", t1)
        if not caps:
            continue
        # si el caption esta en el ultimo 18 % de la hoja, sospechar
        for b in d[i].get_text("blocks"):
            if re.search(r"Tabla \d+:", b[4]) and b[3] / d[i].rect.height > 0.82:
                partidas.append("p.%d" % (i + 1))
    chequeo("Ninguna tabla arranca al pie de una hoja",
            "PASA" if not partidas else "FALLA", ", ".join(partidas))
else:
    chequeo("PDF compilado", "FALLA", "no existe examen-dma-2026.pdf")

# ------------------------------------------------------------ 2b, 5 puntos --
faltan_esquema = []
for e in CONCEPTUALES:
    s = seccion(texto, e)
    if not re.search(r'include_graphics\("[^"]*esquema[^"]*"\)', s):
        faltan_esquema.append(e)
chequeo("Cada ejercicio propio tiene esquema de flujo (2b punto 4)",
        "PASA" if not faltan_esquema else "FALLA",
        "" if not faltan_esquema else "sin esquema: %s" % faltan_esquema)

faltan_compite = []
for e in CONCEPTUALES:
    p = prosa(seccion(texto, e))
    if not re.search(r"XGBoost|LightGBM|Random Forest|UMAP|frente a|compiten|"
                     r"en lugar de|alternativa|contra ", p):
        faltan_compite.append(e)
chequeo("Cada ejercicio nombra contra quien compite (2b punto 3)",
        "PASA" if not faltan_compite else "FALLA",
        "" if not faltan_compite else "sin competidores: %s" % faltan_compite)

faltan_es = []
for e in CONCEPTUALES:
    p = prosa(seccion(texto, e))
    if not re.search(r"[Ll]a entrada (es|del|de la)|[Ll]a salida (es|son|del)", p):
        faltan_es.append(e)
chequeo("Cada ejercicio explicita entrada y salida (2b punto 2)",
        "PASA" if not faltan_es else "FALLA",
        "" if not faltan_es else "sin entrada/salida: %s" % faltan_es)

# pseudocodigo o derivacion paso a paso, que la aclaracion pide evitar
pseudo = []
for e in CONCEPTUALES:
    s = seccion(texto, e)
    s = s.split("**Fuentes del Ejercicio")[0]   # las Fuentes describen la fuente
    if re.search(r"[Pp]seudoc[óo]digo|^\s*\d\.\s+(Inicial|Calcul|Actualic)", s, flags=re.M):
        pseudo.append(e)
chequeo("Ningun ejercicio propio reproduce pseudocodigo",
        "PASA" if not pseudo else "REVISAR",
        "" if not pseudo else "mencion de pseudocodigo en: %s" % pseudo)

# ---------------------------------------------------------------- estilo ----
pruebas = [
    ("Sin em dashes ni en dashes", r"—|–"),
    ("Sin guiones dobles en prosa", r" -- "),
    ("Sin muletillas delatoras",
     r"de modo que|[Cc]onviene señalar|Cabe consignar|En síntesis|"
     r"Corresponde señalar|satisfactorio|contundente|insumo"),
    ("Sin primera persona",
     r"\b(creo|considero|analicé|observé|encontré|nuestro|nuestra|hicimos|usamos|vimos)\b"),
    ("Sin notas de trabajo en texto visible", r"COMPLETAR|TODO|PENDIENTE|XXX"),
    ("Sin tabulaciones", r"\t"),
]
for nombre, pat in pruebas:
    hits = re.findall(pat, texto)
    chequeo(nombre, "PASA" if not hits else "FALLA",
            "" if not hits else "%d coincidencias" % len(hits))

# gerundios fuera de los enunciados citados
ger = []
for i, l in enumerate(texto.split("\n"), 1):
    if l.startswith("*") or l.startswith("#"):
        continue
    for m in re.finditer(r"\b[A-Za-zÁÉÍÓÚáéíóúñ]+(?:ando|iendo)\b", l):
        if m.group(0).lower() in ("cuando", "siendo", "blando"):
            continue
        ger.append("L%d %s" % (i, m.group(0)))
chequeo("Sin gerundios fuera de los enunciados",
        "PASA" if not ger else "FALLA", ", ".join(ger[:6]))

# ------------------------------------------------------------ referencias ---
refs = texto.split("# Referencias")[1] if "# Referencias" in texto else ""
entradas = [" ".join(e.split()) for e in re.split(r"\n\s*\n", refs.strip()) if e.strip()]


def clave(e):
    m = re.match(r"^(.*?)\(", e)
    k = (m.group(1) if m else e).strip().lower()
    return re.sub(r"^van der ", "maaten", k)


ordenadas = [clave(e) for e in entradas]
chequeo("Referencias en orden alfabetico",
        "PASA" if ordenadas == sorted(ordenadas) else "FALLA",
        "" if ordenadas == sorted(ordenadas) else "hay entradas fuera de orden")

sin_loc = [e for e in entradas
           if "doi.org" not in e and "http" not in e and "s.f." not in e]
chequeo("Referencias externas con localizador",
        "PASA" if len(sin_loc) <= 2 else "REVISAR",
        "sin localizador: %d" % len(sin_loc))

# citas del cuerpo que no tienen entrada, y entradas que nadie cita
cuerpo = prosa(texto.split("# Referencias")[0])
apellidos = set()
for e in entradas:
    m = re.match(r"^((?:van der |de |del )?[A-ZÁÉÍÓÚ][\wáéíóúñ'-]+)", e)
    if m:
        ap = m.group(1)
        apellidos.add(ap)
        apellidos.add(ap.split()[-1])      # Maaten, Institute
cuerpo_plano = " ".join(cuerpo.split())
citados = set(re.findall(r"([A-ZÁÉÍÓÚ][\wáéíóúñ'-]+)(?:,? (?:y|&) [A-ZÁÉÍÓÚ][\wáéíóúñ'-]+)?"
                         r"(?: et al\.)?,? \(?(?:s\.f\.|\d{4})", cuerpo_plano))
huerfanas = sorted(a for a in apellidos if a not in citados and a not in ("SAS",))
no_listadas = sorted(c for c in citados if c not in apellidos
                     and c not in ("Elaboración", "Nota", "Material", "Corrida"))
chequeo("Toda entrada de Referencias se cita en el texto",
        "PASA" if not huerfanas else "REVISAR", ", ".join(huerfanas))
chequeo("Toda cita del texto tiene entrada en Referencias",
        "PASA" if not no_listadas else "REVISAR", ", ".join(no_listadas[:8]))

# ---------------------------------------------------------------- codigo ----
scripts = sorted(RAIZ.glob("codigo/*/*.py"))
flojos = []
for f in scripts:
    src = io.open(f, encoding="utf-8").read()
    lineas = [l for l in src.split("\n") if l.strip()]
    com = sum(1 for l in lineas if l.strip().startswith("#"))
    doc = src.count('"""') // 2
    if lineas and (100 * com / len(lineas) < 4) and doc < 2:
        flojos.append("%s (%d%% comentarios, %d docstrings)"
                      % (f.name, 100 * com // len(lineas), doc))
chequeo("Los scripts tienen comentarios propios",
        "PASA" if not flojos else "FALLA", "; ".join(flojos))

# los archivos que el Rmd referencia tienen que existir
faltantes = [r for r in sorted(set(re.findall(r'"(codigo/[^"]+)"', texto)))
             if not (RAIZ / r).exists()]
chequeo("Todo archivo referenciado existe",
        "PASA" if not faltantes else "FALLA", ", ".join(faltantes))

# ------------------------------------------------- numeros desde los datos --
inline = len(re.findall(r"`r [^`]+`", texto))
chequeo("Los numeros del texto salen de los CSV",
        "PASA" if inline > 40 else "REVISAR",
        "%d interpolaciones inline" % inline)

# --------------------------------------------- referencias cruzadas ---------
lineas = texto.split("\n")
tab = fig = 0
mapa = {}
sec = None
for i, l in enumerate(lineas):
    m = re.match(r"^# Ejercicio (\d+)", l)
    if m:
        sec = int(m.group(1))
    if l.startswith("```{r") and "fig.cap=" in l:
        fig += 1
        mapa.setdefault(sec, {"Tabla": [], "Figura": []})["Figura"].append(fig)
    elif l.startswith("```{r") and sec:
        j, cuerpo_ch = i + 1, []
        while j < len(lineas) and not lineas[j].startswith("```"):
            cuerpo_ch.append(lineas[j]); j += 1
        if any("caption" in c for c in cuerpo_ch):
            tab += 1
            mapa.setdefault(sec, {"Tabla": [], "Figura": []})["Tabla"].append(tab)
malas = []
sec, ch, com = None, False, False
for i, l in enumerate(lineas, 1):
    m = re.match(r"^# Ejercicio (\d+)", l)
    if m:
        sec = int(m.group(1))
    if l.startswith("```"):
        ch = not ch; continue
    if "<!--" in l:
        com = True
    if com:
        if "-->" in l:
            com = False
        continue
    if ch or sec not in mapa:
        continue
    for mm in re.finditer(r"(Tabla|Figura) (\d+)", l):
        if int(mm.group(2)) not in mapa[sec][mm.group(1)]:
            malas.append("L%d Ej%d %s" % (i, sec, mm.group(0)))
chequeo("Referencias cruzadas a tablas y figuras correctas",
        "PASA" if not malas else "FALLA", ", ".join(malas))

# ---------------------------------------------------------------- informe ---
ancho = max(len(n) for n, _, _ in resultados)
falla = sum(1 for _, e, _ in resultados if e == "FALLA")
rev = sum(1 for _, e, _ in resultados if e == "REVISAR")
print("=" * (ancho + 26))
for n, e, d in resultados:
    print("%-*s  %-8s %s" % (ancho, n, e, d))
print("=" * (ancho + 26))
print("%d chequeos: %d fallan, %d a revisar a mano, %d pasan"
      % (len(resultados), falla, rev, len(resultados) - falla - rev))
sys.exit(1 if falla else 0)
