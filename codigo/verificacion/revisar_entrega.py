"""
Verificacion de la entrega: corre los controles que la consigna y el CLAUDE.md
imponen, sobre el .Rmd y sobre el PDF ya compilado.

Por que existe: los controles que importan no se ven en el fuente. El limite de
4 hojas por pregunta, el texto que se pega al numero de pagina y el titulo que
queda colgado al pie solo aparecen una vez compilado, y hasta ahora se median a
mano cada vez. Esto los deja corribles por cualquiera del grupo.

Uso, desde la raiz del repo:

    python codigo/verificacion/revisar_entrega.py

Requiere PyMuPDF para los controles de maqueta:

    pip install pymupdf

Sin PyMuPDF igual corre: saltea los controles sobre el PDF, avisa, y hace los
del .Rmd. Devuelve 0 si todo pasa y 1 si algo falla, para poder encadenarlo.

Los umbrales son los que el grupo venia usando a mano:
  - 4 hojas por pregunta (consigna).
  - 0,45 cm minimos entre el ultimo renglon y el numero de pagina. Nace de un
    \\enlargethispage que habia dejado el texto a 0,14 cm.
  - un titulo necesita 2 renglones de contenido debajo para no estar colgado.
"""

import io
import os
import re
import sys
import unicodedata

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, "..", ".."))
RMD = os.path.join(RAIZ, "examen-dma-2026.Rmd")
PDF = os.path.join(RAIZ, "examen-dma-2026.pdf")

HOJAS_MAX = 4
MARGEN_MIN_CM = 0.45
RENGLONES_BAJO_TITULO = 2

# Excepciones al limite de hojas, con el motivo. Van acá y no en un comentario
# para que cada corrida las vuelva a mostrar: una excepcion que no se ve se
# convierte en un incumplimiento olvidado. Si el grupo cambia de criterio, se
# saca la entrada y el control pasa a fallar.
# El Ejercicio 1 estuvo registrado en 5 hojas; se volvio a 4 sin perder clausulas,
# asi que hoy no hay excepciones.
HOJAS_EXCEPCION = {}

CM = 72 / 2.54  # puntos por centimetro

fallas = []
avisos = []


def ok(titulo, detalle=""):
    print("  OK    %-46s %s" % (titulo, detalle))


def mal(titulo, detalle=""):
    print("  FALLA %-46s %s" % (titulo, detalle))
    fallas.append(titulo)


def aviso(titulo, detalle=""):
    print("  AVISO %-46s %s" % (titulo, detalle))
    avisos.append(titulo)


def sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------
# Controles sobre el .Rmd
# --------------------------------------------------------------------------

def controles_rmd():
    print("\n=== .Rmd ===")
    s = io.open(RMD, encoding="utf-8").read()
    if "# Referencias" not in s:
        mal("seccion Referencias", "no esta")
        return
    cuerpo, refs = s.split("# Referencias")

    # Un bloque Fuentes por ejercicio. La consigna pide detallar la fuente de
    # cada respuesta, asi que faltar uno no es un detalle de formato.
    ej = re.findall(r"^# Ejercicio (\d+)", cuerpo, re.M)
    fu = re.findall(r"^\*\*Fuentes del Ejercicio (\d+)\.", cuerpo, re.M)
    if len(ej) == len(fu) == 10 and ej == fu:
        ok("bloques Fuentes", "10, uno por ejercicio")
    else:
        mal("bloques Fuentes", "ejercicios %s, bloques %s" % (ej, fu))

    # Referencias en las dos direcciones. La comparacion es por apellido: una
    # entrada cuyo apellido no aparece en el cuerpo esta huerfana, y una cita
    # cuyo apellido no abre ninguna entrada no tiene entrada. Se ignoran los
    # segundos autores, que en APA van alfabetizados bajo el primero.
    cb, rf = sin_tildes(cuerpo), sin_tildes(refs)
    bloques = [b.strip() for b in re.split(r"\n\s*\n", rf) if b.strip()]
    entradas = [b for b in bloques if re.search(r"\((\d{4}[a-z]?|s\.f\.)", b)]
    apellidos = []
    for b in entradas:
        m = re.match(r"([A-Za-z][A-Za-z\-']*(?:\s+[a-z]+)?)", b)
        if m:
            apellidos.append(m.group(1).split(",")[0].strip())
    huerfanas = sorted({a for a in apellidos
                        if a.split()[-1] not in cb and a not in cb})
    if huerfanas:
        mal("referencias huerfanas", ", ".join(huerfanas))
    else:
        ok("referencias huerfanas", "ninguna sobre %d entradas" % len(entradas))

    # Los nombres de archivo de catedra van en codigo en linea y traen
    # mayusculas (deep_learning_CONV_Ng.pptx): se sacan antes de buscar citas,
    # porque si no "Ng" pasa por apellido.
    cb_citas = re.sub(r"`[^`]*`", "", cb)
    citas = set()
    for m in re.finditer(r"\(([^()]{2,90}?),\s*(?:\d{4}[a-z]?|s\.f\.)", cb_citas):
        for w in re.findall(r"[A-Z][a-zA-Z\-']+", m.group(1)):
            citas.add(w)
    for m in re.finditer(r"\b([A-Z][a-zA-Z\-']+)\s*(?:et al\.|y\s+[A-Z][a-zA-Z\-']+)?"
                         r"\s*\((?:\d{4}[a-z]?|s\.f\.)", cb_citas):
        citas.add(m.group(1))
    # Siglas, rotulos y los segundos autores de las citas de dos.
    conocidos = set()
    for b in entradas:
        for w in re.findall(r"[A-Z][a-zA-Z\-']+", b[:160]):
            conocidos.add(w)
    ruido = {"Nota", "Tabla", "Figura", "Apendice", "Codigo", "El", "La", "Los",
             "En", "Con", "Para", "De", "Se", "Un", "Una", "Volpacchio", "CONV",
             "DEAP", "SVM", "PCA", "RBF", "SOM", "ARI", "MANOVA", "ANOVA",
             "CANDISC", "Software", "Conjunto", "Opcion", "VC"}
    sin_entrada = sorted(c for c in citas if c not in conocidos and c not in ruido)
    if sin_entrada:
        mal("citas sin entrada en Referencias", ", ".join(sin_entrada))
    else:
        ok("citas sin entrada en Referencias", "ninguna")

    # Reglas de redaccion del CLAUDE.md. Los em dashes se cuentan solo en
    # prosa: en el YAML y en los comentarios HTML el guion doble es legitimo.
    prosa = re.sub(r"<!--.*?-->", "", cuerpo, flags=re.S)
    prosa = re.sub(r"^---$.*?^---$", "", prosa, flags=re.S | re.M)
    prosa = re.sub(r"```\{r.*?```", "", prosa, flags=re.S)
    for nombre, patron in [
        ("em dashes y -- en prosa", r"—|--"),
        ("muletillas prohibidas",
         r"de modo que|[Cc]onviene señalar|Cabe consignar|En síntesis|"
         r"[Cc]orresponde señalar|En primer lugar|En segundo lugar|insumo|"
         r"satisfactorio|contundente"),
        ("primera persona",
         r"\b(creo|considero|nuestro|nuestra|nuestros|hemos|hicimos|analizamos|"
         r"observamos|concluimos)\b"),
        ("marcas de trabajo sin cerrar", r"PENDIENTE|COMPLETAR|TODO"),
        ("simbolo de seccion", r"§"),
        ("rotulos Cuadro, Grafico o Imagen", r"\b(Cuadro|Gráfico|Imagen) \d"),
    ]:
        hits = re.findall(patron, prosa)
        if hits:
            mal(nombre, "%d: %s" % (len(hits), ", ".join(sorted(set(hits))[:6])))
        else:
            ok(nombre, "0")

    # Punto decimal en prosa. Se excluye lo que esta en modo matematico, el
    # codigo en linea y las ordenes de LaTeX, donde el punto es correcto, mas
    # dos casos legitimos: el separador de miles en castellano (100.000 USD) y
    # las unidades tipograficas (1.5em, 0.55em).
    plano = re.sub(r"\$[^$]*\$", "", prosa)
    plano = re.sub(r"`[^`]*`", "", plano)
    plano = re.sub(r"\\[a-zA-Z]+(\{[^{}]*\})*", "", plano)
    dec = [m.group(0) for m in re.finditer(r"(?<![\d.])\d+\.\d+(?![\d.])", plano)
           if not re.fullmatch(r"\d{1,3}\.\d{3}", m.group(0))
           and not re.match(r"(em|pt|cm|mm|in|ex)\b", plano[m.end():m.end() + 3])]
    if dec:
        aviso("punto decimal en prosa", ", ".join(dec[:8]))
    else:
        ok("punto decimal en prosa", "0")

    # Numeros tipeados a mano: el CLAUDE.md pide que todo numero de resultado
    # salga de un CSV. No se puede verificar automaticamente cual es resultado,
    # asi que solo se informa cuantas lecturas en linea hay, como referencia.
    inline = len(re.findall(r"`r [^`]+`", cuerpo))
    ok("lecturas de R en linea en el cuerpo", str(inline))


# --------------------------------------------------------------------------
# Controles sobre el PDF compilado
# --------------------------------------------------------------------------

def controles_pdf():
    print("\n=== PDF compilado ===")
    try:
        import fitz
    except ImportError:
        aviso("PyMuPDF", "no instalado: pip install pymupdf (se saltean 3 controles)")
        return
    if not os.path.exists(PDF):
        mal("PDF", "no existe, hay que knitear primero")
        return

    d = fitz.open(PDF)
    toc = d.get_toc()
    if not toc:
        mal("indice del PDF", "sin marcadores, no se puede ubicar cada ejercicio")
        return
    ejs = [(t, p) for _, t, p in toc if re.match(r"^Ejercicio \d+", sin_tildes(t))]
    apend = [p for _, t, p in toc
             if sin_tildes(t).startswith("Ap") or t.startswith("Referencias")]
    fin = min(apend) - 1 if apend else d.page_count
    print("  (%d paginas; el cuerpo termina en la %d)" % (d.page_count, fin))

    # 1. Hojas por pregunta.
    hojas = []
    for i, (t, p) in enumerate(ejs):
        ult = ejs[i + 1][1] - 1 if i + 1 < len(ejs) else fin
        hojas.append((sin_tildes(t).split()[1].rstrip(":"), p, ult, ult - p + 1))
    resumen = "  ".join("E%s %d" % (n, h) for n, _, _, h in hojas)
    excede, previstos = [], []
    for n, _, _, h in hojas:
        if h <= HOJAS_MAX:
            continue
        esperado = HOJAS_EXCEPCION.get(n)
        if esperado and h == esperado[0]:
            previstos.append((n, h, esperado[1]))
        else:
            excede.append((n, h))
    if excede:
        mal("limite de %d hojas por pregunta" % HOJAS_MAX,
            "excede sin excepcion: "
            + ", ".join("E%s con %d" % (n, h) for n, h in excede))
        print("        " + resumen)
    else:
        ok("limite de %d hojas por pregunta" % HOJAS_MAX, resumen)
    for n, h, motivo in previstos:
        aviso("E%s en %d hojas (excepcion registrada)" % (n, h), motivo)

    # 2. Distancia del ultimo renglon al numero de pagina.
    peor, peor_pg = 99.0, None
    for p in range(ejs[0][1], fin + 1):
        pg = d[p - 1]
        bl = [b for b in pg.get_text("blocks") if b[4].strip()]
        num = [b for b in bl if b[4].strip().isdigit()]
        txt = [b for b in bl if b not in num and b[1] > 1.3 * CM]
        if not num or not txt:
            continue
        cm = (min(b[1] for b in num) - max(b[3] for b in txt)) / CM
        if cm < peor:
            peor, peor_pg = cm, p
    if peor < MARGEN_MIN_CM:
        mal("margen al numero de pagina",
            "pag %d a %.2f cm (minimo %.2f)" % (peor_pg, peor, MARGEN_MIN_CM))
    else:
        ok("margen al numero de pagina",
           "el mas ajustado, pag %d a %.2f cm" % (peor_pg, peor))

    # 3. Titulos colgados al pie. Un titulo cuyo tope esta a menos de
    # RENGLONES_BAJO_TITULO renglones del final del bloque de texto queda
    # separado de su contenido.
    colgados = []
    for p in range(ejs[0][1], fin + 1):
        pg = d[p - 1]
        limite = pg.rect.height - 2.4 * CM
        for b in pg.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    if sp["size"] > 12.5 and sp["origin"][1] > limite:
                        colgados.append((p, sp["text"][:40]))
    if colgados:
        mal("titulos colgados al pie",
            "; ".join("pag %d '%s'" % c for c in colgados[:4]))
    else:
        ok("titulos colgados al pie", "ninguno")


def main():
    print("Verificacion de la entrega")
    print("umbrales: %d hojas por pregunta, %.2f cm al numero de pagina, "
          "%d renglones bajo cada titulo"
          % (HOJAS_MAX, MARGEN_MIN_CM, RENGLONES_BAJO_TITULO))
    controles_rmd()
    controles_pdf()
    print("\n=== resultado ===")
    if fallas:
        print("  %d FALLA(S): %s" % (len(fallas), "; ".join(fallas)))
    if avisos:
        print("  %d aviso(s): %s" % (len(avisos), "; ".join(avisos)))
    if not fallas:
        print("  sin fallas")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
