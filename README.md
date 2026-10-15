# Examen Final · Data Mining Avanzado 2026

Examen final de Data Mining Avanzado (Maestría en Explotación de Datos y Gestión del
Conocimiento, Universidad Austral), desarrollado en R Markdown y compilado a PDF.

## Contenido

- `examen-dma-2026.Rmd`: los ejercicios desarrollados, el apéndice de código y las
  referencias (APA 7). La consigna general y los criterios de evaluación están en un
  comentario al inicio del archivo.
- `codigo/`: scripts Python que producen los resultados que el informe lee.
  - `e1_lstm/lstm_serie.py`: LSTM del Ejercicio 1.VI (TensorFlow).
  - `e1_lstm/figuras.py`: diagramas de CNN y celda LSTM del Ejercicio 1.
  - `e4_ag/ag_rastrigin.py`: AG binario contra AG real con DEAP (Ejercicio 4).
  - `e6_supervivencia/supervivencia_rossi.py`: atajos de regresión contra Kaplan-Meier
    y Cox sobre los datos Rossi (Ejercicio 6).
  - `e10_pipeline/pipeline_opcion1.py`: pipeline comparativo del Ejercicio 10.
  - `verificacion/revisar_entrega.py`: controles de la entrega sobre el `.Rmd` y
    el PDF compilado (ver más abajo).
  - `resultados/`: CSV, JSON y PNG generados por esos scripts.
- `Austral logo.png`: logo de la portada. Tiene que quedar en la misma carpeta que el
  `.Rmd`.
- `CLAUDE.md`: reglas de redacción, código y citas del trabajo.

## Reproducir los resultados

Desde la raíz del repo:

```bash
# Ejercicio 1 (requiere TensorFlow; Python 3.12)
.venv-tf/bin/python codigo/e1_lstm/lstm_serie.py
.venv-tf/bin/python codigo/e1_lstm/figuras.py

# Ejercicios 4 y 6 (deap, lifelines)
.venv/bin/python codigo/e4_ag/ag_rastrigin.py
.venv/bin/python codigo/e6_supervivencia/supervivencia_rossi.py

# Ejercicio 10 (scikit-learn, pandas, matplotlib)
.venv/bin/python codigo/e10_pipeline/pipeline_opcion1.py
```

Los entornos `.venv` y `.venv-tf` no se versionan.

## Compilar

Knit a PDF (`xelatex`). Requiere `rmarkdown`, `knitr` y `kableExtra`:

```bash
Rscript -e 'rmarkdown::render("examen-dma-2026.Rmd")'
```

## Verificar la entrega

Después de knitear, desde la raíz del repo:

```bash
python codigo/verificacion/revisar_entrega.py   # necesita: pip install pymupdf
```

Corre de una vez los controles que antes se medían a mano y devuelve 1 si algo
falla, así que se puede encadenar:

- **Sobre el PDF compilado**, que es donde se ven: hojas por pregunta contra el
  límite de la consigna, distancia del último renglón al número de página
  (mínimo 0,45 cm) y títulos que quedan colgados al pie sin su contenido.
- **Sobre el `.Rmd`**: que haya un bloque `Fuentes del Ejercicio N` por
  ejercicio, que la lista de Referencias no tenga huérfanas ni falte ninguna
  cita (se comprueba en las dos direcciones) y las reglas de redacción del
  `CLAUDE.md`: em dashes, muletillas, primera persona, `PENDIENTE`, el símbolo
  de sección, los rótulos "Cuadro" y "Gráfico", y el punto decimal en prosa.

Las excepciones al límite de hojas viven en `HOJAS_EXCEPCION`, arriba del
script, con el motivo escrito. Se muestran en cada corrida a propósito: una
excepción que no se ve se vuelve un incumplimiento olvidado. Hoy hay una, el
Ejercicio 1 en 5 hojas.

Sin PyMuPDF el script igual corre: avisa y saltea los tres controles de maqueta.

## Reglas de la consigna a tener en cuenta

- Entrega en PDF (plazo original: 5 de octubre de 2026, extendido).
- Máximo **4 hojas por pregunta**.
- Portada con los nombres de todos los integrantes.
- Hay penalizaciones explícitas por texto que parezca generado automáticamente.
