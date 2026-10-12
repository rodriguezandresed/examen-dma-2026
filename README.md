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
  - `e10_pipeline/pipeline_opcion1.py`: pipeline comparativo del Ejercicio 10.
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

# Ejercicio 10 (scikit-learn, pandas, matplotlib)
.venv/bin/python codigo/e10_pipeline/pipeline_opcion1.py
```

Los entornos `.venv` y `.venv-tf` no se versionan.

## Compilar

Knit a PDF (`xelatex`). Requiere `rmarkdown`, `knitr` y `kableExtra`:

```bash
Rscript -e 'rmarkdown::render("examen-dma-2026.Rmd")'
```

## Reglas de la consigna a tener en cuenta

- Entrega en PDF (plazo original: 5 de octubre de 2026, extendido).
- Máximo **4 hojas por pregunta**.
- Portada con los nombres de todos los integrantes.
- Hay penalizaciones explícitas por texto que parezca generado automáticamente.
