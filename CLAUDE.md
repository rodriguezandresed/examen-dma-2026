# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio: el examen final de Data Mining
Avanzado (Maestría en Explotación de Datos, Universidad Austral). El entregable es
`examen-dma-2026.Rmd`, que se compila a PDF con `xelatex`.

## Qué hay acá

- `examen-dma-2026.Rmd`: los 10 ejercicios con el enunciado en cursiva y un espacio
  para la respuesta. La consigna general y los criterios de corrección están en el
  comentario HTML del inicio del archivo. Leerlo antes de escribir nada.
- `Austral logo.png`: va en la misma carpeta que el `.Rmd`; no moverlo ni renombrarlo.

## Reglas de la consigna que condicionan todo

- **Se debe detallar la fuente de cada respuesta.** Cada ejercicio cierra con su bloque
  `**Fuentes del Ejercicio N.**` y todo se consolida en `# Referencias` (APA 7,
  orden alfabético). La sección de Referencias es obligatoria en este trabajo.
- **Máximo 4 hojas por pregunta.** El código completo va al apéndice, no al cuerpo.
- **Portada con los nombres de todos los integrantes.**
- **Hay penalizaciones explícitas por texto que parezca generado por IA**: -20 % por
  jerga sin explicar, -30 % por falta de razonamiento personal, -50 % por respuesta
  que parezca automática. Los criterios de detección del profesor son: inconsistencias
  entre secciones, terminología avanzada sin comprensión demostrable, código perfecto
  sin comentarios personales, referencias inexistentes y estilo inconsistente.

## Cómo se escribe

Registro académico en español, voz de un estudiante que cursó la materia, no de un
manual.

- **Sin gerundios** ("analizando", "utilizando"). Usar subordinadas o nominalizaciones:
  "del análisis se desprende", "mediante la aplicación de".
- **Sin primera persona.** Construcciones impersonales con "se". El razonamiento tiene
  que ser visible igual: se explicita qué se eligió, por qué, y qué se descartó.
- **Sin guiones largos (em dashes).** Ni `—`, ni `---` ni `--` dentro de la prosa.
  Reemplazar por comas, paréntesis, punto y coma o dos puntos. Antes de entregar,
  grep de `—` y `--` con cero coincidencias.
- **Coma decimal** en el texto ("0,05"), espacio fino antes de `%` ("35,6 %"). En
  tablas `kable`, `format.args = list(decimal.mark = ",")`.
- **Cada término técnico se explica la primera vez que aparece.** "Dimensión VC",
  "kernel", "pseudo-residuo", "censura": si se usan sin definir, cae la penalización
  del 20 %.
- **Un tema por párrafo.** Un bloque de más de ocho líneas con varios hallazgos se
  parte.
- **No calificar el propio resultado** ("el resultado es satisfactorio",
  "contundente"). Dar el número y seguir.
- **Muletillas prohibidas**: "de modo que", "conviene señalar", "cabe consignar", "en
  síntesis", "corresponde señalar", "en primer lugar / en segundo lugar", el párrafo de
  hoja de ruta ("el informe se organiza..."). Auditoría antes de entregar:

  ```bash
  grep -oE "de modo que|Corresponde señalar|[Cc]onviene señalar|Cabe consignar|En síntesis|insumo|satisfactorio|contundente" examen-dma-2026.Rmd | sort | uniq -c
  ```

  El objetivo es cero.
- **Paráfrasis y cita, nunca texto en inglés pegado.** Si la fuente está en inglés,
  se parafrasea en español y se cita.

## Anclaje al material de la materia

La respuesta tiene que usar **la notación y el vocabulario del profesor**, no los del
libro de texto. Antes de escribir un ejercicio, abrir el apunte o las slides de la
clase correspondiente (no alcanza con un índice de temas):

- La cota de riesgo se escribe $R(\alpha) \le R_{emp}(\alpha) + \text{intervalo de
  confianza VC}$, y la regla operativa de capacidad es $d/l < 0{,}05$.
- Gradient Boosting se explica con la analogía del golfista y los pseudo-residuos.
- Random Forest se explica por la correlación entre árboles y la fuerza de cada árbol,
  con $m$ como el parámetro que mueve ambas.
- En SVM, la cota alternativa es $E[P(\text{error})] \le E[\#\text{vectores
  soporte}]/n$.
- En t-SNE, la función de costo es la divergencia de Kullback-Leibler y $\sigma_i$ sale
  de la perplejidad por búsqueda binaria.
- La cantidad de parámetros de una LSTM es $4 \times [(\text{units} \times
  \text{input\_dim}) + \text{units}^2 + \text{units}]$, según el apunte de cátedra.

Si un tema no está en el material de clase (Kernel PCA y selección de kernel en el
Ejercicio 5 son los dos casos), decirlo explícitamente y traer bibliografía externa.

## Código

- La materia es en Python; el informe es R Markdown por el formato institucional. **El
  código Python no se ejecuta desde el `.Rmd`**: se desarrolla y se corre de verdad en
  scripts aparte (`codigo/`), los resultados se exportan a CSV y PNG, y el `.Rmd` los
  lee con `read.csv()` e `include_graphics()`. Las tablas se calculan en vivo desde los
  CSV, nunca se tipean números a mano.
- **Todo número del texto sale de una corrida real.** La inconsistencia entre
  secciones es criterio de detección.
- **El código lleva comentarios propios**, incluidos los que registran qué se probó y
  qué no funcionó. Un resultado que falló (por ejemplo, un DBSCAN que no encuentra
  clusters) se deja en el informe si prueba una limitante; no se esconde.
- Semilla global fija. Preprocesamiento dentro del `Pipeline` para evitar fuga de
  información. En series de tiempo, partición temporal, nunca aleatoria.
- Si algo no se pudo correr (por ejemplo, falta TensorFlow), se dice que **queda
  pendiente**; nunca se afirma que se corrió.

## Citas y referencias (APA 7)

- Toda cita externa tiene que ser real y verificable. Antes de entregar, se chequea
  una por una: "referencias irrelevantes o inexistentes" es criterio de detección.
- No se citan paquetes de R o Python salvo que la consigna lo pida (el Ejercicio 4 pide
  citar el software open source, ahí sí).
- Las fuentes de clase se citan por archivo y clase (`clase2/SVM Universidad
  Austral.pdf`), y el material de cátedra tiene una entrada única en Referencias.

### Citas en el texto

- **Parentética:** `(Vapnik, 1995)`. **Narrativa:** `Vapnik (1995) sostiene que...`.
- **Dos autores:** `&` en la parentética, `(Freund & Schapire, 1997)`; `y` en la
  narrativa, `Freund y Schapire (1997)`.
- **Tres o más:** solo el primero y `et al.` desde la primera cita, `(Schölkopf et al.,
  1998)`. Sin cursiva, punto solo en `al.`.
- **Sin fecha:** `(Autor, s.f.)`. **Institucional:** `(SAS Institute, 1983)`.
- **Paráfrasis:** solo autor y año, sin página. **Cita textual:** entre comillas y con
  página obligatoria, `(Burges, 1998, p. 124)`. Si la obra no tiene páginas, capítulo o
  párrafo. **Nunca el símbolo `§`**, que en APA es solo para material legal.
- Cita textual de 40 palabras o más: bloque aparte con sangría, sin comillas, con la
  página al final.
- **Sin citas redundantes consecutivas:** si una oración termina con `(Breiman, 2001)`,
  la siguiente no empieza con `Breiman (2001)`. Se unifica en una sola oración.
- Nunca se pega texto en inglés: se parafrasea en español y se cita.

### Lista de referencias

Orden alfabético por apellido del primer autor, sangría francesa. Qué va en cursiva
depende del tipo de fuente:

| Tipo | Cursiva | Formato |
|---|---|---|
| Libro | título | `Vapnik, V. N. (1995). *The nature of statistical learning theory*. Springer.` |
| Artículo | revista **y** volumen | `Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.` |
| Capítulo | título del libro | `Apellido, I. (Año). Título del capítulo. En I. Editor (Ed.), *Título del libro* (pp. 1-20). Editorial.` |
| Actas | nombre del congreso | `Davis, L. (1985). Job shop scheduling with genetic algorithms. *Proceedings of the First International Conference on Genetic Algorithms*, 136-140.` |
| Dataset | título, con `[Conjunto de datos]` | `Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository.` |
| Software | título, con `[Software]` o su paper | `Fortin, F.-A., et al. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research, 13*, 2171-2175.` |

- En artículos el rango de páginas va solo; `pp.` es únicamente para capítulos y actas.
- Hasta 20 autores se listan todos. Con 21 o más, los primeros 19, puntos suspensivos
  y el último.
- DOI o URL al final cuando existe.

### Tablas y figuras

- Siempre "Tabla" y "Figura", nunca "Cuadro", "Gráfico" ni "Imagen". El YAML ya
  fuerza "Tabla" con `\addto\captionsspanish`.
- Numeración correlativa por orden de aparición, título descriptivo, y `*Nota.*`
  debajo (la palabra en cursiva, el texto no). "Elaboración propia" cuando la hizo el
  autor; si usa datos externos, "Elaboración propia con datos de (Fuente, año)".
- En LaTeX el caption de `kable` sale como "Tabla 1: Título" en una línea: es la
  convención aceptada en los TP.
- Los símbolos estadísticos latinos van en cursiva (*p*, *N*, *R²*); las letras griegas
  no. Formato de un estadístico: `χ²(4, N = 1838) = 22,4; p < 0,001`. Para *p* muy
  chicos, `p < 0,001`, nunca notación científica.

## R Markdown y compilación

- YAML con `latex_engine: xelatex` y `number_sections: false`. No tocar el
  `header-includes` salvo necesidad concreta.
- Individual: campo `author:` en el YAML y sin página de integrantes. Grupal: página
  de integrantes y sin `author:`. Nunca los dos.
- Tablas con `kableExtra`: `booktabs = TRUE`, `HOLD_position`, `position = "center"`,
  anchos con `column_spec`. Nunca combinar `scale_down` con `column_spec`.
- Figuras y tablas con título descriptivo y `*Nota.*` debajo.
- **No knitear durante el desarrollo.** Se desarrolla, después se compila. Al
  compilar, revisar cada página renderizada como imagen, no solo el conteo de páginas.
- Las notas de trabajo van en comentarios HTML `<!-- -->`, nunca en texto visible.
  Antes de entregar, grep de `PENDIENTE`, `COMPLETAR` y `TODO`.

## Forma de trabajo

- Las instrucciones de desarrollo van a este archivo, no al chat.
- Ediciones mínimas: se cambia lo que se pidió, no se reescribe alrededor.
- "Stop" significa stop: se detiene el trabajo en el punto donde está.
