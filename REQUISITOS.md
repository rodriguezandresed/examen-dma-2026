# Examen Final · Data Mining Avanzado 2026

Requisitos y limitantes extraídos de `Examen Dataming Avanzado 2026.pdf` (4 páginas,
cátedra Volpacchio / López-Imizcoz). Documento de trabajo interno, no se entrega.

---

## 1. Requisitos formales (página 1 del PDF)

| Requisito | Detalle | Estado |
|---|---|---|
| **Fecha límite** | Antes del **5 de octubre de 2026** | Pendiente |
| **Formato de entrega** | Un archivo **PDF** con el desarrollo de las preguntas | Pendiente |
| **Modalidad** | Grupos de **hasta 4 alumnos**; el examen individual es válido | A definir |
| **Portada** | Debe contener **los nombres de los miembros del grupo** | Pendiente |
| **Fuentes** | **Se debe detallar la fuente de cada respuesta** | Pendiente |
| **Extensión** | **No más de 4 hojas por pregunta** | Pendiente |
| **Defensa oral** | El profesor puede preguntar a **cualquier integrante** por una explicación adicional de una pregunta determinada | Preparar |
| **Rigor** | Respuestas rigurosas **tanto en las expresiones como en las descripciones** | Pendiente |
| **Nota** | Se asigna por calidad del trabajo, **normalizada contra el mejor examen** | Informativo |
| **Objetivo declarado** | Repaso general de los algoritmos de la materia para fijar conceptos | Informativo |

**Nota sobre el PDF fuente:** el octavo bullet de la consigna está **truncado en el
original**: dice *"Las explicaciones deben intentar ex"* y se corta ahí. No es un
problema de extracción del texto, falta en el archivo del profesor. Conviene
preguntarle cómo termina esa frase (probablemente "extenderse" o algo sobre explicar
con palabras propias).

---

## 2. Limitante crítica: criterios de detección de uso de LLM

Esta es la restricción que gobierna **cómo** se escribe todo el examen. La consigna
lista criterios explícitos de detección y penalizaciones porcentuales.

### Criterios de detección declarados

1. Inconsistencias entre diferentes secciones del trabajo
2. Uso de terminología muy avanzada sin comprensión demostrable
3. Código perfecto sin comentarios personales
4. Referencias irrelevantes o inexistentes
5. Estilo de escritura inconsistente

### Penalizaciones por estilo artificial

| Penalización | Motivo |
|---|---|
| **-20 %** por respuesta | Jerga técnica usada sin explicar, como si fuera obvia |
| **-30 %** por respuesta | Respuesta que no muestre proceso de razonamiento personal |
| **-50 %** por respuesta | Respuesta que parezca generada automáticamente |

### Qué implica en la práctica

- **Cada término técnico se explica la primera vez que aparece.** "Dimensión VC",
  "kernel", "pseudo-residuo", "censura a la derecha": si se usan sin definir, cae el
  -20 %.
- **El razonamiento tiene que ser visible.** No alcanza con la conclusión correcta,
  hay que mostrar el paso intermedio, la cuenta, el porqué de la elección.
- **El código lleva comentarios propios**, incluidos los que registran qué se probó y
  qué no funcionó. Código impecable y sin una sola nota personal es criterio de
  detección explícito.
- **Toda referencia se verifica.** El Ejercicio 4 pide papers con autor y fecha, y el 5
  pide fuentes sobre Kernel PCA: una cita inventada activa el criterio 4 directamente.
- **Consistencia entre secciones.** Los números del Ejercicio 10 tienen que coincidir
  con lo que el texto afirma, y la notación de un ejercicio no puede contradecir la de
  otro (por ejemplo, usar C para la regularización de SVM en el 2 y para otra cosa en
  el 10).
- **Estilo uniforme.** Si el trabajo es grupal, hay que uniformar la redacción al final,
  porque el cambio de registro entre secciones es criterio de detección.

Complementa esta lista el catálogo de expresiones delatoras de
`claude_helpers/redaccion_academica.md` §6c, que hay que grepear antes de entregar
(objetivo: cero coincidencias).

---

## 2b. Aclaración del profesor sobre el nivel de desarrollo esperado

Gero envió por escrito, después de publicada la consigna, una aclaración sobre qué
espera cuando pide desarrollar un concepto (nombra autoencoder, Seq2Seq y
encoder-decoder, pero el criterio aplica a todas las preguntas conceptuales).
**Esta aclaración manda sobre cualquier interpretación previa de la consigna.**

### Lo que NO espera

- El algoritmo en detalle.
- Código Python del algoritmo.
- Cada paso matemático con lujo de detalles.
- Que se memorice el pseudocódigo.

### Lo que SÍ espera, punto por punto

| # | Qué pide | Cómo se verifica en el texto |
|---|---|---|
| 1 | **Qué es y cuál es su estructura general** | Las partes y el rol de cada una. Su ejemplo: en un autoencoder, encoder, espacio latente y decoder |
| 2 | **Cómo se usa** | Entrada, salida, qué se pretende aprender y cómo se entrena (función de pérdida, reconstrucción) |
| 3 | **En qué casos es útil, y con quién compite en cada caso** | No alcanza con listar usos: hay que nombrar la alternativa que se descarta en cada uno |
| 4 | **Un diagrama o esquema** | Que muestre **el flujo de datos y las partes principales**. Un gráfico de resultados no cumple este punto |
| 5 | **Matemática macro, no detallada** | Su ejemplo: que el encoder mapea $x \to z$ y el decoder $z \to \hat{x}$, y que se minimiza una pérdida de reconstrucción. Sin derivadas ni detalle por capa |

### Extensión

> "La extensión ideal: no más de 4 hojas, pero puede ser menos."

Esto resuelve la ambigüedad que planteaba la sección 5.2: el límite es **por pregunta**
y **menos es mejor**. Una respuesta que entra en 3 hojas y cubre los cinco puntos vale
más que una de 4 llena de evidencia adicional.

### La frase que resume el criterio de correción

> "Lo importante es que se vea que entienden el qué, para qué, cómo y cuándo, no que
> memoricen el pseudocódigo."

**Los cuatro ejes son: qué, para qué, cómo y cuándo.** Al revisar una respuesta
terminada, corresponde preguntarse por los cuatro por separado. La experiencia de los
Ejercicios 7 a 9 fue que el "cuándo" (en qué casos conviene y contra qué alternativa) es
el que más fácil se omite, porque es el único que no sale de leer el apunte.

### Cómo se combina con los criterios de detección de la sección 2

Los dos documentos tiran en la misma dirección salvo en un punto, y conviene tenerlo
presente:

- La penalización del 30 % castiga la respuesta **sin proceso de razonamiento personal**,
  lo que empuja a mostrar decisiones propias.
- La aclaración pide **no agregar aparato** que no responda la pregunta.

No se resuelve con experimentos extra. El razonamiento personal que la aclaración admite
es el de la propia explicación: elegir una alternativa y decir por qué se descartan las
otras, declarar lo que está fuera del programa, admitir una limitación del método,
señalar dónde el material de clase se contradice. Eso cabe dentro de los cinco puntos y
no agrega páginas.

---

## 3. Requisitos por ejercicio

Entregables concretos exigidos por cada consigna. Los verbos son los del PDF.

### Ejercicio 1 (ocho incisos, I a VIII)

| Inciso | Qué pide exactamente | Entregable extra |
|---|---|---|
| **I** | Papel de las **funciones de activación** en redes profundas: por qué son necesarias, cómo afectan la capacidad de representación, ventajas y desventajas | |
| **II** | Proceso de entrenamiento (**forward pass, función de pérdida, backpropagation, optimización**); factores que determinan la calidad del aprendizaje; diferencias esenciales **Deep vs Shallow** | |
| **III** | Concepto de **embedding** y de **autoencoder** en Deep Learning | **Ejemplos de uso** |
| **IV** | **Un tipo de red neuronal que sirva para clustering**, con explicación breve de cómo funciona | Elegir una (SOM de Kohonen es la vista en clase) |
| **V** | Qué es una **convolución** y cómo aplica en redes neuronales | **Diagrama de las distintas capas** de una CNN |
| **VI** | Qué es una **RNN clásica**; en qué se diferencian las **LSTM**; **cantidad de parámetros según la configuración** | **Diagrama de la topología** más **ejemplo completo en Python** de LSTM para predecir una serie de tiempo arbitraria, **probando distintas configuraciones de capas y neuronas** |
| **VII** | Algoritmo **encoder-decoder** y usos habituales | |
| **VIII** | Arquitectura vs **capacidad de representación** y **capacidad de generalización**: definir ambas; cómo las modifican capas, neuronas, activaciones, regularización y tipo de conexión; rol del **sesgo y la varianza**; **ejemplos conceptuales no numéricos** de underfitting y overfitting por decisiones arquitectónicas; cómo un cambio arquitectónico mejora o empeora el desempeño **sin tocar los datos**; por qué más parámetros no siempre es mejor y su relación con el **principio de mínima complejidad** | Cinco sub-puntos con guion en el PDF |

**El inciso VI y el VIII son los más caros.** El VI exige código Python ejecutado y el
VIII tiene seis preguntas encadenadas.

### Ejercicio 2 (SVM)

- Mencionar y **describir los tipos** de SVM.
- Qué **parámetros externos** necesita el algoritmo en su versión **no lineal separable**.
- Qué **característica especial** tiene SVM respecto del resto de los algoritmos de ML.
- Cómo se adapta **SVM de clasificación a regresión** (SVR).

*El PDF escribe "VSM" en lugar de "SVM" en todo el ejercicio; es un error de tipeo del
profesor. En la respuesta corresponde usar SVM.*

### Ejercicio 3 (Statistical Learning Theory)

- De qué trata el campo de **Statistical Learning Theory**.
- Qué **indicador muestra la complejidad de un dataset** usando SVM.
- Qué es la **capacidad** de un algoritmo.
- Qué **manera heurística** propone Vapnik-Chervonenkis para estimarla, **dado un
  dataset de 1000 observaciones con k variables**, y **dejando de lado el caso
  particular de SVM** donde existe una forma matemática de estimarla.

**Limitante:** la última parte es un cálculo concreto sobre n = 1000, no una
descripción general. Hay que llegar a un número o a una regla operativa (la heurística
h/l < 0,05 del apunte de la Clase 1).

### Ejercicio 4 (Algoritmos genéticos)

- **Pseudocódigo** de algoritmos genéticos.
- Principales diferencias entre **AG binarios y reales**.
- **Tres casos de aplicación** con sus respectivas **fuentes (paper, autor, fecha)** y
  breve explicación de cada uno.
- Cómo se implementaría con **programas open-source**, con cita y breve explicación.

**Limitante:** los tres papers deben ser reales y verificables. Es el ejercicio con
mayor exposición al criterio "referencias irrelevantes o inexistentes".

### Ejercicio 5 (PCA y kernels)

- Utilización de **componentes principales** en Machine Learning.
- Implicancia de usar **Kernel en componentes principales** (Kernel PCA), **citando
  fuentes y explicando con palabras propias**.
- Qué significa el **truco del kernel** y ejemplos de los **kernels más utilizados**.
- Si existen **métodos sencillos para seleccionar el kernel apropiado** para un dataset
  determinado, con fuentes y conclusiones.

**Limitante:** dos de los cuatro puntos piden explícitamente fuentes. "Investigue"
aparece dos veces, así que se espera bibliografía externa, no solo el apunte.

### Ejercicio 6 (Análisis de supervivencia)

- Por qué este enfoque es **superior a los métodos de regresión** para estimar días a un
  evento.
- Qué **tipos** existen.
- Qué **implementaciones en Python/R** se conocen.

### Ejercicio 7 (Clustering y t-SNE)

- **Técnicas de preproceso** que existen en clustering para visualizar la potencial
  existencia de grupos; explicar **una** de ellas brevemente.
- Obtenidos los grupos finales, **de qué manera estimar qué diferencias existen en los
  perfiles** de los grupos, y **explicar los resultados**.
  *Ayuda dada por el profesor: usar el último capítulo de la guía de clustering
  (Assessing Clustering Results).*
- **t-SNE**: en qué consiste, para qué se utiliza, **ejemplo de implementación en
  Python**, **limitantes** y **ventajas**.

**Limitante:** el profesor señala la fuente exacta. No usarla es una decisión que hay
que poder justificar oralmente.

### Ejercicio 8 (Boosting)

- Diferencias entre **AdaBoost y Gradient Boosting**.
- **Explicación intuitiva** de cómo trabaja el Gradient Boosting.

*"Intuitiva" es una instrucción de registro: se espera una analogía o una construcción
paso a paso, no la derivación formal.*

### Ejercicio 9 (Random Forest)

- **Describir el algoritmo**.
- Su **diferencia con el resto de los algoritmos de ensamblaje**.
- **Principales usos** del algoritmo.

### Ejercicio 10 (Pipeline en Python)

Se elige **una** de las dos opciones. Hacer las dos es opcional y no suma a la
evaluación: *"a los fines de la evaluación uno es suficiente"*.

**Opción 1 · Modelos clásicos**

- Dos **datasets públicos**: uno de clasificación, uno de regresión.
- Pipeline completo en Python que compare **cuatro modelos**: SVM, Random Forest,
  Gradient Boosting y MLP.
- Debe incluir: (1) preprocesamiento adecuado, (2) **definición explícita del espacio de
  búsqueda de hiperparámetros para cada modelo**, (3) estrategia de optimización
  (GridSearch o RandomSearch), (4) **validación k-fold**.
- **Comparación clara del rendimiento de cada modelo en cada tarea**.

**Opción 2 · Deep Learning**

- Dos **datasets reales obtenidos de Internet**: uno de clasificación, uno de regresión.
- Pipeline en Python con **TensorFlow o PyTorch** que compare **al menos dos
  arquitecturas por tarea**, obligatoriamente **un MLP** y **al menos una arquitectura
  moderna** (CNN, RNN/LSTM o Transformer, según el tipo de dato).
- Debe incluir: (1) preprocesamiento y normalización, (2) **definición explícita de cada
  arquitectura** (capas, activaciones, regularización), (3) búsqueda de hiperparámetros
  esenciales (learning rate, batch size, número de capas/neuronas o bloques) por
  estrategia sistemática (Grid, Random o Bayesian Search), (4) k-fold CV o hold-out
  robusto, (5) **análisis comparativo del MLP frente a las arquitecturas modernas, con
  discusión de por qué el MLP presenta limitaciones estructurales** según el tipo de dato.
- Entregar **el código completo** y una **interpretación breve** de los resultados.

**Recomendación:** la Opción 1 es la de menor riesgo. El material de las Clases 5 y 6
(`DMA_Clase_5_3_de_3` y `DMA - Clase 6 - 3 de 3`) ya trae el pipeline unificado de
varios modelos, la búsqueda sin fuga de información y la comparación por AUC contra
tiempo. La Opción 2 exige entrenar redes y encontrar dos datasets reales que justifiquen
una CNN o una LSTM, lo que multiplica el trabajo sin mejorar la nota.

---

## 4. Mapa de ejercicios al material de clase

Tomado de `datamining_avanzado/INDEX.md`, que ya trae el mapeo completo.

| Ejercicio | Fuente principal en la materia |
|---|---|
| 1.I, 1.II, 1.VIII | `clase3/Data-Mining-Avanzado-01/02/03`, `clase1/SLT Universidad Austral1.pdf` (sesgo-varianza) |
| 1.III | `clase4/autoencoders.pdf`, `clase4/resumen_redes_recurrentes_2024_.pdf` (embeddings) |
| 1.IV | `clase3/Data-Mining-Avanzado-08-Self Organizing Maps-V07.pdf` (SOM de Kohonen) |
| 1.V | `clase3/Clase3/10 - Redes Neuronales Convolucionales.ipynb`, `material_extra_clase3/Convolutional neural networks_word.pdf`, `Understanding Convolutions.pdf` |
| 1.VI | `clase4/Apunte - Dimensiones y Arquitectura LSTM.pdf` (parámetros), `clase4/resumen_redes_recurrentes_2024_.pdf` |
| 1.VII | `clase4/atencion_mecanismo.txt` (Bahdanau 2014, Transformer 2017) |
| 2 | `clase2/SVM Universidad Austral.pdf`, `clase2/Clase2volpacchio.pptx` |
| 3 | `clase1/SLT Universidad Austral1.pdf`, `material_extra_clase3/Statistical Learning Therory - Notas finales.txt` |
| 4 | `clase5/Algoritmos geneticos Universidad Austral.pdf`, `clase5/DMA_Clase_5_1_de_3`, `clase7/DMA - Clase 7 - 1 de 3` (DEAP) |
| 5 | `clase2/SVM Universidad Austral.pdf` (Mercer, kernels), `clase7/Reducción de la complejidad de los datos con clústeres de variables.pdf` |
| 6 | `clase7/DMA - Clase 7 - 1 de 3` (Kaplan-Meier, log-rank, Cox, AFT con `lifelines`), `clase4/Apunte - Taxonomia de Modelos y Algoritmos.pdf` §13 |
| 7 | `clase7/Preparation for Clustering.pdf`, `clase7/Assessing Clustering Results.pdf`, `clase8/TSNE_Perplejidad_Clustering_NoLineal_Volpacchio.pdf`, `clase7/t_sne.pdf` |
| 8 | `clase6/Ensemble Methods Universidad Austral.pdf`, `clase6/Gradient boosting_explicado 1/2/3.pdf` |
| 9 | `clase6/randon forest explicado.pdf`, `clase6/DMA - Clase 6 - 2 de 3` |
| 10 | Opción 1: `clase5/DMA_Clase_5_3_de_3`, `clase6/DMA - Clase 6 - 3 de 3`. Opción 2: `clase3/Clase3/09` y `10` |

**Regla de anclaje:** usar la notación y el vocabulario del profesor, no los del libro de
texto. Ejemplos concretos de esta materia:

- La cota de riesgo se escribe **R(α) ≤ Remp(α) + intervalo de confianza VC**, y la
  heurística de capacidad es **h/l < 0,05** (apunte de la Clase 1).
- En Gradient Boosting el apunte de cátedra usa la analogía del **golfista** (golpe
  inicial F₀ más correcciones) y llama **pseudo-residuos** a lo que ajusta cada modelo
  débil.
- En Random Forest, la tasa de error depende de **la correlación entre árboles y la
  fuerza de cada árbol**, y **M** es el parámetro que baja las dos a la vez.
- En SVM, la cota alternativa de Vapnik vía leave-one-out es
  **E[P(error)] ≤ E[#vectores soporte]/n**.
- En t-SNE, la función de costo es la **divergencia de Kullback-Leibler** y el ancho de
  banda sale de la **perplejidad por búsqueda binaria** de σᵢ.

---

## 5. Limitantes propias del entorno de trabajo

### 5.1 La materia es en Python, el repositorio es en R

Es la limitante estructural del trabajo. `datamining_avanzado` es la única materia del
repositorio en Python (`scikit-learn`, `keras`, `lifelines`, `deap`). El informe se
arma en R Markdown por el formato institucional (portada con logo, índice, encabezados),
pero **el código Python no se ejecuta desde el `.Rmd`**.

Flujo propuesto:

1. El código de los Ejercicios 1.VI, 7 y 10 se desarrolla y **se corre de verdad** en
   notebooks aparte (`codigo/`), en Colab o en local.
2. Los resultados (métricas, tablas, gráficos) se exportan a PNG y CSV.
3. El `.Rmd` incluye el código en bloques de Python **sin evaluar** y las figuras con
   `include_graphics()`.

Esto evita depender de `reticulate` y de que la máquina tenga TensorFlow instalado al
momento de knitear. **Los números del texto tienen que salir de una corrida real**,
porque la inconsistencia entre secciones es criterio de detección de LLM.

### 5.2 El límite de páginas (resuelto por la aclaración del profesor)

La consigna decía "no deberían tener más de 4 hojas por pregunta", lo que admitía dos
lecturas: por ejercicio o por inciso. **La aclaración de la sección 2b lo resuelve**: el
límite es por pregunta y "puede ser menos". La lectura es por ejercicio,
lo que deja el Ejercicio 1 muy ajustado: ocho incisos, dos de ellos con diagrama y uno
con código Python, en cuatro carillas. Estrategia: el código completo del 1.VI y del 10
va a un **apéndice** o a un archivo adjunto, no al cuerpo de la respuesta.

### 5.3 Interacción con la guía de redacción del repositorio

La consigna **obliga a detallar la fuente de cada respuesta**. Esto pisa dos
convenciones por defecto del repositorio:

- La sección **Referencias es obligatoria** acá, aunque
  `redaccion_academica.md` §6b diga que se borra si el texto no cita nada.
- Se citan las fuentes teóricas **y** las herramientas cuando la consigna lo pide (el
  Ejercicio 4 pide explícitamente citar el software open-source), aunque la regla
  general del repositorio sea no citar paquetes.

Todo lo demás de la guía sigue vigente: sin gerundios, sin primera persona, sin em
dashes, coma decimal, APA 7.

### 5.4 Material sin texto extraíble

Varios PDF de la materia son imágenes y hay que abrirlos a mano si se los necesita:

- `clase3/material_extra_clase3/primal a dual vsm.pdf` (derivación primal a dual de SVM,
  relevante para el Ejercicio 2)
- `clase3/material_extra_clase3/Resumen entrenamiento de Redes Neuronales.pdf`
  (relevante para el Ejercicio 1.II)
- `clase4/Extensión LSTM clase.pdf` (relevante para el Ejercicio 1.VI)
- La sección de Random Forest de `clase6/Ensemble Methods Universidad Austral.pdf`
  (relevante para el Ejercicio 9; el reemplazo es `randon forest explicado.pdf`)

Además, los apuntes de clustering de la Clase 7 son capítulos de un curso de **SAS**: la
teoría aplica, pero el código hay que traducirlo a `scikit-learn`.

### 5.5 Temas del examen que no cubre el material de clase

Auditoría del enunciado contra el índice de la materia:

- **Kernel PCA** (Ejercicio 5) no tiene un apunte propio. El material cubre el truco del
  kernel y la condición de Mercer en `SVM Universidad Austral.pdf`, y PCA en los
  capítulos de clustering, pero la combinación de ambos hay que traerla de bibliografía
  externa. La consigna lo anticipa: dice "investigue" y "citar fuentes". La fuente
  canónica es Schölkopf, Smola y Müller (1998).
- **Métodos de selección de kernel** (Ejercicio 5) tampoco figura en el programa. Es el
  otro punto donde la consigna pide investigación externa.
- El resto de los ejercicios está íntegramente cubierto por el material de la materia.

---

## 5b. Estado del desarrollo

Al 2 de septiembre de 2026, los diez ejercicios están redactados en
`rodriguez-tp.Rmd` (unas 2550 líneas). Estado por ejercicio:

| Ejercicio | Estado | Observación |
|---|---|---|
| 1.I a 1.V | Completo | Figura 1 (capas de CNN) generada |
| 1.VI | **Parcial** | Teoría, diagrama (Figura 2), cuenta de parámetros (Tabla 1) y código completos. **Falta correr el entrenamiento**: no hay TensorFlow en el entorno local |
| 1.VII, 1.VIII | Completo | |
| 2 a 9 | Completo | |
| 10 | Completo | Opción 1, con pipeline corrido de verdad y resultados reales |

**Verificaciones numéricas realizadas.** La cuenta de parámetros de la LSTM se validó
contra el ejemplo del apunte de cátedra (10 features, 20 unidades, 2480 parámetros, con
`assert` en el código). El máximo de $f(x) = x \cdot \mathrm{sen}(10\pi x) + 1$ se verificó
por barrido numérico y coincide con el valor de las slides ($x \approx 1{,}85$,
$f \approx 2{,}85$), igual que el cálculo de los 22 bits. La regla $d/l < 0{,}05$ se aplicó
a $l = 1000$ y da $h \le 50$, es decir $k \le 49$ para una máquina lineal.

**Scripts en `codigo/`.** `ej10_pipeline.py` (corrido, 4 modelos por tarea),
`ej07_tsne.py` (corrido), `ej01_lstm.py` (cuenta de parámetros corrida, entrenamiento
pendiente) y `diagramas.py` (corrido). Las salidas quedaron en `salida_ej*.txt` y en los
CSV que el `.Rmd` lee en vivo.

**Auditoría de estilo.** Cero coincidencias en el grep de expresiones delatoras de
`redaccion_academica.md` §6c, cero em dashes, cero primera persona. Los únicos gerundios
que quedan están dentro de los enunciados citados de la consigna.

---

## 6. Decisiones pendientes

1. **Individual o grupal.** Cambia la portada. El andamiaje quedó armado como
   **individual** (campo `author` en el YAML, sin página de integrantes). Si es grupal,
   se restaura la página de integrantes y se saca el `author`.
2. **Instalar TensorFlow o correr en Colab** para completar el entrenamiento de la LSTM
   del punto 1.VI. Es lo único que falta para cerrar el desarrollo.
3. **Consultar al profesor** por el bullet truncado de la consigna (sección 1).
4. **Verificar una por una las citas externas** antes de entregar. Son 30 entradas en
   Referencias y el criterio 4 de detección penaliza las inexistentes.

Resueltas durante el desarrollo: se eligió la **Opción 1** del Ejercicio 10 (Breast Cancer
Wisconsin para clasificación y California Housing submuestreado a 5000 filas para
regresión) y una **serie sintética** para el punto 1.VI, con estacionalidad de período 50,
tendencia lineal y ruido gaussiano, porque conocer la estructura verdadera permite
distinguir si la red aprendió la estacionalidad o solo copia el último valor.

---

## 7. Checklist antes de entregar

Combina la consigna con `claude_helpers/rmarkdown_pdf_workflow.md` §10.

- [ ] Portada con los nombres de todos los integrantes
- [ ] **Cada respuesta cubre los cinco puntos de la sección 2b**: qué es y estructura,
      cómo se usa, en qué casos sirve y contra quién compite, un esquema del flujo de
      datos, y matemática macro
- [ ] **Cada respuesta contesta los cuatro ejes**: qué, para qué, cómo y cuándo
- [ ] **El diagrama de cada respuesta es un esquema del flujo**, no un gráfico de
      resultados
- [ ] Ninguna respuesta reproduce pseudocódigo ni desarrollo matemático paso a paso
- [ ] **Cada respuesta tiene su fuente detallada**
- [ ] Ningún ejercicio supera las 4 hojas
- [ ] Todas las citas verificadas como existentes y pertinentes
- [ ] Cada término técnico explicado en su primera aparición
- [ ] El código tiene comentarios propios, incluidos los de lo que se probó y se descartó
- [ ] Los números del texto coinciden con la salida real del código
- [ ] Estilo de redacción uniforme en todo el documento
- [ ] Diagramas pedidos: CNN (1.V) y topología LSTM (1.VI)
- [ ] Código Python entregado: LSTM (1.VI), t-SNE (7) y pipeline completo (10)
- [ ] Grep de expresiones delatoras de `redaccion_academica.md` §6c con cero coincidencias
- [ ] Sin em dashes, sin gerundios, sin primera persona
- [ ] Referencias en APA 7, orden alfabético
- [ ] Knit sin errores y revisión visual de cada página renderizada
- [ ] Cada integrante puede defender oralmente cualquier ejercicio
