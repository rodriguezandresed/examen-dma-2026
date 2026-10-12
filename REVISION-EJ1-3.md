# Revisión adversarial de los Ejercicios 1 a 3

No es un chequeo de cobertura: las 16 sub-preguntas están contestadas. Acá la
pregunta es si están **bien** contestadas. Nada está corregido.

Cada punto se verificó contra el material de cátedra en
`data-science-master/datamining_avanzado/clase*/`. Donde no pude verificar,
lo digo.

Archivo: `examen-dma-2026.Rmd`, commit `0dc9e20`.
Ej 1 = págs. 3-6, Ej 2 = págs. 7-9, Ej 3 = págs. 10-11.

---

## Resumen por punto

| Punto | Veredicto |
|---|---|
| 1.I | Correcto pero contesta de costado: Cybenko es un resultado de **una** capa y el punto pregunta por redes **profundas** |
| 1.II | "Analice conceptualmente" contestado con una lista de cuatro sustantivos |
| 1.III | Definiciones bien, "ejemplos de uso" son tres sintagmas. Niega material de cátedra que existe |
| 1.IV | **Bien.** El mejor del ejercicio. Le falta solo el último paso |
| 1.V | La convolución, muy bien. Las capas del diagrama no se explican en la prosa |
| 1.VI | Cuentas impecables, **lectura del experimento equivocada** |
| 1.VII | **Mal.** Esbozo de 4 líneas que saltea el núcleo del tema, que además está en la cátedra |
| 1.VIII | **Error conceptual** en el doble descenso, y desaprovecha la respuesta que estaba en el material |
| Fuentes Ej 1 | **No atribuye nada.** Doce apellidos en fila |
| Ej 2 | **Sólido.** Verificado contra el deck, textual. Una frase falsa |
| Ej 3 | **El mejor del trabajo.** Un matiz menor |

---

## A. Errores

### A1. El doble descenso está usado al revés — Punto VIII, pág. 6

> Belkin et al. (2019) describen el doble descenso, una baja posterior del
> riesgo de test **si la regularización o el descenso estocástico (SGD) acotan
> la complejidad efectiva**.

Dos problemas distintos.

**El condicional es falso.** El doble descenso no está condicionado a que algo
acote la complejidad efectiva. Es que el riesgo de test sube hasta un pico en
el umbral de interpolación y después **baja al seguir agrandando el modelo**.
La frase convierte un fenómeno sobre el tamaño del modelo en una propiedad del
regularizador.

**Está citado para probar lo contrario de lo que dice.** El párrafo quiere
fundamentar el sub-pedido "por qué modelos con más parámetros no siempre son
mejores". El doble descenso dice que pasado el umbral **más parámetros sí
suelen ser mejores**. Es el contraejemplo del punto que se defiende,
presentado como si lo apoyara.

### A2. La respuesta correcta a ese sub-pedido estaba en el deck y no se usó

Esto es lo que vuelve grave a A1. `clase1/SLT Universidad Austral1.pdf` trae
exactamente la respuesta, en la diapositiva de dimensión VC:

> "Intuitivamente podríamos pensar que el número de parametros de un algoritmo
> es directamente proporcional a la capacidad….**pero esto no es verdad** (Ver
> contraejemplo en Vapnik, 1995)"
>
> "No necesariamente alta capacidad significa Pobre perfomance: Ejemplo: k
> Nearest Neighbour Classifier con K=1, Infinita Dimensión y Cero empirical
> risk. **En este caso la cota CI no provee inforamción.**"

Eso contesta "por qué más parámetros no siempre son mejores" de frente, con
contraejemplo, y lo escribió el profesor. En lugar de eso el punto va a buscar
a Belkin y lo invierte.

De paso, esa misma diapositiva contradice la primera oración del punto
(*"la capacidad de representación [...] crece con las capas, las neuronas"*):
el deck dice que la capacidad **no** es proporcional a la cantidad de
parámetros.

### A3. "Ningún otro algoritmo deja una cota así" — Ej 2, pág. 8

> Ningún otro algoritmo de los vistos en la materia deja una cota así, leída
> desde la propia solución.

Lo contradice el Ejercicio 9 **del mismo trabajo** (`.Rmd:964`):

> El resultado teórico central es que la tasa de error del bosque queda acotada
> por $PE^* \le \bar\rho(1-s^2)/s^2$ [...] sirve para leer de dónde viene el
> error.

Una cota de generalización leída de las propiedades del propio bosque. Es
literalmente "una cota así". Inconsistencia entre secciones, criterio 1 de
detección de LLM.

Es tan mía como de Marcos: escribí el 9 sin cruzarlo contra el 2. Se arregla
acotando la frase, o mejor, reconociendo el paralelo con Random Forest.

### A4. Se declara ganador un resultado que es ruido — Punto VI, págs. 5-6

La nota de la Tabla 2 concluye: *"El menor MAE de test es lstm_32 (0,0925)"*.
El segundo es 0,0932. Diferencia de 0,0007, con **una sola semilla**
(`SEED = 42`, una corrida por configuración, verificado en
`codigo/e1_lstm/lstm_serie.py:21`).

Y hay un dato que lo vuelve indefendible. La serie es

```python
np.sin(0.05*t) + 0.3*np.sin(0.12*t) + 0.1*rng.normal(size=n)
```

Ruido gaussiano de desvío 0,1, sin ningún escalado en el script (verificado:
no hay scaler, `model.evaluate` devuelve MSE y MAE en unidades crudas).
Entonces el error irreducible es $\sigma^2 = 0{,}0100$ para el MSE y
$\sigma\sqrt{2/\pi} = 0{,}0798$ para el MAE:

| | MSE | sobre el piso | MAE | sobre el piso |
|---|---|---|---|---|
| piso teórico | 0,0100 | — | 0,0798 | — |
| lstm_32 | 0,0134 | +34 % | 0,0925 | +16 % |
| lstm_64 | 0,0136 | +36 % | 0,0932 | +17 % |
| lstm_64_32 | 0,0145 | +45 % | 0,0951 | +19 % |
| lstm_32_16 | 0,0156 | +56 % | 0,0972 | +22 % |

Las cuatro configuraciones están contra el techo del problema. **Esa es la
lectura que corresponde y no está escrita: ninguna arquitectura puede
distinguirse de las otras porque todas ya alcanzaron el error irreducible de
la serie.** Es un resultado mejor que el que se reporta y es justo el tipo de
razonamiento que la consigna premia, porque demuestra que se entendió qué se
midió. Como está, se lee como sobre-interpretación de ruido con n = 1.

**Arrastre al Punto VIII:** el sub-pedido "fundamente cómo un cambio
arquitectónico puede empeorar el desempeño sin tocar los datos" se apoya en
*"una segunda LSTM no bajó el MAE con los mismos datos"*. Con una semilla y
todo contra el piso de ruido, ese resultado no sostiene la afirmación.

### A5. El PDF nunca dice qué serie es — Punto VI, pág. 5

Se informan semilla, ventana, épocas, corte y `shuffle=False`, pero no la
serie. Sin saber amplitud y ruido, un MSE de 0,0134 no significa nada para el
lector. Una línea ("dos senoides de distinta frecuencia más ruido gaussiano de
desvío 0,1") arregla A5 y habilita A4.

### A6. "La regla $d/l < 0{,}05$ es la mínima complejidad" — Punto VIII

No son lo mismo. El deck titula esa diapositiva *"Minimizar la cota por
minimizar d"*: es un umbral operativo para que la cota VC no sea vacía. La
mínima complejidad (riesgo estructural) es el criterio de elegir la familia
que minimiza riesgo empírico más intervalo de confianza. Decir que una **es**
la otra es impreciso.

Menor, pero del mismo tipo: el Punto VIII usa la notación $d/l$ y el Ejercicio
3 declara explícitamente que unifica en $h/n$ (*"En las diapositivas esa
dimensión aparece como h dentro de la cota y como d en la regla d/l; acá se
unifica en h"*). El Ejercicio 1 no sigue la unificación que el 3 anuncia.

---

## B. Respuestas por debajo de lo que se pregunta

### B1. Punto VII, encoder-decoder — y la cátedra tiene una nota dedicada

Cuatro líneas para "Describa el algoritmo encoder-decoder. Usos habituales". No
describe un algoritmo: da una topología en una oración.

Lo que agrava esto: existe `clase4/atencion_mecanismo.txt`, nota de cátedra
titulada **"MODELO ENCODER-DECODER CON ATENCION Y TRANSFORMERS"**, que arranca
justo donde el punto se queda corto:

> "Fue popularizado por el paper *Neural Machine Translation by Jointly
> Learning to Align and Translate* de Bahdanau, Cho y Bengio. Objetivo:
> **Resolver el 'cuello de botella' de los modelos anteriores, que comprimian
> toda la frase en un solo vector** de longitud [fija]."

Ese cuello de botella del vector de contexto de tamaño fijo, y la atención que
lo resuelve, son el núcleo del tema. No aparecen. Tampoco: teacher forcing,
cómo termina el decodificado, greedy contra beam search, ni por qué hace falta
la arquitectura (entrada y salida de largos distintos, que una RNN sola no
resuelve).

Además cita a Sutskever et al. (2014) sin mencionar el resultado por el que se
lo cita (invertir la secuencia de entrada mejora el BLEU). "voz" es una palabra
suelta: no se sabe si es reconocimiento o síntesis. Y listar el autoencoder
como *uso* del encoder-decoder confunde dos cosas: el autoencoder es un
encoder-decoder cuyo objetivo es su propia entrada, no una aplicación de
seq2seq.

Hay lugar de sobra: el Ejercicio 1 usa 4 hojas y el límite es por pregunta.

### B2. El bloque Fuentes del Ejercicio 1 no atribuye nada — pág. 6

> Volpacchio (s.f.-e); Ng (s.f.); Hochreiter y Schmidhuber (1997); Mikolov et
> al. (2013); Hinton y Salakhutdinov (2006); Sutskever et al. (2014); Cybenko
> (1989); LeCun et al. (1998); Kohonen (1990); Goodfellow et al. (2016); Belkin
> et al. (2019); Vapnik (1995); codigo/e1_lstm/lstm_serie.py.

Doce apellidos en fila sin decir qué sostiene cada uno. La consigna pide
detallar la fuente de **cada respuesta**; el Ejercicio 1 tiene ocho respuestas
y este bloque no atribuye ninguna. Es una bibliografía, no una atribución.

En el mismo documento el Ejercicio 2 lo hace bien (*"Burges (1998) para la
formulación del clasificador; Cortes y Vapnik (1995) para el margen blando;
Vapnik (1995) para la pérdida con zona de ε-insensibilidad"*) y el 3 también.
Dos convenciones distintas en un mismo trabajo: criterio 5, estilo de escritura
inconsistente.

### B3. "El apunte" no identifica ningún documento

El Ejercicio 1 dice "el apunte" una y otra vez, y Referencias tiene una sola
entrada de cátedra para todo: *Volpacchio (s.f.-e). Data Mining Avanzado: redes
neuronales*. Pero lo que se cita vive en documentos distintos, y lo verifiqué
uno por uno:

| Afirmación del trabajo | Dónde está en realidad |
|---|---|
| "El apunte cubre escalón" | `clase3/...03-Intrducción a las Redes Neuronales` (única que tiene) |
| "...sigmoidal, tanh" | `clase3/...01-Repaso-de-Matematica` y docs de clase 4 |
| "...y softmax" | **solo** `clase4/Apunte - Taxonomia de Modelos y Algoritmos` |
| "El apunte frena el sobreajuste con early stopping y weight decay" | **solo** `clase4/Apunte - Taxonomia de Modelos y Algoritmos` |
| "la red de Kohonen del apunte" | `clase3/...08-Self Organizing Maps` |
| "Según el apunte, una capa LSTM tiene 4×[...]" | `clase4/Apunte - Dimensiones y Arquitectura LSTM` |

Seis documentos distintos bajo un mismo "el apunte" y una sola entrada
bibliográfica. El lector no puede verificar nada.

### B4. "Analice conceptualmente" contestado con una enumeración — Punto II

> La calidad depende de los datos frente a la cantidad de pesos, de $\eta$, de
> los mínimos locales y del corte.

Cuatro sustantivos, cero mecanismo. No aparecen inicialización, normalización
de entradas, tamaño de lote ni programación de la tasa, y sobre todo no aparece
el desvanecimiento del gradiente, que es el factor que gobierna la calidad
justamente en las redes profundas. Está nombrado en el pie de la Tabla 1 y no
se usa acá.

### B5. Deep vs shallow resuelto con el eslogan — Punto II

> la profunda (deep), que apila capas y aprende jerarquías, con más datos y más
> riesgo de desvanecimiento.

"Aprende jerarquías" es la frase hecha. La diferencia esencial es de eficiencia
representacional: hay funciones que una red profunda representa con una
cantidad de unidades polinómica y una superficial necesita exponencialmente
más. Sin eso no se explica por qué sirve la profundidad.

### B6. Cybenko contesta la pregunta de al lado — Punto I

El punto pregunta por activaciones **en redes profundas** y se apoya en Cybenko
(1989), que es un teorema sobre **una sola capa oculta**. La cita no está mal,
pero sin aclaración queda al revés: la aproximación universal ya vale con una
capa, así que no dice nada sobre profundidad. Lo interesante, ausente, es
precisamente que la universalidad **no** distingue deep de shallow, y que lo
que las separa es el costo en unidades (enlaza con B5).

La Tabla 1 está bien y verificada (el rango −1/1 del escalón coincide con el
deck, que define $f(x)=1$ si $x>0$ y $-1$ si $x\le 0$), pero resume en dos o
tres palabras por celda. "Escalón / Derivada nula" dice poco: derivada nula
significa que no se puede entrenar por gradiente, que es la razón entera por la
que el escalón quedó afuera.

### B7. El Punto V no explica el diagrama — pág. 4

La consigna pide un diagrama de las capas **y explique**. La Figura 1 está, y la
prosa le dedica una subordinada: *"La Figura 1 sigue ese orden, con pooling por
el máximo de la ventana"*. No se explica para qué está el pooling, qué hace el
aplanado, ni cuál es el rol de la cabeza densa.

Y el material estaba: `clase3/material_extra_clase3/Convolutional neural
networks_word.pdf` cubre *"Pooling layers"*, *"pooling layer can be used to
compress spatial information of our feature map"*, max pooling y el flatten. La
frase *"El esquema no está en los PDF de MLP o de SOM"* es técnicamente cierta
por cómo está acotada, pero deja afuera el documento de la materia que sí lo
trata.

La parte de convolución en sí (filtro, stride, padding, la fórmula
$\lfloor (n+2p-f)/s \rfloor + 1$, pesos compartidos, invariancia) está muy
bien.

### B8. El SOM explica cómo se entrena, no cómo da clusters — Punto IV

La pregunta es por una red "que sirve para obtener clustering". La respuesta
desarrolla bien la grilla, la neurona ganadora, la regla de actualización y el
decaimiento del entorno, y termina en *"Entradas parecidas caen en neuronas
próximas"*. Falta el último paso, que es el que contesta: el mapa entrenado
todavía hay que segmentarlo para leer los grupos (matriz U, o un clustering
sobre los pesos de las neuronas). Sin eso el SOM es una proyección topológica,
no una partición.

Igual es el mejor punto del Ejercicio 1, y el párrafo que justifica por qué
descarta k-means, DBSCAN y competitive learning es exactamente el razonamiento
propio que la consigna premia.

### B9. El Punto III se queda en la definición

"De ejemplos de uso" se contesta con *"reducción de dimensión no lineal,
anomalías y eliminación de ruido"*: tres sintagmas, ningún ejemplo. No se dice
por qué un embedding denso le gana a un one-hot (el one-hot no tiene noción de
similitud y crece con el vocabulario), ni se mencionan los tipos de
autoencoder, que `clase4/autoencoders.pdf` enumera en una sección entera.

---

## C. Tres afirmaciones falsas sobre el material de cátedra

| Dónde | Dice | Existe |
|---|---|---|
| `.Rmd:245`, Punto VI | "La LSTM **no está en los PDF**" | `clase4/Apunte - Dimensiones y Arquitectura LSTM.pdf`, `clase4/Extensión LSTM clase.pdf`, `clase4/resumen_redes_recurrentes_2024_.pdf` |
| `.Rmd:175`, Punto III | "Embedding y autoencoder **no aparecen con este detalle en los PDF de redes**" | `clase4/autoencoders.pdf` (deck entero) |
| `.Rmd:299`, Punto VII | "Dos redes acopladas, **ausentes con este detalle en los PDF de la materia**" | `clase4/atencion_mecanismo.txt`, `clase4/resumen_redes_recurrentes_2024_.pdf` |

**La del LSTM se contradice sola.** Once líneas más abajo (`.Rmd:256`) el
trabajo escribe *"**Según el apunte**, una capa LSTM tiene
$4\times[(units \times input\_dim)+units^2+units]$ parámetros"*, y ese apunte
es el que se acaba de declarar inexistente. Trae la cuenta textual:

```
(20 x 10) + (20 x 20) + 20 = 200 + 400 + 20 = 620
4 x 620 = 2480 parametros
```

y hasta el agregado de la densa, `(20 x 1) + 1 = 21`.

**Un solo documento desmiente dos de las tres.**
`clase4/Apunte - Taxonomia de Modelos y Algoritmos.pdf` contiene, conteo
textual: autoencoder ×8, atención ×20, softmax ×6, sigmoid ×7, weight decay ×3,
early stopping ×2, LSTM ×4, encoder-decoder ×2, dropout ×2, backpropagation ×2.
El trabajo se apoya en ese documento para el softmax del Punto I y para el
early stopping del Punto II, y declara ausente lo que ese mismo documento
cubre.

**La del encoder-decoder es la más incómoda:** los tres usos que lista el
trabajo (traducción, resumen, captioning con CNN más recurrente) salen casi
palabra por palabra de la sección "Arquitectura Encoder Decoder" de
`resumen_redes_recurrentes_2024_.pdf`, que además trae un caso resuelto seq2seq
traduciendo "Let's go" a "Vamos".

---

## D. Lo que verifiqué y está bien

**Ejercicio 3 es el más sólido del trabajo.** Todo verificado contra
`clase1/SLT Universidad Austral1.pdf`, textual:

- *"La relación d/l debería ser menor al 0.05 para garantizar que el riesgo
  actual es cercano al riesgo émpírico"*.
- *"30 veces la cantidad de parámetros en datos con ruido y 5 veces datos sin
  ruido"*.
- *"la dimensión VC de lineas rectas orientadas en R2 es 3"* y *"La dimensión
  VC de hiperplanos orientados en RN es N+1"*.
- El clasificador que memoriza, como ejemplo de inconsistencia del ERM.
- La aplicación numérica da: $d/l<0{,}05$ con $l=1000 \Rightarrow d<50$; con
  $d=k+1$, hasta $k=48$.
- Declara que unifica la notación $d$ de la regla con la $h$ de la cota. Casi
  nadie escribe eso.

Único matiz: *"allí el margen acota la dimensión VC"* es cierto en la
literatura pero **no está en el material de cátedra** y no se cita a nada
puntual. Queda bajo el "Vapnik (1995)" genérico del bloque de fuentes.

**Ejercicio 2 está bien construido.** Verificado contra
`clase2/SVM Universidad Austral.pdf`:

- La cota leave-one-out, textual: *"Vapnik da una Cota alternativa al riesgo
  actual en VSM por medio de Boostrap (Leave-One-Out) E[P(error)] ≤ E[Numeros
  de V Soporte] / Número de Obsrv."*
- Mercer y la matriz de Gram semidefinida positiva, textual.
- Los parámetros $\kappa$ y $\theta$ del kernel sigmoide y que no satisface
  Mercer para toda combinación, textual.
- Los tres pasos de SVR, textual: riesgo empírico con ε-insensibilidad,
  capacidad con $\tfrac12\lVert w\rVert^2$, y el funcional combinado.
- La función de aproximación $F(x)=\sum_i \alpha_i K(x_i,x)+b$ es la del deck
  tal cual, con $-c \le \alpha_i \le c$. **Verifiqué que la forma con un solo
  $\alpha$ es la del deck y no un error de transcripción.**
- Dejar afuera ν-SVM y One-Class está bien justificado: **confirmé que no
  figuran en el deck**. Lo mismo con SVM multiclase, que tampoco está, así que
  omitirlo es coherente con el criterio declarado.

**Las cuentas del Punto VI dan las cuatro.** Con
$4[(u\,d_{in})+u^2+u]$: 4352, 16896, 7488 (4352+3136), 29312 (16896+12416); y
la columna Keras es la fórmula más la densa final: 4385, 16961, 7505, 29345.

**La nota de la prueba fallida en NumPy** (*"sin BPTT, aplanó el pronóstico y
no se reporta"*) es exactamente el tipo de comentario que la consigna busca.

**Tabla 2 y las fórmulas de la pág. 9 no están rotas.** `pdftotext` las
descuadra; las páginas renderizadas están perfectas. No perder tiempo ahí.

---

## E. Correcciones a mi primera versión de este archivo

Para que quede asentado, tres cosas que dije antes y verifiqué que estaban mal:

1. Puse que faltaba SVM multiclase como omisión del Ejercicio 2. **No
   corresponde:** el deck no lo trata y Marcos declaró que se limita a la
   clase.
2. Sospeché que la función de regresión de SVR debía ser
   $\sum(\alpha_i-\alpha_i^*)K$. **Falso:** el deck usa un solo $\alpha$ con
   $-c\le\alpha_i\le c$, y el trabajo lo reproduce bien.
3. Cité la regla como $h/n$ en el Punto VIII. **Es $d/l$** en el original; el
   hallazgo real es la inconsistencia de notación con el Ejercicio 3.

**No pude verificar:**
`clase3/material_extra_clase3/Resumen entrenamiento de Redes Neuronales.pdf` no
tiene texto extraíble (es imagen), así que no sé si cubre los factores de B4.
Tampoco sé si el material de clase 4 estaba publicado cuando Marcos escribió el
Ejercicio 1.

---

## F. Prioridad

1. **A4 y A5**, la lectura del experimento LSTM. Hay un resultado mejor
   escondido detrás del que se reporta, y arrastra al Punto VIII.
2. **A1 y A2**, el doble descenso. Error conceptual, con la respuesta correcta
   disponible en el deck.
3. **C**, las tres frases de "no está en los PDF". El profesor escribió ese
   material.
4. **B1**, el Punto VII. Cuatro líneas para un tema con nota de cátedra propia.
5. **B2 y B3**, atribución: el bloque de Fuentes y el "apunte" sin identificar.
6. **A3**, acotar la frase del Ejercicio 2 o cruzarla con Random Forest.
7. **A6** y el resto de B, por orden de aparición.

---

# G. La misma auditoría aplicada a MIS ejercicios (7, 8, 9)

Andrés señaló, con razón, que le apliqué a los Ejercicios 1-3 un estándar que
no me había aplicado. Mismo método: cada afirmación contra el material.

## G1. Dos apelaciones a "la bibliografía posterior" sin una sola cita — FALTA REAL

`.Rmd:986`, Ejercicio 9:

> El documento va más lejos y llama insesgada a esa estimación, y ahí
> corresponde frenar: una serie de pruebas empíricas no demuestra insesgadez, y
> **la bibliografía posterior mostró** que el error OOB tiene sesgo, en general
> pesimista, que empeora con pocos árboles o con clases muy desbalanceadas.

`.Rmd:999`, Ejercicio 9:

> El documento considera las dos medidas muy consistentes entre sí, aunque **la
> bibliografía posterior le encontró** a la de Gini un sesgo a favor de
> variables con muchos niveles.

Las dos afirmaciones son ciertas en la literatura. **Las dos están sin citar.**
Mi bloque de Fuentes del Ejercicio 9 lista Breiman (1996), Breiman (2001), Ho
(1995) y Geurts et al. (2006): ninguno sostiene ninguna de las dos.

Esto es peor que varias cosas que le marqué a Marcos, por dos motivos. Primero,
"la bibliografía posterior mostró" es una autoridad infalsificable: el lector no
puede verificar nada. Segundo, la uso **para contradecir al documento de la
cátedra**, dos veces. Contradecir al profesor sobre la base de literatura que no
nombro es exactamente "referencias irrelevantes o inexistentes", criterio 4.

**No lo corrijo todavía** porque las citas que tengo en la cabeza (Strobl et al.
para el sesgo de Gini, y el trabajo sobre el sesgo pesimista del OOB) hay que
verificarlas contra las fuentes primarias antes de ponerlas. Citar de memoria
sería cometer el mismo error que estoy señalando.

## G2. Todos mis números del Ejercicio 7 son de una sola corrida

El PDF reporta siluetas, ARI, OOB e importancias con una semilla y **no dice en
ningún lado que sean estables**. Es el mismo hueco metodológico que le marqué en
A4. Lo verifiqué ahora, después del hecho:

| Afirmación | Semillas probadas | Resultado |
|---|---|---|
| Orden de importancia PC1, PC3, PC2, PC4, PC5 | 8 | **8/8 idéntico**; desvíos 0,001 contra brechas de 0,015 |
| OOB 0,915 | 8 | media 0,9152, rango [0,9121; 0,9210] |
| Siluetas 0,396 / 0,486 / 0,479 | 4 de t-SNE | desvío 0,001; brecha p30−p50 de +0,007 en las 4 |
| ARI del embedding 0,845 | 4 de t-SNE | media 0,843, rango [0,842; 0,845] |

**Aguantan todas**, con margen amplio. La diferencia con el caso de la LSTM es
que las mías sobreviven y la suya no. Pero el documento no se gana ese derecho:
lo sé ahora, no lo sabía cuando lo escribí, y el lector no puede saberlo.

## G3. Reporté el extremo del rango, no el valor típico

El ARI del embedding figura como 0,845, que es el máximo del rango [0,842;
0,845]. Es el valor de la corrida única, no una elección, pero queda del lado
favorable. Trivial; lo anoto por consistencia.

## G4. Un párrafo entero defendiéndome de un cargo que la fuente no hace

En el Ejercicio 7 escribí una "precisión de vocabulario" aclarando que agrupo
sobre 30 componentes y no sobre los 64 píxeles, cuando el apunte dice "sobre el
espacio original". Verificado el apunte de la clase 8, dice textual:

> "2. Modelado: Ejecutar DBSCAN, HDBSCAN o clustering espectral sobre el espacio
> de caracteristicas original (**o sobre una reduccion PCA previa si la
> dimensionalidad es extrema**)."

O sea que el apunte lo autoriza explícitamente. El párrafo no está mal, pero
gasta líneas en disculparse por algo permitido, en un ejercicio que está justo
en el límite de 4 hojas.

## G5. Lo que verifiqué de lo mío y está bien

- **No hay ninguna afirmación tipo "no está en los PDF"** en mis tres
  ejercicios: siempre nombro el documento.
- **Mis bloques de Fuentes sí mapean fuente a afirmación**, en los tres, al
  estilo de los Ejercicios 2 y 3.
- **Los nombres de archivo que cito existen**, incluido `randon forest
  explicado.pdf` con el error de tipeo de la cátedra, y `Gradient
  boosting_explicado 1 a 3`.
- **Los números del Ejercicio 8 están en el deck**: 0,30 y 0,42; 0,21 y 0,65;
  0,92. Y mi recálculo (0,4236; 0,6625; ε=0,2142; ε=0,137) da.
- **Las citas del Ejercicio 9 son textuales**: *"Esto ha demostrado ser
  insesgado en muchas pruebas empíricas"* y *"Los RF no se adaptan demasiado.
  Puede ejecutar tantos árboles como desee"*.
- **Las del Ejercicio 7 también**: *"no es un algoritmo de clustering"* aparece
  al inicio del apunte como digo; el pipeline de tres pasos es textual; la
  limitante de DBSCAN figura como *"degradacion de rendimiento en alta
  dimensionalidad debido a la maldicion de la dimensionalidad en metricas de
  distancia"*, que es lo que parafraseo.
- **A3 sigue en pie y es mitad mía**: la frase del Ejercicio 2 sobre que ningún
  otro algoritmo deja una cota así choca con la cota de Breiman de mi Ejercicio
  9. Escribí el 9 sin cruzarlo contra el 2.

## G6. Prioridad de lo mío

1. **G1**, las dos citas faltantes. Es la única falta real y hay que verificar
   las fuentes antes de agregarlas.
2. **G2**, una línea diciendo que los resultados se verificaron estables entre
   semillas, ahora que lo están de verdad.
3. **A3**, acordar con Marcos cómo se acota esa frase.
4. G3 y G4, cosméticos.
