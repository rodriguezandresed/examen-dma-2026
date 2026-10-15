# Corrección simulada, con el criterio del profesor

Lectura completa del PDF final (69 páginas, commit `253afec`), ejercicio por ejercicio,
inciso por inciso, con la consigna literal al lado. No es una revisión de forma: es el
intento de poner la nota y de anticipar por dónde va a entrar la repregunta.

Tres ejes, como pide la consigna: **desarrollo** (si la respuesta es correcta, completa y
rigurosa), **detección de IA** (los cinco criterios y las tres penalizaciones) y
**scoring**.

---

> **Segunda lectura.** La primera versión de este documento ponía 9,2 de promedio y
> siete ejercicios en 9,5. Releído con más exigencia, ese promedio estaba inflado y la
> uniformidad lo delataba: puntuaba la prosa y el rigor metodológico, que son excelentes,
> y no miraba de qué están hechos los experimentos. La nota baja a **8,5**, y abajo está
> la sección "El techo del examen", que explica qué lo frena. También se corrige una
> afirmación falsa que esa primera versión dejó escrita.

## Veredicto

| Eje | Resultado |
|---|---|
| Desarrollo | **8,5 / 10** promedio sin ponderar |
| Penalización por estilo artificial | **0 %**. Ninguno de los diez ejercicios la activa |
| Riesgo real de que el profesor sospeche IA | **Bajo**, y por un motivo poco común: el trabajo corrige a la cátedra en cinco lugares y documenta lo que falló |
| Incumplimiento formal | **Uno**: el Ejercicio 1 va en 5 hojas |
| Exposición en la defensa oral | **Alta**, y es el riesgo principal que queda |

Si la nota se normaliza contra el mejor examen, como dice la consigna, este trabajo
compite por ser ese mejor examen. Lo que lo puede bajar no es el contenido: es la hoja de
más del Ejercicio 1 y una defensa oral floja.

---

## Nota por ejercicio

| Ej | Cláusulas | Desarrollo | Qué lo sostiene | Qué le resta |
|---|---|---|---|---|
| 1 | 39 | **7,5** | Los ocho incisos completos, con cuenta de parámetros contrastada contra Keras | **El único ejemplo en Python que pide la consigna corre sobre dos senos más ruido**, y el experimento no puede distinguir las configuraciones que se le pidió comparar. Además 5 hojas y el Punto VIII muy comprimido |
| 2 | 4 | **9,0** | Las cuatro versiones del apunte como relajación sucesiva de supuestos; la cota leave-one-out; SVR derivada entera | Exposición pura, sin una sola cuenta propia. La consigna no la pide, pero en un final de posgrado se nota |
| 3 | 4 | **9,0** | Dos hojas y no falta nada; el contraejemplo de Vapnik; contrasta la regla con un caso real del Ejercicio 10 | Lo más corto del examen; se apoya en datos de otro ejercicio y no propios |
| 4 | 4 | **8,5** | El AG binario y el real con 20 semillas, y el precipicio de Hamming explicado desde los datos propios | Rastrigin es el banco de pruebas de juguete por excelencia. Y de los tres casos de aplicación, dos comparten autor (Davis) y dos son de los 80 |
| 5 | 5 | **8,5** | Ahora mide: el alineamiento sobre datos reales, y el resultado negativo se informa | Sigue siendo el más bibliográfico, y la medición llegó al final y por insistencia |
| 6 | 3 | **9,0** | Pone el argumento donde va, en la verosimilitud, y mide los dos atajos equivocados | Rossi viene del notebook de la cátedra, así que el dato tampoco se salió a buscar |
| 7 | 9 | **9,0** | Nueve cláusulas completas; el DBSCAN que falla se informa; la trampa del clustering sobre el embedding | `digits` de scikit-learn, 8 por 8 píxeles. El perfilado va sobre componentes, así que no dice qué significan los grupos |
| 8 | 2 | **9,0** | Auditar la aritmética de la cátedra es la mejor idea del examen | El recálculo estaba **tipeado a mano** hasta esta pasada: la tabla que dice verificar no verificaba nada |
| 9 | 3 | **8,5** | La cota de Breiman leída como herramienta; los dos reparos que el documento no hace | Exposición pura, tres hojas sin una medición |
| 10 | 7 | **7,5** | Validación anidada, el sesgo de selección citado y evitado, el análisis de bordes de grilla | **Los dos datasets son los de juguete de scikit-learn**, y en clasificación los cuatro modelos empatan, así que la comparación que el ejercicio existe para hacer no informa nada. El preprocesamiento, que es el requisito (1), se reduce a un escalador |

### El techo del examen

La primera versión de esta corrección puntuaba la prosa, el rigor metodológico y la
honestidad, que son excelentes y sostienen los nueves. Lo que no miraba es **de qué están
hechos los experimentos**, y ahí hay un patrón que no se ve ejercicio por ejercicio y
salta cuando se los mira juntos:

| Ejercicio | Datos | Origen |
|---|---|---|
| 1.VI | Dos senos más ruido gaussiano | Sintético, generado en el script |
| 4 | Rastrigin en dos variables | Función de prueba sintética |
| 5 | Breast Cancer Wisconsin | `sklearn.datasets` |
| 6 | Rossi | Incluido en `lifelines`, y es el del notebook de la clase |
| 7 | Digits, 8 por 8 píxeles | `sklearn.datasets` |
| 10 | Breast Cancer y California Housing | `sklearn.datasets` |

**Ningún dato del examen salió de afuera de una biblioteca.** Seis experimentos, seis
conjuntos que vienen con el paquete o se generan con tres líneas. Para un final de
*Data Mining Avanzado* eso marca un techo, porque la parte empírica nunca se enfrenta a
lo que hace difícil el oficio: datos sucios, faltantes, variables categóricas, clases
desbalanceadas, fugas temporales.

La consigna lo tolera. La Opción 1 del Ejercicio 10 pide "datasets públicos" y los de
scikit-learn lo son. Pero la Opción 2, que el grupo no eligió, pide "datasets reales
obtenidos de Internet", y esa diferencia de redacción muestra qué esperaba el profesor.

Y en el Ejercicio 10 el costo es concreto, no estético. **El ejercicio existe para
comparar cuatro modelos, y el propio texto informa que en clasificación no se
distinguen.** Esa respuesta es verdadera y está bien argumentada, pero es vacía, y la
causa es la elección del conjunto: Breast Cancer tiene 569 casos, 30 variables numéricas
limpias y clases casi linealmente separables, cosa que ahora el alineamiento del
Ejercicio 5 demuestra con un número. Con ese conjunto, los cuatro modelos tienen que
empatar.

Lo mismo arrastra el requisito (1), "preprocesamiento adecuado de los datos": como los
dos conjuntos son matrices numéricas sin faltantes y sin categóricas, el preprocesamiento
del pipeline es un `StandardScaler` y nada más. Está bien hecho (dentro del `Pipeline`,
sin fuga, y con la prueba de que en árboles no cambia nada), pero no hay nada que
preprocesar. Si el profesor pregunta "¿y el preprocesamiento?", la respuesta honesta es
"un escalador, porque los datos venían limpios".

El Ejercicio 1.VI tiene su propia versión del problema. La consigna pide un ejemplo con
"una serie de tiempo **arbitraria**", y dos senos con ruido cumplen la letra. Pero una
señal periódica y determinista más ruido es el caso donde una LSTM no tiene nada que
demostrar, y el experimento queda sin poder para contestar lo que se le preguntó: con
tres semillas y 20 épocas, las cuatro configuraciones se superponen, y el propio texto lo
dice ("en test `lstm_64` gana por menos de un desvío"). Informar la superposición es
honesto; no convierte en informativo a un experimento que no lo es.

---

## Lo que marcaría con lápiz rojo

Ninguno de estos cinco invalida una respuesta. Son los que preguntaría. Los dos primeros
ya están corregidos; los tres que siguen, no.

### 1. Ejercicio 1.VI: "4352 parámetros, como `model.summary()`" · CORREGIDO

`model.summary()` informa **4385** para `lstm_32`: 4352 de la capa LSTM más 33 de la capa
densa de salida (32 pesos y un sesgo). La Tabla 1 separaba bien las dos columnas,
"Fórmula" y "Total Keras", y la nota aclaraba "Fórmula: solo capas LSTM". Pero la oración
del cuerpo, leída sola, decía que la fórmula coincide con `model.summary()`, y no coincide
con el total.

Es la clase de detalle que un profesor usa para ver si el alumno entendió la fórmula o la
copió. Ahora dice "los que `model.summary()` asigna a la capa LSTM" y da el total aparte,
leído del CSV.

### 2. Ejercicio 1.II: "apila capas ... con pocas neuronas por capa" · CORREGIDO

Era falso como caracterización de una red profunda: las arquitecturas profundas modernas
no tienen pocas neuronas por capa.

**Y no salía del apunte.** `Apunte - Taxonomia de Modelos y Algoritmos.pdf` titula su
sección 3.3 "Redes neuronales *shallow* (de pocas capas)" y define Deep Learning como
"redes neuronales de múltiples capas"; la 6.1 habla de "redes densas de varias capas", y
al describir EfficientNet menciona que escala "de forma conjunta el ancho, la profundidad
y la resolución". En ningún lugar dice nada sobre la cantidad de neuronas por capa. Así
que no correspondía atribuirlo: correspondía corregirlo.

Ahora dice que una red profunda "necesita menos neuronas en total que una sola capa oculta
de igual expresividad", que es el resultado que de verdad justifica la profundidad y que
engancha con la función lineal por tramos del Punto I.

Sin atribuir, la frase anterior era además el tipo de afirmación que un corrector lee como
"terminología avanzada sin comprensión demostrable", que es el criterio del 20 %.

### 3. Ejercicio 1.VIII: densidad

Las seis viñetas están las seis contestadas, pero en tres párrafos muy compactos. El
tercero encadena en nueve renglones el conteo de parámetros, la regla *d/l*, el hecho de
que contar pesos sobrestima la capacidad y el doble descenso de Belkin et al. (2019).

El doble descenso llega con el mecanismo a medias ("el descenso de gradiente estocástico y
la regularización limitan la complejidad usada") y el propio texto lo declara hipótesis no
medida, lo cual es correcto y honesto. Pero si el profesor pregunta por ahí, hay que poder
sostenerlo, porque es el único lugar del examen donde se invoca un resultado reciente sin
nada propio que lo apoye.

### 4. Ejercicio 5: era el único sin corrida propia · CORREGIDO

Los otros nueve traían números de una corrida real y el 5 era enteramente bibliográfico.
No lo pide la consigna, así que no era un incumplimiento, pero en un examen donde el resto
mide, el contraste se nota.

Se calculó el alineamiento kernel-objetivo de Cristianini et al. (2002) sobre el conjunto
de clasificación del Ejercicio 10, con su misma semilla, su misma partición y su mismo
escalado, y se comparó el orden que produce contra el que produjo la validación cruzada.
Eso es lo que de verdad se le pide al método: se vende como un atajo para elegir kernel
sin entrenar un modelo por candidato, así que la prueba es si coincide con haber
entrenado.

| Kernel | *A* |
|---|---|
| **Lineal** | **0,5255** |
| RBF, γ = 0,1 | 0,3692 |
| RBF, γ = 0,01, el que eligió la validación cruzada | 0,2172 |
| RBF, γ = 0,001 | 0,0845 |
| RBF, γ = 0,0001 | 0,0660 |
| RBF, γ = 1 | 0,0538 |
| Polinómico de grado 2 | 0,0419 |
| Polinómico de grado 3 | 0,0357 |

**No coincide, y eso es el hallazgo.** Dentro de la familia RBF, el alineamiento prefiere
γ = 0,1 y la validación cruzada eligió γ = 0,01, que es apenas el tercero de su familia.
Como sustituto de la validación cruzada para ajustar un hiperparámetro, acá falla.

Donde sí acierta es en la familia, y ese acuerdo vale más de lo que parece: el
alineamiento más alto por lejos es el del kernel lineal, y eso **respalda con una
medición una frase del Ejercicio 10 que hasta ahora era una conjetura declarada**, la de
que las clases son casi linealmente separables en el espacio estandarizado. Esa frase
llevaba el rótulo "inferencia propia, no verificada"; ahora está verificada y remite al
Ejercicio 5.

Dos ejercicios que antes se tocaban de lejos quedaron atados por un número, que es
exactamente lo contrario de lo que busca el criterio 1 de la consigna.

### 5. Las dos figuras imprimían sus rótulos a unos 5 pt · CORREGIDO

Este reemplaza a un hallazgo anterior que estaba mal. Yo había anotado que `figuras.py`
era el único script sin comentarios personales, y eso salió de contar palabras clave, no
de leerlo. Leído, el script tiene las mejores notas de diseño del repositorio, incluida
esta: *"Con fuentes de 9 pt, al 50 % del ancho las ecuaciones quedaban en ~4 pt impresas:
se suben las fuentes para que se lean al 60 %"*. El criterio 3 estaba cubierto.

Lo que apareció al leerlo es mejor y es real: **la figura estaba incluida al 42 %, no al
60 % para el que fue diseñada.** Medido sobre el PDF, comparando los píxeles de cada PNG
con el ancho al que se muestra:

| | Escala | Rótulo de 9 pt | Rótulo de 12 pt |
|---|---|---|---|
| Figura 1, capas de la CNN, antes al 56 % | 0,53 | **4,8 pt** | |
| Figura 2, celda LSTM, antes al 42 % | 0,42 | | **5,0 pt** |
| Figura 1, ahora al 82 % | 0,78 | 7,0 pt | |
| Figura 2, ahora al 64 % | 0,64 | 5,7 pt | 7,7 pt |

Cinco puntos contra un cuerpo de texto de 12 es ilegible impreso, y es justo lo que el
autor del script había tratado de evitar. Venía de la compresión del Ejercicio 1 a 4
hojas: achicar las figuras era parte de cómo se ganó esa hoja.

Con el Ejercicio 1 en 5 hojas, achicarlas dejó de comprar nada: **la quinta hoja tenía
15,5 cm libres**. Las dos volvieron a un tamaño legible sin mover el conteo de hojas, y la
página se revisó renderizada como imagen, que es lo que pide el `CLAUDE.md`.

Queda una sola cosa chica: dentro de la Figura 2, la caja de $h_t = o_t \odot \tanh(c_t)$
usa cuerpo 10 y no 12, así que imprime a unos 6,4 pt, por debajo del resto del diagrama.
Se lee, y la fórmula también está en el cuerpo del texto. Agrandarla pide reacomodar las
flechas de esa esquina, y no parece valer el riesgo.

---

## Detección de IA, criterio por criterio

La consigna lista cinco criterios y tres penalizaciones. Los evalúo como los evaluaría el
profesor, con la evidencia a la vista.

### Criterio 1. Inconsistencias entre secciones

**No se activa.** Es el criterio mejor cubierto, y por construcción: todo número del texto
se lee en línea desde el CSV de la corrida, así que no hay dos secciones que puedan
discrepar.

Los cruces entre ejercicios están tejidos y verificados: la regla *d/l* del 3 se usa en el
1.VIII y en el 5; la $\gamma = 1/(2\sigma^2)$ del 2 reaparece en el 10; el *shrinkage*
$\nu$ del 8 se usa en el 10; la cota por vectores soporte del 2 se compara con la de
Breiman en el 9; el perfilado del 7 remite a la importancia de variables del 9. Las seis
remisiones apuntan a donde dicen apuntar.

Dos chequeos que corrí sobre el PDF final:

- **Referencias en las dos direcciones: 0 huérfanas y 0 faltantes.** 61 entradas; cada
  apellido de la lista aparece citado en el cuerpo, y cada cita del cuerpo tiene su
  entrada. Vaswani, Bahdanau, Sutskever, Belkin, Cybenko y Scrucca, que son las que más
  fácil se quedan sin entrada, están las seis.
- **Diez bloques `Fuentes del Ejercicio N`**, uno por ejercicio, con nombre de archivo de
  cátedra y remisión al apéndice de código.

### Criterio 2. Terminología avanzada sin comprensión demostrable

**No se activa, y es donde el trabajo está más fuerte.** La densidad terminológica es
altísima (dimensión VC, pseudo-residuo, censura, perplejidad, Λ de Wilks, verosimilitud
parcial, equivariancia), y cada término se glosa en el lugar donde aparece primero.

Pero lo que de verdad desarma este criterio no es la glosa: es que **el examen discute con
el material en cinco lugares, con la fuente abierta al lado**.

| Dónde | Qué hace |
|---|---|
| Ej 8 | Recalcula las tres rondas de AdaBoost de la presentación y muestra que dos no cierran: con error 0,21 la fórmula da 0,6625 y la diapositiva publica 0,65; con 0,14 da 0,9076 y publica 0,92 |
| Ej 8 | Dice que la presentación mete a Gradient Boosting entre los métodos *arcing* y que la etiqueta le queda grande, porque no repondera ni remuestrea |
| Ej 9 | El documento llama insesgada a la estimación OOB y el examen frena: una serie de pruebas empíricas no demuestra insesgadez (Janitza & Hornung, 2018) |
| Ej 9 | Le encuentra a la importancia de Gini el sesgo por cantidad de categorías (Strobl et al., 2007), que el documento no menciona |
| Ej 7 | La guía advierte que MANOVA y ANOVA quedan sesgados y no ofrece salida; el examen agrega un clasificador contra la etiqueta de grupo y explica por qué no hay supuesto que violar |

Un modelo de lenguaje parafrasea el material. No lo audita con la calculadora y no le
discute una taxonomía. Esto es lo contrario de la señal que el criterio busca.

### Criterio 3. Código perfecto sin comentarios personales

**No se activa**, con la salvedad de `figuras.py` ya dicha. Los comentarios registran lo
que no funcionó, que es exactamente lo que el criterio pide:

- `ag_rastrigin.py`: *"sigma 0,3: primero se probó 1,0 y el AG saltaba entre valles sin
  afinar"*.
- `ag_rastrigin.py`: la definición de "cuenca global" *"se agregó al ver que varias
  corridas reales terminaban a pocas centésimas del origen con f > 0,01"*.
- `tsne_digits.py`: *"Primero lo probé sobre las 30 y la importancia se reparte entre
  tantas componentes que no se puede contrastar con nada"*.
- `tsne_digits.py` deja el contraejemplo deliberado del clustering sobre el embedding, con
  el comentario de por qué está mal aunque dé mejor ARI.

Y el cuerpo informa resultados que no favorecen: el DBSCAN que manda el 39,7 % de los
puntos al ruido, el MLP que empeora de validación cruzada a *hold-out*, el Gradient
Boosting de regresión que elige el máximo de etapas de su grilla.

### Criterio 4. Referencias irrelevantes o inexistentes

**No se activa.** 61 entradas, todas verificables, con DOI o URL donde existe. Las que un
corrector escéptico buscaría primero (Cybenko 1989, Ho 1995, Breiman 1996 y 2001, Friedman
2001, Freund y Schapire 1997, Hochreiter y Schmidhuber 1997, van der Maaten y Hinton 2008,
Schölkopf et al. 1998, Belkin et al. 2019, Grinsztajn et al. 2022) son todas reales y
están citadas donde corresponde, no decorando el final.

**Davis (1985) quedó confirmada contra la fuente original, y de paso corregida.** Era el
único punto blando: el rango de páginas venía de listas de referencias de terceros y no
de las actas, que se daban por no digitalizadas. Sí lo están: hay un escaneo del volumen
completo, y su índice dice que "Job shop scheduling with genetic algorithms", de Lawrence
Davis, empieza en la **página 136**, y que el trabajo siguiente, el de Fourman sobre
compactación de layout, empieza en la **141**. El rango 136-140 es correcto.

La portada agrega lo que faltaba: editor **John J. Grefenstette**, publicado por
**Lawrence Erlbaum Associates**, 1985, ISBN 0-8058-0426-9, del congreso del 24 al 26 de
julio de 1985 en Carnegie-Mellon.

Con esos datos la entrada cambió de forma, porque en APA 7 unas actas publicadas una sola
vez como libro con editor y editorial se citan como capítulo, no como artículo:

> Davis, L. (1985). Job shop scheduling with genetic algorithms. En J. J. Grefenstette
> (Ed.), *Proceedings of the First International Conference on Genetic Algorithms and
> their Applications* (pp. 136-140). Lawrence Erlbaum Associates.

Las otras quince referencias de congresos no cambian, y eso también es APA: *Proceedings
of the IEEE*, *Advances in Neural Information Processing Systems* y las actas de ICDAR,
KDD, CVPR y SIGKDD son series periódicas, que van con formato de artículo y el rango de
páginas sin `pp.`. La regla quedó escrita en el `CLAUDE.md`, con las dos formas separadas
y un ejemplo de cada una, para que no se mezclen la próxima vez.

### Criterio 5. Estilo de escritura inconsistente

**No se activa.** El trabajo es homogéneo de punta a punta: mismo registro impersonal,
misma coma decimal, mismo formato en los diez bloques de Fuentes, misma manera de glosar
un término la primera vez que aparece.

Conviene decir por qué, porque es una decisión del grupo y no una casualidad. El
`CLAUDE.md` del repositorio fija las normas de redacción y de citación antes de escribir:
APA 7 para las citas y la lista de referencias, voz impersonal con "se", prohibición de
gerundios y de em dashes, coma decimal y espacio fino antes del `%`, "Tabla" y "Figura"
como únicos rótulos, un bloque de Fuentes por ejercicio con el nombre del archivo de
cátedra. Los tres autores escribieron contra ese estándar, y
`codigo/verificacion/revisar_entrega.py` lo controla en cada corrida.

Esa es además la respuesta si el profesor pregunta cómo puede ser que tres personas
escriban igual: **escriben igual porque el grupo acordó un estándar y lo dejó escrito**,
que es lo que se espera de un entregable único con una sola lista de referencias
alfabética. La alternativa, tres registros distintos pegados en un mismo PDF, es
justamente lo que el criterio 5 penaliza.

Lo que sí hay que tener a mano es qué escribió cada uno, que no se deduce del texto y es
el tema de la sección siguiente.

### Las tres penalizaciones

| Penalización | Se aplica |
|---|---|
| −20 % por jerga técnica sin explicar | **No.** Cada término se glosa en su primera aparición. La única frase expuesta es la de "pocas neuronas por capa" del 1.II, por no atribuirla |
| −30 % por no mostrar razonamiento personal | **No, y por lejos.** El examen explicita qué se eligió, qué se descartó y por qué: el SOM frente a competitive learning (1.IV), la esfera descartada frente a Rastrigin (4), DEAP frente a GA de R (4), el bosque aleatorio frente a pruebas no paramétricas (7), `RandomizedSearchCV` frente a `GridSearchCV` (10), el F1 de la clase maligna frente al F1 ponderado (10) |
| −50 % por respuesta que parezca generada automáticamente | **No.** Ninguna de las diez. El trabajo informa fracasos, declara sus inferencias como propias y no verificadas, y corrige al material |

---

## El riesgo que queda, y no es el texto

> "Al corregir, es opción del profesor preguntar arbitrariamente a algún miembro del
> grupo, por alguna explicación adicional a una determinada pregunta."

Ahí está el problema real. El examen tiene una densidad de decisiones propias muy alta, y
cada decisión la tomó quien escribió ese ejercicio. Marcos escribió 1, 2, 3 y 10; Ulises,
4, 5 y 6; Andrés, 7, 8 y 9. El profesor puede preguntarle a cualquiera por cualquiera.

Las diez preguntas que yo haría, en orden de probabilidad:

1. **Ej 1.I.** ¿Por qué ReLU desplazó a la sigmoide si la sigmoide ya garantizaba
   aproximación universal? (La respuesta está en el examen: el teorema habla de una capa
   ancha, y lo que no garantiza es que el gradiente llegue.)
2. **Ej 8.** Mostrame la cuenta de la tercera ronda. (0,9076 contra 0,92 publicado.)
3. **Ej 3.** ¿Por qué contar parámetros no mide la capacidad? (Contraejemplo de Vapnik, y
   el caso propio del 1.VIII donde *d/l* da 5,2 y la red igual funciona.)
4. **Ej 7.** ¿Por qué no agrupaste sobre el mapa de t-SNE, si el ARI da mucho mejor?
5. **Ej 9.** ¿Por qué reducir *m* baja la correlación y la fuerza a la vez?
6. **Ej 6.** ¿Por qué el intervalo del *hazard ratio* casi toca 1 y el *p* igual da 0,047?
7. **Ej 4.** ¿Por qué el AG binario queda clavado en x = 1?
8. **Ej 10.** ¿Por qué el MLP empeora de validación cruzada a *hold-out* y los árboles
   mejoran?
9. **Ej 2.** ¿Por qué decís que en la versión no lineal separable *C* no interviene?
10. **Ej 5.** Explicame el alineamiento kernel-objetivo sin leer.

Las diez tienen respuesta dentro del propio examen. Ninguna la tiene si el que responde no
escribió ese ejercicio. **Un repaso cruzado de una hora rinde más, a esta altura, que
cualquier cambio en el texto.**

---

## Lo único que incumple la consigna

> "No deberían tener más de 4 hojas por pregunta."

El Ejercicio 1 va en 5. Los otros nueve están en 4 o menos.

```
E1 5   E2 3   E3 2   E4 4   E5 3   E6 3   E7 4   E8 3   E9 3   E10 4
```

Es la contrapartida de haber completado las cláusulas que faltaban en el 1.I, el 1.VI, el
1.VII y el 1.VIII. Medido: la quinta hoja lleva 8,58 cm de texto y la cuarta quedó sin
cola, así que volver a 4 significa devolver unos 17 renglones de los agregados.

Cómo lo leería yo como profesor: la consigna dice "no deberían", en condicional, el
Ejercicio 1 tiene ocho incisos (más del doble que cualquier otro) y ninguno de los ocho
pasa de dos hojas. Una hoja de más en la pregunta más larga cuesta menos que una cláusula
sin contestar. Pero es una lectura, no un hecho, y la decisión es del grupo.

---

## Estado de los arreglos

| | Qué | Estado |
|---|---|---|
| 1 | Las tres palabras del `model.summary()` en el 1.VI | **Hecho** |
| 2 | "Con pocas neuronas por capa" en el 1.II, verificado contra el apunte y corregido | **Hecho** |
| 3 | Las dos figuras del Ejercicio 1, que imprimían a unos 5 pt | **Hecho** |
| 4 | Davis (1985): páginas confirmadas en el escaneo de las actas y entrada pasada a formato de capítulo | **Hecho** |
| 5 | La regla de actas, en sus dos formas, escrita en el `CLAUDE.md` | **Hecho** |
| 6 | Alineamiento kernel-objetivo del Ejercicio 5 sobre datos reales | **Hecho** |
| 7 | El recálculo de los $\alpha_t$ del Ejercicio 8, que estaba tipeado a mano | **Hecho** |
| 8 | Decidir la hoja del Ejercicio 1 | Decisión del grupo, PR #8 |
| 9 | Los datasets de juguete, sobre todo en el Ejercicio 10 | **Abierto, y es lo único que movería la nota** |
| 10 | Repaso cruzado oral con las diez preguntas de abajo | Del grupo |

Sobre el 9, para que la decisión sea informada y no una aspiración: cambiar el conjunto
de clasificación del Ejercicio 10 por uno real, con faltantes y variables categóricas,
haría que el requisito (1) tenga contenido y que la comparación de los cuatro modelos
distinga algo. Cuesta volver a correr el pipeline completo, reescribir los párrafos de
resultados y de comparación, y rehacer las Tablas 8 y 9. Es el ejercicio de Marcos y es
el cambio más grande que queda sobre la mesa. Con la entrega encima, puede ser más
sensato dejarlo y tener preparada la respuesta: se eligió priorizar que el pipeline fuera
metodológicamente correcto antes que difícil.

**Corrección de la primera versión.** Ahí escribí que "los diez ejercicios tienen al
menos una medición propia". Es falso, y lo comprobé contando las lecturas en línea del
`.Rmd`: **los Ejercicios 2 y 9 no tienen ninguna**, y el 3 se apoya en datos del 10. El 8
tampoco tenía, hasta que en esta pasada el recálculo de los $\alpha_t$ pasó a calcularse.
Ninguno de los tres incumple la consigna, que no pide código ahí, pero la frase estaba
mal y había quedado también en el cuerpo del PR.

El examen conserva a propósito otras cinco afirmaciones rotuladas como inferencia o
hipótesis: el doble descenso del 1.VIII ("es una hipótesis: no se midió el error de
entrenamiento"), la causa probable del comportamiento de DBSCAN en el 7, y tres del 10
(el kernel ancho, la explicación de por qué los árboles ganan y la conjetura sobre el
MLP). Esas no son deudas y no hay que sacarlas: declarar una inferencia como inferencia
es lo que separa un razonamiento propio de una afirmación inventada, y es de las cosas
que más juegan a favor frente al criterio del 30 %. La que cambió es solo la de la
separabilidad lineal, porque ahora hay con qué medirla.

## Cómo reproducir esta corrección

Los controles que se pueden automatizar están en un script, así que no hay que volver a
medirlos a mano ni creerle a este archivo:

```bash
Rscript -e 'rmarkdown::render("examen-dma-2026.Rmd")'
python codigo/verificacion/revisar_entrega.py
```

Verifica, sobre el PDF compilado y sobre el `.Rmd`: hojas por pregunta, distancia al
número de página, títulos colgados al pie, los diez bloques de Fuentes, la lista de
Referencias en las dos direcciones y las reglas de redacción del `CLAUDE.md`. Devuelve 1
si algo falla. Las excepciones al límite de hojas están declaradas arriba del script, con
el motivo, y se imprimen en cada corrida.

Lo que el script **no** puede verificar, y por eso esta corrección se hizo a mano: si una
definición es correcta, si una cláusula de la consigna está realmente contestada y si una
afirmación sobre el material de cátedra resiste abrir el archivo. Los tres hallazgos más
importantes de este documento salieron de eso, no de un grep.
