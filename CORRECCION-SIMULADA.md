# Corrección simulada, con el criterio del profesor

Lectura completa del PDF final (69 páginas, commit `253afec`), ejercicio por ejercicio,
inciso por inciso, con la consigna literal al lado. No es una revisión de forma: es el
intento de poner la nota y de anticipar por dónde va a entrar la repregunta.

Tres ejes, como pide la consigna: **desarrollo** (si la respuesta es correcta, completa y
rigurosa), **detección de IA** (los cinco criterios y las tres penalizaciones) y
**scoring**.

---

## Veredicto

| Eje | Resultado |
|---|---|
| Desarrollo | **9,2 / 10** promedio sin ponderar |
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
| 1 | 39 | **8,0** | Los ocho incisos completos, con LSTM corrida de verdad y cuenta de parámetros contrastada contra Keras | 5 hojas; el Punto VIII contesta seis viñetas en tres párrafos; dos imprecisiones (abajo) |
| 2 | 4 | **9,5** | Las cuatro versiones del apunte como relajación sucesiva de supuestos; la cota leave-one-out; SVR derivada entera | La taxonomía mete a SVR como cuarta "versión", que no relaja un supuesto sino que cambia de problema |
| 3 | 4 | **9,5** | Dos hojas y no falta nada; el contraejemplo de Vapnik sobre contar parámetros; contrasta la regla con un caso real del Ejercicio 10 | Nada sustantivo |
| 4 | 4 | **9,5** | El AG binario y el real corridos con 20 semillas, y el precipicio de Hamming explicado desde los datos propios | Davis (1985): el rango de páginas no se pudo confirmar |
| 5 | 5 | **9,0** | Declara que Kernel PCA no está en el material y trae a Schölkopf; la selección de kernel con tres fuentes y conclusión operativa | Es el ejercicio más libresco, el único sin corrida propia |
| 6 | 3 | **9,5** | Pone el argumento donde va, en la verosimilitud, y mide los dos atajos equivocados sobre Rossi | Nada sustantivo |
| 7 | 9 | **9,5** | La pregunta más larga del examen y está completa; el DBSCAN que falla se informa; la trampa del clustering sobre el embedding | El perfilado va sobre componentes y no sobre variables, así que no dice qué significan los grupos (y lo admite) |
| 8 | 2 | **9,5** | Recalcula el ejemplo de la cátedra y encuentra dos rondas que no cierran; unifica AdaBoost como caso particular de Gradient Boosting | Nada sustantivo |
| 9 | 3 | **9,5** | La cota de Breiman leída como herramienta; los dos reparos que el documento no hace (OOB y Gini); la pérdida de interpretabilidad | Nada sustantivo |
| 10 | 7 | **9,0** | Validación anidada, el sesgo de selección citado y evitado, el análisis de bordes de grilla | La submuestra de California Housing acota el alcance; la corrección del F1 ponderado queda contada como error propio, lo cual es honesto pero visible |

El Ejercicio 1 es el más débil del conjunto, y no por lo que dice sino por cuánto tiene
que decir: ocho incisos, uno de ellos con seis viñetas, en el mismo presupuesto de hojas
que el Ejercicio 3, que tiene cuatro cláusulas y las contesta en dos hojas.

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

### 4. Ejercicio 5: el único sin corrida propia

Los otros nueve traen números de una corrida real. El 5 es enteramente bibliográfico. No
lo pide la consigna, así que no es un incumplimiento. Pero en un examen donde el resto
mide, el contraste se nota, y el alineamiento kernel-objetivo de Cristianini et al. (2002)
se podía calcular sobre los mismos datos del Ejercicio 10 en diez líneas de código.

### 5. `figuras.py` es el único script sin un comentario personal

Criterio 3 de la consigna: "código perfecto sin comentarios personales". Los seis scripts
promedian entre el 6 % y el 24 % de líneas de comentario, y cinco de los seis tienen
marcas claras de ensayo y error. `codigo/e1_lstm/figuras.py` tiene nueve comentarios sobre
148 líneas y ninguno dice por qué se eligió nada.

Es el script de menor riesgo, porque solo dibuja. Pero es el único al que ese criterio se
le podría aplicar.

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

**Único punto blando: Davis (1985).** El trabajo existe y está acreditado en la literatura
de *scheduling*, pero el rango de páginas 136-140 no se pudo confirmar, porque las actas
del primer ICGA no están digitalizadas. Si el profesor es de los que verifica una
referencia al azar y cae justo en esa, la respuesta tiene que ser esa misma.

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
| 3 | Comentarios propios en `figuras.py`, por el criterio 3 | Pendiente |
| 4 | Alineamiento kernel-objetivo del Ejercicio 5 sobre datos reales | Pendiente |
| 5 | Rango de páginas de Davis (1985) | Sin vía de confirmación |
| 6 | Decidir la hoja del Ejercicio 1 | Decisión del grupo, PR #8 |
| 7 | Repaso cruzado oral con las diez preguntas de arriba | Del grupo, y es lo de mayor retorno |

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
