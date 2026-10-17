# Cambios propuestos, los que no son de texto

Lo que sigue **no está hecho**. Son los dos cambios que quedan sobre la mesa después de
la corrección simulada, los dos requieren volver a correr un script, y los dos son del
ejercicio de otro integrante. Se dejan acá especificados para que la decisión sea del
grupo y, si se toma, no haya que volver a pensarla.

Lo que sí está hecho es todo lo que se podía arreglar tocando solo texto, incluido
declarar las dos limitaciones que estas propuestas resolverían de fondo.

## Por qué, en una tabla

Las notas por ejercicio están en `CORRECCION-SIMULADA.md`. Lo que importa para decidir
no es cuál ejercicio está peor, sino cuánto examen hay detrás de cada uno: la consigna
tiene **82 cláusulas**, y están muy mal repartidas.

| Escenario | Promedio simple | Ponderado por cláusulas |
|---|---|---|
| Como está hoy | 8,55 | **8,09** |
| Con la propuesta 1 (Ejercicio 1.VI) | 8,68 | **8,68** |
| Con la propuesta 2 (Ejercicio 10) | 8,70 | **8,21** |
| Con las dos | 8,82 | **8,81** |

| | Cláusulas | Aporte ponderado |
|---|---|---|
| **Propuesta 1**, Ejercicio 1 | 39 de 82 | **+0,59** |
| Propuesta 2, Ejercicio 10 | 7 de 82 | +0,13 |

La propuesta 1 vale unas cuatro veces y media más que la 2, y es la más barata de las
dos. Si se hace una sola, es esa.

Vale mirar también la brecha entre las dos columnas. El promedio simple favorece al
examen porque las dos preguntas más flojas son también dos de las tres más pesadas. Cuál
de los dos números aplica depende de cómo corrija el profesor: una nota por ejercicio da
8,5; leer inciso por inciso da 8,1.

---

## Propuesta 1. Una serie de tiempo real en el Ejercicio 1.VI

### El problema

El enunciado pide "un ejemplo completo en Python de aplicación de una red LSTM para
predecir una serie de tiempo arbitraria" y "pruebe diferentes configuraciones de capas y
cantidad de neuronas". Hoy la serie son dos senos más ruido gaussiano.

Cumple la letra, pero una señal periódica y determinista con ruido es el caso donde una
LSTM no tiene nada que demostrar, y el experimento se queda sin poder: con tres semillas
y 20 épocas, las cuatro configuraciones caen dentro de su propia dispersión. El texto ya
lo dice ("en test `lstm_64` gana por menos de un desvío"), así que **la segunda mitad del
inciso queda sin respuesta firme**.

### Qué serie

Hace falta una serie real, pública, con tendencia y estacionalidad, de al menos unos mil
puntos (con la ventana de 20 y el corte 80/20, menos que eso deja un test ridículo).
Candidatas, en orden de preferencia:

1. **Temperaturas mínimas diarias de Melbourne**, 3.650 puntos, diez años. Estacionalidad
   anual limpia, ruido diario alto, y es chica: el CSV entra en el repo, así que la
   corrida no depende de la red.
2. **Bike Sharing por hora** (UCI), 17.379 puntos. Doble estacionalidad, diaria y
   semanal, que es donde la memoria larga de una LSTM se luce frente a un modelo lineal.
   Más cara de entrenar.
3. **Manchas solares mensuales**. Ciclo largo de unos once años, pero viene con
   `statsmodels`, así que no sale de la crítica de usar datos que trae la biblioteca.

Recomendada la 1. El CSV se versiona en `codigo/e1_lstm/datos/` y se cita en Referencias
como conjunto de datos, con el formato que ya usa Wolberg et al. (1993).

### Qué tocar

En `codigo/e1_lstm/lstm_serie.py`:

- Reemplazar el generador de senos por la lectura del CSV. Son unas veinte líneas.
- Subir las épocas: 20 es poco y con una serie real va a quedar corto. Conviene parada
  temprana sobre el bloque de validación en lugar de un tope fijo.
- **No tocar** el resto del montaje, que está bien y es lo que da valor: corte temporal
  80/20 sin mezclar, el último 10 % del entrenamiento como validación, las mismas cuatro
  configuraciones y las tres semillas.

En el `.Rmd`, Punto VI:

- Los números se actualizan solos: todos se leen en línea desde `lstm_configs.csv`,
  `lstm_runs.csv` y `meta.json`.
- **Hay que reescribir a mano dos cosas.** La descripción del montaje ("dos senos más
  ruido gaussiano de desvío 0,1") y, sobre todo, **la comparación contra el piso de
  ruido**, que desaparece: el piso se calcula comparando la señal limpia con la
  observada, y en una serie real no hay señal limpia. La persistencia sobrevive como
  línea de base y alcanza.
- Sacar el párrafo que hoy declara la limitación de la serie sintética, que deja de
  aplicar.
- Revisar la conclusión sobre las configuraciones: si con una serie real se separan, hay
  que decir cuál gana y por qué; si siguen empatadas, eso también es un resultado, pero
  esta vez uno que significa algo.

### Lo que no se rompe

`d_over_l_best` de `meta.json` alimenta el Ejercicio 1.VIII (la regla $d/l$). Se recalcula
solo y la lectura no cambia, porque depende de la cantidad de pesos y del tamaño de
entrenamiento, no de qué serie sea.

### Costo y riesgo

Un par de horas con la verificación incluida. El riesgo es bajo y hay lugar: el Ejercicio
1 está en 5 hojas con **9,1 cm libres** en la quinta, así que el texto nuevo entra sin
pelear con el límite. TensorFlow 2.21 está instalado y corre en esta máquina.

---

## Propuesta 2. Un conjunto de clasificación real en el Ejercicio 10

### El problema

El ejercicio existe para comparar cuatro modelos, y el propio texto informa que en
clasificación no se distinguen. Es verdad y está bien argumentado, pero es una respuesta
vacía, y la causa es la elección del conjunto: Breast Cancer tiene 569 casos, 30
variables numéricas limpias y clases casi linealmente separables, cosa que el alineamiento
del Ejercicio 5 ahora demuestra con un número. Con ese conjunto los cuatro **tienen** que
empatar.

Arrastra al requisito (1) de la consigna, "preprocesamiento adecuado de los datos": como
no hay faltantes ni variables categóricas, el preprocesamiento es un `StandardScaler` y
nada más. Está bien hecho, dentro del `Pipeline` y sin fuga, pero no hay nada que
preprocesar.

### Qué conjunto

Tiene que traer lo que Breast Cancer no tiene: variables categóricas, y ojalá faltantes y
desbalance. Y tiene que ser chico, porque el SVR ya obligó a submuestrear en regresión.

1. **German Credit (`credit-g`, OpenML)**, 1.000 casos, 20 variables de las cuales 13 son
   categóricas, 30 % de la clase mala. Chico para SVM, y el costo asimétrico entre los
   dos errores es real y da de qué hablar, igual que el maligno no detectado de ahora.
2. **Bank Marketing** (UCI), 45.211 casos, 11 % de positivos. Más realista y más caro.
3. **Adult / Census Income** (UCI), 48.842 casos, categóricas y faltantes. El más caro de
   los tres para SVM.

Recomendado el 1, por tamaño. Se baja con `fetch_openml`, que además responde a la
crítica de que ningún dato del examen sale de afuera de una biblioteca.

### Qué tocar

En `codigo/e10_pipeline/pipeline_opcion1.py`:

- El preproceso pasa a ser un `ColumnTransformer` de verdad: `OneHotEncoder` para las
  categóricas y `StandardScaler` para las numéricas, todo dentro del `Pipeline` para
  mantener la ausencia de fuga, que es lo mejor que tiene el ejercicio.
- Las grillas de hiperparámetros no cambian.
- La métrica se mantiene: F1 de la clase costosa, con la clase mala como positiva.

En el `.Rmd`, Ejercicio 10: la viñeta de datos, el párrafo de preprocesamiento y los dos
párrafos de comparación. Las Tablas 8 y 9 se rehacen solas desde el CSV.

### El costo escondido, que es lo que la vuelve cara

**Cambiar el conjunto del Ejercicio 10 rompe dos remisiones de otros ejercicios.**

- El **Ejercicio 3** cierra contrastando la regla $d/l < 0{,}05$ con un caso real, y ese
  caso es Breast Cancer: lee `k` y el tamaño de entrenamiento desde `meta.json` del 10.
  Con otro conjunto cambian los dos números y hay que revisar que la lectura siga en pie.
- El **Ejercicio 5** calcula el alineamiento kernel-objetivo **sobre la partición del
  Ejercicio 10**, y todo el valor de esa comparación es que los dos miran el mismo
  conjunto. O se recalcula el alineamiento sobre el conjunto nuevo, o el Ejercicio 5
  pierde el anclaje.

Y hay una pérdida fina: con un conjunto no separable linealmente, el alineamiento del
Ejercicio 5 probablemente deje de coronar al kernel lineal, con lo cual se cae el
respaldo que hoy le da a la frase del Ejercicio 10 sobre la separabilidad. No es grave,
pero el párrafo hay que rehacerlo.

### Costo y riesgo

Medio día. Toca tres ejercicios de dos autores distintos y el Ejercicio 10 está en 4
hojas con **0,03 cm libres**, o sea que cualquier línea de más lo empuja a 5. Es el
cambio más grande que queda y el de menor retorno por hora.

---

## Lo que conviene no hacer

- **Cambiar el conjunto del Ejercicio 7.** `digits` es de juguete, pero el enunciado solo
  pide "un ejemplo de implementación en Python" de t-SNE, y el ejercicio ya usa esos
  datos para algo que vale: mostrar que DBSCAN falla y que agrupar sobre el embedding es
  una trampa. Cambiarlo cuesta y no suma.
- **Agregar corridas a los Ejercicios 2 y 9.** No tienen ninguna medición propia, pero la
  consigna no pide código ahí y los dos están entre los mejores del examen por la calidad
  del razonamiento. Meter un experimento decorativo los empeoraría.
- **Volver el Ejercicio 1 a 4 hojas.** Está medido en `CORRECCION-SIMULADA.md`: son 17
  renglones, y salen casi todos de las cláusulas que estaban sin contestar.

## Si no se hace nada

El examen queda en 8,5 de promedio simple y 8,1 ponderado, sin ninguna cláusula sin
responder, sin penalización por estilo artificial y con las dos limitaciones **declaradas
en el propio texto**, que es la diferencia entre que el profesor las encuentre y que se
las cuenten. Para la defensa oral, la respuesta preparada es la misma en los dos casos:
se priorizó que el montaje fuera metodológicamente correcto antes que difícil, y las dos
limitaciones están dichas en el informe.
