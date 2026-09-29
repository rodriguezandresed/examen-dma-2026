# Revisión del Ejercicio 9

Revisor: autor del Ejercicio 8. Material revisado: `staging/e9.Rmd`,
`staging/e9_referencias.md`, `codigo/e9_rf/rf_breiman.py`, `codigo/resultados/e9/`.

Verificaciones ejecutadas de verdad: se volvió a correr el script (37 s), se compararon
los md5 de los siete CSV y los dos PNG, se evaluaron los 36 valores interpolados con R
inline contra los CSV, se compiló el fragmento a PDF y se revisó cada página como imagen,
y se cotejaron las afirmaciones contra `clase6/randon forest explicado.pdf` y
`clase6/Ensemble Methods Universidad Austral [...].pdf`.

## Puntaje por criterio

| Criterio | Resultado | Evidencia |
|---|---|---|
| 1. Inconsistencias entre secciones | **no cumple** | L124: "No hay un punto a partir del cual convenga frenar, que es lo que separa a RF del gráfico equivalente de boosting en las diapositivas". El Ejercicio 8 corre AdaBoost sobre la misma partición y reporta que el error de test **no** sube. Además L121-123 "El error de test sigue el mismo patrón y tampoco vuelve a subir" contra `oob_vs_arboles.csv`, donde sube dos veces. Y L221-222 "las 30 variables reales, que acá aportan todas algo" contra `importancias.csv`, donde 7 reales tienen permutación exactamente 0 |
| 2. Terminología sin comprensión demostrable | **cumple** | Todo término se glosa en su primera aparición: "out of bag (OOB, fuera de la bolsa)" L68, "decision stump (tocón de decisión, un árbol de una sola división)" L107, "Arcing (Adaptative Resampling and Combining)" L89-90, "$s$ la fuerza, esperanza de la función de margen del bosque" L60-61. No encontré jerga sin definir |
| 3. Código sin comentarios personales | **cumple** | Tres registros de intentos descartados con su razón: estimador de correlación por pares abandonado por costo (script L17-22), segundo eje con OOB descartado porque "el eje comprimido convertia ese ruido en picos enormes" (L163-166), barras Gini/permutación lado a lado descartadas por escalas incomparables (L251-256) |
| 4. Referencias irrelevantes o inexistentes | **parcial** | Las cinco entradas existen y las verifiqué contra las fuentes. Pero L250-253 atribuye al documento de Breiman y Cutler un par de cifras (0,037 → 0,043) que en el documento corresponden a una corrida distinta de la regla que se enuncia (ver hallazgo 3) |
| 5. Estilo inconsistente | **cumple** | Greps en cero: 0 guiones largos, 0 guiones dobles, 0 muletillas, 0 primera persona, 0 gerundios, 0 PENDIENTE. Coma decimal en prosa y en la tabla. El registro y el largo de oración igualan a los Ejercicios 1 a 3. Dos defectos de presentación, no de redacción: hallazgos 10 y 17 |

Reproducibilidad (parte mecánica del criterio 1): **cumple sin reservas**. Los siete CSV
y los dos PNG salen con md5 idéntico al volver a correr.

## Cobertura del enunciado

| Pregunta de la consigna | ¿Dónde se responde? |
|---|---|
| Describir el algoritmo | L39-74. Voto por mayoría, bootstrap de filas, sorteo de $m$ variables por nodo, sin poda, correlación y fuerza, bolsa OOB, importancias y proximidad. **Cumple** |
| Diferencia con el resto de los algoritmos de ensamblaje | L76-112 y Tabla 1. Contra bagging: **cumple**, y con evidencia propia (aislar `max_features` es la forma correcta de hacerlo). Contra boosting: **parcial**. Ver hallazgos 1 y 4. Además, subagging aparece en el mismo deck (p. 70, submuestras sin reemplazo con $k = 0{,}5M$) y no se menciona: es el pariente más cercano de bagging y RF y es lo primero que un oral puede pedir |
| Principales usos | L229-253. Cinco grupos. **Cumple**, con la enumeración rota del hallazgo 7 y con una omisión: el documento dedica sección propia a la detección experimental de interacciones entre variables y no figura |

## Hallazgos, del más grave al más leve

1. **[bloqueante]** **Contradice al Ejercicio 8 sobre el sobreajuste de boosting, con los
   mismos datos y la misma partición.** L108-112: "las diapositivas advierten que en
   boosting el error de entrenamiento cae a cero mientras el de test vuelve a subir";
   L124-125: "No hay un punto a partir del cual convenga frenar, que es lo que separa a RF
   del gráfico equivalente de boosting en las diapositivas".
   El Ejercicio 8 corre AdaBoost sobre `load_breast_cancer` con `test_size=0.30`,
   `random_state=42`, `stratify=target`: **la partición idéntica a la de este ejercicio**
   (398 / 171 en los dos). Y reporta, con corrida propia, que con las etiquetas originales
   el error de test **no vuelve a subir** en 400 rondas, y que la subida solo aparece
   después de dar vuelta el 10 % de las etiquetas de entrenamiento. Sobre esa misma
   partición AdaBoost deja 7 casos mal clasificados de 171 y el RF de este ejercicio deja
   10.
   Un corrector que lea las dos secciones seguidas ve que el Ejercicio 9 apoya su
   diferencia central en un comportamiento que la sección anterior demostró que no ocurre
   con estos datos. Es exactamente el criterio 1 del profesor.
   **Cede el 9**, porque el 8 tiene corrida y el 9 solo tiene la cita de la diapositiva.
   Corrección: reemplazar el tercer punto del párrafo por la diferencia que sí se sostiene
   (en RF la cantidad de árboles no es un parámetro de regularización, el voto converge al
   sumar árboles; en boosting $T$ sí lo es) y remitir al Ejercicio 8, que muestra que esa
   regularización recién pesa con ruido de etiquetas. Y borrar la cláusula final de L124.

2. **[bloqueante]** **"El error de test ... tampoco vuelve a subir" es falso contra el
   propio CSV.** L121-123. En `oob_vs_arboles.csv` el error de test sube de 0,0643 con 30
   árboles a 0,0760 con 50, y de 0,0585 con 75 a 0,0643 con 100. El OOB sube de 0,0503 con
   10 a 0,0553 con 20, y de 0,0327 con 300 a 0,0352 con 400. Los dos picos se ven con toda
   claridad en el panel izquierdo de la Figura 1, o sea que la prosa contradice a la figura
   que tiene al lado, en la misma página.
   Corrección literal: "a partir de los 150 árboles ninguna de las dos curvas vuelve a
   subir". Eso sí es cierto y sigue sirviendo al argumento.

3. **[bloqueante]** **Cifra mal atribuida a la regla que se enuncia.** L250-253: "el manejo
   de clases desbalanceadas por pesos inversamente proporcionales al tamaño de cada clase,
   con un costo registrado en el documento de clase: en su ejemplo con 1000 y 50 casos,
   equilibrar los errores por clase subió la tasa de error general de 0,037 a 0,043".
   En `randon forest explicado.pdf` (p. 7) el peso inversamente proporcional es 20
   (1000/50) y esa corrida da **12,1 %** de error general, que el propio documento descarta:
   "El peso de 20 en la clase 2 es demasiado alto". El 4,3 % sale de la corrida con peso
   **10**, hallada por tanteo. La regla y la cifra que el texto pega juntas son de corridas
   distintas. Si el profesor pregunta "¿y cuánto dio con el peso inversamente
   proporcional?", la respuesta que el texto sugiere es la equivocada.
   Corrección: separarlas. "El documento propone el peso inversamente proporcional como
   guía inicial; en su ejemplo ese peso (20) desbalancea hacia el otro lado y lleva el error
   general a 0,121, y recién un peso de 10, hallado por tanteo, equilibra los errores por
   clase a costa de subir el error general de 0,037 a 0,043."

4. **[importante]** **La diferencia de clasificador base contra boosting es falsa contra
   las propias diapositivas y contra el Ejercicio 8.** L106-108: "boosting parte de reglas
   débiles, del tipo decision stump [...], y RF de árboles crecidos sin poda".
   La sección "Clasificadores base" del deck de ensambles lista para boosting: redes
   neuronales, árboles CART "Simples: Decision Stump" **y "Complejos sin podar"**, Naive
   Bayes, SVM y vecino más cercano, ordenados de menos a más estables, con la regla "cuanto
   más inestable es el algoritmo mayor es la mejora por aplicar boosting". Los árboles sin
   podar están explícitamente entre las bases de boosting. La Tabla 1 del Ejercicio 8 dice
   lo mismo: "Cualquier aprendiz débil: árbol, red, Naive Bayes, SVM".
   **Cede el 9.** La diferencia que sí se sostiene: RF **fija** el clasificador base (CART
   sin podar, siempre el mismo, por definición del algoritmo), mientras que en boosting la
   base es una elección del usuario que las diapositivas guían por inestabilidad.
   En el mismo párrafo: el 9 habla de "boosting" en bloque y nunca distingue AdaBoost de
   Gradient Boosting, que es justamente la distinción que construye el Ejercicio 8. Las
   tres diferencias que enumera son ciertas de AdaBoost con stumps y no de Gradient Boosting
   con árboles profundos. Una cláusula que remita al Ejercicio 8 cierra la cobertura y cose
   las dos secciones.

5. **[importante]** **Contradicción interna sobre qué variables aportan algo.** L220-222:
   "el ruido continuo queda en el puesto 31 de 33 por Gini, por debajo de las 30 variables
   reales, que acá aportan todas algo".
   En `importancias.csv` hay **10 variables con permutación exactamente 0,000** y solo 3
   son el ruido plantado. Las otras siete son reales: `mean compactness`, `mean symmetry`,
   `mean fractal dimension`, `texture error`, `smoothness error`, `concavity error` y
   `concave points error`, todas empatadas con el ruido en el puesto 28. Se ven en el panel
   izquierdo de la Figura 2 como la fila horizontal de puntos azules a la altura de los
   rombos del ruido. El párrafo afirma que todas aportan algo dos oraciones después de
   haber establecido que la permutación es la medida confiable, y según esa medida siete no
   aportan nada.
   Corrección, que además fortalece el argumento en vez de debilitarlo: acotar la frase a
   Gini ("por debajo de las 30 variables reales en la escala Gini") y agregar que la
   permutación tampoco distingue al ruido de siete variables reales, porque medida en
   accuracy sobre 171 casos su resolución es 1/171 = 0,006 y todo lo que pese menos de un
   caso cae a cero exacto.

6. **[importante]** **La conclusión del experimento de insesgadez acepta la hipótesis
   nula.** L193-201: "con *t*(19) = -0,82; *p* = 0,420. El signo repetido venía de la
   partición, no del estimador".
   Un *p* de 0,420 no es evidencia de ausencia de sesgo. Con $n = 20$ y sd de la brecha
   0,0176 el error estándar es 0,0039, así que el intervalo de confianza del 95 % de la
   brecha media va de -0,011 a +0,005: no descarta un sesgo de media unidad de caso. Lo que
   la corrida sí muestra es que el signo deja de repetirse (11 brechas negativas y 9
   positivas en `oob_particiones.csv`) y que el sesgo sistemático de -0,022 del barrido
   queda fuera del intervalo.
   Corrección: "el signo deja de repetirse al cambiar la partición (11 negativas y 9
   positivas) y la prueba no detecta sesgo, *t*(19) = -0,82; *p* = 0,420. Con 20
   particiones el intervalo de confianza del 95 % de la brecha media, de -0,011 a 0,005, no
   descarta un sesgo pequeño, pero sí descarta el -0,022 del barrido." Esa última cláusula
   es la que cierra el argumento y hoy falta.

7. **[importante]** **Vara de evidencia asimétrica: el error estándar binomial se usa para
   demoler una afirmación del documento y no para las otras dos.**
   L145-149 aplica el error estándar binomial (0,009) al barrido de $m$ y concluye, bien,
   que el rango óptimo "es todo el rango" porque el conjunto es demasiado fácil. Pero:
   - la banda de la curva de árboles es 0,010 (L121), del mismo orden que ese error
     estándar, y ahí se concluye que RF no se sobreajusta sin ninguna salvedad;
   - la cota de Breiman recorre 0,0161 a 0,0178 en todo el barrido, y su mínimo en $m = 8$
     se presenta como "cerca de la raíz de 30" (L131-132), cuando el mínimo del OOB está en
     $m = 3$ y $\sqrt{30} = 5{,}5$: tres respuestas distintas, las tres dentro del ruido.
     En el panel derecho de la Figura 1 la curva de la cota sube y baja sin forma.
   El autor ya reconoce en L148 que el conjunto es demasiado fácil para discriminar $m$;
   esa misma limitación vale para la afirmación de no sobreajuste y no se declara.
   Corrección: una cláusula en el párrafo de la curva ("la banda de 0,010 es del orden del
   error estándar binomial, así que la corrida muestra ausencia de deterioro y no podría
   detectar un deterioro de menos de un caso") y bajar "cerca de la raíz de 30" a la
   constatación de que el mínimo del OOB, el de la cota y $\sqrt{M}$ no coinciden.

8. **[importante]** **La advertencia sobre Gini se queda corta y desaprovecha una
   contradicción verificable.** L227: "Esta advertencia no está en el material de la
   materia".
   El documento de clase no calla sobre el punto: dice que la suma de disminuciones de Gini
   "usualmente es muy consistente con la medida de importancia de permutación". La corrida
   da un Spearman de 0,65 entre los dos ordenamientos y la variable más importante por
   permutación, `worst texture`, es la 14.ª por Gini. Eso **contradice** al documento, no lo
   completa.
   Corrección: "El documento de clase afirma que la importancia Gini es muy consistente con
   la de permutación; la corrida no lo sostiene." Es el hallazgo más fuerte del ejercicio y
   está enterrado en una frase defensiva.

9. **[importante]** **El texto define una importancia por permutación y la corrida mide
   otra.** L71-73 la define como "cuántos votos correctos se pierden al permutar al azar
   una variable en los casos OOB", que es la definición del documento. La corrida usa
   `permutation_importance` de sklearn sobre el **test reservado** y en **accuracy**
   (`rf_breiman.py` L235-237). El script lo justifica en un comentario; el `.Rmd` solo dice
   "calculada sobre el test con 30 repeticiones" y no avisa que es un estimador distinto
   del que definió tres párrafos antes.
   Corrección: una oración que diga que la corrida sustituye la bolsa por el test porque
   sklearn no expone la versión OOB, y que el cambio de métrica es lo que produce los ceros
   exactos del hallazgo 5.

10. **[importante]** **Figura 2 ilegible al tamaño de inclusión.** L213: `out.width="50%"`
    sobre una figura de 8,6 × 2,9 pulgadas. En el PDF compilado los rótulos de eje, la
    leyenda y las anotaciones "0,000" no se leen. Es la figura que carga toda la evidencia
    del contraste Gini contra permutación. Subirla a 95-100 %. (La Figura 1 va a 82 % y
    queda justo en el límite de lo legible.)

11. **[menor]** **Números tipeados a mano habiendo `meta.json` que los trae.** L18
    `/ 398` (tamaño de entrenamiento) y L156 `round(e9rf$err_test * 171)`. `meta.json` ya
    guarda `n_train: 398`, `n_test: 171` y hasta el `ee_binomial: 0,008572` calculado. En
    prosa, L115-116 repiten "569 casos, 30 variables" y "398 casos de entrenamiento". El
    `.Rmd` no lee `meta.json` en ningún momento. Fix: `jsonlite::fromJSON()` en el chunk de
    carga, o que el script escriba un `dimensiones.csv`.

12. **[menor]** **Los mismos estadísticos se calculan dos veces, en Python y en R.** La
    prueba t está en el script (scipy, L380-382, va a `meta.json`) y otra vez en el `.Rmd`
    (L20, `t.test`). Lo mismo el Spearman (script L378, `.Rmd` L23) y el error estándar
    binomial (script L375, `.Rmd` L18). Hoy coinciden, lo verifiqué. Son tres oportunidades
    de que dejen de coincidir en una edición futura, y la inconsistencia entre secciones es
    criterio de detección. Una sola fuente.

13. **[menor]** **La proximidad se describe sin la normalización.** L73-74: "la matriz de
    proximidad $N \times N$ cuenta en cuántos árboles cae en el mismo nodo terminal cada
    par de casos". El documento agrega "Al final, normalizar las proximidades dividiendo por
    el número de árboles", y esa normalización es la que hace que prox esté acotada por 1 y
    que el $1 - \text{prox}(k, r)$ de L239 sea una distancia. Tal como está descripta,
    $1 - \text{prox}$ sería negativo.

14. **[menor]** **"las dos cantidades se mueven juntas" se sostiene solo en los extremos.**
    L128-131 compara $m = 1$ contra $m = 30$. En `barrido_m.csv` la fuerza sube de 0,768 a
    0,839 entre $m = 1$ y $m = 8$ y después queda plana (0,836 a 0,839 hasta $m = 30$),
    mientras la correlación sigue subiendo hasta 0,413. Esa saturación es justamente el
    mecanismo por el cual $m$ grande no conviene, y la lectura de extremos lo tapa. Se ve
    en el panel derecho de la Figura 1. Una cláusula lo arregla y mejora el argumento.

15. **[menor]** **Choque de notación con el Ejercicio 8.** El 9 usa $M$ = total de
    variables y $m$ = variables sorteadas por nodo. El 8 usa $M$ = cantidad de iteraciones
    de Gradient Boosting y $m$ = cantidad de observaciones de entrenamiento (viene del
    pseudocódigo de AdaBoost del profesor, $D_1(i) = 1/m$). Las dos son la notación de su
    fuente, así que no corresponde renombrar ninguna; pero el documento ensamblado usa las
    mismas dos letras para cuatro cosas sin avisar.
    **Cede el 9**, no por estar equivocado sino porque ya tiene el párrafo de blanqueo de
    L56-65 y ahí entra media oración sin costo: "$M$ y $m$ designan acá el total de
    variables y las sorteadas por nodo, distinto del uso que tienen en el Ejercicio 8".

16. **[menor]** **"Es todo el rango" contra el propio resumen del script.** L148.
    `meta.json` da `n_plano: 13` sobre `n_puntos: 15`: dos puntos ($m = 2$ y $m = 27$)
    quedan fuera de la banda de un error estándar. "Trece de los quince puntos" es más
    preciso, no cuesta nada y evita que el número del texto discrepe del que el propio
    script dejó escrito.

17. **[menor]** **Tres defectos de las figuras.**
    - En `bosque.png`, la entrada "Correlación $\bar{\rho}$" de la leyenda (`figura_bosque`
      L194-195, `loc="lower right"`) queda encima de la curva azul entre $m = 21$ y
      $m = 30$.
    - En `importancias.png`, la leyenda del panel derecho tiene una entrada "Permutación"
      sin ninguna barra visible, porque los tres valores son exactamente cero. La nota al
      pie debería decirlo; hoy dice solo "Derecha: el ruido en su escala".
    - En el panel izquierdo de `importancias.png` las anotaciones "mean texture" y
      "worst texture" se superponen (`exp_importancias` L264-267).

18. **[menor]** **Dos notas de `e9_referencias.md` que pueden confundir al ensamblado.**
    Dice "La letra `a` no colisiona con las que ya usan los Ejercicios 1 a 3 (`b`, `c`,
    `e`)". Es cierto, pero **el Ejercicio 8 usa la misma letra `a` para la misma
    presentación**, que es lo correcto: una entrada por fuente. Conviene decirlo, para que
    al consolidar nadie renumere una de las dos y rompa las citas. Verifiqué que el texto de
    la entrada es idéntico palabra por palabra al del Ejercicio 8, así que la consolidación
    es directa.
    La otra nota, "No se cita a Freund y Schapire", queda desactualizada: el Ejercicio 8 sí
    los cita, así que la entrada va a estar en la lista final. No es un problema del 9, pero
    la nota induce a error.

19. **[menor]** **Una oportunidad de blanqueo desaprovechada.** El autor blanquea muy bien
    el $M$/$m$ de la traducción (L56-65), pero deja pasar otra torpeza del mismo documento:
    la afirmación de no sobreajuste está traducida como "Los RF no se adaptan demasiado",
    que no dice lo que el texto le hace decir en L111-112. Dado que el patrón ya está
    establecido dos párrafos antes, la omisión se nota.

## Lo que está bien y no hay que tocar

- **La reproducibilidad, sin reservas.** Volví a correr `rf_breiman.py` (37 s) y los siete
  CSV y los dos PNG salen con md5 idéntico, imágenes incluidas. Es más de lo que exige la
  rúbrica.
- **Los 36 valores interpolados con R inline.** Evalúan sin error y los verifiqué uno a uno
  contra los CSV: todos correctos, incluidos los redondeos finos (0,161 para la cota, 2,8
  para el factor de exceso, 5,8 para la razón Gini continua/binaria, 0,65 para el Spearman,
  puesto 31 de 33 para el ruido continuo, 10 casos de 171). No encontré un solo número
  inventado.
- **El blanqueo del $M$/$m$ de la traducción** (L56-65). Lo verifiqué en el PDF: el
  documento escribe "La reducción de M reduce tanto la correlación como la fuerza" y dos
  líneas después "un valor de m". La corrección es correcta y está declarada como
  corresponde. Es exactamente el gesto que el brief premia.
- **La declaración de que la sección de RF del deck son imágenes sin texto** (L29-30 y
  L34-37). Verificado: las páginas 72 a 74 del PDF no tienen texto extraíble. La declaración
  es cierta y se usa para justificar de dónde sale el núcleo del ejercicio, que es
  precisamente para lo que sirve.
- **El experimento de bagging contra RF** (Tabla 1 y `exp_bagging`). Aislar el efecto
  cambiando solo `max_features` es la forma correcta de responder con evidencia propia la
  segunda pregunta de la consigna, y descomponer el saldo en correlación (-0,045) y acierto
  por árbol suelto (-0,004) es literalmente la lectura del documento de clase, medida.
- **El diseño de las tres variables de ruido con cardinalidad creciente.** Es un
  experimento propio, bien pensado: la razón de 5,8 entre la Gini de la continua y la de la
  binaria es evidencia limpia del sesgo por cantidad de cortes.
- **El reemplazo del estimador de correlación** (docstring del script, L17-22). Documenta un
  intento descartado por costo, su sustituto y la ventaja adicional de habilitar la cota. Es
  el tipo de comentario que el criterio 3 busca, y los otros dos (figuras descartadas) están
  al mismo nivel.
- **Las mecánicas de estilo.** Cero en todos los greps de la rúbrica, coma decimal
  consistente, y la advertencia sobre `escape = FALSE` aplicada y documentada en el chunk
  (L170-172), con la notación matemática movida a la nota al pie, que es donde funciona.
- **Compila en 4 páginas de contenido**, dentro del límite de la consigna.

## Nota para el ensamblado

El fragmento numera sus propias tablas y figuras ("Tabla 1", "Figura 1", "Figura 2"). En el
documento completo serán otras. Lo mismo vale para mi Ejercicio 8. No es un defecto del
autor, pero hay que renumerar los dos al integrar.
