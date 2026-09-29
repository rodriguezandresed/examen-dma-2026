# Revisión del Ejercicio 8

Revisor: autor del Ejercicio 7. Material revisado: `staging/e8.Rmd` (280 líneas),
`staging/e8_referencias.md`, `codigo/e8_boosting/adaboost_vs_gb.py`,
`codigo/e8_boosting/figuras.py`, `codigo/resultados/e8/`.

Todo lo que sigue se corrió de verdad: el script completo, `figuras.py`, los greps de la
rúbrica, el knit del ejercicio suelto contra el YAML del examen, y la aritmética del
ejemplo de Schapire rehecha desde cero sin mirar el CSV del autor.

## Puntaje por criterio

| Criterio | Resultado | Evidencia |
|---|---|---|
| 1. Inconsistencias entre secciones | **cumple** | El script reproduce los siete CSV y el `meta.json` byte a byte (`diff` sin diferencias), y `figuras.py` reproduce los dos PNG con el mismo md5. Verifiqué contra los CSV los quince valores interpolados del texto (L217-224, L235-242, L259-267) y todos dan: `e8m0`=18, `e8tefin`=0,0409, `e8temin`=0,0292, ruido mín. 0,0643 en m=161 y final 0,1053, pesos 7,84 y razón 10,06, `share_top10` 81,2 % contra 9,8 %, sd 78,4→33,5, grilla 2714 / 2739 / 5584. Números tipeados a mano: solo constantes de configuración (ver hallazgo 5) |
| 2. Terminología sin glosar | **parcial** | `Huber-M` (L96 y L120) y `log-verosimilitud` (L96) no se glosan nunca. `MAE` aparece en la Tabla 2 (fila "Gradiente de la MAE contra el vector de signos") y no se glosa en ningún lado. `MSE` aparece en esa misma Tabla 2 (L167-179) pero recién se glosa en la nota de la Tabla 3 (L256), doce líneas después. También sin glosar: `estratificada` (L163) y `diferencias finitas` (L183) |
| 3. Código sin comentarios personales | **cumple** | Los comentarios explican decisiones, no líneas: L4-8 ("La idea del script NO es comparar accuracy: con estos datos los dos empatan y no se aprende nada"), L80-81 (guarda para `eps=0`, con el motivo), L114-115 (por qué no se pasa `algorithm=SAMME` en sklearn 1.7), L268-269 (por qué hace falta ensuciar el train). Hay resultado negativo registrado en el código y sostenido en el texto |
| 4. Referencias | **cumple** con una corrección | Verifiqué las cinco entradas: Freund & Schapire (1997) *JCSS 55*(1), 119-139, DOI 10.1006/jcss.1997.1504; Friedman (2001) *Annals of Statistics 29*(5), 1189-1232, DOI 10.1214/aos/1013203451; Friedman (2002) *CSDA 38*(4), 367-378, DOI 10.1016/S0167-9473(01)00065-2. Las tres correctas en autor, año, revista, volumen, páginas y DOI. Ninguna de adorno: las tres sostienen afirmaciones concretas. Problema en el título de Volpacchio (hallazgo 3) y cita colgada de Schapire (hallazgo 4) |
| 5. Estilo | **cumple** | Los cinco greps de la rúbrica dan 0: em/en dashes 0, ` -- ` 0, muletillas 0, primera persona 0, gerundios 0 (ni siquiera dentro del enunciado, porque la consigna del 8 no trae ninguno). Coma decimal en toda la prosa. El registro es el mismo de los ejercicios 1 a 3 |

## Verificación del ejemplo de Schapire

Rehíce la cuenta a mano, sin abrir `schapire.csv`, y después extraje las diapositivas
para confirmar la transcripción.

Con `alpha = 1/2 · ln((1-eps)/eps)` sobre los eps impresos:

| eps de la slide | alpha calculado | alpha de la slide |
|---|---|---|
| 0,30 | 0,42365 | 0,42 |
| 0,21 | 0,66246 | 0,65 |
| 0,14 | 0,90764 | 0,92 |

Despeje inverso `eps = 1/(1+e^(2·alpha))` sobre los alpha publicados: 0,30153, 0,21417 y
0,13705, que redondeados a dos decimales dan exactamente 0,30, 0,21 y 0,14.

**La afirmación del autor es correcta y está bien fundada.** Coincide con
`schapire.csv` en los cinco decimales. Además confirmé la transcripción contra el PDF
(`clase6/Ensemble Methods Universidad Austral [Modo de compatibilidad] [Reparado].pdf`):
la página 25 trae eps=0,30 y alpha=0,42, la 26 eps=0,21 y alpha=0,65, la 28 eps=0,14 y
alpha=0,92, y la 29 repite los tres alpha. No hay error de transcripción ni de cuenta.

Igual de importante: el texto **no acusa a la cátedra de un error**. L204-205 cierra con
"la discrepancia está en el redondeo de la presentación, no en el algoritmo", que es la
lectura correcta, porque los eps impresos son a su vez el redondeo de los eps reales. El
encuadre es seguro para el oral. Sin cambios en este punto.

## Cobertura del enunciado

| Pregunta de la consigna | ¿Dónde se responde? |
|---|---|
| "Qué diferencias existen entre el algoritmo Adaboost y Gradiente Boosting" | L49-123: mecanismo de AdaBoost (L58-68), de Gradient Boosting (L70-78), Tabla 1 con seis ejes de comparación (L80-111) y las dos consecuencias derivadas (L116-123). **Cumple, y con sobra** |
| "Explique brevemente" | **Parcial.** El bloque de diferencias ocupa unas dos hojas contando Tabla 1 y la evidencia empírica asociada. Ver hallazgo 1: el ejercicio entero mide 5 hojas contra un tope de 4 |
| "De una explicación intuitiva de cómo trabaja el Gradiente Boosting" | L125-159. **Cumple y es genuinamente intuitiva**: analogía del golfista (L125-131), ejemplo numérico de los cinco departamentos con la media 1418 como primer golpe (L133-140), por qué se llama gradiente (L142-150) y tamaño del paso (L152-159). Cubre el ciclo completo: modelo inicial, residuo, árbol sobre el residuo, suma, repetición, shrinkage y variante estocástica |

Sobre la pregunta de si la explicación intuitiva es una derivación formal disfrazada:
**no lo es.** El único elemento formal es `sign(y - F_{m-1}(x))` en L149, y llega después
de la frase que hace el trabajo intuitivo ("El residuo no es solo una magnitud: es un
vector que apunta desde la predicción actual hacia el valor verdadero", L142-144). La
formalidad está al servicio de la intuición y no al revés.

## Hallazgos, del más grave al más leve

1. **[importante] El ejercicio mide 5 hojas y el tope de la consigna es 4.** Compilé
   `staging/e8.Rmd` sola contra el YAML del examen: 5 páginas de contenido, las cinco
   llenas hasta el final de la caja de texto. La consigna dice "No más de 4 hojas por
   pregunta" y el BRIEF apunta a 1,5-2,5 páginas de prosa más tablas y figuras.
   *Corrección concreta*, por orden de menor daño: (a) la Tabla 2 de verificaciones
   (L167-184, nueve filas más caption más nota de tres líneas, unos 7 cm) puede bajar a
   las cuatro filas que el texto efectivamente cita y el resto pasar al apéndice;
   (b) el párrafo de la grilla de tasa de aprendizaje (L259-268) y su Tabla 3 se pueden
   fundir dejando la tabla y dos oraciones; (c) la Figura 2 a `out.width="85%"`.
   Aviso de contexto: **mi Ejercicio 7 tiene el mismo problema** y también quedó en 5
   hojas, así que conviene que el coordinador decida un criterio único para los dos.

2. **[importante] `meta.json` guarda `NaN` en dos campos, por un error de índice.**
   `adaboost_vs_gb.py` L387-388 hace `float(df_D["eps"].iloc[0])` y
   `float(df_D["alpha"].iloc[0])`, pero la fila 0 de `df_D` es el punto de partida
   `D_1(i)=1/m`, que no tiene eps ni alpha. El propio autor lo documenta en L320-322
   ("Fila 0: el punto de partida D_1(i) = 1/m, antes de la primera reponderacion"), así
   que el índice correcto es `.iloc[1]`. Resultado: `"eps_ronda1": NaN, "alpha_ronda1":
   NaN` junto a un `"status": "ok"`, y el archivo deja de ser JSON estricto (lo confirmé
   con `json.loads(..., parse_constant=...)`, que levanta `ValueError`).
   *Impacto acotado*: el `.Rmd` no lee `meta.json`, así que ningún número publicado está
   mal. Pero es un artefacto de la entrega con `NaN` a la vista.
   *Corrección*: cambiar los dos `.iloc[0]` por `.iloc[1]`, que dan eps=0,0729 y
   alpha=1,2718 (los tengo verificados contra `adaboost_pesos.csv`).

3. **[importante] El título de la presentación de Volpacchio en las referencias no es el
   del archivo.** `e8_referencias.md` L20-21 pone
   "*Ensemble methods in machine learning*". El archivo de la clase 6 se llama
   `Ensemble Methods Universidad Austral` (hay un `.ppt` y un
   `.pdf [Modo de compatibilidad] [Reparado]`), y el propio `e8.Rmd` L271 lo cita así.
   Dos problemas: la entrada de Referencias contradice al bloque de Fuentes del mismo
   ejercicio, y "Ensemble methods in machine learning" es además el título de un trabajo
   conocido de Dietterich (2000), con lo cual un corrector puede leerlo como una
   referencia inventada. El exemplar cita la otra presentación como
   "*SVM Universidad Austral*", es decir por el nombre del archivo.
   *Corrección*: reemplazar el título por "*Ensemble Methods Universidad Austral*".

4. **[menor] Cita colgada de Schapire (1998).** L195 abre con "El ejemplo de Schapire de
   1998 que reproducen las diapositivas". `e8_referencias.md` L47-49 decide, con buen
   criterio, no agregar entrada propia porque la fuente consultada es la presentación.
   El problema es que la prosa igual deja "Schapire de 1998" con forma de cita y sin
   entrada en Referencias, que es exactamente el patrón que el profesor marca como
   referencia inexistente.
   *Corrección*: reformular a "El ejemplo que las diapositivas atribuyen a Schapire
   (Volpacchio, s.f.-a)", que dice lo mismo y deja la cita apuntando a una entrada que
   sí existe.

5. **[menor] Constantes de configuración tipeadas a mano teniendo `meta.json`.** L162
   "con semilla 42", L163 "partición estratificada 70/30", L231 y L238 "10 %" de
   etiquetas dadas vuelta. Los tres valores están en `meta.json` (`seed`,
   `dataset_clasificacion`, `nota_ruido`) y en las constantes `SEED`, `RUIDO` del script.
   No están mal hoy, pero si alguien toca `RUIDO` el texto queda mintiendo en silencio, y
   el criterio 1 del profesor es justamente ese. En el Ejercicio 7 estos valores se
   interpolan desde un `meta.csv`.
   *Corrección*: exportar un `meta.csv` de una fila como hace el 7 y leerlo con
   `read.csv()`; evita además la dependencia de `jsonlite` en el documento compartido.

6. **[menor] Dos caracterizaciones en prosa que el dato no sostiene del todo.**
   - L217-218: "se estabilizan alrededor de `r fm(e8razon, 1)` veces el peso de las
     fáciles". `e8razon` (L26) es la razón de la **última fila**, 10,06. La serie
     completa oscila entre 7,03 y 16,97 (últimas diez rondas: 13,8, 8,2, 10,1, 8,1,
     10,3, 13,3, 8,7, 9,8, 7,7, 10,1), y eso se ve en el panel izquierdo de la Figura 1,
     que es cualquier cosa menos una meseta. La media de la serie es 10,03, así que el
     número que se publica es correcto por casualidad.
     *Corrección*: usar la media (`mean`) y escribir "oscila alrededor de", o dar el
     rango. Con el valor final, si cambia `M_ADA` el número salta.
   - L236-237: "oscila alrededor de `r fm(e8tefin, 3)`" también toma la última fila.
     Acá el 0,041 sí coincide con la meseta que muestra la figura, así que el problema es
     solo de robustez, no de lectura.

7. **[menor] Notación `D_t`, `F_0` y `alpha` en crudo dentro de la Tabla 2.** Las filas
   vienen de `verificaciones.csv` y en el PDF salen "D\_ t con la convención...",
   "F\_ 0 con pérdida cuadrática..." y "alpha de sklearn sobre alpha de las
   diapositivas", mientras la prosa de al lado usa $D_t$, $F_0$ y $\alpha_t$ bien
   compuestos. La decisión de no usar `escape = FALSE` es correcta y está bien
   documentada en L81-82; el arreglo no es tocar el escape sino el texto de origen.
   *Corrección*: reescribir las cadenas del CSV en castellano llano ("Distribución de la
   ronda t con las dos convenciones", "Predicción inicial con pérdida cuadrática",
   "alfa de sklearn sobre alfa de las diapositivas").

8. **[menor] `v` y `\eta` para la misma idea, sin blanquearlo.** L154 llama a $v$ "tasa
   de aprendizaje", que es lo correcto porque es la palabra de las diapositivas
   (confirmado: la slide dice "Parámetro Shrincage (tasa de aprendizaje. En general
   <0.1)"). Pero el Ejercicio 1 ya usa $\eta$ para la tasa de aprendizaje. Un corrector
   que lea los dos ejercicios seguidos ve dos símbolos para el mismo concepto.
   *Corrección*: una subordinada del tipo "la tasa de aprendizaje $v$, que el apunte de
   ensambles llama shrinkage y que en el Ejercicio 1 aparece como $\eta$". Es además el
   movimiento de blanquear contradicciones que el BRIEF premia, y sale gratis.

9. **[menor] "shrinkage" y "tasa de aprendizaje" se usan como sinónimos sin decirlo.**
   L99 y L113-114 introducen shrinkage, L154 introduce "tasa de aprendizaje $v$" y L159
   vuelve a "el shrinkage". El lazo se infiere de la Tabla 1 ("más el shrinkage v") pero
   nunca se afirma. Se arregla junto con el hallazgo 8.

10. **[menor, de ensamblado] "(Apéndice)" todavía no tiene destino.** L161 promete el
    código en el apéndice, pero `examen-dma-2026.Rmd` solo tiene A.1 y A.2 con los
    scripts del Ejercicio 1. Hay que agregar `adaboost_vs_gb.py` y `figuras.py`. Le pasa
    lo mismo a mi Ejercicio 7; es tarea del ensamblado, pero si se olvida queda una
    promesa colgada.

## Lo que está bien y no hay que tocar

- **La verificación de Schapire.** Es el mejor movimiento del ejercicio: agarra un número
  de las diapositivas, lo recalcula, encuentra una brecha, hace el despeje inverso y
  concluye que es redondeo. Demuestra comprensión del algoritmo sin decir "comprendo el
  algoritmo", y no acusa a la cátedra. Dejarlo tal cual.
- **El resultado negativo de las curvas de error.** L233-242 contradice al material de
  clase con evidencia propia ("Las diapositivas avisan... La segunda no") y después
  muestra bajo qué condición el aviso sí se cumple. Exactamente lo que pide el BRIEF, y
  la Figura 2 lo sostiene visualmente.
- **La Figura 1.** Los dos paneles normalizados a su valor inicial son la respuesta
  gráfica a "qué diferencias existen": izquierda el peso se mueve y el target no,
  derecha el target se achica y el peso no. Una figura que argumenta.
- **El cotejo contra `AdaBoostClassifier`, incluida la diferencia de convención.** L186-193
  explica el factor 2 entre el alpha del apunte y el de sklearn y demuestra que no altera
  ni el signo del voto ni $D_t$ renormalizada. Lo verifiqué a mano y es correcto: con
  $\alpha_{sk} = 2\alpha_{ap}$, la razón entre el peso de una mal clasificada y el de una
  bien clasificada es $e^{L}$ en las dos convenciones, así que las distribuciones
  coinciden después de normalizar. La fila 1 de la Tabla 2 (8,33e-17) lo confirma.
- **Las declaraciones de límite.** L123 (la unificación AdaBoost / GB con pérdida
  exponencial queda fuera porque no está en el material), L205-206 (no se reconstruyeron
  las coordenadas de los diez puntos porque las slides no las dan), L259 ("acompaña en
  parte la estimación de Friedman"). No hay exceso de pulido: hay tres limitaciones
  admitidas y un resultado que contradice a la cátedra.
- **Las tres claves verificadas contra las diapositivas.** Confirmé en el PDF de la clase
  6 que dicen lo que el texto les atribuye: "Friedman estimo que valores de v<0.1 generan
  los menores errores de generalización", "Usando NO=0.5 N, es equivalente a muestras
  boostrap", "Los tres parámetros fundamentales son: Número de Boosting o iteraciones /
  Parámetro Shrincage / Proporción de entrenamiento", y "El error de test aumenta cuando
  H final se torna demasiado compleja, pudiendo haber overfitting". Las cuatro atribuciones
  son fieles.
