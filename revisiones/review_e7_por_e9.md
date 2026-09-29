# Revisión del Ejercicio 7

Revisor: autor del Ejercicio 9. Material revisado: `staging/e7.Rmd` (354 líneas),
`staging/e7_referencias.md`, `codigo/e7_tsne/tsne_clustering.py`,
`codigo/resultados/e7/`, contra `clase7/Preparation for Clustering.pdf`,
`clase7/Assessing Clustering Results.pdf` y
`clase8/TSNE_Perplejidad_Clustering_NoLineal_Volpacchio.pdf`.

## Puntaje por criterio

| Criterio | Resultado | Evidencia |
|---|---|---|
| 1. Inconsistencias entre secciones | **no cumple** | L275-285: el texto concluye que el clustering sobre el embedding es "el camino equivocado" con un ARI de 0,766 impreso en la misma oración contra 0,534 del espacio original. L142-143: "el pseudo-*F* decrece en forma monótona desde $k = 2$", falso en `pseudo_f.csv` (k=14: 105,48; k=15: 107,08) |
| 2. Terminología sin comprensión demostrable | **cumple** | Glosa ARI (L70), MANOVA (L147), $\eta^2$ (L151), CCC (L132), pseudo-*T²* (L135), aglomeración (L211), confiabilidad (L258), perplejidad (L219-221). No encontré término técnico sin definir |
| 3. Código sin comentarios personales | **cumple** | `tsne_clustering.py` L28-38 registra tres pruebas fallidas (t-SNE sin estandarizar, MDS sobre 1797 puntos, `MinPts = 2*d`). L227-231 explica por qué se cambió la regla de selección de DBSCAN. L111-112 y L208-210 justifican decisiones, no describen líneas |
| 4. Referencias irrelevantes o inexistentes | **parcial** | Las 8 citas del texto están en `e7_referencias.md` y las 8 entradas se usan. Pero `t_sne.pdf` aparece en el bloque de Fuentes (L350) sin cita en el texto ni entrada en la lista. Las letras de SAS Institute están invertidas respecto del orden alfabético por título que pide APA |
| 5. Estilo inconsistente | **cumple** | 0 guiones largos, 0 guiones dobles, 0 muletillas de la lista, 0 primera persona, 0 gerundios fuera del enunciado citado, 0 `PENDIENTE/TODO`. Registro indistinguible de los Ejercicios 1 a 3 |

**Reproducción del código: cumple.** Corrí `python codigo/e7_tsne/tsne_clustering.py`
de nuevo (exit 0) contra una copia previa de `codigo/resultados/e7/`. Diez de los once
CSV y los tres PNG son idénticos byte a byte. El único distinto es `pseudo_f.csv`, en
dos dígitos finales de punto flotante (`81678.46669890721` contra `81678.4666989072`);
es ruido de coma flotante, no un resultado distinto.

## Cobertura del enunciado

| Pregunta de la consigna | ¿Dónde se responde? |
|---|---|
| Mencione las técnicas de pre-proceso para visualizar la potencial existencia de grupos | L75-80 (VARCLUS, STDIZE, ACECLUS, GPLOT) y Tabla 1 (PRINCOMP, MDS, CANDISC). **Completo** |
| Explique brevemente 1 de ellas | L107-114, componentes principales, con la justificación de por qué se eligió esa y no las otras dos. **Completo** |
| ¿De qué manera puede estimar qué diferencias existen en los perfiles de los grupos obtenidos? | L147-152: medias por grupo, MANOVA, ANOVA más discriminante canónico, poda. **Parcial**: falta la segunda variante de perfilado del capítulo, ver hallazgo 6 |
| Explicar los resultados | L170-193, Tabla 2, más los dos controles. **Completo y es lo mejor de la sección** |
| Ayuda: usar el último capítulo (Assessing Clustering) | Citado como SAS Institute (s.f.-b) en L130, L344-347. **Completo** |
| ¿En qué consiste el algoritmo t-SNE? | L197-227. **Completo** |
| ¿Para qué se utiliza? | L229-236. **Completo** |
| Ejemplo de implementación en Python | L238-245. **Completo**, y el fragmento coincide exactamente con la llamada real del script (L106-107) |
| ¿Qué limitantes tiene? | L253-267. **Completo** |
| ¿Qué ventajas tiene? | L269-273. **Completo** |

No falta ninguna parte del enunciado. La única laguna es la variante 2 del perfilado.

## Hallazgos, del más grave al más leve

### 1. [bloqueante] La conclusión del Punto III está refutada por un número del mismo párrafo

L275-285 dice:

> k-means sobre las 61 variables da una silueta de 0,139 y un ARI de 0,534 contra el
> dígito real; sobre el embedding, 0,573, 4,1 veces más, y un ARI de 0,766. Ahí está la
> trampa: (...) El índice interno, único criterio disponible sin etiquetas, señala el
> camino equivocado.

El ARI es el juez externo que el propio texto montó en L70 ("El dígito queda fuera del
modelo, como juez externo"). Ese juez dice que la partición del embedding se parece
**más** al dígito real: 0,766 contra 0,534. En la Tabla 3 pasa lo mismo y más fuerte con
DBSCAN: 0,848 sobre el mapa contra 0,001 en el espacio original. El texto no puede
llamar "camino equivocado" a la opción que su propia validación externa prefiere, y en
un oral esa es la primera pregunta.

El texto sí blanquea la tensión para DBSCAN (L336-338, "Queda una tensión con la
recomendación de modelar sobre el espacio original") y la pasa por alto para k-means,
que es donde el contraste es el argumento central.

**Corrección concreta.** El argumento defendible es más angosto y está a mano en el
mismo párrafo: la silueta no es comparable entre espacios, porque la misma partición
evaluada en el espacio original cae a 0,110 (L283), por debajo de la nativa de 0,139.
Eso descalifica a la silueta como criterio para **elegir el espacio**, que es lo que hay
que decir. Lo demás son dos razones independientes y sólidas: no hay `transform`, así
que la partición del mapa no se puede aplicar a datos nuevos, y las coordenadas no
tienen interpretación métrica. Y el ARI hay que reportarlo como lo que es: en este
conjunto la partición del embedding recupera mejor el dígito, lo cual va contra la
recomendación del apunte y merece quedar escrito. El BRIEF premia exactamente eso.

### 2. [bloqueante] Excede el límite de 4 hojas

Compilé la sección sola con el YAML y el chunk de setup de `examen-dma-2026.Rmd`, sin
portada ni índice, para que las páginas sean de cuerpo puro: **da 5 páginas**
(453 / 445 / 586 / 454 / 519 palabras). La consigna dice "no más de 4 hojas por
pregunta". Hay que sacar aproximadamente una página.

Dónde sobra, por orden de rendimiento: la Figura 1 a `out.width="100%"` con su Nota
ocupa cerca de un quinto de página y es ilegible igual (hallazgo 7); el párrafo L188-193
repite en prosa lo que ya dijeron los dos controles; el bloque de Fuentes (L344-354) son
11 líneas contra 5 o 6 en los Ejercicios 2 y 3.

### 3. [importante] Dos afirmaciones sobre el barrido de $k$ que los datos no sostienen

L142-144: "el pseudo-*F* decrece en forma monótona desde $k = 2$ y la silueta crece
hasta el extremo, $k = 15$".

En `pseudo_f.csv` el pseudo-*F* sube en el último paso: k=14 da 105,48 y k=15 da 107,08.
No es monótono. La silueta tampoco crece: 0,1056 / 0,1053 / 0,0954 / 0,1023 / 0,0988 y
recién después sube, con otra caída en k=14. Lo que sí es cierto es que **no hay pico
interior** y que el **máximo** de la silueta está en el extremo.

Agrava el hallazgo que `meta.csv` ya trae el dato correcto y más débil,
`pseudo_f_pico_interior = False`, y el `.Rmd` no lo lee. Además `pseudo_f.csv` y
`pseudo_f.png` se generan y no se usan en ningún lado, así que el lector no puede
comprobar la afirmación.

**Corrección.** "el pseudo-*F* no presenta el pico interior que el capítulo manda buscar
(`r e7meta$pseudo_f_pico_interior`) y el máximo de la silueta cae en el extremo del
barrido, $k = `r e7meta$silhouette_k_max`$".

### 4. [importante] La lectura de los controles es más fuerte que los controles

L188-190: "lo que dispara la significación no es la existencia de grupos sino el haberlos
construido".

Los números del párrafo anterior dicen otra cosa: sobre datos uniformes sin estructura,
k-means deja **5** variables significativas y una *F* mediana de **1,04**; sobre los
dígitos reales, **56** y **134,5**. O sea que la prueba discrimina bien en la masa de la
distribución y solo se rompe en la cola, donde el *F* máximo (545 contra 594) y el
$\eta^2$ máximo (0,73 contra 0,75) sí son indistinguibles.

La conclusión honesta, y sigue siendo fuerte: construir la partición alcanza para
producir **unas pocas** variables con significación y tamaño de efecto grandes aun sin
estructura, así que un $\eta^2$ alto aislado no prueba nada; lo que distingue el caso
real del artificial es la masa, no el extremo.

### 5. [importante] Describe Tukey y corre Bonferroni, sin declarar el cambio

L149: "un ANOVA por variable con corrección de Tukey". L172: "Con el umbral de
Bonferroni". El script no implementa Tukey en ningún lado; solo calcula
`alfa_bonferroni`.

El capítulo 5 usa Tukey (`means cluster / tukey;`) y además anticipa el reemplazo: "A
Bonferroni correction could also have been used. However, the Bonferroni correction is
considered to be more conservative than the Tukey adjustment". Es decir, la sustitución
es legítima y la justificación está servida en la fuente, pero el texto no la declara y
deja al lector con dos correcciones distintas en dos párrafos seguidos. Es justo el
gesto 2 del BRIEF, justificar una elección contra la alternativa descartada, y está sin
usar.

### 6. [importante] Falta la segunda variante de perfilado del capítulo que indicó el profesor

El capítulo 5 abre la sección con "At least two forms of cluster profiling exist:
1. Comparing the clusters means 2. Comparing the clusters means with respect to a
derived (or given) class", y cierra con "The second variant is similar to the method
outlined above, except that the centroids are assessed with respect to a (given or
derived) class".

L147-152 desarrolla solo la variante 1. La variante 2 no aparece. Pesa porque el
profesor señaló ese capítulo por su nombre en la propia consigna, y porque la corrida ya
tiene la clase derivada (el dígito) y ya la usa para el ARI: es una oración de trabajo,
no un experimento nuevo.

### 7. [importante] La Figura 1 es ilegible y es la única evidencia de la conclusión del Punto I

Cinco paneles en 17 cm de ancho: los títulos de panel quedan en cuerpo mínimo y las
nubes de puntos no se distinguen entre sí. Sobre esa figura se apoyan las tres
afirmaciones de L123-126 ("Las diez clases quedan superpuestas bajo componentes
principales y bajo escalado multidimensional", "Solo t-SNE muestra unas diez islas") y
la de L341-342 ("sus etiquetas sobre el mapa coinciden con las islas salvo en dos grupos
que las cortan"). Ninguna se puede verificar mirando la figura impresa.

**Corrección.** Dos filas de paneles más grandes, o bajar a los tres paneles que sostienen
el argumento (PCA, CDA, t-SNE) y dejar MDS descrito en la Tabla 1. Resuelve además parte
del hallazgo 2.

### 8. [importante] Endurece lo que dice el apunte sobre dónde clusterizar

El apunte de Volpacchio dice: "Sobre la representación proyectada, **o preferentemente
sobre el espacio original**, pueden aplicarse algoritmos de clustering como K-means,
DBSCAN o HDBSCAN". Es una preferencia, no una prohibición.

L232-236 lo convierte en regla absoluta ("modelar con un algoritmo de clustering sobre
el espacio original"). Por sí solo sería menor; combinado con el hallazgo 1, no lo es:
el texto endurece la fuente justo en el punto donde su propia evidencia va en contra.
Corresponde citar el "preferentemente" tal cual y discutirlo.

### 9. [menor] Coma decimal dentro de modo matemático: se imprime "0, 749"

L172: `$\eta^2 = `r fmt_e7(e7med$eta2[1], 3)`$` produce `$\eta^2 = 0,749$`, y LaTeX trata
la coma en modo matemático como separador y le mete espacio: en el PDF se lee
**"η² = 0, 749"**. El propio autor usa el remedio correcto en L226,
`$\theta \approx 0{,}5$`. Es la única ocurrencia; L143-145 interpolan enteros y no sufren
el problema.

**Corrección.** Sacar el número del modo matemático: `$\eta^2$ = 0,749`, o envolver la
coma en llaves.

### 10. [menor] "siete de los diez" está tipeado a mano y es frágil

L171: "cae a cero en siete de los diez". En `perfil_medias.csv` el píxel `p7,7` vale
exactamente 0 en cinco grupos (G1, G2, G4, G7, G9); los otros dos que el texto cuenta
(G0 = 0,008 y G8 = 0,045) redondean a 0,0 con el decimal de la Tabla 2. La afirmación
es defendible pero depende del redondeo y no sale del CSV. Es el único número tipeado a
mano que encontré en toda la sección; el resto se interpola.

### 11. [menor] Números calculados que no llegan al texto, y punto decimal dentro de las figuras

`meta.csv` trae `var_pca2 = 0,2160` y el Punto I, que eligió componentes principales
justamente como la técnica a explicar, nunca dice cuánta varianza retiene el par de
componentes, pese a cerrar con "la varianza que retiene mide cuánta información queda
afuera" (L113-114). El 21,6 % existe, pero solo dentro del PNG y escrito **"21.6 %"**,
con punto decimal, igual que "KL = 0.98" y "Trustworthiness = 0.978" en la Figura 2. En
prosa y tablas el trabajo usa coma en todos lados.

Lo mismo con `tsne_tiene_transform = False`, que está en `meta.csv` y el texto afirma en
prosa (L263-265) en vez de interpolar, y con `alfa` / `alfa_bonferroni`, disponibles
mientras L172 escribe "0,05" a mano.

### 12. [menor] Indexado posicional donde el propio autor se cuidó de no usarlo

L56-59 del `.Rmd` trae un comentario ejemplar: las tablas de DBSCAN y de concentración
se unen por nombre "para que el orden de las filas no pueda desfasarse". Pero
`e7ctrl$F_mediana[2]`, `e7ctrl$n_signif[3]` (L181-186), `e7med[1:5, ]` (L50) y
`e7perp$kl_final[1]` / `[3]` (L255-256) dependen del orden de filas. Hoy funcionan
porque el script ordena por *F* descendente y escribe los escenarios en ese orden; si
alguien toca el script, el texto miente sin fallar. Filtrar por nombre, como ya se hace
con DBSCAN.

### 13. [menor] `t_sne.pdf` citado en Fuentes, ausente de las Referencias

L350: "con el complemento `t_sne.pdf` de la clase 7". El archivo existe en
`clase7/`, pero no se lo cita en ninguna parte del texto y no tiene entrada en
`e7_referencias.md`. La consigna exige detallar la fuente de cada respuesta y el criterio
4 del profesor es "referencias inexistentes": o se lo cita en el cuerpo y se le arma
entrada, o se saca la mención.

### 14. [menor] Letras de SAS Institute invertidas respecto de APA

APA asigna las letras por orden alfabético del título entre obras del mismo autor y año.
"Assessing clustering results" precede a "Preparation for clustering", así que
*Assessing* debería ser `s.f.-a` y *Preparation* `s.f.-b`. Están al revés en el texto y
en `e7_referencias.md`. El propio archivo de referencias exige ese criterio para
Volpacchio (L51-53) y no lo aplica acá.

### 15. [menor] Dos cosas para el ensamblado, no son culpa del autor

- `e7_referencias.md` L50-53 pide renumerar **todas** las letras de Volpacchio desde `a`
  por título. Eso pisa la letra `s.f.-a` que el Ejercicio 9 tiene asignada para
  `Ensemble Methods Universidad Austral`. Hay que renumerar una sola vez, al final, en
  los cinco ejercicios a la vez.
- El nombre del apunte en L348 está truncado: el archivo es
  `TSNE_Perplejidad_Clustering_NoLineal_Volpacchio.pdf`.
- No pude verificar el DOI `10.24432/C50P49` de Alpaydin y Kaynak. El formato es el de
  UCI y probablemente esté bien, pero conviene abrirlo antes de entregar; ante la duda,
  la entrada funciona sin DOI.

## Lo que está bien y no hay que tocar

- **La advertencia sobre MANOVA y ANOVA.** El capítulo dice "The results of MANOVA or
  ANOVA should be interpreted with caution. The process of clustering violates key
  assumptions underlying these tests" y L178-179 lo traduce sin ablandarlo. Pero lo
  importante es que no se queda ahí: monta dos controles para medir la advertencia en
  vez de repetirla. Es lo mejor de la sección y es exactamente lo que el profesor no
  espera encontrar en un texto generado. No lo toquen al arreglar el hallazgo 4, que es
  solo la oración de cierre.
- **No se clusteriza sobre el embedding como resultado válido.** El punto que pedía
  vigilarse está cubierto: el texto argumenta explícitamente en contra y da un argumento
  técnicamente correcto (la silueta 0,573 cae a 0,110 al evaluar la misma partición en
  el espacio original). El problema del hallazgo 1 es de honestidad con el ARI, no de
  haber caído en la trampa.
- **La reproducción.** El script corre limpio y devuelve los mismos CSV.
- **Los comentarios del script**, sobre todo el bloque de L28-38 con tres pruebas
  fallidas y la regla de DBSCAN reescrita en L227-231.
- **"Lo que falló"** (L287-292): reporta una expectativa que no se cumplió y ajusta el
  argumento en consecuencia, en lugar de esconderla.
- **La fidelidad al capítulo 2**: la limitante de CANDISC ("CDA requires a classification
  variable. This can limit...") está tomada literal del capítulo, y la descripción de PCA
  parafrasea sin copiar.
- **El fragmento de Python** coincide exactamente con la llamada real del script,
  incluido `max_iter`, que es el nombre correcto del parámetro en scikit-learn 1.7.

## Prioridad sugerida

Los hallazgos 1 y 2 son de entrega. El 1 se arregla con un párrafo reescrito y mejora el
trabajo, porque convierte un argumento débil en un hallazgo declarado. El 7 resuelve
parte del 2. Los hallazgos 3, 4 y 5 son tres frases. El 6 es una oración.
