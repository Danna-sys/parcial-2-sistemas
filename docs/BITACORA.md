# Bitácora

Aquí queda el camino del proyecto en el orden en que ocurrió. Cada entrada separa lo que se observó de lo que se interpreta y dice de qué figura o tabla sale.

```text
observación → pregunta → hipótesis → experimento controlado → resultado → argumento
```

## 1. Observaciones del EDA

Fuente: `notebooks/01_eda.ipynb`.

| # | Observación | Evidencia |
|---|---|---|
| 1 | Hay 2543 imágenes en 11 clases. 2332 miden 400×400 y 211 miden 256×256. Las de 256 están solo en Cc, Cs y Ns. | sección 1, tabla de tamaños |
| 2 | Sc es la clase más grande con 340 imágenes y Ci la más pequeña con 139. Predecir siempre Sc da 13.4 % de accuracy en validation. | sección 2, conteos |
| 3 | En validation una imagen vale 0.26 puntos de accuracy y hasta 4.5 puntos de recall en Ci. | sección 2 |
| 4 | En 106 grupos la misma foto aparece con etiquetas distintas. Son 221 imágenes, el 8.7 % del dataset. El par más común es Ns/St con 18 grupos. | sección 3, tabla de conflictos |
| 5 | El detector no agrupó algunas copias que cambian un poco de color. Unas 6 imágenes de validation y 7 de test tienen una copia en train, casi siempre con otra etiqueta. | sección 3, pares más parecidos |
| 6 | El brillo de arriba menos el de abajo es alto en Sc, St, Ns y Cb y negativo en Ci, Cc y Ct. | sección 4, boxplots y medianas |
| 7 | Un k vecinos con seis rasgos globales logra 24.3 % en validation y con píxeles de 16×16 logra 25.7 %. | sección 5 |
| 8 | Los pares que más confunde el k vecinos son Cb/Sc, Ac/Cc, As/Cs y Cc/Cs. | sección 5, matriz de confusión |
| 9 | Al reducir a 32 píxeles, Ac, Ns, Ct y Cc son las clases que más detalle pierden. | sección 6 |

Decisiones que salen del EDA:

- **Augmentation que conserva el significado de la imagen:** volteo horizontal, recorte moderado y cambios suaves de brillo y contraste. El volteo vertical no la conserva porque deja el suelo arriba. Por eso lo usamos como variable del experimento y no como augmentation normal.
- **Clases que se parecen antes de entrenar:** Ac/Cc, Cc/Cs, Cb/Sc y Ns/St. Las dos últimas también comparten fotos repetidas.
- **Resolución mínima a la que las clases siguen siendo distinguibles a ojo:** 64 píxeles. A 32 se pierde la textura de Ac y Cc.

## 2. Observaciones de la línea base

Fuente: `notebooks/02_linea_base.ipynb`.

| # | Observación | Evidencia |
|---|---|---|
| 1 | En el piloto de 5 épocas la pérdida de validation seguía bajando al final. 5 épocas no alcanzan para que el modelo se estabilice. | sección 2, curva del piloto |
| 2 | Con 30 épocas la mejor época fue la 26. La accuracy de validation llegó a 0.414 y el gap entre train y validation fue 0.083. | sección 3 |
| 3 | El recall es muy desigual entre clases. Es alto en Sc (0.686), Cc (0.651) y Ct (0.633) y bajo en Cs (0.119), Ac (0.121) y St (0.185). | sección 4, tabla por clase |
| 4 | La confusión más fuerte es Ac predicha como Cc: 15 imágenes, el 45 % de Ac. Le siguen Ns/Sc con 15 errores y Cc/Cs con 14. | sección 4 |
| 5 | Los errores de la CNN se parecen a los del k vecinos, con una correlación de Spearman de 0.549 entre pares de clases. | sección 5 |
| 6 | Los errores se quedan dentro del mismo nivel de altura en el 28.6 % de los casos. Al azar sería 21.8 %. | sección 4, errores por familia |

Resultado de la línea base, una semilla y 30 épocas:

| config | seeds | val_macro_f1 | val_acc | train_acc | gap | best_epoch |
|---|---|---|---|---|---|---|
| base | 1 | 0.378 | 0.414 | 0.496 | 0.083 | 26 |

## 3. Pregunta del proyecto

¿La CNN aprende a reconocer el tipo de nube o se apoya en su posición respecto al horizonte?

De qué observación sale: en el EDA las nubes bajas aparecen con el horizonte en la parte baja de la foto y las altas no (observación 6 del EDA). Si la red aprende eso, podría estar clasificando por el encuadre de la foto y no por la forma de la nube.

## 4. Experimentos

Hipótesis general: la CNN utiliza la posición de las nubes respecto al horizonte como una pista para clasificarlas. Por ello, esperamos que el volteo vertical reduzca más el recall de las clases de nubes bajas que el de las nubes altas. Asimismo, esperamos que una cabeza de clasificación que conserve información espacial (head_grid = 2) favorezca el reconocimiento de las nubes bajas frente a una cabeza que no conserva esa distribución espacial (head_grid = 1).

Grupos que se comparan:

- **Nubes bajas:** Sc, St, Ns y Cb. Son las clases que en el EDA aparecen con el horizonte abajo.
- **Nubes altas:** Ci, Cc, Cs y Ct. Son las que se fotografían mirando hacia arriba.
- Ac, As y Cu no entran en la comparación porque su diferencia de brillo entre arriba y abajo es intermedia.

### Experimento 1: volteo vertical

| Campo | Contenido |
|---|---|
| Fecha | 18 de octubre de 2026 |
| Observación de partida | observación 6 del EDA |
| Hipótesis | si la red usa el horizonte, voltear las imágenes de arriba a abajo le quita esa pista y le afecta más a las nubes bajas |
| Qué espero ver si es cierta | con `aug_flip_hv` el recall de las nubes bajas baja más que el de las altas, y la diferencia es mayor que la desviación entre semillas |
| Qué vería si no lo es | el recall de los dos grupos cambia parecido, o la diferencia queda dentro del ruido entre semillas |
| Variable que cambia | `augment`: `flip` contra `flip_hv`. Las dos voltean de izquierda a derecha; solo `flip_hv` voltea también de arriba a abajo |
| Qué queda fijo | todo lo demás de `configs/base.toml` |
| Configuraciones | `aug_flip` (referencia) y `aug_flip_hv` |
| Semillas | 11, 22, 33 |
| Épocas | 5 |

Resultado:

| config | seeds | val_macro_f1 | val_acc | train_acc | gap |
|---|---|---|---|---|---|
| aug_flip | 3 | 0.183 ± 0.011 | 0.258 ± 0.008 | 0.273 ± 0.009 | 0.015 ± 0.016 |
| aug_flip_hv | 3 | 0.175 ± 0.001 | 0.250 ± 0.006 | 0.271 ± 0.007 | 0.021 ± 0.010 |

Cambio de recall con `aug_flip_hv` frente a `aug_flip`, promedio por grupo:

| grupo | cambio de recall |
|---|---|
| bajas | −0.004 |
| altas | −0.020 |

| Campo | Contenido |
|---|---|
| Qué se observó | Con el volteo vertical el macro F1 bajó de 0.183 a 0.175 y la accuracy de 0.258 a 0.250. Las dos diferencias son menores que la desviación entre semillas. El recall de las nubes bajas cambió −0.004 y el de las altas −0.020. Con 5 épocas la red casi no reconoce Ns (0.016), St (0.012) ni Ci (0.000) en ninguna de las dos configuraciones. |
| Qué se interpreta | El volteo vertical no tuvo un efecto que podamos distinguir del ruido. Tampoco afectó más a las nubes bajas: las altas bajaron un poco más. |
| Veredicto | no se puede decidir |
| Qué limita la conclusión | Con 5 épocas la red todavía no aprende varias clases bajas. Si una clase tiene recall cercano a 0, el volteo no puede bajarlo más, así que el experimento no alcanza a medir si la red usa el horizonte. |
| Qué pregunta nueva deja | ¿Aparece el efecto cuando la red se entrena hasta que reconoce las nubes bajas, por ejemplo con 30 épocas? |

### Experimento 2: cabeza que conserva la posición

| Campo | Contenido |
|---|---|
| Fecha | 18 de octubre de 2026 |
| Observación de partida | observación 6 del EDA |
| Hipótesis | si la red usa el horizonte, una cabeza que sabe qué estaba arriba y qué abajo le ayuda más a las nubes bajas |
| Qué espero ver si es cierta | con `head_grid = 2` el recall de las nubes bajas sube más que el de las altas, y la diferencia es mayor que la desviación entre semillas |
| Qué vería si no lo es | el recall de los dos grupos cambia parecido, o la diferencia queda dentro del ruido entre semillas |
| Variable que cambia | `head_grid`: 1 contra 2 |
| Qué queda fijo | todo lo demás de `configs/base.toml` |
| Configuraciones | `grid1` (referencia) y `grid2` |
| Semillas | 11, 22, 33 |
| Épocas | 5 |

Resultado:

| config | seeds | val_macro_f1 | val_acc | train_acc | gap |
|---|---|---|---|---|---|
| grid1 | 3 | 0.174 ± 0.016 | 0.247 ± 0.028 | 0.274 ± 0.005 | 0.027 ± 0.023 |
| grid2 | 3 | 0.211 ± 0.010 | 0.269 ± 0.009 | 0.309 ± 0.008 | 0.041 ± 0.006 |

Cambio de recall con `grid2` frente a `grid1`, promedio por grupo:

| grupo | cambio de recall |
|---|---|
| bajas | +0.037 |
| altas | +0.058 |

| Campo | Contenido |
|---|---|
| Qué se observó | Con `head_grid = 2` el macro F1 subió de 0.174 a 0.211. Esa diferencia es mayor que la desviación entre semillas. El recall subió +0.037 en las nubes bajas y +0.058 en las altas. Dentro de las bajas subieron Cb (+0.135) y Ns (+0.079) pero bajó Sc (−0.078). La clase que más mejoró fue Ct (+0.289), que es alta. |
| Qué se interpreta | Conservar la posición sí ayuda a la red, pero no ayuda más a las nubes bajas. La mejora más grande fue en las estelas, que tienen una forma alargada y una posición en el cielo muy marcada. |
| Veredicto | no se sostiene |
| Qué limita la conclusión | `grid2` tiene 521 035 parámetros y `grid1` 422 731, así que parte de la mejora puede venir de tener una cabeza más grande y no solo de conservar la posición. Además son 5 épocas. |
| Qué pregunta nueva deja | ¿La mejora se mantiene si se compara con una cabeza de 1x1 que tenga el mismo número de parámetros? |

Nota sobre las épocas: la línea base se entrenó 30 épocas con una semilla. Por tiempo de cómputo, los experimentos se entrenaron 5 épocas con tres semillas. Cada experimento se compara solo contra su propia referencia de 5 épocas. Con 5 épocas la red todavía está aprendiendo, así que estos resultados muestran una tendencia y no el efecto final.

## 5. Evaluación final en test

| Campo | Contenido |
|---|---|
| Configuración elegida con validation | `grid2` del experimento de la cabeza |
| Por qué esa | tuvo el mayor macro F1 de validation entre todas las configuraciones de los experimentos, 0.211 ± 0.010 |
| Fecha de la evaluación | 18 de octubre de 2026 |

Se evaluó una sola vez el mejor checkpoint de cada una de las tres semillas.

| seed | val_acc | test_acc | val_macro_f1 | test_macro_f1 |
|---|---|---|---|---|
| 11 | 0.278 | 0.309 | 0.209 | 0.220 |
| 22 | 0.259 | 0.298 | 0.202 | 0.235 |
| 33 | 0.270 | 0.293 | 0.222 | 0.227 |
| media | 0.269 | 0.300 | 0.211 | 0.227 |

| Campo | Contenido |
|---|---|
| Qué se observó | En test la accuracy media fue 0.300 y el macro F1 0.227, un poco por encima de validation (0.269 y 0.211). La diferencia en accuracy equivale a unas 12 imágenes de 382. Igual que en validation, la red no reconoce Ci (recall 0.000) ni St (0.011) y reconoce mejor Ct (0.720) y Cb (0.675). |
| Qué se interpreta | El resultado de validation no era optimista: el modelo se comporta parecido con imágenes que nunca se usaron para decidir nada. La diferencia entre validation y test es del tamaño de la variación entre semillas, así que no indica nada más. |
| Qué limita la conclusión | Es un modelo de 5 épocas, así que este número describe una red que todavía está aprendiendo y no el mejor resultado posible de la arquitectura. Test tiene unas 7 imágenes con una copia en train, casi siempre con otra etiqueta. |

## 6. Qué se aprendió

**Sobre el problema:** Clasificar nubes en 11 géneros es difícil incluso antes de entrenar. Varias clases se parecen mucho, como Ac y Cc o Ns y St, y la diferencia entre ellas está en la textura y el tamaño de las nubes más que en el color o el brillo. Un k vecinos con rasgos globales solo llegó a 24.3 % y la CNN a 41.4 % después de 30 épocas, así que la red sí aprende algo que los rasgos simples no capturan, pero sigue lejos de separar bien todas las clases.

**Sobre el dato:** El dataset tiene problemas que afectan cualquier resultado. En 106 grupos la misma foto aparece con etiquetas distintas, lo que equivale al 8.7 % de las imágenes, y algunas copias con cambios de color quedaron repartidas entre train, validation y test. Además, la forma en que se tomaron las fotos cambia según la clase: las nubes bajas aparecen con el horizonte y las altas mirando al cielo. Esto le da a la red una pista que no tiene que ver con la forma de la nube.

**Sobre la CNN:** Con 5 épocas la red todavía no reconoce varias clases, como Ns, St y Ci, y eso impidió medir el efecto del volteo vertical. Conservar la posición con `head_grid = 2` mejoró el macro F1 de 0.174 a 0.211, pero la mejora fue mayor en las nubes altas, sobre todo en las estelas, y no en las bajas como esperábamos. Con tres semillas pudimos ver que varias diferencias quedaban dentro del ruido, algo que con una sola semilla habríamos tomado como un efecto real.

**Qué haría distinto:** Repetiríamos los dos experimentos con 30 épocas, para que la red alcance a reconocer las nubes bajas antes de medir el efecto del volteo. Compararíamos `head_grid = 2` con una cabeza de 1x1 que tenga el mismo número de parámetros, para separar el efecto de la posición del efecto de tener una cabeza más grande. También mejoraríamos la detección de duplicados para que encuentre copias con cambios de color antes de hacer el split.
