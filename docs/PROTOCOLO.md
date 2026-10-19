# Protocolo

Decisiones que quedan fijas para todas las corridas. Se toman antes de comparar configuraciones y no se cambian después de ver resultados. Si alguna cambia se anota aquí con fecha y motivo y se repiten las corridas afectadas.

## Datos

| Decisión | Valor | Motivo |
|---|---|---|
| Split | 70 / 15 / 15 estratificado por clase | mantiene la proporción de cada género en los tres conjuntos |
| `split_seed` | 2026 | fija el split para todo el proyecto |
| Duplicados | cada grupo de copias va completo a un solo split | evita medir en validation fotografías que la red ya vio |
| Normalización | media y desviación por canal de train | usar validation o test filtraría información al preprocesamiento |
| Exploración | figuras y rasgos con train | validation solo donde se mide algo y test solo se cuenta |

## Entrenamiento

| Decisión | Valor | Motivo |
|---|---|---|
| Pérdida | cross entropy sobre logits | función de pérdida adecuada para clasificación multiclase |
| Optimizador | AdamW con `lr = 1e-3` y `weight_decay = 0` | punto de partida estándar. Cambia solo si es la variable del experimento |
| Scheduler y early stopping | ninguno | mezclarlos con la variable estudiada impediría atribuir el efecto |
| Épocas | 30 | en la corrida de línea base con semilla 11, el mínimo de `val_loss` se alcanzó en la época 26. Se conserva un presupuesto inicial común de 30 épocas para comparar configuraciones |
| Checkpoint | época con menor `val_loss` | criterio fijado para seleccionar el modelo |
| Semillas | 11, 22 y 33 | tres corridas por configuración para medir la dispersión |
| Determinismo | `deterministic = true` y `num_workers = 0` | favorece la reproducibilidad de las corridas |

## Evaluación

| Decisión | Valor | Motivo |
|---|---|---|
| Métrica primaria | macro F1 de validation en el mejor checkpoint, media entre semillas | pondera igual a las 11 clases aunque no tengan el mismo tamaño |
| Métricas secundarias | accuracy, recall por clase, matriz de confusión, generalization gap | una métrica global puede esconder lo que pasa en una clase |
| Generalization gap | accuracy de train en modo eval menos accuracy de validation | compara los dos conjuntos en las mismas condiciones |
| Diferencia interpretable | mayor que la desviación entre semillas de la referencia | permite valorar si la mejora supera la variabilidad de las corridas |
| Test | una sola vez, al final, para la configuración elegida con validation | test no es una segunda validación |

## Piloto

Los resultados disponibles corresponden a la corrida de línea base con semilla 11 y no al promedio entre las tres semillas.

| Dato | Valor |
|---|---|
| Época del mínimo de `val_loss` | 26 |
| `val_loss` y `val_acc` en esa época | 1.790 y 0.414 |
| Comportamiento después del mínimo | la pérdida de validación fluctúa y aumenta ligeramente al final, mientras la pérdida de entrenamiento sigue disminuyendo |
| Épocas elegidas para todas las corridas | 30 como presupuesto inicial común |
| Tiempo por época y equipo usado | 222 segundos en promedio, en CPU y sin GPU |

## Observaciones de la línea base

La corrida con semilla 11 obtuvo una accuracy de validación de 0.4136 y un generalization gap de 0.0827. El rendimiento varió entre clases: Cc, Ct y Cb presentaron mejores valores de recall que Ac, Cs y St. La matriz de confusión mostró errores frecuentes entre algunas clases, como Ac/Cc y Ns/Sc.

Estos resultados sirven como referencia para describir el comportamiento de la red. La línea base se corrió con una sola semilla, así que no conocemos su variabilidad. La variabilidad entre semillas se midió en los experimentos.

## Hipótesis y experimento: el horizonte como pista

**Pregunta:** ¿la red reconoce la nube o el ángulo de la cámara?

**Hipótesis:** el volteo vertical reduce el recall de las nubes bajas, que dependen de la posición del horizonte, más que el de las altas. Una cabeza que conserva parte de la información espacial, con `head_grid = 2`, podría mejorar el reconocimiento de las nubes bajas frente a `head_grid = 1`.

**Experimentos previstos:**

- comparar `aug_flip` contra `aug_flip_hv` para estudiar el efecto del volteo vertical
- comparar `head_grid = 1` contra `head_grid = 2` para estudiar el efecto de conservar información espacial

En cada comparación se mantendrán fijas las demás variables. Se analizarán el macro F1, el recall por clase y la matriz de confusión, prestando especial atención a las nubes de nivel bajo. Las comparaciones se realizarán con las mismas semillas.

## Cambios al protocolo

| Fecha | Cambio | Motivo | Corridas repetidas |
|---|---|---|---|
| 2026-10-05 | se establece inicialmente un presupuesto de 60 épocas para el piloto | se buscaba observar la evolución del modelo durante más épocas | ninguna |
| 2026-10-14 | se mantiene el piloto de 60 épocas | se busca decidir el presupuesto común a partir de la evolución de las métricas de validación | ninguna |
| 2026-10-15 | se interrumpe el piloto de 60 épocas y se prepara una versión corta de 5 épocas | el entrenamiento tardaba demasiado en el computador disponible | ninguna |
| 2026-10-17 | se recuperan los resultados guardados de la línea base de 30 épocas | se evita repetir el entrenamiento y se aprovecha la corrida existente con semilla 11 | ninguna |
| 2026-10-18 | se establece un presupuesto inicial común de 30 épocas | la corrida recuperada alcanzó el mínimo de `val_loss` en la época 26 | ninguna |
| 2026-10-18 | los experimentos de volteo y cabeza se corren con 5 épocas y tres semillas en vez de 30 épocas | con unos 222 segundos por época en CPU, las 12 corridas de 30 épocas no alcanzaban a terminar antes de la entrega | ninguna. Cada experimento se compara solo contra su propia referencia de 5 épocas y no contra la línea base de 30 |
