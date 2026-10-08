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
| Pérdida | cross entropy sobre logits | |
| Optimizador | AdamW con `lr = 1e-3` y `weight_decay = 0` | punto de partida estándar. Cambia solo si es la variable del experimento |
| Scheduler y early stopping | ninguno | mezclarlos con la variable estudiada impediría atribuir el efecto |
| Épocas | *por decidir con el piloto* | |
| Checkpoint | época con menor `val_loss` | criterio fijado de antemano |
| Semillas | 11, 22 y 33 | tres corridas por configuración para medir la dispersión |
| Determinismo | `deterministic = true` y `num_workers = 0` | misma semilla, mismo resultado |

## Evaluación

| Decisión | Valor | Motivo |
|---|---|---|
| Métrica primaria | macro F1 de validation en el mejor checkpoint, media entre semillas | pondera igual a las 11 clases aunque no tengan el mismo tamaño |
| Métricas secundarias | accuracy, recall por clase, matriz de confusión, generalization gap | una métrica global puede esconder lo que pasa en una clase |
| Generalization gap | accuracy de train en modo eval menos accuracy de validation | compara los dos conjuntos en las mismas condiciones |
| Diferencia interpretable | mayor que la desviación entre semillas de la referencia | con unas 380 imágenes de validation una imagen vale cerca de 0.26 puntos de accuracy |
| Test | una sola vez, al final, para la configuración elegida con validation | test no es una segunda validación |

## Piloto

*por escribir después de correr la sección 2 de `notebooks/02_linea_base.ipynb`*

| Dato | Valor |
|---|---|
| Época del mínimo de `val_loss` | |
| `val_loss` y `val_acc` en esa época | |
| Comportamiento después del mínimo | |
| Épocas elegidas para todas las corridas | |
| Tiempo por época y equipo usado | |

## Cambios al protocolo

| Fecha | Cambio | Motivo | Corridas repetidas |
|---|---|---|---|
| | | | |
