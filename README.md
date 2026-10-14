# Nubes CCSN: una CNN propia y experimentos controlados

Proyecto del curso **Sistemas Inteligentes II + Introducción al Deep Learning** de la Universidad El Bosque, periodo 2026-2. Tema 1: leer la atmósfera desde la forma de las nubes.

El objetivo no es la mayor accuracy. Es usar una CNN sencilla para entender qué necesita ver una red para separar géneros de nubes que se parecen entre sí.

## El problema

[CCSN](https://doi.org/10.7910/DVN/CADDPD) reúne 2543 fotografías de nubes tomadas desde el suelo. Son imágenes JPEG de 400×400, salvo 211 de 256×256 que están en `Cc`, `Cs` y `Ns`. Hay 11 clases: los diez géneros de la clasificación meteorológica y las estelas de condensación.

| Código | Género | Nivel | Forma | Imágenes |
|---|---|---|---|---|
| `Ac` | Altocumulus | medio | cumuliforme | 221 |
| `As` | Altostratus | medio | estratiforme | 188 |
| `Cb` | Cumulonimbus | vertical | cumuliforme | 242 |
| `Cc` | Cirrocumulus | alto | cumuliforme | 268 |
| `Ci` | Cirrus | alto | cirriforme | 139 |
| `Cs` | Cirrostratus | alto | estratiforme | 287 |
| `Ct` | Contrail | alto | artificial | 200 |
| `Cu` | Cumulus | bajo | cumuliforme | 182 |
| `Ns` | Nimbostratus | medio | estratiforme | 274 |
| `Sc` | Stratocumulus | bajo | cumuliforme | 340 |
| `St` | Stratus | bajo | estratiforme | 202 |

Son pocas imágenes para 11 clases. Con un split 70 / 15 / 15 quedan unas 380 imágenes en validation y entre 20 y 50 por clase. Una sola imagen mueve varios puntos el recall de una clase. Por eso cada configuración se entrena con tres semillas y las diferencias se leen contra la dispersión entre semillas.

## Cómo está organizado el trabajo

```text
EDA → observación → pregunta → hipótesis → experimento → evidencia → argumento
```

| Paso | Dónde |
|---|---|
| Entender el dato | `notebooks/01_eda.ipynb` |
| Línea base y análisis de errores | `notebooks/02_linea_base.ipynb` |
| Un experimento controlado | `notebooks/03_experimentos.ipynb` |
| Decisiones fijas del protocolo | `docs/PROTOCOLO.md` |
| Observaciones, pregunta, hipótesis y veredictos | `docs/BITACORA.md` |

Los notebooks no tienen loops de entrenamiento. Todo lo hace el paquete `nubes` y la línea de comandos llama a las mismas funciones.

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) y git. uv instala Python 3.12 si falta.

```powershell
git clone <url-del-repositorio> ccsn-nubes-cnn
cd ccsn-nubes-cnn
uv sync
uv run python -c "import torch, nubes; print(torch.__version__, torch.cuda.is_available())"
```

`uv sync` crea `.venv` e instala el paquete en modo editable. En Windows y Linux instala PyTorch con CUDA 13.2, que necesita un driver NVIDIA de la serie 580 o posterior. Sin GPU el código corre en CPU.

## Datos

1. Descargar CCSN desde <https://doi.org/10.7910/DVN/CADDPD> y descomprimirlo dentro de `data/raw/`. Deben quedar las 11 carpetas `Ac`, `As`, `Cb`, `Cc`, `Ci`, `Cs`, `Ct`, `Cu`, `Ns`, `Sc` y `St`. No importa si quedan dentro de otra carpeta.
2. Preparar el inventario y el split:

```powershell
uv run nubes preparar
```

La preparación no copia imágenes. Escribe en `data/ccsn/` un `manifest.csv` con una fila por imagen y un `stats.json` con conteos, duplicados y la media y desviación por canal calculadas solo con train.

El split es estratificado por clase con `split_seed = 2026`. Las imágenes que son la misma fotografía se detectan antes de dividir y van juntas al mismo split para que no haya fuga entre train y validation.

`data/`, `runs/`, `mlruns/` y `mlflow.db` no se versionan.

## Tests y estilo

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Los tests fabrican un CCSN sintético. No necesitan el dataset ni GPU.

## Entrenar

Cada corrida se define con un archivo de `configs/`. Los archivos declaran `extends = "base.toml"` y solo escriben lo que cambian, así que la diferencia con la línea base se lee en el propio archivo.

```powershell
uv run nubes train --config configs/base.toml --seeds 11 22 33
uv run nubes resumen --experiment linea-base
```

Cada corrida queda en `runs/<experimento>/<configuración>/seed_<semilla>/` con:

| Archivo | Contenido |
|---|---|
| `config.json` | la configuración completa |
| `history.csv` | una fila por época |
| `metrics.json` | métricas de train y validation con el mejor checkpoint, por clase y matriz de confusión |
| `predictions_val.csv` | la predicción de cada imagen de validation |
| `best.pt` | pesos de la época con menor `val_loss` |

Una corrida terminada no se sobrescribe. `train` nunca toca test. La evaluación final es un comando aparte que solo se puede ejecutar una vez por corrida:

```powershell
uv run nubes test --run runs/linea-base/base/seed_11
```

Las corridas también se registran en MLflow:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Qué se puede variar

| Clave del TOML | Qué controla |
|---|---|
| `image_size` | resolución de entrada |
| `blocks` | profundidad y con ella el campo receptivo |
| `base_channels`, `max_channels` | ancho de la red |
| `kernel_size` | tamaño del kernel |
| `pooling` | `max`, `avg` o `stride` |
| `head_grid` | rejilla a la que se promedia el último mapa. Con 1 se pierde la posición |
| `batchnorm`, `dropout`, `weight_decay` | regularización |
| `augment` | `none`, `flip`, `flip_crop`, `flip_crop_color` o `flip_hv` |
| `class_weights` | pérdida ponderada por clase |
| `epochs`, `learning_rate`, `batch_size` | entrenamiento |

`configs/` trae cinco familias listas: `res_*`, `prof_*`, `cabeza_*`, `reg_*` y `aug_*`. No son una lista para correr completa. Se corre la familia que responde a la hipótesis de `docs/BITACORA.md`.

## Estructura

| Quiero cambiar | Archivo |
|---|---|
| Valores por defecto y validación de una configuración | `src/nubes/config.py` |
| Inventario, duplicados y split | `src/nubes/preparacion.py` |
| Transformaciones, augmentation y `DataLoader` | `src/nubes/datos.py` |
| Arquitectura | `src/nubes/modelo.py` |
| Loop de entrenamiento y evaluación | `src/nubes/motor.py` |
| Qué se guarda de cada corrida, MLflow y evaluación en test | `src/nubes/experimento.py` |
| Métricas y tablas de confusión | `src/nubes/metricas.py` |
| Tablas entre semillas y configuraciones | `src/nubes/resumen.py` |
| Rasgos simples, vecinos y pérdida de detalle | `src/nubes/eda.py` |
| Campo receptivo, formas y feature maps | `src/nubes/inspeccion.py` |
| Figuras | `src/nubes/figuras.py` |
| Comandos | `src/nubes/cli.py` |

## Relación con el repositorio de referencia del curso

El punto de partida fue Galaxy3, el repositorio del profesor Mauricio Suárez Mantilla sobre morfología de galaxias. De ahí vienen las ideas que este proyecto conserva:

- un paquete instalable con módulos de responsabilidad única y notebooks que solo orquestan
- una configuración inmutable que viaja dentro del checkpoint
- normalización calculada solo con train
- número fijo de épocas decidido con un piloto y checkpoint por menor `val_loss`
- generalization gap medido con train en modo eval
- test fuera del entrenamiento y evaluado una sola vez
- tests con datos sintéticos, Ruff y MLflow

Lo que cambia responde a que CCSN es otro problema:

- **Varias semillas por configuración.** Galaxy3 tiene 917 imágenes de validation para 3 clases. Aquí hay unas 380 para 11. Con una sola semilla casi ninguna diferencia sería interpretable.
- **Duplicados antes del split.** La preparación busca copias y recompresiones de una misma fotografía y no las separa entre splits.
- **Manifest en vez de carpetas exportadas.** Las imágenes se leen en su tamaño original y la resolución es un parámetro de la corrida. Un experimento de resolución no obliga a regenerar datos.
- **Arquitectura parametrizada.** Profundidad, ancho, kernel, pooling y la rejilla de la cabeza son claves de configuración porque son las variables del problema.
- **Campo receptivo como dato.** `inspeccion.receptive_field` calcula qué fracción de la imagen ve cada bloque. Es la medida que conecta la arquitectura con la pregunta sobre textura local y organización global.
- **Referencia sin red.** Un k vecinos sobre rasgos globales da un piso y una matriz de confusión previa para comparar con la de la CNN.
- **Errores por familia.** Las confusiones se agrupan por nivel de altura y por forma de la nube.
- **Configuraciones en archivos.** Cada corrida es un TOML que hereda de la línea base y deja su carpeta propia con todo lo necesario para rehacer tablas y figuras sin reentrenar.

## Estado

El protocolo y los resultados están en `docs/`. Las secciones marcadas *por escribir* se completan con las corridas reales.
