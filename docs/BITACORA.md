# Bitácora

Aquí queda el camino del proyecto en el orden en que ocurrió. Cada entrada separa lo que se observó de lo que se interpreta y dice de qué figura o tabla sale.

```text
observación → pregunta → hipótesis → experimento controlado → resultado → argumento
```

## 1. Observaciones del EDA

Fuente: `notebooks/01_eda.ipynb`.

| # | Observación | Evidencia |
|---|---|---|
| 1 | *por escribir* | |
| 2 | | |
| 3 | | |

Decisiones que salen del EDA:

- Augmentation que conserva el significado de la imagen: *por escribir*
- Clases que se parecen antes de entrenar: *por escribir*
- Resolución mínima a la que las clases siguen siendo distinguibles a ojo: *por escribir*

## 2. Observaciones de la línea base

Fuente: `notebooks/02_linea_base.ipynb`.

| # | Observación | Evidencia |
|---|---|---|
| 1 | *por escribir* | |
| 2 | | |
| 3 | | |

Resultado de la línea base:

*pegar aquí la salida de `uv run nubes resumen --experiment linea-base`*

## 3. Pregunta del proyecto

*por escribir*

De qué observación sale: *por escribir*

## 4. Experimentos

Una entrada por experimento. La hipótesis se escribe antes de correr.

### Experimento 1: *nombre*

| Campo | Contenido |
|---|---|
| Fecha | |
| Observación de partida | |
| Hipótesis | |
| Qué espero ver si es cierta | |
| Qué vería si no lo es | |
| Variable que cambia | |
| Qué queda fijo | todo lo demás de `configs/base.toml` y `docs/PROTOCOLO.md` |
| Configuraciones | |
| Semillas | 11, 22, 33 |

Resultado:

*pegar aquí la salida de `uv run nubes resumen --experiment <nombre>`*

| Campo | Contenido |
|---|---|
| Qué se observó | |
| Qué se interpreta | |
| Veredicto | se sostiene / no se sostiene / no se puede decidir |
| Qué limita la conclusión | |
| Qué pregunta nueva deja | |

## 5. Evaluación final en test

| Campo | Contenido |
|---|---|
| Configuración elegida con validation | |
| Por qué esa | |
| Fecha de la evaluación | |

*pegar aquí la tabla de la sección 8 de `notebooks/03_experimentos.ipynb`*

## 6. Qué se aprendió

Sobre el problema: *por escribir*

Sobre el dato: *por escribir*

Sobre la CNN: *por escribir*

Qué haría distinto: *por escribir*
