"""tablas comparativas construidas desde las carpetas de corridas"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nubes.experimento import METRICS_FILE, RunResult, load_result

# columnas que se utilizan para comparar el rendimiento de las configuraciones
SUMMARY_COLUMNS: tuple[str, ...] = (
    "val_macro_f1",
    "val_acc",
    "val_loss",
    "train_acc",
    "gap",
    "best_epoch",
    "seconds_per_epoch",
)


def load_experiment(runs_dir: Path, experiment: str) -> dict[str, list[RunResult]]:
    """corridas terminadas de un experimento, por configuración y ordenadas por semilla"""
    root = runs_dir / experiment  # construye la ruta de la carpeta del experimento
    if not root.is_dir():
        raise FileNotFoundError(f"No existe {root}")  # informa si la carpeta no existe

    results: dict[str, list[RunResult]] = {}  # almacena las corridas agrupadas por configuración
    for metrics in sorted(root.glob(f"*/seed_*/{METRICS_FILE}")):  # busca los archivos de métricas de las corridas
        result = load_result(metrics.parent)  # carga los resultados de cada corrida
        results.setdefault(result.config.name, []).append(result)  # agrupa el resultado según el nombre de configuración

    for runs in results.values():
        runs.sort(key=lambda r: r.config.seed)  # ordena las corridas por semilla

    return results  # devuelve las corridas organizadas por configuración


def runs_table(results: dict[str, list[RunResult]]) -> pd.DataFrame:
    """una fila por corrida"""
    rows = [
        {"config": name, "seed": result.config.seed, **result.summary()}  # combina la configuración, la semilla y sus métricas
        for name, runs in results.items()  # recorre las configuraciones
        for result in runs  # recorre las corridas de cada configuración
    ]
    return pd.DataFrame(rows)  # convierte los resultados en una tabla


def aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """media y desviación estándar entre semillas para cada configuración"""
    grouped = table.groupby("config", sort=False)[list(SUMMARY_COLUMNS)]  # agrupa las métricas por configuración
    summary = grouped.agg(["mean", "std"])  # calcula la media y la desviación estándar
    summary.insert(0, "seeds", table.groupby("config", sort=False).size())  # agrega la cantidad de corridas por configuración
    return summary  # devuelve la tabla comparativa


def per_class_recall(
    results: dict[str, list[RunResult]], class_names: list[str], stat: str = "mean"
) -> pd.DataFrame:
    """recall de validation por clase, resumido entre semillas con ``mean`` o ``std``

    una columna por configuración
    """
    return pd.DataFrame(
        {
            name: pd.DataFrame([r.val.recall for r in runs], columns=class_names).agg(stat)  # calcula la estadística elegida del recall por clase
            for name, runs in results.items()  # repite el cálculo para cada configuración
        }
    )  # organiza los resultados en una tabla con una columna por configuración


def total_confusion(runs: list[RunResult]) -> pd.DataFrame:
    """matriz de confusión de validation sumada sobre las semillas"""
    return sum(pd.DataFrame(r.val.confusion) for r in runs)  # suma las matrices de confusión de todas las corridas


def to_markdown(table: pd.DataFrame) -> str:
    """tabla markdown con media ± desviación a partir de ``aggregate``"""
    header = ["config", "seeds", *SUMMARY_COLUMNS]  # define los encabezados de la tabla
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]  # crea las filas del encabezado en markdown

    for name, row in table.iterrows():  # recorre cada configuración de la tabla
        cells = [str(name), str(int(row[("seeds", "")]))]  # agrega el nombre y el número de semillas

        for column in SUMMARY_COLUMNS:
            mean, std = row[(column, "mean")], row[(column, "std")]  # obtiene la media y la desviación de cada métrica
            digits = 1 if column in ("best_epoch", "seconds_per_epoch") else 3  # define los decimales según el tipo de métrica
            cells.append(
                f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"  # muestra la media sola o acompañada de la desviación
            )

        lines.append("| " + " | ".join(cells) + " |")  # agrega la fila de resultados en formato markdown

    return "\n".join(lines)  # une las filas en una sola cadena de texto