"""Tablas comparativas construidas desde las carpetas de corridas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nubes.experimento import METRICS_FILE, RunResult, load_result

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
    """Corridas terminadas de un experimento, por configuración y ordenadas por semilla."""
    root = runs_dir / experiment
    if not root.is_dir():
        raise FileNotFoundError(f"No existe {root}")
    results: dict[str, list[RunResult]] = {}
    for metrics in sorted(root.glob(f"*/seed_*/{METRICS_FILE}")):
        result = load_result(metrics.parent)
        results.setdefault(result.config.name, []).append(result)
    for runs in results.values():
        runs.sort(key=lambda r: r.config.seed)
    return results


def runs_table(results: dict[str, list[RunResult]]) -> pd.DataFrame:
    """Una fila por corrida."""
    rows = [
        {"config": name, "seed": result.config.seed, **result.summary()}
        for name, runs in results.items()
        for result in runs
    ]
    return pd.DataFrame(rows)


def aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar entre semillas para cada configuración."""
    grouped = table.groupby("config", sort=False)[list(SUMMARY_COLUMNS)]
    summary = grouped.agg(["mean", "std"])
    summary.insert(0, "seeds", table.groupby("config", sort=False).size())
    return summary


def per_class_recall(
    results: dict[str, list[RunResult]], class_names: list[str], stat: str = "mean"
) -> pd.DataFrame:
    """Recall de validation por clase, resumido entre semillas con ``mean`` o ``std``.

    Una columna por configuración.
    """
    return pd.DataFrame(
        {
            name: pd.DataFrame([r.val.recall for r in runs], columns=class_names).agg(stat)
            for name, runs in results.items()
        }
    )


def total_confusion(runs: list[RunResult]) -> pd.DataFrame:
    """Matriz de confusión de validation sumada sobre las semillas."""
    return sum(pd.DataFrame(r.val.confusion) for r in runs)


def to_markdown(table: pd.DataFrame) -> str:
    """Tabla Markdown con media ± desviación a partir de ``aggregate``."""
    header = ["config", "seeds", *SUMMARY_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in table.iterrows():
        cells = [str(name), str(int(row[("seeds", "")]))]
        for column in SUMMARY_COLUMNS:
            mean, std = row[(column, "mean")], row[(column, "std")]
            digits = 1 if column in ("best_epoch", "seconds_per_epoch") else 3
            cells.append(
                f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
