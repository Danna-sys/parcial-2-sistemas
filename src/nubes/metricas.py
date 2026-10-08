"""Métricas de clasificación a partir de etiquetas reales y predichas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class Report:
    """Métricas de un split.

    ``confusion[i][j]`` cuenta las imágenes de la clase real ``i`` predichas como ``j``.
    """

    loss: float
    accuracy: float
    macro_f1: float
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    confusion: list[list[int]]

    @property
    def errors(self) -> int:
        matrix = np.asarray(self.confusion)
        return int(matrix.sum() - np.trace(matrix))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        return cls(**data)

    def per_class(self, class_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "support": self.support,
            },
            index=class_names,
        )


def build_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, loss: float) -> Report:
    labels = list(range(num_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return Report(
        loss=float(loss),
        accuracy=float(np.trace(matrix) / matrix.sum()),
        macro_f1=float(f1.mean()),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1=f1.tolist(),
        support=support.astype(int).tolist(),
        confusion=matrix.astype(int).tolist(),
    )


def top_confusions(confusion: Any, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Los ``k`` pares real y predicha con más errores.

    ``share`` es la fracción de las imágenes de la clase real que cayó en ese error.
    """
    matrix = np.asarray(confusion, dtype=float)
    support = matrix.sum(axis=1)
    rows = [
        {
            "real": class_names[i],
            "predicha": class_names[j],
            "errores": int(matrix[i, j]),
            "share": matrix[i, j] / support[i] if support[i] else 0.0,
        }
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and matrix[i, j] > 0
    ]
    table = pd.DataFrame(rows, columns=["real", "predicha", "errores", "share"])
    return table.sort_values("errores", ascending=False).head(k).reset_index(drop=True)


def symmetric_confusions(confusion: Any, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Pares de clases ordenados por errores sumados en ambos sentidos."""
    matrix = np.asarray(confusion)
    rows = [
        {
            "par": f"{class_names[i]} / {class_names[j]}",
            "a→b": int(matrix[i, j]),
            "b→a": int(matrix[j, i]),
            "total": int(matrix[i, j] + matrix[j, i]),
        }
        for i in range(len(class_names))
        for j in range(i + 1, len(class_names))
        if matrix[i, j] + matrix[j, i] > 0
    ]
    table = pd.DataFrame(rows, columns=["par", "a→b", "b→a", "total"])
    return table.sort_values("total", ascending=False).head(k).reset_index(drop=True)


def group_confusion(confusion: Any, groups: list[str]) -> pd.DataFrame:
    """Colapsa la matriz de confusión según una agrupación de clases.

    ``groups[i]`` es el grupo de la clase ``i``. Sirve para ver si los errores se quedan
    dentro del mismo nivel de altura o de la misma forma de nube.
    """
    matrix = pd.DataFrame(np.asarray(confusion), index=groups, columns=groups)
    return matrix.groupby(level=0).sum().T.groupby(level=0).sum().T


def within_group_share(confusion: Any, groups: list[str]) -> float:
    """Fracción de los errores que ocurre entre clases del mismo grupo."""
    matrix = np.asarray(confusion)
    same = np.equal.outer(np.asarray(groups), np.asarray(groups))
    off_diagonal = ~np.eye(len(groups), dtype=bool)
    errors = matrix[off_diagonal].sum()
    return float(matrix[same & off_diagonal].sum() / errors) if errors else 0.0
