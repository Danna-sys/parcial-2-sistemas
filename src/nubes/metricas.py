
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
        matrix = np.asarray(self.confusion)  # convierte la matriz de confusión en un arreglo numérico
        return int(matrix.sum() - np.trace(matrix))  # calcula el total de errores fuera de la diagonal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # convierte el reporte en un diccionario

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        return cls(**data)  # reconstruye el reporte a partir de un diccionario

    def per_class(self, class_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precision": self.precision,  # registra la precisión de cada clase
                "recall": self.recall,  # registra la sensibilidad de cada clase
                "f1": self.f1,  # registra la medida f1 de cada clase
                "support": self.support,  # registra la cantidad de ejemplos reales de cada clase
            },
            index=class_names,  # utiliza los nombres de las clases como índice
        )


def build_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, loss: float) -> Report:
    labels = list(range(num_classes))  # crea la lista de etiquetas numéricas de las clases
    matrix = confusion_matrix(y_true, y_pred, labels=labels)  # calcula la matriz de confusión
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0  # calcula las métricas por clase y evita divisiones por cero
    )
    return Report(
        loss=float(loss),  # guarda la pérdida del modelo
        accuracy=float(np.trace(matrix) / matrix.sum()),  # calcula la proporción de predicciones correctas
        macro_f1=float(f1.mean()),  # calcula el promedio del f1 entre las clases
        precision=precision.tolist(),  # convierte las precisiones en una lista
        recall=recall.tolist(),  # convierte las sensibilidades en una lista
        f1=f1.tolist(),  # convierte los valores f1 en una lista
        support=support.astype(int).tolist(),  # convierte los soportes en una lista de enteros
        confusion=matrix.astype(int).tolist(),  # guarda la matriz de confusión como una lista de listas
    )


def top_confusions(confusion: Any, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Los ``k`` pares real y predicha con más errores.

    ``share`` es la fracción de las imágenes de la clase real que cayó en ese error.
    """
    matrix = np.asarray(confusion, dtype=float)  # convierte la matriz de confusión a números decimales
    support = matrix.sum(axis=1)  # calcula el total de imágenes reales de cada clase
    rows = [
        {
            "real": class_names[i],  # guarda la clase real
            "predicha": class_names[j],  # guarda la clase predicha
            "errores": int(matrix[i, j]),  # registra la cantidad de errores para ese par
            "share": matrix[i, j] / support[i] if support[i] else 0.0,  # calcula la proporción de errores respecto a la clase real
        }
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and matrix[i, j] > 0  # excluye las predicciones correctas y los pares sin errores
    ]
    table = pd.DataFrame(rows, columns=["real", "predicha", "errores", "share"])  # organiza los errores en una tabla
    return table.sort_values("errores", ascending=False).head(k).reset_index(drop=True)  # devuelve los k errores más frecuentes


def symmetric_confusions(confusion: Any, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Pares de clases ordenados por errores sumados en ambos sentidos."""
    matrix = np.asarray(confusion)  # convierte la matriz de confusión en un arreglo
    rows = [
        {
            "par": f"{class_names[i]} / {class_names[j]}",  # identifica el par de clases comparadas
            "a→b": int(matrix[i, j]),  # registra los errores de la clase a hacia la clase b
            "b→a": int(matrix[j, i]),  # registra los errores de la clase b hacia la clase a
            "total": int(matrix[i, j] + matrix[j, i]),  # suma los errores en ambas direcciones
        }
        for i in range(len(class_names))
        for j in range(i + 1, len(class_names))
        if matrix[i, j] + matrix[j, i] > 0  # conserva solo los pares que presentan errores
    ]
    table = pd.DataFrame(rows, columns=["par", "a→b", "b→a", "total"])  # organiza las confusiones por pares
    return table.sort_values("total", ascending=False).head(k).reset_index(drop=True)  # devuelve los k pares con más errores totales


def group_confusion(confusion: Any, groups: list[str]) -> pd.DataFrame:
    """Colapsa la matriz de confusión según una agrupación de clases.

    ``groups[i]`` es el grupo de la clase ``i``. Sirve para ver si los errores se quedan
    dentro del mismo nivel de altura o de la misma forma de nube.
    """
    matrix = pd.DataFrame(np.asarray(confusion), index=groups, columns=groups)  # crea una tabla de confusión con los grupos como etiquetas
    return matrix.groupby(level=0).sum().T.groupby(level=0).sum().T  # suma las filas y columnas que pertenecen al mismo grupo


def within_group_share(confusion: Any, groups: list[str]) -> float:
    """Fracción de los errores que ocurre entre clases del mismo grupo."""
    matrix = np.asarray(confusion)  # convierte la matriz de confusión en un arreglo
    same = np.equal.outer(np.asarray(groups), np.asarray(groups))  # identifica los pares de clases que pertenecen al mismo grupo
    off_diagonal = ~np.eye(len(groups), dtype=bool)  # crea una máscara que excluye la diagonal de aciertos
    errors = matrix[off_diagonal].sum()  # calcula el total de errores de clasificación
    return float(matrix[same & off_diagonal].sum() / errors) if errors else 0.0  # calcula la proporción de errores dentro del mismo grupo
