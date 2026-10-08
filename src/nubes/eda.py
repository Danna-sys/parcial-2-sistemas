"""Exploración de CCSN antes de entrenar: rasgos simples, vecinos y similitud entre clases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from nubes.preparacion import CLASSES, CODES


def class_table() -> pd.DataFrame:
    """Código, género, nivel de altura y forma de cada clase."""
    return pd.DataFrame([vars(c) for c in CLASSES]).set_index("code")


def split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Imágenes por clase y split, con el total y la proporción de cada clase."""
    table = manifest.groupby(["code", "split"]).size().unstack(fill_value=0)
    table = table.reindex(index=list(CODES), columns=["train", "val", "test"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table["share"] = table["total"] / table["total"].sum()
    return table


def load_thumbnails(manifest: pd.DataFrame, root: Path, size: int = 32) -> np.ndarray:
    """Todas las imágenes como arreglo ``(N, size, size, 3)`` en ``[0, 1]``."""
    thumbs = np.empty((len(manifest), size, size, 3), dtype=np.float32)
    for i, path in enumerate(manifest["path"]):
        with Image.open(root / path) as image:
            thumbs[i] = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
    return thumbs / 255.0


def simple_features(thumbnails: np.ndarray) -> pd.DataFrame:
    """Rasgos globales por imagen que no dependen de ninguna red.

    - ``brightness``: luminancia media.
    - ``contrast``: desviación estándar de la luminancia.
    - ``saturation``: diferencia media entre el canal más alto y el más bajo.
    - ``blue_excess``: azul menos rojo. Alto cuando domina el cielo despejado.
    - ``texture``: magnitud media del gradiente de la luminancia.
    - ``top_minus_bottom``: brillo del tercio superior menos el del tercio inferior.
    """
    gray = thumbnails @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    third = gray.shape[1] // 3
    dy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    dx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    return pd.DataFrame(
        {
            "brightness": gray.mean(axis=(1, 2)),
            "contrast": gray.std(axis=(1, 2)),
            "saturation": (thumbnails.max(axis=3) - thumbnails.min(axis=3)).mean(axis=(1, 2)),
            "blue_excess": (thumbnails[..., 2] - thumbnails[..., 0]).mean(axis=(1, 2)),
            "texture": dx + dy,
            "top_minus_bottom": gray[:, :third].mean(axis=(1, 2))
            - gray[:, -third:].mean(axis=(1, 2)),
        }
    )


def class_mean_images(thumbnails: np.ndarray, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Imagen promedio de cada clase."""
    codes = manifest["code"].to_numpy()
    return {code: thumbnails[codes == code].mean(axis=0) for code in CODES}


def knn_baseline(
    features: np.ndarray, manifest: pd.DataFrame, k: int = 5
) -> tuple[float, pd.DataFrame]:
    """k vecinos entrenado con train y evaluado en validation.

    Devuelve la accuracy y la matriz de confusión de validation. Es la referencia de lo
    que se logra sin aprender filtros. No usa test.
    """
    split = manifest["split"].to_numpy()
    labels = manifest["label"].to_numpy()
    train, val = split == "train", split == "val"
    scaler = StandardScaler().fit(features[train])
    model = KNeighborsClassifier(n_neighbors=k).fit(
        scaler.transform(features[train]), labels[train]
    )
    predicted = model.predict(scaler.transform(features[val]))
    matrix = confusion_matrix(labels[val], predicted, labels=list(range(len(CODES))))
    table = pd.DataFrame(matrix, index=list(CODES), columns=list(CODES))
    return float((predicted == labels[val]).mean()), table


def centroid_distances(features: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """Distancia euclidiana entre los centroides de clase, con rasgos estandarizados.

    Solo usa train. Un valor pequeño señala dos clases parecidas en esos rasgos.
    """
    train = (manifest["split"] == "train").to_numpy()
    scaled = StandardScaler().fit_transform(features[train])
    codes = manifest.loc[train, "code"].to_numpy()
    centroids = np.stack([scaled[codes == code].mean(axis=0) for code in CODES])
    distances = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
    return pd.DataFrame(distances, index=list(CODES), columns=list(CODES))


def closest_pairs(
    thumbnails: np.ndarray, manifest: pd.DataFrame, split_a: str, split_b: str, k: int = 12
) -> pd.DataFrame:
    """Los ``k`` pares de imágenes más parecidos entre dos splits.

    Sirve para revisar a ojo si quedaron fotografías casi iguales a ambos lados del split.
    """
    flat = thumbnails.reshape(len(thumbnails), -1)
    index_a = np.flatnonzero((manifest["split"] == split_a).to_numpy())
    index_b = np.flatnonzero((manifest["split"] == split_b).to_numpy())
    a, b = flat[index_a], flat[index_b]
    squared = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None] - 2 * a @ b.T
    order = np.argsort(squared, axis=None)[:k]
    rows, cols = np.unravel_index(order, squared.shape)
    return pd.DataFrame(
        {
            "index_a": index_a[rows],
            "index_b": index_b[cols],
            "code_a": manifest["code"].to_numpy()[index_a[rows]],
            "code_b": manifest["code"].to_numpy()[index_b[cols]],
            "distance": np.sqrt(np.maximum(squared[rows, cols], 0) / flat.shape[1]),
        }
    )


def resolution_ladder(image: Image.Image, sizes: tuple[int, ...]) -> list[Image.Image]:
    """La misma imagen reducida a cada resolución y devuelta al tamaño original."""
    return [image.resize((s, s)).resize(image.size, Image.Resampling.NEAREST) for s in sizes]


def detail_loss(image: Image.Image, sizes: tuple[int, ...]) -> list[float]:
    """Error RMS en luminancia al reducir la imagen a cada tamaño y volver a ampliarla.

    Mide cuánto detalle de la imagen original no sobrevive a esa resolución.
    """
    gray = image.convert("L")
    reference = np.asarray(gray, dtype=np.float32) / 255.0
    losses = []
    for size in sizes:
        restored = gray.resize((size, size)).resize(gray.size, Image.Resampling.BILINEAR)
        difference = np.asarray(restored, dtype=np.float32) / 255.0 - reference
        losses.append(float(np.sqrt((difference**2).mean())))
    return losses
