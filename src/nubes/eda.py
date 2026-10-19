"""Exploración de CCSN antes de entrenar: rasgos simples, vecinos y similitud entre clases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image  #abre y redimensiona imagenes
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier  #k vecinos cercanos
from sklearn.preprocessing import StandardScaler  #estandariza

from nubes.preparacion import CLASSES, CODES  #las fichas de las clases y sus codigos


def class_table() -> pd.DataFrame:
    """Código, género, nivel de altura y forma de cada clase."""
    return pd.DataFrame([vars(c) for c in CLASSES]).set_index("code")     #vars(c) convierte cada ficha en diccionario. La tabla usa el codigo como indice


def split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Imágenes por clase y split, con el total y la proporción de cada clase."""
    table = manifest.groupby(["code", "split"]).size().unstack(fill_value=0)          #conteo de imagenes segun (clase,split), y pone los split en columnas
    table = table.reindex(index=list(CODES), columns=["train", "val", "test"], fill_value=0)     #fija las coliumnas
    table["total"] = table.sum(axis=1)     #agrega el total de imagenes por clase
    table["share"] = table["total"] / table["total"].sum()          #que fraccion de las imagenes tiene cada clase (para ver el balance)
    return table


def load_thumbnails(manifest: pd.DataFrame, root: Path, size: int = 32) -> np.ndarray:
    """Todas las imágenes como arreglo ``(N, size, size, 3)`` en ``[0, 1]``."""
    thumbs = np.empty((len(manifest), size, size, 3), dtype=np.float32)     #arregla vacio en el que entrarán las imagenes
    for i, path in enumerate(manifest["path"]):      #se van agregando las imagenes
        with Image.open(root / path) as image:        #abre la imagen
            thumbs[i] = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)    #las pone en canales rgb, tamaño size x size, y convertidas a numeros
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
    gray = thumbnails @ np.array([0.299, 0.587, 0.114], dtype=np.float32)    #escala de grsises con luminancia
    third = gray.shape[1] // 3     #alto de  un tercio de la imagen
    dy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))     #cuanto cambia en promedio cada pixel con el de abajo (bordes horizontales)
    dx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))     #cuanto cambia en promedio cada pixel con el de al lado (bordes verticales)
    return pd.DataFrame(        #rasgo por columna e imagen por fila
        {
            "brightness": gray.mean(axis=(1, 2)),    #brillo de la imag
            "contrast": gray.std(axis=(1, 2)),     #que tanto cambia el brillo de la img
            "saturation": (thumbnails.max(axis=3) - thumbnails.min(axis=3)).mean(axis=(1, 2)),   #que tanto color tiene la img
            "blue_excess": (thumbnails[..., 2] - thumbnails[..., 0]).mean(axis=(1, 2)),     #azul menos rojo. cielo despejado da valores altos, nubes grises valores bajos
            "texture": dx + dy,        #mira cantidad de bordes (textura)
            "top_minus_bottom": gray[:, :third].mean(axis=(1, 2))    # brillo de arriba menos brillo de abajo
            - gray[:, -third:].mean(axis=(1, 2)),
        }
    )


def class_mean_images(thumbnails: np.ndarray, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Imagen promedio de cada clase."""
    codes = manifest["code"].to_numpy()     #codigo por imagen
    return {code: thumbnails[codes == code].mean(axis=0) for code in CODES}   #por clase, promedia pixel a pixel las imagenes


def knn_baseline(
    features: np.ndarray, manifest: pd.DataFrame, k: int = 5
) -> tuple[float, pd.DataFrame]:
    """k vecinos entrenado con train y evaluado en validation.

    Devuelve la accuracy y la matriz de confusión de validation. Es la referencia de lo
    que se logra sin aprender filtros. No usa test.
    """
    split = manifest["split"].to_numpy()     #split de cada imagen
    labels = manifest["label"].to_numpy()     #etiquetas de cada imagen
    train, val = split == "train", split == "val"     #mascaras de train y val
    scaler = StandardScaler().fit(features[train])     #estandarizador de train
    model = KNeighborsClassifier(n_neighbors=k).fit(    #clasificador aprende de los train ya estandarizados
        scaler.transform(features[train]), labels[train]     
    )
    predicted = model.predict(scaler.transform(features[val]))   # predice val por votacion de sus 5 vecinos mas cercanos en train
    matrix = confusion_matrix(labels[val], predicted, labels=list(range(len(CODES))))     #matriz de confusion en val
    table = pd.DataFrame(matrix, index=list(CODES), columns=list(CODES))    #la converte en tabla
    return float((predicted == labels[val]).mean()), table       #accuracy y tabla


def centroid_distances(features: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    """Distancia euclidiana entre los centroides de clase, con rasgos estandarizados.

    Solo usa train. Un valor pequeño señala dos clases parecidas en esos rasgos.
    """
    train = (manifest["split"] == "train").to_numpy()       #   mascara de train
    scaled = StandardScaler().fit_transform(features[train])      # estandarizacion de los features de train
    codes = manifest.loc[train, "code"].to_numpy()        #   codigos de las imagenes de train
    centroids = np.stack([scaled[codes == code].mean(axis=0) for code in CODES])     # los rasgos promedio de cada clase son centroides
    distances = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)      #distancias entre cada par de centroides
    return pd.DataFrame(distances, index=list(CODES), columns=list(CODES))    #tabla con nombres e clases


def closest_pairs(
    thumbnails: np.ndarray, manifest: pd.DataFrame, split_a: str, split_b: str, k: int = 12
) -> pd.DataFrame:
    """Los ``k`` pares de imágenes más parecidos entre dos splits.

    Sirve para revisar a ojo si quedaron fotografías casi iguales a ambos lados del split.
    """
    flat = thumbnails.reshape(len(thumbnails), -1)     #cada miniatura aplanada en un vector
    index_a = np.flatnonzero((manifest["split"] == split_a).to_numpy())     #posiciones de las imagenes del primer split
    index_b = np.flatnonzero((manifest["split"] == split_b).to_numpy())     #posiciones del segundo
    a, b = flat[index_a], flat[index_b]      #vectores de cada lado
    squared = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None] - 2 * a @ b.T     #distancia al cuadrado entre todos los pares, con la misma identidad que en preparacion.py
    order = np.argsort(squared, axis=None)[:k]      #posiciones de las k distancias mas pequeñas en la matriz aplanada
    rows, cols = np.unravel_index(order, squared.shape)     #convierte esas posiciones en fila y columna
    return pd.DataFrame(     #tabla de pares
        {
            #indice en el manifest de cada imagen del par
            "index_a": index_a[rows],     
            "index_b": index_b[cols],     
            "code_a": manifest["code"].to_numpy()[index_a[rows]],    #clase de cada una
            "code_b": manifest["code"].to_numpy()[index_b[cols]],
            "distance": np.sqrt(np.maximum(squared[rows, cols], 0) / flat.shape[1]),    #diferencia RMS por pixel. maximum evita negativos por redondeo antes de la raiz
        }
    )


def resolution_ladder(image: Image.Image, sizes: tuple[int, ...]) -> list[Image.Image]:
    """La misma imagen reducida a cada resolución y devuelta al tamaño original."""
    return [image.resize((s, s)).resize(image.size, Image.Resampling.NEAREST) for s in sizes]    #eeduce a (s x s) y vuelve a ampliar sin suavizar (NEAREST) para que se vean los "pixeles".


def detail_loss(image: Image.Image, sizes: tuple[int, ...]) -> list[float]:
    """Error RMS en luminancia al reducir la imagen a cada tamaño y volver a ampliarla.

    Mide cuánto detalle de la imagen original no sobrevive a esa resolución.
    """
    gray = image.convert("L")    #imagen en escala de grises
    reference = np.asarray(gray, dtype=np.float32) / 255.0    #original como arreglo en [0, 1]. es la referencia
    losses = []     #errores para cada tmaño
    for size in sizes:
        restored = gray.resize((size, size)).resize(gray.size, Image.Resampling.BILINEAR)    #reduce y vuelve a ampliar suavizando (BILINEAR)
        difference = np.asarray(restored, dtype=np.float32) / 255.0 - reference    #cual es la diferencia con la original
        losses.append(float(np.sqrt((difference**2).mean())))    #raiz del error cuadratico medio
    return losses   #un error por tamaño
