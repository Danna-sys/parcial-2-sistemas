"""Inventario de CCSN, detección de duplicados y split estratificado.

No se copian imágenes. La preparación escribe ``manifest.csv`` con una fila por imagen y
``stats.json`` con los conteos y la normalización calculada solo con train.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from nubes.utilidades import PROJECT_ROOT, write_json

MANIFEST_NAME = "manifest.csv"  # nombre del archivo que guarda el inventario de imágenes
STATS_NAME = "stats.json"  # nombre del archivo que guarda las estadísticas
SPLITS: tuple[str, ...] = ("train", "val", "test")  # particiones del conjunto de datos
FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}  # proporciones para cada partición
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})  # extensiones de imagen aceptadas
STATS_SIZE = 128  # tamaño usado para calcular la normalización
DUPLICATE_SIZE = 32  # tamaño de las imágenes para comparar posibles duplicados
DUPLICATE_TOLERANCE = 3.0  # diferencia máxima para considerar dos imágenes duplicadas


@dataclass(frozen=True)
class CloudClass:
    code: str  # código que identifica la clase
    name: str  # nombre completo del tipo de nube
    level: str  # altura a la que se encuentra la nube
    form: str  # forma o categoría de la nube


CLASSES: tuple[CloudClass, ...] = (
    CloudClass("Ac", "Altocumulus", "medio", "cumuliforme"),
    CloudClass("As", "Altostratus", "medio", "estratiforme"),
    CloudClass("Cb", "Cumulonimbus", "vertical", "cumuliforme"),
    CloudClass("Cc", "Cirrocumulus", "alto", "cumuliforme"),
    CloudClass("Ci", "Cirrus", "alto", "cirriforme"),
    CloudClass("Cs", "Cirrostratus", "alto", "estratiforme"),
    CloudClass("Ct", "Contrail", "alto", "artificial"),
    CloudClass("Cu", "Cumulus", "bajo", "cumuliforme"),
    CloudClass("Ns", "Nimbostratus", "medio", "estratiforme"),
    CloudClass("Sc", "Stratocumulus", "bajo", "cumuliforme"),
    CloudClass("St", "Stratus", "bajo", "estratiforme"),
)

CODES: tuple[str, ...] = tuple(c.code for c in CLASSES)  # códigos de las clases en el orden definido


PUBLISHED_COUNTS: dict[str, int] = {
    "Ac": 221, "As": 188, "Cb": 242, "Cc": 268, "Ci": 139, "Cs": 287,
    "Ct": 200, "Cu": 182, "Ns": 274, "Sc": 340, "St": 202,
}  # cantidades de imágenes reportadas en la publicación original


def find_class_dirs(source: Path) -> dict[str, Path]:
    """Ubica las 11 carpetas de clase dentro de ``source``, sin importar cuán anidadas estén."""
    wanted = {code.lower(): code for code in CODES}  # permite identificar los códigos sin distinguir mayúsculas
    found: dict[str, Path] = {}  # almacena la ruta encontrada para cada clase
    directories = sorted(p for p in source.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts)  # busca carpetas e ignora archivos auxiliares de macOS

    for directory in directories:
        code = wanted.get(directory.name.lower())  # comprueba si el nombre corresponde a una clase
        if code is not None and code not in found:
            found[code] = directory  # guarda la ruta de la clase encontrada

    missing = [code for code in CODES if code not in found]  # identifica las clases que faltan
    if missing:
        raise FileNotFoundError(f"En {source} faltan las carpetas de clase {missing}")  # informa si faltan carpetas

    return found  # devuelve las rutas de las clases


def scan_images(source: Path) -> pd.DataFrame:
    """Una fila por imagen legible: ruta relativa, clase, tamaño y modo."""
    rows = []  # almacena la información de cada imagen

    for label, (code, directory) in enumerate(find_class_dirs(source).items()):  # recorre las clases y asigna etiquetas numéricas
        files = sorted(
            p
            for p in directory.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")  # selecciona imágenes y excluye archivos auxiliares
        )

        for file in files:
            with Image.open(file) as image:  # abre la imagen para consultar sus propiedades
                rows.append(
                    {
                        "path": file.relative_to(source).as_posix(),  # guarda la ruta relativa de la imagen
                        "code": code,  # registra el código de la clase
                        "label": label,  # registra la etiqueta numérica
                        "width": image.width,  # guarda el ancho original
                        "height": image.height,  # guarda el alto original
                        "mode": image.mode,  # registra el modo de color
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No hay imágenes en {source}")  # detiene el proceso si no encuentra imágenes

    return pd.DataFrame(rows)  # convierte los registros en una tabla


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:  # continúa mientras el elemento no sea la raíz de su grupo
        parent[i] = parent[parent[i]]  # acorta el camino para acelerar futuras búsquedas
        i = parent[i]  # avanza hacia la raíz del grupo

    return i  # devuelve el identificador de la raíz


def assign_groups(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Agrega ``group``: las imágenes que son la misma fotografía comparten grupo.

    Cada imagen se reduce a 32x32 y se comparan todos los pares. Dos imágenes se unen
    si su diferencia RMS no supera ``DUPLICATE_TOLERANCE`` niveles de 255. Eso cubre
    copias exactas y recompresiones, no recortes ni reencuadres.
    """
    frame = frame.reset_index(drop=True)  # reinicia los índices para trabajar con posiciones consecutivas
    size = (DUPLICATE_SIZE, DUPLICATE_SIZE)  # define el tamaño de las miniaturas
    thumbs = np.empty((len(frame), DUPLICATE_SIZE * DUPLICATE_SIZE * 3), dtype=np.float32)  # reserva espacio para los píxeles de cada miniatura

    for i, path in enumerate(frame["path"]):
        with Image.open(source / path) as image:  # abre la imagen original
            thumb = image.convert("RGB").resize(size, Image.Resampling.BOX)  # convierte a RGB y reduce su tamaño
        thumbs[i] = np.asarray(thumb, dtype=np.float32).ravel()  # almacena los píxeles en una fila

    norms = (thumbs**2).sum(axis=1)  # calcula la suma de cuadrados de los píxeles de cada imagen
    limit = DUPLICATE_TOLERANCE**2 * thumbs.shape[1]  # calcula el límite de distancia permitido
    parent = np.arange(len(frame))  # inicia cada imagen como un grupo independiente

    for start in range(0, len(frame), 512):  # compara las imágenes en bloques para reducir el uso de memoria
        block = slice(start, start + 512)  # define el bloque actual
        squared = norms[block, None] + norms[None] - 2 * thumbs[block] @ thumbs.T  # calcula las distancias cuadráticas entre las imágenes

        for a, b in zip(*np.nonzero(squared <= limit), strict=True):  # obtiene los pares que cumplen la tolerancia
            root_a, root_b = _find(parent, start + int(a)), _find(parent, int(b))  # busca la raíz de cada grupo
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)  # une los grupos si son diferentes

    roots = np.array([_find(parent, i) for i in range(len(frame))])  # obtiene la raíz de cada imagen
    frame["group"] = pd.factorize(roots)[0]  # asigna identificadores consecutivos a los grupos

    return frame  # devuelve la tabla con los grupos de duplicados


def split_groups(
    frame: pd.DataFrame, seed: int, fractions: dict[str, float] | None = None
) -> pd.DataFrame:
    """Split estratificado por clase que nunca separa un grupo de duplicados.

    Un grupo con imágenes en varias clases se asigna completo según su primera imagen.
    """
    fractions = fractions or FRACTIONS  # utiliza las proporciones predeterminadas si no se especifican otras

    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions debe cubrir {SPLITS} y sumar 1")  # verifica las particiones y sus proporciones

    rng = np.random.default_rng(seed)  # crea un generador aleatorio reproducible
    owner = frame.groupby("group")["label"].first()  # asigna a cada grupo la etiqueta de su primera imagen
    sizes = frame.groupby("group").size()  # calcula el tamaño de cada grupo
    assignment: dict[int, str] = {}  # almacena la partición asignada a cada grupo

    for label in sorted(frame["label"].unique()):  # procesa cada clase por separado
        groups = owner.index[owner == label].to_numpy().copy()  # obtiene los grupos asociados a la clase
        rng.shuffle(groups)  # mezcla los grupos para evitar un orden fijo
        total = int(sizes.loc[groups].sum())  # calcula el total de imágenes de esos grupos
        quota = {"test": round(total * fractions["test"]), "val": round(total * fractions["val"])}  # define las cuotas para prueba y validación
        filled = {"test": 0, "val": 0}  # inicializa los conteos de las particiones

        for group in groups:
            size = int(sizes.loc[group])  # consulta cuántas imágenes tiene el grupo
            target = next((s for s in ("test", "val") if filled[s] + size <= quota[s]), "train")  # asigna el grupo a la partición donde cabe
            if target != "train":
                filled[target] += size  # actualiza el conteo de la partición elegida
            assignment[int(group)] = target  # registra la partición del grupo completo

    frame = frame.copy()  # crea una copia para conservar la tabla original
    frame["split"] = frame["group"].map(assignment)  # asigna a cada imagen la partición de su grupo

    return frame  # devuelve la tabla con las particiones


def channel_stats(frame: pd.DataFrame, source: Path) -> tuple[list[float], list[float]]:
    """Media y desviación por canal de las imágenes de train, a 128x128 y en [0, 1]."""
    train = frame[frame["split"] == "train"]  # selecciona únicamente las imágenes de entrenamiento
    total = np.zeros(3)  # acumula los valores de los canales RGB
    total_sq = np.zeros(3)  # acumula los cuadrados de los valores de los canales RGB
    pixels = 0  # cuenta los píxeles procesados

    for path in train["path"]:
        with Image.open(source / path) as image:  # abre cada imagen de entrenamiento
            array = np.asarray(
                image.convert("RGB").resize((STATS_SIZE, STATS_SIZE)), dtype=np.float64
            )  # convierte la imagen a RGB, ajusta el tamaño y obtiene los píxeles

        array = array.reshape(-1, 3) / 255.0  # organiza los píxeles por canal y normaliza sus valores
        total += array.sum(axis=0)  # suma los valores de cada canal
        total_sq += (array**2).sum(axis=0)  # suma los cuadrados de cada canal
        pixels += array.shape[0]  # actualiza la cantidad de píxeles procesados

    mean = total / pixels  # calcula la media de cada canal
    std = np.sqrt(total_sq / pixels - mean**2)  # calcula la desviación estándar de cada canal

    return mean.tolist(), std.tolist()  # devuelve la media y la desviación como listas


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    sizes = frame.groupby("group").size()  # calcula cuántas imágenes tiene cada grupo
    classes = frame.groupby("group")["code"].nunique()  # cuenta las clases diferentes presentes en cada grupo

    return {
        "duplicate_groups": int((sizes > 1).sum()),  # cuenta los grupos con más de una imagen
        "images_in_duplicate_groups": int(sizes[sizes > 1].sum()),  # cuenta las imágenes que pertenecen a esos grupos
        "largest_group": int(sizes.max()),  # registra el tamaño del grupo más grande
        "groups_across_classes": int((classes > 1).sum()),  # cuenta los grupos que aparecen en varias clases
    }


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()  # convierte la ruta a una ruta relativa al proyecto
    except ValueError:
        return str(path.resolve())  # conserva la ruta absoluta si está fuera del proyecto


def prepare(
    source: Path, output: Path, split_seed: int = 2026, overwrite: bool = False
) -> dict[str, Any]:
    """Escribe ``manifest.csv`` y ``stats.json`` en ``output`` y devuelve las estadísticas."""
    manifest_path = output / MANIFEST_NAME  # construye la ruta del archivo de inventario

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {manifest_path}. Usa --overwrite para regenerarlo.")  # evita sobrescribir el inventario sin autorización

    frame = split_groups(assign_groups(scan_images(source), source), seed=split_seed)  # inventaría las imágenes, agrupa duplicados y divide los datos
    mean, std = channel_stats(frame, source)  # calcula la normalización usando solo train
    counts = frame.groupby(["split", "code"]).size().unstack(fill_value=0)  # cuenta las imágenes por partición y clase
    counts = counts.reindex(index=list(SPLITS), columns=list(CODES), fill_value=0)  # garantiza que aparezcan todas las particiones y clases
    per_class = frame["code"].value_counts()  # cuenta las imágenes de cada clase

    stats = {
        "source": _portable(source),  # registra la ruta de origen
        "split_seed": split_seed,  # guarda la semilla utilizada para dividir los datos
        "fractions": FRACTIONS,  # registra las proporciones previstas
        "classes": list(CODES),  # guarda los códigos de las clases
        "total": int(len(frame)),  # registra el número total de imágenes
        "counts": {split: counts.loc[split].astype(int).to_dict() for split in SPLITS},  # guarda los conteos por partición y clase
        "differs_from_published": {
            code: int(per_class.get(code, 0)) - expected  # calcula la diferencia con el conteo publicado
            for code, expected in PUBLISHED_COUNTS.items()
            if int(per_class.get(code, 0)) != expected  # incluye solo las clases con diferencias
        },
        "duplicates": duplicate_report(frame),  # incorpora el resumen de duplicados
        "stats_size": STATS_SIZE,  # registra el tamaño usado para calcular la normalización
        "mean_train": mean,  # guarda las medias RGB de entrenamiento
        "std_train": std,  # guarda las desviaciones estándar RGB de entrenamiento
    }

    empty = [
        f"{split}/{code}" for split in SPLITS for code in CODES if counts.loc[split, code] == 0
    ]  # identifica las clases que no tienen imágenes en alguna partición

    if empty:
        warnings.warn(
            f"Hay clases sin imágenes en algún split: {empty}. Revisa los grupos de duplicados.",
            stacklevel=2,
        )  # advierte si alguna partición quedó sin imágenes de una clase

    output.mkdir(parents=True, exist_ok=True)  # crea la carpeta de salida si no existe
    frame.to_csv(manifest_path, index=False)  # guarda el inventario en formato CSV
    write_json(output / STATS_NAME, stats)  # guarda las estadísticas en formato JSON

    return stats  # devuelve las estadísticas generadas