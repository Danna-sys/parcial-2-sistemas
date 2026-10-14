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

MANIFEST_NAME = "manifest.csv"
STATS_NAME = "stats.json"
SPLITS: tuple[str, ...] = ("train", "val", "test")
FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
STATS_SIZE = 128
DUPLICATE_SIZE = 32
DUPLICATE_TOLERANCE = 3.0


@dataclass(frozen=True)
class CloudClass:
    code: str
    name: str
    level: str
    form: str


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
CODES: tuple[str, ...] = tuple(c.code for c in CLASSES)

PUBLISHED_COUNTS: dict[str, int] = {
    "Ac": 221, "As": 188, "Cb": 242, "Cc": 268, "Ci": 139, "Cs": 287,
    "Ct": 200, "Cu": 182, "Ns": 274, "Sc": 340, "St": 202,
}  # fmt: skip


def find_class_dirs(source: Path) -> dict[str, Path]:
    """Ubica las 11 carpetas de clase dentro de ``source``, sin importar cuán anidadas estén."""
    wanted = {code.lower(): code for code in CODES}
    found: dict[str, Path] = {}
    directories = sorted(p for p in source.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts)
    for directory in directories:
        code = wanted.get(directory.name.lower())
        if code is not None and code not in found:
            found[code] = directory
    missing = [code for code in CODES if code not in found]
    if missing:
        raise FileNotFoundError(f"En {source} faltan las carpetas de clase {missing}")
    return found


def scan_images(source: Path) -> pd.DataFrame:
    """Una fila por imagen legible: ruta relativa, clase, tamaño y modo."""
    rows = []
    for label, (code, directory) in enumerate(find_class_dirs(source).items()):
        files = sorted(
            p
            for p in directory.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
        )
        for file in files:
            with Image.open(file) as image:
                rows.append(
                    {
                        "path": file.relative_to(source).as_posix(),
                        "code": code,
                        "label": label,
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No hay imágenes en {source}")
    return pd.DataFrame(rows)


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def assign_groups(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Agrega ``group``: las imágenes que son la misma fotografía comparten grupo.

    Cada imagen se reduce a 32x32 y se comparan todos los pares. Dos imágenes se unen
    si su diferencia RMS no supera ``DUPLICATE_TOLERANCE`` niveles de 255. Eso cubre
    copias exactas y recompresiones, no recortes ni reencuadres.
    """
    frame = frame.reset_index(drop=True)
    size = (DUPLICATE_SIZE, DUPLICATE_SIZE)
    thumbs = np.empty((len(frame), DUPLICATE_SIZE * DUPLICATE_SIZE * 3), dtype=np.float32)
    for i, path in enumerate(frame["path"]):
        with Image.open(source / path) as image:
            thumb = image.convert("RGB").resize(size, Image.Resampling.BOX)
        thumbs[i] = np.asarray(thumb, dtype=np.float32).ravel()
    norms = (thumbs**2).sum(axis=1)
    limit = DUPLICATE_TOLERANCE**2 * thumbs.shape[1]
    parent = np.arange(len(frame))
    for start in range(0, len(frame), 512):
        block = slice(start, start + 512)
        squared = norms[block, None] + norms[None] - 2 * thumbs[block] @ thumbs.T
        for a, b in zip(*np.nonzero(squared <= limit), strict=True):
            root_a, root_b = _find(parent, start + int(a)), _find(parent, int(b))
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    roots = np.array([_find(parent, i) for i in range(len(frame))])
    frame["group"] = pd.factorize(roots)[0]
    return frame


def split_groups(
    frame: pd.DataFrame, seed: int, fractions: dict[str, float] | None = None
) -> pd.DataFrame:
    """Split estratificado por clase que nunca separa un grupo de duplicados.

    Un grupo con imágenes en varias clases se asigna completo según su primera imagen.
    """
    fractions = fractions or FRACTIONS
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions debe cubrir {SPLITS} y sumar 1")
    rng = np.random.default_rng(seed)
    owner = frame.groupby("group")["label"].first()
    sizes = frame.groupby("group").size()
    assignment: dict[int, str] = {}
    for label in sorted(frame["label"].unique()):
        groups = owner.index[owner == label].to_numpy().copy()
        rng.shuffle(groups)
        total = int(sizes.loc[groups].sum())
        quota = {"test": round(total * fractions["test"]), "val": round(total * fractions["val"])}
        filled = {"test": 0, "val": 0}
        for group in groups:
            size = int(sizes.loc[group])
            target = next((s for s in ("test", "val") if filled[s] + size <= quota[s]), "train")
            if target != "train":
                filled[target] += size
            assignment[int(group)] = target
    frame = frame.copy()
    frame["split"] = frame["group"].map(assignment)
    return frame


def channel_stats(frame: pd.DataFrame, source: Path) -> tuple[list[float], list[float]]:
    """Media y desviación por canal de las imágenes de train, a 128x128 y en [0, 1]."""
    train = frame[frame["split"] == "train"]
    total = np.zeros(3)
    total_sq = np.zeros(3)
    pixels = 0
    for path in train["path"]:
        with Image.open(source / path) as image:
            array = np.asarray(
                image.convert("RGB").resize((STATS_SIZE, STATS_SIZE)), dtype=np.float64
            )
        array = array.reshape(-1, 3) / 255.0
        total += array.sum(axis=0)
        total_sq += (array**2).sum(axis=0)
        pixels += array.shape[0]
    mean = total / pixels
    std = np.sqrt(total_sq / pixels - mean**2)
    return mean.tolist(), std.tolist()


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    sizes = frame.groupby("group").size()
    classes = frame.groupby("group")["code"].nunique()
    return {
        "duplicate_groups": int((sizes > 1).sum()),
        "images_in_duplicate_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "groups_across_classes": int((classes > 1).sum()),
    }


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def prepare(
    source: Path, output: Path, split_seed: int = 2026, overwrite: bool = False
) -> dict[str, Any]:
    """Escribe ``manifest.csv`` y ``stats.json`` en ``output`` y devuelve las estadísticas."""
    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {manifest_path}. Usa --overwrite para regenerarlo.")
    frame = split_groups(assign_groups(scan_images(source), source), seed=split_seed)
    mean, std = channel_stats(frame, source)
    counts = frame.groupby(["split", "code"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=list(SPLITS), columns=list(CODES), fill_value=0)
    per_class = frame["code"].value_counts()
    stats = {
        "source": _portable(source),
        "split_seed": split_seed,
        "fractions": FRACTIONS,
        "classes": list(CODES),
        "total": int(len(frame)),
        "counts": {split: counts.loc[split].astype(int).to_dict() for split in SPLITS},
        "differs_from_published": {
            code: int(per_class.get(code, 0)) - expected
            for code, expected in PUBLISHED_COUNTS.items()
            if int(per_class.get(code, 0)) != expected
        },
        "duplicates": duplicate_report(frame),
        "stats_size": STATS_SIZE,
        "mean_train": mean,
        "std_train": std,
    }
    empty = [
        f"{split}/{code}" for split in SPLITS for code in CODES if counts.loc[split, code] == 0
    ]
    if empty:
        warnings.warn(
            f"Hay clases sin imágenes en algún split: {empty}. Revisa los grupos de duplicados.",
            stacklevel=2,
        )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    write_json(output / STATS_NAME, stats)
    return stats
