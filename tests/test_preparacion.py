from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nubes.datos import load_manifest
from nubes.preparacion import CODES, SPLITS, assign_groups, find_class_dirs, prepare, scan_images
from tests.conftest import IMAGES_PER_CLASS


def test_finds_nested_class_dirs(raw_dir: Path):
    found = find_class_dirs(raw_dir)  # busca las carpetas de las clases aunque estén anidadas
    assert list(found) == list(CODES)  # verifica que se encuentren todas las clases en el orden esperado


def test_missing_class_raises(tmp_path: Path):
    (tmp_path / "Ac").mkdir()  # crea una carpeta para una sola clase
    with pytest.raises(FileNotFoundError):
        find_class_dirs(tmp_path)  # comprueba que se genere un error cuando faltan clases


def test_manifest_covers_every_image_once(data_dir: Path):
    frame, stats = load_manifest(data_dir)  # carga el inventario y las estadísticas del conjunto
    assert len(frame) == len(CODES) * IMAGES_PER_CLASS + 2 == stats["total"]  # verifica que el total coincida con las imágenes esperadas
    assert frame["path"].is_unique  # comprueba que cada ruta aparezca una sola vez
    assert set(frame["split"]) == set(SPLITS)  # verifica que estén presentes las tres particiones


def test_split_is_stratified(data_dir: Path):
    frame, _ = load_manifest(data_dir)  # carga el inventario de imágenes
    counts = frame.groupby(["code", "split"]).size().unstack()  # cuenta las imágenes por clase y partición
    assert (counts["val"] >= 1).all() and (counts["test"] >= 1).all()  # verifica que cada clase tenga imágenes en validación y prueba
    assert (counts["train"] > counts["val"]).all()  # comprueba que cada clase tenga más imágenes de entrenamiento que de validación


def test_duplicates_stay_in_one_split(data_dir: Path):
    frame, stats = load_manifest(data_dir)  # carga el inventario y las estadísticas de duplicados
    assert frame.groupby("group")["split"].nunique().max() == 1  # verifica que las imágenes de un mismo grupo no se separen entre particiones
    assert stats["duplicates"]["duplicate_groups"] == 1  # comprueba la cantidad de grupos duplicados
    assert stats["duplicates"]["images_in_duplicate_groups"] == 3  # verifica cuántas imágenes pertenecen a grupos duplicados
    assert stats["duplicates"]["groups_across_classes"] == 1  # comprueba que exista un grupo con imágenes de distintas clases


def test_split_is_reproducible(raw_dir: Path, tmp_path: Path):
    prepare(raw_dir, tmp_path / "a", split_seed=3)  # prepara los datos con la semilla tres
    prepare(raw_dir, tmp_path / "b", split_seed=3)  # repite la preparación con la misma semilla
    prepare(raw_dir, tmp_path / "c", split_seed=4)  # prepara los datos con una semilla diferente
    a, b, c = (pd.read_csv(tmp_path / n / "manifest.csv")["split"] for n in "abc")  # carga las particiones generadas
    assert a.equals(b)  # verifica que la misma semilla produzca la misma división
    assert not a.equals(c)  # comprueba que la semilla diferente produzca otra división


def test_stats_use_train_only_and_look_sane(data_dir: Path):
    _, stats = load_manifest(data_dir)  # carga las estadísticas calculadas durante la preparación
    assert len(stats["mean_train"]) == len(stats["std_train"]) == 3  # verifica que haya un valor por cada canal RGB
    assert all(0 < m < 1 for m in stats["mean_train"])  # comprueba que las medias estén dentro del intervalo de normalización
    assert all(s > 0 for s in stats["std_train"])  # verifica que las desviaciones estándar sean positivas


def test_prepare_refuses_to_overwrite(raw_dir: Path, data_dir: Path):
    with pytest.raises(FileExistsError):
        prepare(raw_dir, data_dir)  # comprueba que no se sobrescriba un inventario existente sin autorización


def test_only_true_copies_are_grouped(tmp_path: Path):
    rng = np.random.default_rng(1)  # inicializa un generador aleatorio con una semilla fija
    level = 40  # define el nivel inicial de intensidad de los píxeles

    for code in CODES:
        (tmp_path / code).mkdir()  # crea la carpeta correspondiente a cada clase
        for i in range(3):
            level += 5  # aumenta la intensidad para generar imágenes diferentes
            pixels = np.clip(level + rng.normal(0, 6, (128, 128, 3)), 0, 255).astype(np.uint8)  # genera una imagen casi uniforme, parecida a un cielo liso
            Image.fromarray(pixels).save(tmp_path / code / f"{i}.png")  # guarda la imagen en la carpeta de la clase
    Image.open(tmp_path / "St" / "0.png").save(tmp_path / "St" / "copia.jpg", quality=90)  # crea una copia real recomprimida como jpg
    grouped = assign_groups(scan_images(tmp_path), tmp_path)  # hace el inventario y agrupa los duplicados
    sizes = grouped.groupby("group").size()  # cuenta cuántas imágenes tiene cada grupo
    assert sizes.max() == 2 and (sizes > 1).sum() == 1  # verifica que solo exista un grupo de dos imágenes
    pair = grouped[grouped["group"] == sizes.idxmax()]["path"].tolist()  # obtiene las rutas de ese grupo
    assert sorted(pair) == ["St/0.png", "St/copia.jpg"]  # verifica que el grupo sea la original y su copia


def test_macos_metadata_is_ignored(tmp_path: Path):
    for code in CODES:
        for root in (tmp_path / "CCSN_v2", tmp_path / "__MACOSX" / "CCSN_v2"):
            (root / code).mkdir(parents=True)  # crea las carpetas reales y las de macos
        Image.new("RGB", (32, 32), (10 * len(code), 90, 200)).save(
            tmp_path / "CCSN_v2" / code / "a.jpg"
        )  # guarda una imagen real por clase
        (tmp_path / "CCSN_v2" / code / "._a.jpg").write_bytes(b"\x00\x05\x16\x07")  # crea un archivo basura de macos junto a la imagen
        (tmp_path / "__MACOSX" / "CCSN_v2" / code / "._a.jpg").write_bytes(b"\x00\x05\x16\x07")  # crea otro dentro de __MACOSX
    frame = scan_images(tmp_path)  # hace el inventario de imágenes
    assert len(frame) == len(CODES)  # verifica que solo se cuenten las imágenes reales
    assert not frame["path"].str.contains("MACOSX|/\\._").any()  # verifica que no entre ningún archivo de macos
