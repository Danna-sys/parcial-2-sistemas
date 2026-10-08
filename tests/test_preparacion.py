from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nubes.datos import load_manifest
from nubes.preparacion import CODES, SPLITS, assign_groups, find_class_dirs, prepare, scan_images
from tests.conftest import IMAGES_PER_CLASS


def test_finds_nested_class_dirs(raw_dir: Path):
    found = find_class_dirs(raw_dir)
    assert list(found) == list(CODES)


def test_missing_class_raises(tmp_path: Path):
    (tmp_path / "Ac").mkdir()
    with pytest.raises(FileNotFoundError):
        find_class_dirs(tmp_path)


def test_manifest_covers_every_image_once(data_dir: Path):
    frame, stats = load_manifest(data_dir)
    assert len(frame) == len(CODES) * IMAGES_PER_CLASS + 2 == stats["total"]
    assert frame["path"].is_unique
    assert set(frame["split"]) == set(SPLITS)


def test_split_is_stratified(data_dir: Path):
    frame, _ = load_manifest(data_dir)
    counts = frame.groupby(["code", "split"]).size().unstack()
    assert (counts["val"] >= 1).all() and (counts["test"] >= 1).all()
    assert (counts["train"] > counts["val"]).all()


def test_duplicates_stay_in_one_split(data_dir: Path):
    frame, stats = load_manifest(data_dir)
    assert frame.groupby("group")["split"].nunique().max() == 1
    assert stats["duplicates"]["duplicate_groups"] == 1
    assert stats["duplicates"]["images_in_duplicate_groups"] == 3
    assert stats["duplicates"]["groups_across_classes"] == 1


def test_split_is_reproducible(raw_dir: Path, tmp_path: Path):
    prepare(raw_dir, tmp_path / "a", split_seed=3)
    prepare(raw_dir, tmp_path / "b", split_seed=3)
    prepare(raw_dir, tmp_path / "c", split_seed=4)
    a, b, c = (pd.read_csv(tmp_path / n / "manifest.csv")["split"] for n in "abc")
    assert a.equals(b)
    assert not a.equals(c)


def test_stats_use_train_only_and_look_sane(data_dir: Path):
    _, stats = load_manifest(data_dir)
    assert len(stats["mean_train"]) == len(stats["std_train"]) == 3
    assert all(0 < m < 1 for m in stats["mean_train"])
    assert all(s > 0 for s in stats["std_train"])


def test_prepare_refuses_to_overwrite(raw_dir: Path, data_dir: Path):
    with pytest.raises(FileExistsError):
        prepare(raw_dir, data_dir)


def test_only_true_copies_are_grouped(tmp_path: Path):
    rng = np.random.default_rng(1)
    level = 40
    for code in CODES:
        (tmp_path / code).mkdir()
        for i in range(3):
            level += 5
            pixels = np.clip(level + rng.normal(0, 6, (128, 128, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(tmp_path / code / f"{i}.png")
    Image.open(tmp_path / "St" / "0.png").save(tmp_path / "St" / "copia.jpg", quality=90)
    grouped = assign_groups(scan_images(tmp_path), tmp_path)
    sizes = grouped.groupby("group").size()
    assert sizes.max() == 2 and (sizes > 1).sum() == 1
    pair = grouped[grouped["group"] == sizes.idxmax()]["path"].tolist()
    assert sorted(pair) == ["St/0.png", "St/copia.jpg"]
