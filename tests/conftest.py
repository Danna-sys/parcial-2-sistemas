"""Fixtures con un CCSN sintético. Ningún test necesita el dataset real ni GPU."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")

from nubes.config import RunConfig  # noqa: E402
from nubes.preparacion import CODES, prepare  # noqa: E402

IMAGES_PER_CLASS = 14


def _fake_image(rng: np.random.Generator, label: int) -> Image.Image:
    base = np.full((48, 48, 3), 20 * label + 15, dtype=np.float64)
    noise = rng.normal(0, 25, size=base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw") / "CCSN_v2"
    rng = np.random.default_rng(0)
    for label, code in enumerate(CODES):
        folder = root / code
        folder.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            _fake_image(rng, label).save(folder / f"{code}-N{i:03d}.png")
    duplicate = Image.open(root / "Ac" / "Ac-N000.png")
    duplicate.save(root / "Ac" / "Ac-N900.png")
    duplicate.save(root / "As" / "As-N900.png")
    return root.parent


@pytest.fixture(scope="session")
def data_dir(raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("ccsn")
    prepare(raw_dir, output, split_seed=7)
    return output


@pytest.fixture
def config(data_dir: Path, tmp_path: Path) -> RunConfig:
    return RunConfig(
        name="tiny",
        experiment="pruebas",
        image_size=32,
        blocks=2,
        base_channels=4,
        hidden_units=8,
        batch_size=16,
        epochs=2,
        device="cpu",
        track_mlflow=False,
        data_dir=data_dir,
        runs_dir=tmp_path / "runs",
    )
