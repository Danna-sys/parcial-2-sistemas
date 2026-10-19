"""Fixtures con un CCSN sintético. Ningún test necesita el dataset real ni GPU."""

#Nota: conftest.py es un archivo especial de pytest que carga automáticamente antes de los tests.
# Aquí se definen las "fixtures" que los tests piden como argumento y pytest les entrega.


from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image  # PIL para guardar las imagenes falsas como archivos

matplotlib.use("Agg")   # dibuja figuras en memoria sin abrir ventanas.  los tests de figuras fallarian en un computador sin pantalla


from nubes.config import (
    RunConfig,  # noqa: E402    le dice a Ruff que no se queje de que no estén al principio del archivo
)
from nubes.preparacion import CODES, prepare  # noqa: E402

IMAGES_PER_CLASS = 14     # Cuántas imágenes falsas se crean por clase


def _fake_image(rng: np.random.Generator, label: int) -> Image.Image:
    base = np.full((48, 48, 3), 20 * label + 15, dtype=np.float64)    # imagen de 48x48 de un solo color que depende de la clase: clase 0 es oscura, clase 10 clara
    noise = rng.normal(0, 25, size=base.shape)    # ruido aleatorio para que dos img de la misma clase no sean identicas
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))   # suma, pasa a enteros de 8 bits y convierte en imagen


@pytest.fixture(scope="session")    # significa que se crea una sola vez para todos los tests, no una vez por test
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw") / "CCSN_v2"    # carpeta temporal que pytest borra al terminar
    rng = np.random.default_rng(0)    # generador con semilla fija
    for label, code in enumerate(CODES):     # Una carpeta por clase
        folder = root / code        # crea la carpeta de la clase
        folder.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):         # guarda 14 img falsas con nombres como los del CCSN real
            _fake_image(rng, label).save(folder / f"{code}-N{i:03d}.png")
    duplicate = Image.open(root / "Ac" / "Ac-N000.png")   # se planta un duplicado a proposito para probar la deteccion
    duplicate.save(root / "Ac" / "Ac-N900.png")    # una copia en la misma clase
    duplicate.save(root / "As" / "As-N900.png")         # Y otra copia en una clase distinta
    return root.parent      # devuelve la carpeta que contiene a CCSN_v2


@pytest.fixture(scope="session")
def data_dir(raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("ccsn")      #pide la fixture raw_dir y corre la preparacion sobre ella
    prepare(raw_dir, output, split_seed=7)       # escribe manifest.csv y stats.json en la carpeta temporal

    return output     # devuelve la carpeta preparada.


@pytest.fixture    # sin scope, se crea de nuevo para cada test, asi cada uno tiene su propia carpeta de corridas
def config(data_dir: Path, tmp_path: Path) -> RunConfig:
     return RunConfig(       #configuracion diminuta para que entrenar tome segundos en CPU
        name="tiny",
        experiment="pruebas",
        image_size=32,  # imgs de 32 pixeles
        blocks=2,        # Solo 2 bloques convolucionales
        base_channels=4,        # Solo 4 canales en el primer bloque
        hidden_units=8,        # Capa oculta de 8 neuronas
        batch_size=16,        
        epochs=2,            # dos épocas bastan para probar que todo funciona
        device="cpu",         # para que funcione en cualquier computador
        track_mlflow=False,        # sin MLflow para no ensuciar la base de datos real
        data_dir=data_dir,        # datos sinteticos preparados
        runs_dir=tmp_path / "runs",        #corridas en una carpeta temporal propia de cada test

    )