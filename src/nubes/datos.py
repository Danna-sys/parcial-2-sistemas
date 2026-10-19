"""Dataset, transformaciones y DataLoaders a partir del manifest."""

# Anotaciones de tipo modernas.
from __future__ import annotations

# dataclass para agrupar los tres loaders de una corrida.
from dataclasses import dataclass

# Path para rutas.
from pathlib import Path

# Any para anotar valores de cualquier tipo.
from typing import Any

# pandas para leer el manifest.
import pandas as pd

# PyTorch.
import torch

# PIL para abrir imágenes.
from PIL import Image

# DataLoader arma batches. Dataset es la clase base de la que hereda nuestro dataset.
from torch.utils.data import DataLoader, Dataset

# transforms: operaciones sobre imágenes (redimensionar, voltear, normalizar, etc.).
from torchvision import transforms

# Opciones de augmentation y la configuración de la corrida.
from nubes.config import AUGMENTATIONS, RunConfig

# Nombres de archivos y splits definidos en la preparación.
from nubes.preparacion import MANIFEST_NAME, SPLITS, STATS_NAME

# Utilidades para JSON, rutas y semillas.
from nubes.utilidades import read_json, resolve_path, seed_worker, seeded_generator


def load_manifest(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee ``manifest.csv`` y ``stats.json``."""
    # Ruta esperada del manifest.
    manifest_path = data_dir / MANIFEST_NAME
    # Si no existe es porque no se corrió la preparación. Se avisa cómo arreglarlo.
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Primero corre: uv run nubes preparar --source <carpeta>"
        )
    # Devuelve la tabla de imágenes y el diccionario de estadísticas.
    return pd.read_csv(manifest_path), read_json(data_dir / STATS_NAME)


def image_root(stats: dict[str, Any]) -> Path:
    # Carpeta donde están las imágenes, según lo que guardó la preparación.
    return resolve_path(stats["source"])


# Dataset propio. PyTorch exige que tenga __len__ y __getitem__.
class CloudDataset(Dataset):
    """Imágenes de un split. Devuelve ``(tensor, etiqueta)``.

    La posición ``i`` corresponde a la fila ``i`` de ``frame``.
    """

    def __init__(self, frame: pd.DataFrame, root: Path, transform: Any, classes: list[str]):
        # Filas del manifest de este split, con índice reiniciado a 0, 1, 2, ...
        self.frame = frame.reset_index(drop=True)
        # Carpeta raíz de las imágenes.
        self.root = root
        # Transformaciones que se aplican a cada imagen al leerla.
        self.transform = transform
        # Nombres de las clases en orden de etiqueta.
        self.classes = classes
        # Rutas como lista simple. Es más rápido que consultar la tabla en cada lectura.
        self._paths = self.frame["path"].tolist()
        # Etiquetas numéricas como lista simple.
        self._labels = self.frame["label"].tolist()

    def __len__(self) -> int:
        # Cuántas imágenes hay. El DataLoader lo usa para saber cuántos batches armar.
        return len(self._paths)

    def load_image(self, index: int) -> Image.Image:
        # Abre la imagen número index.
        with Image.open(self.root / self._paths[index]) as image:
            # La convierte a RGB para que todas tengan 3 canales, aunque alguna venga en gris.
            return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Lo que pide el DataLoader: la imagen transformada en tensor y su etiqueta.
        return self.transform(self.load_image(index)), self._labels[index]


def build_transform(
    image_size: int, mean: list[float], std: list[float], augment: str = "none"
) -> transforms.Compose:
    """Pipeline de entrada. ``augment`` solo debe ser distinto de ``none`` en train.

    ``flip_hv`` voltea también en vertical. Pone el cielo abajo y existe para medir qué
    pasa cuando la transformación no respeta la escena.
    """
    # Se valida que la opción exista.
    if augment not in AUGMENTATIONS:
        raise ValueError(f"augment debe estar en {AUGMENTATIONS}")
    # Tamaño final cuadrado.
    size = (image_size, image_size)
    # Lista de pasos que se aplicarán en orden.
    steps: list[Any] = []
    # Con recorte aleatorio: toma entre el 60 % y el 100 % del área con proporción casi cuadrada
    # y lo lleva al tamaño final. Simula acercar la cámara.
    if augment in ("flip_crop", "flip_crop_color"):
        steps.append(transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)))
    # Sin recorte: solo se redimensiona la imagen completa.
    else:
        steps.append(transforms.Resize(size))
    # Cualquier augmentation incluye volteo horizontal con probabilidad 0.5.
    # Una nube volteada de izquierda a derecha sigue siendo el mismo género.
    if augment != "none":
        steps.append(transforms.RandomHorizontalFlip())
    # Volteo vertical: pone el suelo arriba. Sirve para probar si la red usa el horizonte.
    if augment == "flip_hv":
        steps.append(transforms.RandomVerticalFlip())
    # Cambios aleatorios de brillo y contraste de hasta 20 %. Simula distinta iluminación.
    if augment == "flip_crop_color":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    # Siempre al final:
    # ToTensor pasa la imagen a tensor (canales, alto, ancho) con valores en [0, 1].
    # Normalize resta la media y divide por la desviación de train, canal por canal.
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    # Compose junta todos los pasos en una sola función.
    return transforms.Compose(steps)


def make_dataset(config: RunConfig, split: str, augment: bool = False) -> CloudDataset:
    """Dataset de un split. ``augment=True`` aplica ``config.augment`` y solo vale en train."""
    # El split debe ser train, val o test.
    if split not in SPLITS:
        raise ValueError(f"split debe estar en {SPLITS}")
    # Validation y test siempre se ven sin modificaciones aleatorias.
    if augment and split != "train":
        raise ValueError("La augmentation solo se aplica a train")
    # Lee el manifest y las estadísticas.
    frame, stats = load_manifest(config.data_dir)
    # Construye las transformaciones.
    transform = build_transform(
        # Resolución de entrada de la configuración.
        config.image_size,
        # Media y desviación de train calculadas en la preparación.
        stats["mean_train"],
        stats["std_train"],
        # La augmentation de la configuración solo si se pidió. Si no, ninguna.
        config.augment if augment else "none",
    )
    # Crea el dataset solo con las filas de este split.
    return CloudDataset(
        frame[frame["split"] == split], image_root(stats), transform, stats["classes"]
    )


def make_loader(
    config: RunConfig, split: str, device: torch.device, shuffle: bool = False
) -> DataLoader:
    """Loader de un split. Con ``shuffle=True`` baraja y aplica la augmentation de la config."""
    # El DataLoader toma el dataset y entrega batches de imágenes.
    return DataLoader(
        # El dataset. Barajar y aumentar van juntos: es el loader de entrenamiento.
        make_dataset(config, split, augment=shuffle),
        # Cuántas imágenes por batch.
        batch_size=config.batch_size,
        # Si se barajan en cada época. Solo en train.
        shuffle=shuffle,
        # Procesos paralelos que cargan imágenes. 0 = todo en el proceso principal.
        num_workers=config.num_workers,
        # Con GPU, reserva memoria fija que acelera la copia de CPU a GPU.
        pin_memory=device.type == "cuda",
        # Con workers, los mantiene vivos entre épocas en vez de crearlos de nuevo.
        persistent_workers=config.num_workers > 0,
        # Siembra random y NumPy dentro de cada worker.
        worker_init_fn=seed_worker,
        # Generador propio para barajar de forma reproducible. Sin barajar no hace falta.
        generator=seeded_generator(config.seed) if shuffle else None,
    )


# Agrupa los tres loaders que usa una corrida.
@dataclass(frozen=True)
class Loaders:
    """Loaders de una corrida. Ninguno es de test.

    ``train`` baraja y aumenta. ``train_eval`` recorre train sin barajar ni aumentar y
    sirve para medir el generalization gap en las mismas condiciones que validation.
    """

    # Para entrenar: barajado y con augmentation.
    train: DataLoader
    # Train otra vez, sin barajar ni aumentar, para medirlo igual que validation.
    train_eval: DataLoader
    # Validation sin barajar, para que la posición i sea siempre la misma imagen.
    val: DataLoader


def make_loaders(config: RunConfig, device: torch.device) -> Loaders:
    # Crea los tres loaders. El de test no se crea aquí a propósito.
    return Loaders(
        train=make_loader(config, "train", device, shuffle=True),
        train_eval=make_loader(config, "train", device),
        val=make_loader(config, "val", device),
    )
