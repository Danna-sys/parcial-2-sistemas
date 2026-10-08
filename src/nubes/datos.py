"""Dataset, transformaciones y DataLoaders a partir del manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from nubes.config import AUGMENTATIONS, RunConfig
from nubes.preparacion import MANIFEST_NAME, SPLITS, STATS_NAME
from nubes.utilidades import read_json, resolve_path, seed_worker, seeded_generator


def load_manifest(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee ``manifest.csv`` y ``stats.json``."""
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Primero corre: uv run nubes preparar --source <carpeta>"
        )
    return pd.read_csv(manifest_path), read_json(data_dir / STATS_NAME)


def image_root(stats: dict[str, Any]) -> Path:
    return resolve_path(stats["source"])


class CloudDataset(Dataset):
    """Imágenes de un split. Devuelve ``(tensor, etiqueta)``.

    La posición ``i`` corresponde a la fila ``i`` de ``frame``.
    """

    def __init__(self, frame: pd.DataFrame, root: Path, transform: Any, classes: list[str]):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.classes = classes
        self._paths = self.frame["path"].tolist()
        self._labels = self.frame["label"].tolist()

    def __len__(self) -> int:
        return len(self._paths)

    def load_image(self, index: int) -> Image.Image:
        with Image.open(self.root / self._paths[index]) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.load_image(index)), self._labels[index]


def build_transform(
    image_size: int, mean: list[float], std: list[float], augment: str = "none"
) -> transforms.Compose:
    """Pipeline de entrada. ``augment`` solo debe ser distinto de ``none`` en train.

    ``flip_hv`` voltea también en vertical. Pone el cielo abajo y existe para medir qué
    pasa cuando la transformación no respeta la escena.
    """
    if augment not in AUGMENTATIONS:
        raise ValueError(f"augment debe estar en {AUGMENTATIONS}")
    size = (image_size, image_size)
    steps: list[Any] = []
    if augment in ("flip_crop", "flip_crop_color"):
        steps.append(transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)))
    else:
        steps.append(transforms.Resize(size))
    if augment != "none":
        steps.append(transforms.RandomHorizontalFlip())
    if augment == "flip_hv":
        steps.append(transforms.RandomVerticalFlip())
    if augment == "flip_crop_color":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def make_dataset(config: RunConfig, split: str, augment: bool = False) -> CloudDataset:
    """Dataset de un split. ``augment=True`` aplica ``config.augment`` y solo vale en train."""
    if split not in SPLITS:
        raise ValueError(f"split debe estar en {SPLITS}")
    if augment and split != "train":
        raise ValueError("La augmentation solo se aplica a train")
    frame, stats = load_manifest(config.data_dir)
    transform = build_transform(
        config.image_size,
        stats["mean_train"],
        stats["std_train"],
        config.augment if augment else "none",
    )
    return CloudDataset(
        frame[frame["split"] == split], image_root(stats), transform, stats["classes"]
    )


def make_loader(
    config: RunConfig, split: str, device: torch.device, shuffle: bool = False
) -> DataLoader:
    """Loader de un split. Con ``shuffle=True`` baraja y aplica la augmentation de la config."""
    return DataLoader(
        make_dataset(config, split, augment=shuffle),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=seeded_generator(config.seed) if shuffle else None,
    )


@dataclass(frozen=True)
class Loaders:
    """Loaders de una corrida. Ninguno es de test.

    ``train`` baraja y aumenta. ``train_eval`` recorre train sin barajar ni aumentar y
    sirve para medir el generalization gap en las mismas condiciones que validation.
    """

    train: DataLoader
    train_eval: DataLoader
    val: DataLoader


def make_loaders(config: RunConfig, device: torch.device) -> Loaders:
    return Loaders(
        train=make_loader(config, "train", device, shuffle=True),
        train_eval=make_loader(config, "train", device),
        val=make_loader(config, "val", device),
    )
