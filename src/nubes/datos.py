"""Dataset, transformaciones y DataLoaders a partir del manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image  #PIL abre imagenes
from torch.utils.data import DataLoader, Dataset  #batcches
from torchvision import (
    transforms,  #opera sobre las imagenes: redimensionar, voltear, normalizar, etc.
)

from nubes.config import AUGMENTATIONS, RunConfig
from nubes.preparacion import MANIFEST_NAME, SPLITS, STATS_NAME
from nubes.utilidades import read_json, resolve_path, seed_worker, seeded_generator


def load_manifest(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee ``manifest.csv`` y ``stats.json``."""
    manifest_path = data_dir / MANIFEST_NAME     #la ruta del manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(      #avisa si no existe porque no se ha corrido preparacion.py con anterioridad
            f"No existe {manifest_path}. Primero corre: uv run nubes preparar --source <carpeta>"
        )
    return pd.read_csv(manifest_path), read_json(data_dir / STATS_NAME)


def image_root(stats: dict[str, Any]) -> Path:
    return resolve_path(stats["source"])      #carpeta de las imagenes segun las haya guardado la preparación


class CloudDataset(Dataset):       #este es el dataset propio, pytorch exige que tenga __len__ y __getitem__
    """Imágenes de un split. Devuelve ``(tensor, etiqueta)``.

    La posición ``i`` corresponde a la fila ``i`` de ``frame``.
    """

    def __init__(self, frame: pd.DataFrame, root: Path, transform: Any, classes: list[str]):
        self.frame = frame.reset_index(drop=True)      #filas del manifest de este split, con indice reiniciado a 0
        self.root = root     #carpeta raiz de las imagenes
        self.transform = transform     #las transformaciones que se aplican al leer cada imagen
        self.classes = classes      #mobres de las etiquetas en orden
        self._paths = self.frame["path"].tolist()     #rutas como lista simple
        self._labels = self.frame["label"].tolist()    #etiquetas numericas como lista simples

    def __len__(self) -> int:    #cantidad de imagenes, para saber como dividir batches
        return len(self._paths)

    def load_image(self, index: int) -> Image.Image:
        with Image.open(self.root / self._paths[index]) as image:    #abre la imagen numero [index]
            return image.convert("RGB")     #asegura los tres canales rgb para TODAs las imagenes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.load_image(index)), self._labels[index]   #la imagen transformada en tensor y su etiqueta, como los necesita el Dataloader


def build_transform(
    image_size: int, mean: list[float], std: list[float], augment: str = "none"
) -> transforms.Compose:
    """Pipeline de entrada. ``augment`` solo debe ser distinto de ``none`` en train.

    ``flip_hv`` voltea también en vertical. Pone el cielo abajo y existe para medir qué
    pasa cuando la transformación no respeta la escena.
    """
    if augment not in AUGMENTATIONS:     #se confirma que sea valida
        raise ValueError(f"augment debe estar en {AUGMENTATIONS}")
    size = (image_size, image_size)    #tamaño final del cuadrado
    steps: list[Any] = []     #la lista de pasos a seguir
    if augment in ("flip_crop", "flip_crop_color"):     #recorta aleatoriamente para obtener tamaño final. como hacerle zoom
        steps.append(transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)))
    else:     #sin recortar, solo se redimensiona la imagen completa
        steps.append(transforms.Resize(size))
    if augment != "none":         #cualquier augmentation incluye volteo horizontal con probabilidad 0.5
        steps.append(transforms.RandomHorizontalFlip())
    if augment == "flip_hv":        #volteo vertical que pone el suelo arriba. sirve para probar si la red usa el horizonte
        steps.append(transforms.RandomVerticalFlip())
    if augment == "flip_crop_color":          #cambios aleatorios de brillo y contraste
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]    #Totensor pasa imagen a tensor, y luego se normaliza canal a canal
    return transforms.Compose(steps)


def make_dataset(config: RunConfig, split: str, augment: bool = False) -> CloudDataset:
    """Dataset de un split. ``augment=True`` aplica ``config.augment`` y solo vale en train."""
    if split not in SPLITS:     #tiene que ser uno valido: train, test o val
        raise ValueError(f"split debe estar en {SPLITS}")
    if augment and split != "train":     #train es el unico que tiene modificaciones aleatorias
        raise ValueError("La augmentation solo se aplica a train")
    frame, stats = load_manifest(config.data_dir)    #lee manifest y estadisticas
    transform = build_transform(
        config.image_size,        #resolucion de entrada segun config
        stats["mean_train"],   #media calculada en la preparacion
        stats["std_train"],    #desviacion calculada en la preparacion
        config.augment if augment else "none",    #si se pidio la argumentacion de la configuracion
    )
    return CloudDataset(    #crea un dataset exclusivo para las filas de este split
        frame[frame["split"] == split], image_root(stats), transform, stats["classes"]
    )


def make_loader(
    config: RunConfig, split: str, device: torch.device, shuffle: bool = False
) -> DataLoader:
    """Loader de un split. Con ``shuffle=True`` baraja y aplica la augmentation de la config."""
    return DataLoader(       #el DataLoader toma el dataset y entrega batches de imagenes
        make_dataset(config, split, augment=shuffle),   #data set para entrenar
        batch_size=config.batch_size,          #num de imagenes por batch 
        shuffle=shuffle,            #si se barajan las imagenes (en train)
        num_workers=config.num_workers,           #trbajos simultaneos para agregar imagenes
        pin_memory=device.type == "cuda",      #configuracion de corrida en GPU
        persistent_workers=config.num_workers > 0,      #los mantiene vivos entre epocas en vez de crearlos otra vez
        worker_init_fn=seed_worker,        #siembra random y numpy dentro de cada worker
        generator=seeded_generator(config.seed) if shuffle else None,      #generador para barajar de forma reproducible
    )


@dataclass(frozen=True)    #une los tres loaders de una corrida
class Loaders:
    """Loaders de una corrida. Ninguno es de test.

    ``train`` baraja y aumenta. ``train_eval`` recorre train sin barajar ni aumentar y
    sirve para medir el generalization gap en las mismas condiciones que validation.
    """

    train: DataLoader         # entrena, barajando y aumentando
    train_eval: DataLoader    # entrena otra vez pero sin barajar ni aumentar
    val: DataLoader           # validacion sun barajar


def make_loaders(config: RunConfig, device: torch.device) -> Loaders:
    return Loaders(       #crea los tres loaders. el de test no se crea aqui a proosito
        train=make_loader(config, "train", device, shuffle=True),
        train_eval=make_loader(config, "train", device),
        val=make_loader(config, "val", device),
    )
