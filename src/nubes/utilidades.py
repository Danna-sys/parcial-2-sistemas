"""Rutas, device, semillas y lectura de JSON."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEVICES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")


def pick_device(preference: str = "auto") -> torch.device:
    """Elige el device. Con ``auto`` el orden es CUDA, MPS, CPU.

    Si se pide un device concreto y no está disponible se lanza ``RuntimeError``.
    """
    if preference not in DEVICES:
        raise ValueError(f"device inválido: {preference!r}. Opciones: {DEVICES}")
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if preference == "auto":
        return torch.device(next(name for name, ok in available.items() if ok))
    if not available[preference]:
        raise RuntimeError(f"Se pidió {preference!r} y no está disponible.")
    return torch.device(preference)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Siembra ``random``, NumPy y PyTorch y fija el modo de cuDNN."""
    if seed < 0:
        raise ValueError(f"La semilla debe ser no negativa. Se recibió {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    """Generador propio para el shuffle del ``DataLoader``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` que propaga la semilla del worker a ``random`` y NumPy."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_path(path: str | Path) -> Path:
    """Una ruta relativa se interpreta desde la raíz del proyecto."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
