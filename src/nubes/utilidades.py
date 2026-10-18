"""rutas, device, semillas y lectura de json"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # obtiene la raíz del proyecto a partir de la ubicación de este archivo
DEVICES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")  # dispositivos de ejecución permitidos


def pick_device(preference: str = "auto") -> torch.device:
    """elige el device. con ``auto`` el orden es cuda, mps, cpu

    si se pide un device concreto y no está disponible se lanza ``RuntimeError``
    """
    if preference not in DEVICES:
        raise ValueError(f"device inválido: {preference!r}. Opciones: {DEVICES}")  # verifica que el dispositivo solicitado sea válido

    available = {
        "cuda": torch.cuda.is_available(),  # comprueba si hay una gpu compatible con cuda
        "mps": torch.backends.mps.is_available(),  # comprueba si está disponible el acelerador de apple
        "cpu": True,  # la cpu siempre se considera disponible
    }

    if preference == "auto":
        return torch.device(next(name for name, ok in available.items() if ok))  # selecciona el primer dispositivo disponible según el orden definido

    if not available[preference]:
        raise RuntimeError(f"Se pidió {preference!r} y no está disponible.")  # informa si el dispositivo solicitado no está disponible

    return torch.device(preference)  # devuelve el dispositivo seleccionado


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """siembra ``random``, numpy y pytorch y fija el modo de cudnn"""
    if seed < 0:
        raise ValueError(f"La semilla debe ser no negativa. Se recibió {seed}")  # evita utilizar semillas negativas

    random.seed(seed)  # fija la semilla de los números aleatorios de python
    np.random.seed(seed)  # fija la semilla de numpy
    torch.manual_seed(seed)  # fija la semilla de pytorch
    torch.backends.cudnn.deterministic = deterministic  # controla el comportamiento determinista de cudnn
    torch.backends.cudnn.benchmark = False  # evita que cudnn seleccione algoritmos según pruebas de rendimiento


def seeded_generator(seed: int) -> torch.Generator:
    """generador propio para el shuffle del ``dataloader``"""
    generator = torch.Generator()  # crea un generador independiente de números aleatorios
    generator.manual_seed(seed)  # establece la semilla para que el orden sea reproducible
    return generator  # devuelve el generador configurado


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` que propaga la semilla del worker a ``random`` y numpy"""
    worker_seed = torch.initial_seed() % 2**32  # obtiene una semilla compatible con python y numpy
    random.seed(worker_seed)  # fija la semilla de python para el worker
    np.random.seed(worker_seed)  # fija la semilla de numpy para el worker


def resolve_path(path: str | Path) -> Path:
    """una ruta relativa se interpreta desde la raíz del proyecto"""
    path = Path(path)  # convierte la entrada en un objeto de ruta
    return path if path.is_absolute() else PROJECT_ROOT / path  # conserva las rutas absolutas y completa las relativas desde la raíz del proyecto


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta de destino si no existe
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")  # convierte los datos a json y los guarda con formato legible


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))  # lee el archivo json y convierte su contenido en un diccionario