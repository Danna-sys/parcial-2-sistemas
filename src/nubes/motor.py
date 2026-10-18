
"""Loop de entrenamiento y evaluación. No conoce MLflow ni recibe nunca el loader de test."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Pass:
    """Resultado de recorrer un loader una vez.

    ``probabilities`` tiene forma ``(N, K)`` y ``targets`` forma ``(N,)``, en el orden
    del loader.
    """

    loss: float
    accuracy: float
    probabilities: np.ndarray
    targets: np.ndarray

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)  # obtiene la clase con mayor probabilidad para cada imagen


def _run(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Pass:
    non_blocking = device.type == "cuda"  # permite transferencias asíncronas cuando se utiliza una gpu compatible
    loss_sum = torch.zeros((), device=device)  # inicializa el acumulador de pérdida
    logits_seen: list[torch.Tensor] = []  # almacena las salidas del modelo de cada lote
    targets_seen: list[torch.Tensor] = []  # almacena las etiquetas reales de cada lote
    for images, targets in loader:
        images = images.to(device, non_blocking=non_blocking)  # mueve las imágenes al dispositivo de ejecución
        targets = targets.to(device, non_blocking=non_blocking)  # mueve las etiquetas al mismo dispositivo
        logits = model(images)  # obtiene las puntuaciones de clasificación del modelo
        loss = loss_fn(logits, targets)  # calcula la pérdida entre las predicciones y las etiquetas reales
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)  # limpia los gradientes de la iteración anterior
            loss.backward()  # calcula los gradientes de la pérdida
            optimizer.step()  # actualiza los parámetros del modelo
        loss_sum += loss.detach() * targets.size(0)  # acumula la pérdida ponderada por el tamaño del lote
        logits_seen.append(logits.detach())  # guarda las salidas sin conservar el historial de gradientes
        targets_seen.append(targets)  # guarda las etiquetas reales del lote
    if not targets_seen:
        raise ValueError("El loader no produjo muestras")  # informa si el conjunto de datos está vacío
    logits_all = torch.cat(logits_seen)  # combina las salidas de todos los lotes
    targets_all = torch.cat(targets_seen)  # combina las etiquetas de todos los lotes
    count = targets_all.numel()  # calcula el número total de muestras procesadas
    return Pass(
        loss=loss_sum.item() / count,  # calcula la pérdida media por imagen
        accuracy=(logits_all.argmax(dim=1) == targets_all).float().mean().item(),  # calcula la proporción de predicciones correctas
        probabilities=logits_all.softmax(dim=1).cpu().numpy(),  # convierte las puntuaciones en probabilidades y las lleva a cpu
        targets=targets_all.cpu().numpy(),  # convierte las etiquetas reales en un arreglo de numpy
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Pass:
    """Una época de entrenamiento. Las métricas se acumulan mientras los pesos cambian."""
    model.train()  # activa el modo de entrenamiento
    return _run(model, loader, loss_fn, device, optimizer)  # recorre los lotes y actualiza los parámetros


def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> Pass:
    """Evalúa en modo eval, sin gradientes y sin modificar el modelo."""
    model.eval()  # activa el modo de evaluación
    with torch.inference_mode():  # desactiva el cálculo de gradientes durante la evaluación
        return _run(model, loader, loss_fn, device, optimizer=None)  # calcula las métricas sin actualizar los parámetros


def save_checkpoint(
    path: Path, epoch: int, model: nn.Module, val_loss: float, config: dict[str, Any]
) -> None:
    """Checkpoint de tipos primitivos y tensores, cargable con ``weights_only=True``."""
    path.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta de destino si todavía no existe
    torch.save(
        {
            "epoch": epoch,  # guarda la época del checkpoint
            "model_state": model.state_dict(),  # guarda los parámetros aprendidos por el modelo
            "val_loss": float(val_loss),  # guarda la pérdida de validación asociada
            "config": config,  # guarda la configuración utilizada en el entrenamiento
        },
        path,
    )  # almacena el checkpoint en la ruta indicada


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    config: dict[str, Any],
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Entrena un número fijo de épocas y conserva el checkpoint de menor ``val_loss``.

    Devuelve una fila por época. ``best`` vale 1 en las épocas que mejoraron el mínimo,
    así que la última fila con ``best == 1`` es la del checkpoint guardado. Al terminar,
    ``model`` queda con los pesos de la última época.
    """
    history: list[dict[str, float]] = []  # inicializa el historial de métricas por época
    best_val_loss = math.inf  # establece el menor valor de pérdida de validación como infinito
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()  # registra el instante de inicio de la época
        train = train_epoch(model, train_loader, loss_fn, optimizer, device)  # entrena el modelo durante una época
        val = evaluate(model, val_loader, loss_fn, device)  # evalúa el modelo con los datos de validación
        seconds = time.perf_counter() - start  # calcula el tiempo empleado en la época
        improved = val.loss < best_val_loss  # comprueba si la pérdida de validación mejoró
        if improved:
            best_val_loss = val.loss  # actualiza la mejor pérdida de validación
            save_checkpoint(checkpoint_path, epoch, model, val.loss, config)  # guarda el modelo cuando mejora la validación
        history.append(
            {
                "epoch": epoch,  # registra el número de época
                "train_loss": train.loss,  # registra la pérdida de entrenamiento
                "train_acc": train.accuracy,  # registra la exactitud de entrenamiento
                "val_loss": val.loss,  # registra la pérdida de validación
                "val_acc": val.accuracy,  # registra la exactitud de validación
                "val_macro_f1": float(
                    f1_score(val.targets, val.predictions, average="macro", zero_division=0)
                ),  # calcula el f1 macro de validación
                "seconds": seconds,  # registra la duración de la época
                "best": int(improved),  # indica si esta época mejoró la mejor pérdida de validación
            }
        )
        if verbose:
            print(
                f"época {epoch:>3}/{epochs}  "
                f"train {train.loss:.4f} / {train.accuracy:.3f}  "
                f"val {val.loss:.4f} / {val.accuracy:.3f}  "
                f"{seconds:.1f}s{'  *' if improved else ''}"
            )  # muestra las métricas y el tiempo de la época
    if math.isinf(best_val_loss):
        raise RuntimeError("val_loss nunca fue finita y no se guardó checkpoint")  # informa si no se pudo guardar un checkpoint válido
    return history  # devuelve el historial de métricas de todas las épocas
