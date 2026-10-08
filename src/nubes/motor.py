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
        return self.probabilities.argmax(axis=1)


def _run(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Pass:
    non_blocking = device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    logits_seen: list[torch.Tensor] = []
    targets_seen: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        logits = model(images)
        loss = loss_fn(logits, targets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += loss.detach() * targets.size(0)
        logits_seen.append(logits.detach())
        targets_seen.append(targets)
    if not targets_seen:
        raise ValueError("El loader no produjo muestras")
    logits_all = torch.cat(logits_seen)
    targets_all = torch.cat(targets_seen)
    count = targets_all.numel()
    return Pass(
        loss=loss_sum.item() / count,
        accuracy=(logits_all.argmax(dim=1) == targets_all).float().mean().item(),
        probabilities=logits_all.softmax(dim=1).cpu().numpy(),
        targets=targets_all.cpu().numpy(),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Pass:
    """Una época de entrenamiento. Las métricas se acumulan mientras los pesos cambian."""
    model.train()
    return _run(model, loader, loss_fn, device, optimizer)


def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> Pass:
    """Evalúa en modo eval, sin gradientes y sin modificar el modelo."""
    model.eval()
    with torch.inference_mode():
        return _run(model, loader, loss_fn, device, optimizer=None)


def save_checkpoint(
    path: Path, epoch: int, model: nn.Module, val_loss: float, config: dict[str, Any]
) -> None:
    """Checkpoint de tipos primitivos y tensores, cargable con ``weights_only=True``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": float(val_loss),
            "config": config,
        },
        path,
    )


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
    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)
        seconds = time.perf_counter() - start
        improved = val.loss < best_val_loss
        if improved:
            best_val_loss = val.loss
            save_checkpoint(checkpoint_path, epoch, model, val.loss, config)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train.loss,
                "train_acc": train.accuracy,
                "val_loss": val.loss,
                "val_acc": val.accuracy,
                "val_macro_f1": float(
                    f1_score(val.targets, val.predictions, average="macro", zero_division=0)
                ),
                "seconds": seconds,
                "best": int(improved),
            }
        )
        if verbose:
            print(
                f"época {epoch:>3}/{epochs}  "
                f"train {train.loss:.4f} / {train.accuracy:.3f}  "
                f"val {val.loss:.4f} / {val.accuracy:.3f}  "
                f"{seconds:.1f}s{'  *' if improved else ''}"
            )
    if math.isinf(best_val_loss):
        raise RuntimeError("val_loss nunca fue finita y no se guardó checkpoint")
    return history
