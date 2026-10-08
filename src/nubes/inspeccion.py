"""Lectura de la arquitectura: formas, campo receptivo y feature maps."""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn


def receptive_field(model: nn.Module, image_size: int) -> pd.DataFrame:
    """Campo receptivo teórico a la salida de cada bloque.

    ``rf`` es el lado en píxeles de entrada que ve una posición del mapa de salida.
    ``fraction`` es ese lado dividido por el lado de la imagen, acotado a 1.
    """
    rf, jump, size = 1, 1, image_size
    rows = []
    for name, stage in model.stages.named_children():
        for layer in stage.children():
            if isinstance(layer, nn.Conv2d | nn.MaxPool2d | nn.AvgPool2d):
                kernel = (
                    layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0]
                )
                stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                rf += (kernel - 1) * jump
                jump *= stride
                size //= stride
        rows.append(
            {
                "stage": name,
                "map_size": size,
                "rf": rf,
                "fraction": min(rf / image_size, 1.0),
                "input_pixels_per_cell": jump,
            }
        )
    return pd.DataFrame(rows).set_index("stage")


def shape_trace(model: nn.Module, image_size: int) -> pd.DataFrame:
    """Forma del tensor después de cada bloque, de ``squeeze`` y de ``head``."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    rows = []
    with torch.inference_mode():
        for name, stage in model.stages.named_children():
            x = stage(x)
            rows.append({"module": f"stages.{name}", "shape": tuple(x.shape[1:])})
        x = model.squeeze(x)
        rows.append({"module": "squeeze", "shape": tuple(x.shape[1:])})
        x = model.head(x)
        rows.append({"module": "head", "shape": tuple(x.shape[1:])})
    model.train(was_training)
    return pd.DataFrame(rows).set_index("module")


def feature_maps(model: nn.Module, images: torch.Tensor, layer: str) -> torch.Tensor:
    """Salida de la capa ``layer`` para ``images``, capturada con un forward hook temporal.

    ``layer`` es un nombre de ``model.named_modules()``, por ejemplo ``stages.s1.act``.
    Devuelve un tensor ``(B, C, H, W)`` en CPU.
    """
    modules = dict(model.named_modules())
    if layer not in modules:
        options = ", ".join(name for name in modules if name.endswith(".act"))
        raise ValueError(f"No existe la capa {layer!r}. Opciones: {options}")
    captured: list[torch.Tensor] = []
    handle = modules[layer].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().cpu())
    )
    was_training = model.training
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.inference_mode():
            model(images.to(device))
    finally:
        handle.remove()
        model.train(was_training)
    return captured[0].clone()
