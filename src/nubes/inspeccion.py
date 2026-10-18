
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
    rf, jump, size = 1, 1, image_size  # inicializa el campo receptivo, el salto entre posiciones y el tamaño de la imagen
    rows = []  # almacena los resultados de cada bloque
    for name, stage in model.stages.named_children():
        for layer in stage.children():
            if isinstance(layer, nn.Conv2d | nn.MaxPool2d | nn.AvgPool2d):  # identifica las capas que modifican el campo receptivo
                kernel = (
                    layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0]
                )  # obtiene el tamaño del kernel de la capa
                stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]  # obtiene el salto de la capa
                rf += (kernel - 1) * jump  # actualiza el tamaño del campo receptivo
                jump *= stride  # actualiza la distancia entre posiciones consecutivas del mapa
                size //= stride  # calcula el tamaño espacial de salida
        rows.append(
            {
                "stage": name,  # registra el nombre del bloque
                "map_size": size,  # registra el tamaño del mapa de salida
                "rf": rf,  # registra el campo receptivo en píxeles
                "fraction": min(rf / image_size, 1.0),  # calcula la proporción de la imagen cubierta por el campo receptivo
                "input_pixels_per_cell": jump,  # registra cuántos píxeles de entrada corresponden al salto entre celdas
            }
        )
    return pd.DataFrame(rows).set_index("stage")  # devuelve los resultados en una tabla indexada por bloque


def shape_trace(model: nn.Module, image_size: int) -> pd.DataFrame:
    """Forma del tensor después de cada bloque, de ``squeeze`` y de ``head``."""
    was_training = model.training  # guarda el estado actual del modelo
    model.eval()  # cambia el modelo al modo de evaluación
    device = next(model.parameters()).device  # obtiene el dispositivo donde está el modelo
    x = torch.zeros(1, 3, image_size, image_size, device=device)  # crea una imagen ficticia para comprobar las dimensiones
    rows = []  # almacena las dimensiones obtenidas en cada etapa
    with torch.inference_mode():  # desactiva el cálculo de gradientes durante la inspección
        for name, stage in model.stages.named_children():
            x = stage(x)  # pasa el tensor por el bloque actual
            rows.append({"module": f"stages.{name}", "shape": tuple(x.shape[1:])})  # registra la forma de salida del bloque
        x = model.squeeze(x)  # aplica la operación squeeze para reducir las dimensiones espaciales según la arquitectura
        rows.append({"module": "squeeze", "shape": tuple(x.shape[1:])})  # registra la forma después de squeeze
        x = model.head(x)  # pasa el tensor por la capa de salida
        rows.append({"module": "head", "shape": tuple(x.shape[1:])})  # registra la forma final de salida
    model.train(was_training)  # restaura el estado de entrenamiento que tenía el modelo
    return pd.DataFrame(rows).set_index("module")  # devuelve las dimensiones en una tabla


def feature_maps(model: nn.Module, images: torch.Tensor, layer: str) -> torch.Tensor:
    """Salida de la capa ``layer`` para ``images``, capturada con un forward hook temporal.

    ``layer`` es un nombre de ``model.named_modules()``, por ejemplo ``stages.s1.act``.
    Devuelve un tensor ``(B, C, H, W)`` en CPU.
    """
    modules = dict(model.named_modules())  # crea un diccionario con los módulos del modelo
    if layer not in modules:
        options = ", ".join(name for name in modules if name.endswith(".act"))  # reúne los nombres de las capas de activación disponibles
        raise ValueError(f"No existe la capa {layer!r}. Opciones: {options}")  # informa si la capa solicitada no existe
    captured: list[torch.Tensor] = []  # almacena la salida capturada de la capa
    handle = modules[layer].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().cpu())  # guarda una copia de la salida en la cpu
    )  # registra temporalmente un hook para capturar la salida de la capa
    was_training = model.training  # guarda el estado actual del modelo
    device = next(model.parameters()).device  # obtiene el dispositivo donde está el modelo
    try:
        model.eval()  # cambia el modelo al modo de evaluación
        with torch.inference_mode():  # desactiva el cálculo de gradientes durante la inspección
            model(images.to(device))  # ejecuta el modelo con las imágenes de entrada
    finally:
        handle.remove()  # elimina el hook temporal para no dejarlo registrado
        model.train(was_training)  # restaura el estado anterior del modelo
    return captured[0].clone()  # devuelve una copia de los mapas de características capturados en la cpu
