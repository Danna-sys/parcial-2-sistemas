
"""CloudCNN: CNN propia con profundidad, ancho, pooling y regularización configurables."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nubes.config import POOLINGS


def stage_channels(blocks: int, base_channels: int, max_channels: int) -> list[int]:
    """Canales de salida de cada bloque: se duplican hasta el tope."""
    return [min(base_channels * 2**i, max_channels) for i in range(blocks)]  # calcula los canales de salida de cada bloque según la profundidad y los límites establecidos


def _stage(
    in_channels: int, out_channels: int, kernel_size: int, pooling: str, batchnorm: bool
) -> nn.Sequential:
    """Conv, BatchNorm o Identity, ReLU y reducción a la mitad.

    Con ``pooling="stride"`` la reducción la hace la propia convolución con stride 2.
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=2 if pooling == "stride" else 1,
        padding=kernel_size // 2,
        bias=not batchnorm,
    )  # crea la capa convolucional con los canales, el kernel y el stride configurados
    down: nn.Module = {
        "max": nn.MaxPool2d(2),
        "avg": nn.AvgPool2d(2),
        "stride": nn.Identity(),
    }[pooling]  # selecciona el método de reducción espacial según la configuración
    return nn.Sequential(
        OrderedDict(
            conv=conv,  # aplica la convolución para extraer características
            norm=nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),  # normaliza las activaciones si batchnorm está habilitado
            act=nn.ReLU(),  # aplica la función de activación relu
            down=down,  # reduce las dimensiones espaciales cuando corresponde
        )
    )


class CloudCNN(nn.Module):
    """Bloques convolucionales, promedio a una rejilla y clasificador denso.

    Con los valores por defecto y entrada de 128x128::

        (B, 3, 128, 128)
        stages.s1 -> (B,  32, 64, 64)
        stages.s2 -> (B,  64, 32, 32)
        stages.s3 -> (B, 128, 16, 16)
        stages.s4 -> (B, 256,  8,  8)
        squeeze   -> (B, 256,  1,  1)
        head      -> (B, num_classes)

    Devuelve logits. ``squeeze`` es adaptativo, así que la misma red acepta otras
    resoluciones de entrada.
    """

    def __init__(
        self,
        num_classes: int = 11,
        blocks: int = 4,
        base_channels: int = 32,
        max_channels: int = 256,
        kernel_size: int = 3,
        pooling: str = "max",
        batchnorm: bool = False,
        head_grid: int = 1,
        hidden_units: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser >= 2")  # verifica que haya al menos dos clases para clasificar
        if pooling not in POOLINGS:
            raise ValueError(f"pooling debe estar en {POOLINGS}")  # comprueba que el método de pooling esté permitido
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")  # valida que la probabilidad de dropout esté dentro del intervalo permitido

        outputs = stage_channels(blocks, base_channels, max_channels)  # calcula los canales de salida de los bloques convolucionales
        inputs = [3, *outputs[:-1]]  # define los canales de entrada de cada bloque considerando las imágenes rgb
        self.stages = nn.Sequential(
            OrderedDict(
                (f"s{i}", _stage(c_in, c_out, kernel_size, pooling, batchnorm))
                for i, (c_in, c_out) in enumerate(zip(inputs, outputs, strict=True), start=1)
            )
        )  # construye los bloques convolucionales en el orden establecido
        self.squeeze = nn.AdaptiveAvgPool2d(head_grid)  # ajusta el mapa de características a una rejilla espacial fija
        self.head = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),  # transforma los mapas de características en un vector
                fc1=nn.Linear(outputs[-1] * head_grid**2, hidden_units),  # conecta las características con una capa densa oculta
                act=nn.ReLU(),  # aplica relu a la salida de la capa oculta
                drop=nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # aplica dropout si está configurado
                fc2=nn.Linear(hidden_units, num_classes),  # produce una salida por cada clase
            )
        )  # construye la cabeza clasificadora de la cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.squeeze(self.stages(x)))  # pasa la entrada por los bloques, el pooling adaptativo y el clasificador


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # cuenta los parámetros que pueden actualizarse durante el entrenamiento
