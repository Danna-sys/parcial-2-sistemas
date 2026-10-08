"""CloudCNN: CNN propia con profundidad, ancho, pooling y regularización configurables."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nubes.config import POOLINGS


def stage_channels(blocks: int, base_channels: int, max_channels: int) -> list[int]:
    """Canales de salida de cada bloque: se duplican hasta el tope."""
    return [min(base_channels * 2**i, max_channels) for i in range(blocks)]


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
    )
    down: nn.Module = {
        "max": nn.MaxPool2d(2),
        "avg": nn.AvgPool2d(2),
        "stride": nn.Identity(),
    }[pooling]
    return nn.Sequential(
        OrderedDict(
            conv=conv,
            norm=nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            act=nn.ReLU(),
            down=down,
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
            raise ValueError("num_classes debe ser >= 2")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling debe estar en {POOLINGS}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")

        outputs = stage_channels(blocks, base_channels, max_channels)
        inputs = [3, *outputs[:-1]]
        self.stages = nn.Sequential(
            OrderedDict(
                (f"s{i}", _stage(c_in, c_out, kernel_size, pooling, batchnorm))
                for i, (c_in, c_out) in enumerate(zip(inputs, outputs, strict=True), start=1)
            )
        )
        self.squeeze = nn.AdaptiveAvgPool2d(head_grid)
        self.head = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                fc1=nn.Linear(outputs[-1] * head_grid**2, hidden_units),
                act=nn.ReLU(),
                drop=nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                fc2=nn.Linear(hidden_units, num_classes),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.squeeze(self.stages(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
