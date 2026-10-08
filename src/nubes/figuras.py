"""Figuras con matplotlib. Cada función devuelve la figura y no la guarda."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from nubes.experimento import RunResult

LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")


def _color(index: int) -> str:
    return f"C{index % 10}"


def _image_axes(ax: plt.Axes, title: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def plot_class_counts(counts: pd.DataFrame) -> Figure:
    """Barras apiladas de imágenes por clase y split."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bottom = np.zeros(len(counts))
    for i, split in enumerate(("train", "val", "test")):
        ax.bar(counts.index, counts[split], bottom=bottom, label=split, color=_color(i))
        bottom += counts[split].to_numpy()
    for x, total in enumerate(counts["total"]):
        ax.text(x, total, str(total), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("imágenes")
    ax.set_title("Imágenes por clase y split", pad=24)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_image_grid(
    images: Sequence, titles: Sequence[str], ncols: int = 6, cell: float = 1.8
) -> Figure:
    """Rejilla de imágenes PIL o arreglos ``(H, W, 3)`` en ``[0, 1]``."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell * 1.12))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, image, title in zip(np.atleast_1d(axes).ravel(), images, titles, strict=False):
        ax.axis("on")
        ax.imshow(image)
        _image_axes(ax, title)
    fig.tight_layout()
    return fig


def plot_feature_boxes(
    features: pd.DataFrame, codes: Sequence[str], order: Sequence[str]
) -> Figure:
    """Un boxplot por rasgo, con una caja por clase."""
    names = list(features.columns)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 2.8))
    codes = np.asarray(codes)
    for ax, name in zip(axes.ravel(), names, strict=False):
        ax.boxplot(
            [features.loc[codes == code, name] for code in order],
            tick_labels=list(order),
            showfliers=False,
        )
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_matrix(
    matrix: pd.DataFrame | np.ndarray,
    labels: Sequence[str],
    title: str = "",
    normalize: bool = False,
    fmt: str = "d",
    cmap: str = "Blues",
    diverging: bool = False,
    xlabel: str = "predicha",
    ylabel: str = "real",
) -> Figure:
    """Mapa de calor anotado con los valores de ``matrix``.

    Con ``normalize`` el color de cada fila se divide por su suma y el texto conserva el
    conteo. Con ``diverging`` la escala de color es simétrica alrededor de cero y los
    ceros no se anotan. En una
    matriz de confusión las filas son la clase real y las columnas la predicha.
    """
    values = np.asarray(matrix, dtype=float)
    shown = values / np.maximum(values.sum(axis=1, keepdims=True), 1) if normalize else values
    limit = float(np.abs(shown).max()) or 1.0
    low, high = (-limit, limit) if diverging else (float(shown.min()), float(shown.max()))
    side = 0.55 * len(labels) + 1.8
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = int(values[i, j]) if fmt.endswith("d") else values[i, j]
            if diverging and value == 0:
                continue
            level = abs(shown[i, j]) if diverging else shown[i, j]
            dark = level > (0.6 * limit if diverging else (low + high) / 2)
            ax.text(
                j, i, format(value, fmt), ha="center", va="center", fontsize=8,
                color="white" if dark else "black",
            )  # fmt: skip
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_history(history: pd.DataFrame, title: str = "") -> Figure:
    """Pérdida y accuracy de train y validation por época. La línea vertical es el checkpoint."""
    best_epoch = int(history.loc[history["best"] == 1, "epoch"].iloc[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, label in zip(axes, ("loss", "acc"), ("loss", "accuracy"), strict=True):
        ax.plot(history["epoch"], history[f"train_{metric}"], label="train", color=_color(0))
        ax.plot(history["epoch"], history[f"val_{metric}"], label="validation", color=_color(1))
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("época")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(title or f"Mejor época: {best_epoch}")
    fig.tight_layout()
    return fig


def plot_comparison(
    results: Mapping[str, Sequence[RunResult]],
    metrics: Sequence[str] = ("val_loss", "val_macro_f1"),
) -> Figure:
    """Curvas de validation de varias configuraciones.

    La línea es la media entre semillas y la banda va del mínimo al máximo.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.8))
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        for i, (name, runs) in enumerate(results.items()):
            curves = np.stack([r.history[metric].to_numpy() for r in runs])
            epochs = np.arange(1, curves.shape[1] + 1)
            ax.plot(
                epochs, curves.mean(axis=0), label=f"{name} (n={len(runs)})",
                color=_color(i), linestyle=LINESTYLES[i % len(LINESTYLES)],
            )  # fmt: skip
            ax.fill_between(
                epochs, curves.min(axis=0), curves.max(axis=0), color=_color(i), alpha=0.15
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("época")
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_seed_points(table: pd.DataFrame, metric: str = "val_macro_f1") -> Figure:
    """Un punto por corrida y una raya en la media de cada configuración.

    ``table`` es la salida de ``resumen.runs_table``.
    """
    names = list(dict.fromkeys(table["config"]))
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2.5, 3.6))
    for x, name in enumerate(names):
        values = table.loc[table["config"] == name, metric].to_numpy()
        ax.scatter(np.full(len(values), x), values, color=_color(x), zorder=3)
        ax.hlines(values.mean(), x - 0.25, x + 0.25, color=_color(x))
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por semilla")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_recall_by_class(recall: pd.DataFrame) -> Figure:
    """Barras agrupadas de recall por clase. Una serie por configuración."""
    fig, ax = plt.subplots(figsize=(1.0 * len(recall) + 2, 3.8))
    width = 0.8 / len(recall.columns)
    positions = np.arange(len(recall))
    for i, name in enumerate(recall.columns):
        ax.bar(positions + i * width, recall[name], width=width, label=name, color=_color(i))
    ax.set_xticks(positions + 0.4 - width / 2, recall.index)
    ax.set_ylabel("recall en validation")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncols=min(len(recall.columns), 4), loc="upper center",
              bbox_to_anchor=(0.5, 1.18))  # fmt: skip
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Deshace ``Normalize`` y devuelve ``(B, H, W, 3)`` en ``[0, 1]`` para mostrar."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (images.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def plot_feature_maps(maps: torch.Tensor, title: str = "", max_maps: int = 16) -> Figure:
    """Los primeros ``max_maps`` canales de un feature map ``(C, H, W)``.

    Cada mapa se escala a su propio rango, así que los niveles no son comparables entre mapas.
    """
    count = min(max_maps, maps.shape[0])
    ncols = 8
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, channel in zip(np.atleast_1d(axes).ravel(), maps[:count], strict=False):
        ax.imshow(channel.numpy(), cmap="viridis")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
