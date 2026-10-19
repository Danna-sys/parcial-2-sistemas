"""Figuras con matplotlib. Cada función devuelve la figura y no la guarda."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure  # Figure: el tipo de objeto que devuelve cada funcion
from matplotlib.ticker import (
    MaxNLocator,  # MaxNLocator para que el eje de épocas muestre solo números enteros
)

from nubes.experimento import RunResult  #tipo de resultado de una corrida

LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")


def _color(index: int) -> str:
    return f"C{index % 10}"     #llama a lso 1o colores de la paleta por defecto


def _image_axes(ax: plt.Axes, title: str = "") -> None:
    #quita los ejes que no son necesarios en una foto
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)   #titulo


def plot_class_counts(counts: pd.DataFrame) -> Figure:
    """Barras apiladas de imágenes por clase y split."""
    fig, ax = plt.subplots(figsize=(8, 3.5))    #crea la figura y un eje
    bottom = np.zeros(len(counts))   #altura donde empieza cada barra
    for i, split in enumerate(("train", "val", "test")):   # Una capa de barras por split, una encima de la otra
        ax.bar(counts.index, counts[split], bottom=bottom, label=split, color=_color(i))
        bottom += counts[split].to_numpy()    #la siguiente capa empieza donde termina esta
    for x, total in enumerate(counts["total"]):    #escribe el total encima de las barras
        ax.text(x, total, str(total), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("imágenes")   #etiqueta de y
    ax.set_title("Imágenes por clase y split", pad=24)    #titulo
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))    #leyenda
    ax.spines[["top", "right"]].set_visible(False)   #quita los bordes arriba y derecha
    return fig


def plot_image_grid(
    images: Sequence, titles: Sequence[str], ncols: int = 6, cell: float = 1.8
) -> Figure:
    """Rejilla de imágenes PIL o arreglos ``(H, W, 3)`` en ``[0, 1]``."""
    nrows = int(np.ceil(len(images) / ncols))    # Filas necesarias, redondeando hacia arriba
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell * 1.12))   # Crea la grid de ejes. cell es el tamaño de cada casilla en pulgadas
    for ax in np.atleast_1d(axes).ravel():   # Apaga todas las casillas. atleast_1d y ravel la vuelven una lista plana
        ax.axis("off")
    for ax, image, title in zip(np.atleast_1d(axes).ravel(), images, titles, strict=False):   # Enciende y dibuja solo las que tienen imagen
        ax.axis("on")    
        ax.imshow(image)     #muestra la imagen
        _image_axes(ax, title)   #quita marcas y pone el titulo
    fig.tight_layout()   #margenes
    return fig


def plot_feature_boxes(
    features: pd.DataFrame, codes: Sequence[str], order: Sequence[str]
) -> Figure:
    """Un boxplot por rasgo, con una caja por clase."""
    names = list(features.columns)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))   #filas necesarias
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 2.8))   #crea la grid
    codes = np.asarray(codes)    #clase de cada imagen
    for ax, name in zip(axes.ravel(), names, strict=False):      #una fig por rasgo
        ax.boxplot(    # un diagrama de caja por clase
            [features.loc[codes == code, name] for code in order],
            tick_labels=list(order),
            showfliers=False,
        )
        ax.set_title(name, fontsize=10)    # Titulo de acuerdo al rasgo
        ax.spines[["top", "right"]].set_visible(False)    #quita los bordes top y right
    for ax in axes.ravel()[len(names) :]:   #apaga las casillas que sobraron
        ax.axis("off")
    fig.tight_layout()   #margenes
    return fig


def plot_matrix(
    matrix: pd.DataFrame | np.ndarray,
    labels: Sequence[str],   #nombres de filas y de columnas
    title: str = "",
    normalize: bool = False,   #si se ciolorea por fraccion de cada fila
    fmt: str = "d",    # formato del número escrito en cada casilla: "d" entero, ".1f" un decimal, "+d" con signo
    cmap: str = "Blues",   #oaleta de colores
    diverging: bool = False,    # Si la escala es simetrica alrededor de 0, para matrices de diferencias
    xlabel: str = "predicha",
    ylabel: str = "real",
) -> Figure:
    """Mapa de calor anotado con los valores de ``matrix``.

    Con ``normalize`` el color de cada fila se divide por su suma y el texto conserva el
    conteo. Con ``diverging`` la escala de color es simétrica alrededor de cero y los
    ceros no se anotan. En una
    matriz de confusión las filas son la clase real y las columnas la predicha.
    """
    values = np.asarray(matrix, dtype=float)   # Matriz como arreglo decimal
    shown = values / np.maximum(values.sum(axis=1, keepdims=True), 1) if normalize else values    # lo que define el color: cada fila dividida por su suma si se normaliza
    limit = float(np.abs(shown).max()) or 1.0    # valor absoluto más grande
    low, high = (-limit, limit) if diverging else (float(shown.min()), float(shown.max()))     # Rango de la escala de color: simétrico si es divergente
    side = 0.55 * len(labels) + 1.8     # tamaño de la figura segun el número de clases
    fig, ax = plt.subplots(figsize=(side, side))       # crea la figura
    image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)         # dibuja la matriz como imagen de colores
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = int(values[i, j]) if fmt.endswith("d") else values[i, j]   # Entero si el formato es de enteros
            if diverging and value == 0:    # En matrices de diferencias los ceros no se escriben para que resalten los cambios
                continue
            level = abs(shown[i, j]) if diverging else shown[i, j]    # Que tan intenso es el color de la casilla
            dark = level > (0.6 * limit if diverging else (low + high) / 2)    # Si es oscuro, el texto va en blanco para que se lea
            ax.text(        # Escribe el número en el centro de la casilla
                j, i, format(value, fmt), ha="center", va="center", fontsize=8,
                color="white" if dark else "black",
            )  # fmt: skip
    # Nombres de las clases en los ejes.
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    # Etiquetas y título.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)         # Barra de colores al lado
    fig.tight_layout()  #margenes

    return fig


def plot_history(history: pd.DataFrame, title: str = "") -> Figure:
    """Pérdida y accuracy de train y validation por época. La línea vertical es el checkpoint."""
    best_epoch = int(history.loc[history["best"] == 1, "epoch"].iloc[-1])    # Época del checkpoint, que seria la última fila que mejoró
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))      # Dos gráficas lado a lad (pérdida y accuracy)
    for ax, metric, label in zip(axes, ("loss", "acc"), ("loss", "accuracy"), strict=True):       # Recorre las dos graficas con su metrica y etiqueta
        ax.plot(history["epoch"], history[f"train_{metric}"], label="train", color=_color(0))   #curva de train
        ax.plot(history["epoch"], history[f"val_{metric}"], label="validation", color=_color(1))  #curva de val
        
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)   # Linea punteada en la época del checkpoint.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # Solo épocas enteras en el eje x
        ax.set_xlabel("época")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)  #otra vez sin los bordes
    axes[0].legend(frameon=False)
    fig.suptitle(title or f"Mejor época: {best_epoch}")   #titulo grande
    fig.tight_layout() #margenes

    return fig


def plot_comparison(
    results: Mapping[str, Sequence[RunResult]],
    metrics: Sequence[str] = ("val_loss", "val_macro_f1"),
) -> Figure:
    """Curvas de validation de varias configuraciones.

    La línea es la media entre semillas y la banda va del mínimo al máximo.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.8))    #una por metrica
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):   #recorre las metricas
        for i, (name, runs) in enumerate(results.items()):       #Recorre las configuraciones
            curves = np.stack([r.history[metric].to_numpy() for r in runs])    # matriz semillas x épocas con la metrica
            epochs = np.arange(1, curves.shape[1] + 1)   # numeros de época 
            ax.plot(     # linea con la media entre semillas
                epochs, curves.mean(axis=0), label=f"{name} (n={len(runs)})",
                color=_color(i), linestyle=LINESTYLES[i % len(LINESTYLES)],
            )  
            ax.fill_between(     # banda sombreada entre la peor y la mejor semilla
                epochs, curves.min(axis=0), curves.max(axis=0), color=_color(i), alpha=0.15
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # Épocas enteras
        ax.set_xlabel("época")     # Etiquetas
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)   # leyenda en la primera grafica
    fig.tight_layout()
    return fig


def plot_seed_points(table: pd.DataFrame, metric: str = "val_macro_f1") -> Figure:
    """Un punto por corrida y una raya en la media de cada configuración.

    ``table`` es la salida de ``resumen.runs_table``.
    """
    names = list(dict.fromkeys(table["config"]))    # Configuraciones en el orden en que aparecen
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2.5, 3.6))     # Crea la figura
    for x, name in enumerate(names):   # Una columna de puntos por config
        values = table.loc[table["config"] == name, metric].to_numpy()    # valores de la metrica de cada semilla
        ax.scatter(np.full(len(values), x), values, color=_color(x), zorder=3)   #punto por semila
        ax.hlines(values.mean(), x - 0.25, x + 0.25, color=_color(x))   # Raya horizontal en la media
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")     # nombres de las configuraciones en el eje x, inclinados
    ax.set_ylabel(metric)      
    ax.set_title(f"{metric} por semilla")
    ax.spines[["top", "right"]].set_visible(False)  #quito los bordes que no necesitp
    fig.tight_layout()

    return fig


def plot_recall_by_class(recall: pd.DataFrame) -> Figure:
    """Barras agrupadas de recall por clase. Una serie por configuración."""
    fig, ax = plt.subplots(figsize=(1.0 * len(recall) + 2, 3.8))   # crea la figura, mas ancha si hay ma clases
    width = 0.8 / len(recall.columns)     # ancho de cada barra para que las del grupo quepan en 0.8 unidades.
    positions = np.arange(len(recall))        # Posicion de cada clase en el eje x
    for i, name in enumerate(recall.columns):       # una serie de barras por configuracion, corrida un poco a la derecha cada vez
        ax.bar(positions + i * width, recall[name], width=width, label=name, color=_color(i))
    ax.set_xticks(positions + 0.4 - width / 2, recall.index)      # Nombre de la clase centrado bajo su grupo de barras
    ax.set_ylabel("recall en validation")   #etiqueta eje y
    ax.set_ylim(0, 1)  # el recall va de 0 a 1
    ax.legend(frameon=False, fontsize=8, ncols=min(len(recall.columns), 4), loc="upper center",   #leyenda
              bbox_to_anchor=(0.5, 1.18)) 
    ax.spines[["top", "right"]].set_visible(False) # sin bordes de arriba y derecha
    fig.tight_layout()

    return fig


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Deshace ``Normalize`` y devuelve ``(B, H, W, 3)`` en ``[0, 1]`` para mostrar."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)   # media como tensor para que se aplique por canal a todo el batch
    std_t = torch.tensor(std).view(1, 3, 1, 1)      # Lo mismo con la desviacion

    return (images.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()       # normalize hizo (x - media) / std. Aqui se hace al revés: x * std + media


def plot_feature_maps(maps: torch.Tensor, title: str = "", max_maps: int = 16) -> Figure:
    """Los primeros ``max_maps`` canales de un feature map ``(C, H, W)``.

    Cada mapa se escala a su propio rango, así que los niveles no son comparables entre mapas.
    """
    count = min(max_maps, maps.shape[0])       # cuantos mapas mostrar
    ncols = 8
    nrows = int(np.ceil(count / ncols))    #filas necesarias
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))   #crea la grid
    for ax in np.atleast_1d(axes).ravel():       # Apaga todas las casilla
        ax.axis("off")
    for ax, channel in zip(np.atleast_1d(axes).ravel(), maps[:count], strict=False):    # dibuja cada canal como imagen. Zonas claras = el filtro respondió fuerte 
        ax.imshow(channel.numpy(), cmap="viridis")
    fig.suptitle(title)
    fig.tight_layout()
    
    return fig