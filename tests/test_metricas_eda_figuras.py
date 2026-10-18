import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from nubes import eda, figuras
from nubes.datos import image_root, load_manifest
from nubes.metricas import (
    build_report,
    group_confusion,
    symmetric_confusions,
    top_confusions,
    within_group_share,
)

Y_TRUE = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # etiquetas reales de las ocho imágenes
Y_PRED = np.array([0, 0, 1, 1, 0, 2, 2, 0])  # etiquetas predichas por el modelo


@pytest.fixture(autouse=True)
def close_figures():
    yield  # permite ejecutar la prueba antes de cerrar las figuras
    plt.close("all")  # cierra todas las figuras para evitar que se acumulen entre pruebas


def test_report_values():
    report = build_report(Y_TRUE, Y_PRED, 3, loss=0.5)  # calcula las métricas para tres clases
    assert report.confusion == [[2, 1, 0], [1, 1, 0], [1, 0, 2]]  # verifica los valores de la matriz de confusión
    assert report.accuracy == pytest.approx(5 / 8)  # comprueba la exactitud de las predicciones
    assert report.errors == 3  # verifica la cantidad total de predicciones incorrectas
    assert report.recall == pytest.approx([2 / 3, 1 / 2, 2 / 3])  # comprueba el recall de cada clase
    assert report.macro_f1 == pytest.approx(np.mean(report.f1))  # verifica el promedio de los valores f1 por clase
    assert report.per_class(["a", "b", "c"]).shape == (3, 4)  # comprueba las filas y columnas de la tabla por clase


def test_absent_class_does_not_break_report():
    report = build_report(np.array([0, 0]), np.array([0, 1]), 3, loss=1.0)  # calcula métricas aunque falten clases reales
    assert report.support == [2, 0, 0]  # verifica cuántas muestras reales hay por clase
    assert report.f1[2] == 0.0  # comprueba que la clase ausente tenga f1 igual a cero


def test_confusion_tables():
    report = build_report(Y_TRUE, Y_PRED, 3, loss=0.5)  # genera el reporte de clasificación
    top = top_confusions(report.confusion, ["a", "b", "c"])  # obtiene los errores más frecuentes entre clases
    assert len(top) == 3 and top["errores"].sum() == 3  # verifica la cantidad de errores identificados
    pairs = symmetric_confusions(report.confusion, ["a", "b", "c"])  # compara los errores en ambos sentidos entre clases
    assert pairs.iloc[0]["par"] == "a / b" and pairs.iloc[0]["total"] == 2  # comprueba el par con mayor cantidad de errores


def test_group_confusion_preserves_total_and_share():
    confusion = [[2, 1, 0], [1, 1, 0], [1, 0, 2]]  # define una matriz de confusión de ejemplo
    grouped = group_confusion(confusion, ["x", "x", "y"])  # agrupa las dos primeras clases bajo x y la tercera bajo y
    assert grouped.loc["x", "x"] == 5 and grouped.loc["y", "x"] == 1  # verifica los conteos de la matriz agrupada
    assert grouped.to_numpy().sum() == 8  # comprueba que se conserve el total de imágenes
    assert within_group_share(confusion, ["x", "x", "y"]) == pytest.approx(2 / 3)  # verifica la proporción de errores entre clases del mismo grupo


def test_eda_pipeline(data_dir):
    manifest, stats = load_manifest(data_dir)  # carga el inventario de imágenes y las estadísticas del conjunto
    counts = eda.split_counts(manifest)  # calcula los conteos de imágenes por partición
    assert counts["total"].sum() == len(manifest)  # comprueba que los conteos coincidan con el inventario
    assert counts["share"].sum() == pytest.approx(1.0)  # verifica que las proporciones sumen uno
    assert eda.class_table().shape == (11, 3)  # comprueba la tabla descriptiva de las once clases

    thumbs = eda.load_thumbnails(manifest, image_root(stats), size=16)  # carga miniaturas de 16 por 16 píxeles
    assert thumbs.shape == (len(manifest), 16, 16, 3)  # verifica las dimensiones de las miniaturas
    features = eda.simple_features(thumbs)  # extrae características sencillas de las imágenes
    assert len(features) == len(manifest) and not features.isna().any().any()  # comprueba que todas las imágenes tengan características sin valores faltantes
    assert set(eda.class_mean_images(thumbs, manifest)) == set(stats["classes"])  # verifica que se calcule una imagen media por clase

    accuracy, confusion = eda.knn_baseline(features.to_numpy(), manifest, k=3)  # evalúa una línea base de clasificación con tres vecinos
    assert 0 <= accuracy <= 1  # verifica que la exactitud esté entre cero y uno
    assert confusion.to_numpy().sum() == (manifest["split"] == "val").sum()  # comprueba que la matriz incluya todas las imágenes de validación
    distances = eda.centroid_distances(features.to_numpy(), manifest)  # calcula las distancias entre los centroides de las clases
    assert np.allclose(np.diag(distances), 0) and np.allclose(distances, distances.T)  # verifica que la diagonal sea cero y la matriz sea simétrica

    pairs = eda.closest_pairs(thumbs, manifest, "train", "val", k=5)  # encuentra cinco pares cercanos entre entrenamiento y validación
    assert len(pairs) == 5 and pairs["distance"].is_monotonic_increasing  # comprueba la cantidad de pares y su orden por distancia
    assert set(manifest["split"].iloc[pairs["index_a"]]) == {"train"}  # verifica que las imágenes del primer conjunto pertenezcan a train

    figuras.plot_class_counts(counts)  # genera la gráfica de cantidades por clase
    figuras.plot_feature_boxes(features, manifest["code"], stats["classes"])  # genera diagramas de caja de las características por clase
    figuras.plot_matrix(confusion, stats["classes"], normalize=True)  # visualiza la matriz de confusión normalizada
    figuras.plot_matrix(distances, stats["classes"], fmt=".1f")  # visualiza las distancias entre centroides
    figuras.plot_matrix(confusion - confusion.T, stats["classes"], fmt="+d", diverging=True)  # visualiza las diferencias entre errores en ambos sentidos


def test_resolution_ladder_keeps_size():
    image = Image.new("RGB", (64, 64), "white")  # crea una imagen blanca de 64 por 64 píxeles
    ladder = eda.resolution_ladder(image, (8, 16))  # genera versiones con las resoluciones indicadas
    assert [im.size for im in ladder] == [(64, 64)] * 2  # comprueba que las imágenes resultantes conserven el tamaño original


def test_detail_loss_grows_as_resolution_drops():
    rng = np.random.default_rng(0)  # inicializa el generador aleatorio con una semilla fija
    image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))  # crea una imagen aleatoria en RGB
    losses = eda.detail_loss(image, (8, 32, 64))  # calcula la pérdida de detalle para distintas resoluciones
    assert losses[0] > losses[1] > losses[2] == 0.0  # verifica que la pérdida disminuya al aumentar la resolución y sea cero en la original


def test_training_figures(config):
    from nubes.experimento import run_seeds
    from nubes.resumen import load_experiment, per_class_recall, runs_table

    run_seeds(config, [1, 2], verbose=False)  # ejecuta el experimento con dos semillas
    run_seeds(config.with_changes(name="otra", dropout=0.2), [1, 2], verbose=False)  # ejecuta otra configuración con dropout
    results = load_experiment(config.runs_dir, config.experiment)  # carga los resultados de las configuraciones ejecutadas
    figuras.plot_history(results["tiny"][0].history)  # grafica la evolución de las métricas durante el entrenamiento
    figuras.plot_comparison(results)  # compara los resultados de las configuraciones
    figuras.plot_seed_points(runs_table(results))  # representa los resultados individuales de las semillas
    figuras.plot_recall_by_class(per_class_recall(results, [f"c{i}" for i in range(11)]))  # grafica el recall de cada clase
    figuras.plot_feature_maps(torch.rand(6, 8, 8))  # visualiza seis mapas de características de ejemplo
    figuras.plot_image_grid([np.zeros((4, 4, 3))] * 3, ["a", "b", "c"], ncols=2)  # crea una cuadrícula con tres imágenes de ejemplo
    shown = figuras.denormalize(torch.zeros(2, 3, 4, 4), [0.5] * 3, [0.2] * 3)  # revierte la normalización de dos imágenes
    assert shown.shape == (2, 4, 4, 3) and np.allclose(shown, 0.5)  # comprueba las dimensiones y los valores de las imágenes recuperadas