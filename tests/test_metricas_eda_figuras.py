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

Y_TRUE = np.array([0, 0, 0, 1, 1, 2, 2, 2])
Y_PRED = np.array([0, 0, 1, 1, 0, 2, 2, 0])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_report_values():
    report = build_report(Y_TRUE, Y_PRED, 3, loss=0.5)
    assert report.confusion == [[2, 1, 0], [1, 1, 0], [1, 0, 2]]
    assert report.accuracy == pytest.approx(5 / 8)
    assert report.errors == 3
    assert report.recall == pytest.approx([2 / 3, 1 / 2, 2 / 3])
    assert report.macro_f1 == pytest.approx(np.mean(report.f1))
    assert report.per_class(["a", "b", "c"]).shape == (3, 4)


def test_absent_class_does_not_break_report():
    report = build_report(np.array([0, 0]), np.array([0, 1]), 3, loss=1.0)
    assert report.support == [2, 0, 0]
    assert report.f1[2] == 0.0


def test_confusion_tables():
    report = build_report(Y_TRUE, Y_PRED, 3, loss=0.5)
    top = top_confusions(report.confusion, ["a", "b", "c"])
    assert len(top) == 3 and top["errores"].sum() == 3
    pairs = symmetric_confusions(report.confusion, ["a", "b", "c"])
    assert pairs.iloc[0]["par"] == "a / b" and pairs.iloc[0]["total"] == 2


def test_group_confusion_preserves_total_and_share():
    confusion = [[2, 1, 0], [1, 1, 0], [1, 0, 2]]
    grouped = group_confusion(confusion, ["x", "x", "y"])
    assert grouped.loc["x", "x"] == 5 and grouped.loc["y", "x"] == 1
    assert grouped.to_numpy().sum() == 8
    assert within_group_share(confusion, ["x", "x", "y"]) == pytest.approx(2 / 3)


def test_eda_pipeline(data_dir):
    manifest, stats = load_manifest(data_dir)
    counts = eda.split_counts(manifest)
    assert counts["total"].sum() == len(manifest)
    assert counts["share"].sum() == pytest.approx(1.0)
    assert eda.class_table().shape == (11, 3)

    thumbs = eda.load_thumbnails(manifest, image_root(stats), size=16)
    assert thumbs.shape == (len(manifest), 16, 16, 3)
    features = eda.simple_features(thumbs)
    assert len(features) == len(manifest) and not features.isna().any().any()
    assert set(eda.class_mean_images(thumbs, manifest)) == set(stats["classes"])

    accuracy, confusion = eda.knn_baseline(features.to_numpy(), manifest, k=3)
    assert 0 <= accuracy <= 1
    assert confusion.to_numpy().sum() == (manifest["split"] == "val").sum()
    distances = eda.centroid_distances(features.to_numpy(), manifest)
    assert np.allclose(np.diag(distances), 0) and np.allclose(distances, distances.T)

    pairs = eda.closest_pairs(thumbs, manifest, "train", "val", k=5)
    assert len(pairs) == 5 and pairs["distance"].is_monotonic_increasing
    assert set(manifest["split"].iloc[pairs["index_a"]]) == {"train"}

    figuras.plot_class_counts(counts)
    figuras.plot_feature_boxes(features, manifest["code"], stats["classes"])
    figuras.plot_matrix(confusion, stats["classes"], normalize=True)
    figuras.plot_matrix(distances, stats["classes"], fmt=".1f")
    figuras.plot_matrix(confusion - confusion.T, stats["classes"], fmt="+d", diverging=True)


def test_resolution_ladder_keeps_size():
    image = Image.new("RGB", (64, 64), "white")
    ladder = eda.resolution_ladder(image, (8, 16))
    assert [im.size for im in ladder] == [(64, 64)] * 2


def test_detail_loss_grows_as_resolution_drops():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    losses = eda.detail_loss(image, (8, 32, 64))
    assert losses[0] > losses[1] > losses[2] == 0.0


def test_training_figures(config):
    from nubes.experimento import run_seeds
    from nubes.resumen import load_experiment, per_class_recall, runs_table

    run_seeds(config, [1, 2], verbose=False)
    run_seeds(config.with_changes(name="otra", dropout=0.2), [1, 2], verbose=False)
    results = load_experiment(config.runs_dir, config.experiment)
    figuras.plot_history(results["tiny"][0].history)
    figuras.plot_comparison(results)
    figuras.plot_seed_points(runs_table(results))
    figuras.plot_recall_by_class(per_class_recall(results, [f"c{i}" for i in range(11)]))
    figuras.plot_feature_maps(torch.rand(6, 8, 8))
    figuras.plot_image_grid([np.zeros((4, 4, 3))] * 3, ["a", "b", "c"], ncols=2)
    shown = figuras.denormalize(torch.zeros(2, 3, 4, 4), [0.5] * 3, [0.2] * 3)
    assert shown.shape == (2, 4, 4, 3) and np.allclose(shown, 0.5)
