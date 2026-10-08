import numpy as np
import pytest
import torch

from nubes.experimento import (
    CHECKPOINT_FILE,
    evaluate_test,
    load_model,
    load_result,
    run,
    run_seeds,
)
from nubes.resumen import aggregate, load_experiment, per_class_recall, runs_table, to_markdown


def test_run_writes_every_file_and_no_test(config):
    result = run(config, verbose=False)
    files = {p.name for p in result.run_dir.iterdir()}
    assert files == {
        "config.json", "history.csv", "metrics.json", "predictions_val.csv", CHECKPOINT_FILE
    }  # fmt: skip
    assert len(result.history) == config.epochs
    assert result.history["best"].sum() >= 1
    assert result.best_epoch == result.history.loc[result.history["best"] == 1, "epoch"].iloc[-1]
    assert np.asarray(result.val.confusion).sum() == sum(result.val.support)
    assert result.gap == pytest.approx(result.train.accuracy - result.val.accuracy)


def test_run_is_reproducible_with_same_seed(config):
    first = run(config, verbose=False)
    second = run(config.with_changes(experiment="otra"), verbose=False)
    assert first.history["val_loss"].tolist() == second.history["val_loss"].tolist()
    third = run(config.with_changes(experiment="tercera", seed=99), verbose=False)
    assert first.history["val_loss"].tolist() != third.history["val_loss"].tolist()


def test_finished_run_is_not_overwritten(config):
    run(config, verbose=False)
    with pytest.raises(FileExistsError):
        run(config, verbose=False)
    run(config, verbose=False, overwrite=True)


def test_load_result_matches_run(config):
    result = run(config, verbose=False)
    loaded = load_result(result.run_dir)
    assert loaded.config == result.config
    assert loaded.val == result.val
    assert loaded.summary() == pytest.approx(result.summary())


def test_checkpoint_loads_with_weights_only(config):
    result = run(config, verbose=False)
    checkpoint = torch.load(result.run_dir / CHECKPOINT_FILE, weights_only=True)
    assert checkpoint["epoch"] == result.best_epoch
    model, loaded_config, classes = load_model(result.run_dir, torch.device("cpu"))
    assert loaded_config == config and len(classes) == 11 and not model.training


def test_test_split_is_evaluated_only_once(config):
    result = run(config, verbose=False)
    report = evaluate_test(result.run_dir)
    assert 0.0 <= report.accuracy <= 1.0
    assert (result.run_dir / "predictions_test.csv").is_file()
    with pytest.raises(FileExistsError):
        evaluate_test(result.run_dir)


def test_class_weights_and_regularized_variant_run(config):
    variant = config.with_changes(
        name="variante", class_weights=True, batchnorm=True, dropout=0.3, augment="flip_crop",
        pooling="stride", head_grid=2, weight_decay=1e-4,
    )  # fmt: skip
    assert np.isfinite(run(variant, verbose=False).val.loss)


def test_seeds_and_summary_tables(config):
    run_seeds(config, [1, 2], verbose=False)
    run_seeds(config.with_changes(name="bn", batchnorm=True), [1, 2], verbose=False)
    reloaded = run_seeds(config, [1, 2], verbose=False)
    assert [r.config.seed for r in reloaded] == [1, 2]

    results = load_experiment(config.runs_dir, config.experiment)
    assert set(results) == {"tiny", "bn"}
    table = runs_table(results)
    assert len(table) == 4
    summary = aggregate(table)
    assert summary[("seeds", "")].tolist() == [2, 2]
    markdown = to_markdown(summary)
    assert markdown.count("\n") == 3 and "±" in markdown
    recall = per_class_recall(results, [f"c{i}" for i in range(11)])
    assert recall.shape == (11, 2)
    assert per_class_recall(results, [f"c{i}" for i in range(11)], stat="std").shape == (11, 2)


def test_run_seeds_refuses_stale_runs(config):
    run_seeds(config, [1], verbose=False)
    with pytest.raises(ValueError, match="otra configuración"):
        run_seeds(config.with_changes(epochs=3), [1], verbose=False)


def test_mlflow_logging(config, tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.chdir(tmp_path)
    run(config.with_changes(track_mlflow=True), verbose=False)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[config.experiment])
    assert len(runs) == 1
    assert runs.iloc[0]["params.blocks"] == "2"
    assert "metrics.final_val_macro_f1" in runs.columns
