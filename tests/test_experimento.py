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
    result = run(config, verbose=False)  # ejecuta el experimento sin mostrar mensajes
    files = {p.name for p in result.run_dir.iterdir()}  # obtiene los nombres de los archivos generados
    assert files == {
        "config.json", "history.csv", "metrics.json", "predictions_val.csv", CHECKPOINT_FILE
    }  # verifica que se generen todos los archivos esperados y no se evalúe test
    assert len(result.history) == config.epochs  # comprueba que se registre una fila por época
    assert result.history["best"].sum() >= 1  # verifica que se guarde al menos una mejora del modelo
    assert result.best_epoch == result.history.loc[result.history["best"] == 1, "epoch"].iloc[-1]  # comprueba que la mejor época coincida con el historial
    assert np.asarray(result.val.confusion).sum() == sum(result.val.support)  # verifica que la matriz de confusión coincida con el total de muestras
    assert result.gap == pytest.approx(result.train.accuracy - result.val.accuracy)  # comprueba la diferencia entre las exactitudes de train y validation


def test_run_is_reproducible_with_same_seed(config):
    first = run(config, verbose=False)  # ejecuta el experimento con la configuración original
    second = run(config.with_changes(experiment="otra"), verbose=False)  # repite el experimento con la misma semilla y otro nombre
    assert first.history["val_loss"].tolist() == second.history["val_loss"].tolist()  # verifica que la pérdida de validación sea reproducible
    third = run(config.with_changes(experiment="tercera", seed=99), verbose=False)  # ejecuta el experimento con una semilla diferente
    assert first.history["val_loss"].tolist() != third.history["val_loss"].tolist()  # comprueba que cambiar la semilla pueda cambiar los resultados


def test_finished_run_is_not_overwritten(config):
    run(config, verbose=False)  # genera una corrida inicial
    with pytest.raises(FileExistsError):
        run(config, verbose=False)  # comprueba que una corrida existente no se sobrescriba por defecto
    run(config, verbose=False, overwrite=True)  # verifica que se permita sobrescribir explícitamente


def test_load_result_matches_run(config):
    result = run(config, verbose=False)  # ejecuta el experimento y guarda sus resultados
    loaded = load_result(result.run_dir)  # carga los resultados desde la carpeta de la corrida
    assert loaded.config == result.config  # verifica que se conserve la configuración
    assert loaded.val == result.val  # comprueba que las métricas de validación coincidan
    assert loaded.summary() == pytest.approx(result.summary())  # compara los resúmenes de resultados


def test_checkpoint_loads_with_weights_only(config):
    result = run(config, verbose=False)  # ejecuta el experimento y genera el checkpoint
    checkpoint = torch.load(result.run_dir / CHECKPOINT_FILE, weights_only=True)  # carga el checkpoint de forma segura usando solo pesos y tipos admitidos
    assert checkpoint["epoch"] == result.best_epoch  # comprueba que el checkpoint corresponda a la mejor época
    model, loaded_config, classes = load_model(result.run_dir, torch.device("cpu"))  # carga el modelo en la cpu
    assert loaded_config == config and len(classes) == 11 and not model.training  # verifica la configuración, las once clases y el modo de evaluación


def test_test_split_is_evaluated_only_once(config):
    result = run(config, verbose=False)  # ejecuta el entrenamiento sin evaluar test
    report = evaluate_test(result.run_dir)  # evalúa el modelo con el conjunto de prueba
    assert 0.0 <= report.accuracy <= 1.0  # verifica que la exactitud esté entre cero y uno
    assert (result.run_dir / "predictions_test.csv").is_file()  # comprueba que se guarden las predicciones de test
    with pytest.raises(FileExistsError):
        evaluate_test(result.run_dir)  # verifica que no se evalúe test otra vez sobre la misma corrida


def test_class_weights_and_regularized_variant_run(config):
    variant = config.with_changes(
        name="variante", class_weights=True, batchnorm=True, dropout=0.3, augment="flip_crop",
        pooling="stride", head_grid=2, weight_decay=1e-4,
    )  # crea una variante con pesos por clase, normalización, dropout, aumento y regularización
    assert np.isfinite(run(variant, verbose=False).val.loss)  # comprueba que el entrenamiento produzca una pérdida de validación finita


def test_seeds_and_summary_tables(config):
    run_seeds(config, [1, 2], verbose=False)  # ejecuta la configuración con dos semillas
    run_seeds(config.with_changes(name="bn", batchnorm=True), [1, 2], verbose=False)  # ejecuta otra configuración con batch normalization
    reloaded = run_seeds(config, [1, 2], verbose=False)  # vuelve a cargar las corridas existentes
    assert [r.config.seed for r in reloaded] == [1, 2]  # verifica que las corridas estén ordenadas por semilla

    results = load_experiment(config.runs_dir, config.experiment)  # carga las corridas del experimento
    assert set(results) == {"tiny", "bn"}  # comprueba que estén las dos configuraciones
    table = runs_table(results)  # crea una tabla con una fila por corrida
    assert len(table) == 4  # verifica que existan cuatro corridas en total
    summary = aggregate(table)  # calcula la media y desviación estándar por configuración
    assert summary[("seeds", "")].tolist() == [2, 2]  # comprueba que cada configuración tenga dos semillas
    markdown = to_markdown(summary)  # convierte el resumen a formato markdown
    assert markdown.count("\n") == 3 and "±" in markdown  # verifica la estructura y la presencia de media más desviación
    recall = per_class_recall(results, [f"c{i}" for i in range(11)])  # calcula el recall medio de cada clase
    assert recall.shape == (11, 2)  # comprueba que haya once clases y dos configuraciones
    assert per_class_recall(results, [f"c{i}" for i in range(11)], stat="std").shape == (11, 2)  # verifica la tabla de desviaciones estándar del recall


def test_run_seeds_refuses_stale_runs(config):
    run_seeds(config, [1], verbose=False)  # crea una corrida para la configuración original
    with pytest.raises(ValueError, match="otra configuración"):
        run_seeds(config.with_changes(epochs=3), [1], verbose=False)  # verifica que se rechacen corridas existentes con otra configuración


def test_mlflow_logging(config, tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # define la ubicación temporal de la base de datos de mlflow
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)  # configura la dirección de seguimiento de mlflow
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))  # configura la carpeta temporal de artefactos
    monkeypatch.chdir(tmp_path)  # establece la carpeta temporal como directorio de trabajo
    run(config.with_changes(track_mlflow=True), verbose=False)  # ejecuta el experimento registrando sus resultados en mlflow
    mlflow.set_tracking_uri(uri)  # conecta mlflow con la base de datos configurada
    runs = mlflow.search_runs(experiment_names=[config.experiment])  # consulta las corridas registradas
    assert len(runs) == 1  # verifica que se haya registrado una sola corrida
    assert runs.iloc[0]["params.blocks"] == "2"  # comprueba que se haya guardado el número de bloques
    assert "metrics.final_val_macro_f1" in runs.columns  # verifica que se registre la métrica macro f1 de validación