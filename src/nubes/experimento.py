"""Orquesta una corrida: datos, modelo, entrenamiento, métricas, archivos y MLflow.

Cada corrida deja en su carpeta ``config.json``, ``history.csv``, ``metrics.json``,
``predictions_val.csv`` y ``best.pt``. El test solo se toca en ``evaluate_test``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from nubes.config import RunConfig, diff
from nubes.datos import CloudDataset, load_manifest, make_loader, make_loaders
from nubes.metricas import Report, build_report  #reporte de metricas
from nubes.modelo import CloudCNN, count_parameters  #modelo y conteo de parametros.
from nubes.motor import Pass, evaluate, fit  #loop de entrenamiento y evaluaciaocn
from nubes.utilidades import PROJECT_ROOT, pick_device, read_json, seed_everything, write_json

CONFIG_FILE = "config.json"    #la configuracion usada
HISTORY_FILE = "history.csv"   #una fila por epoca
METRICS_FILE = "metrics.json"     #metricas finales de train y valudation
CHECKPOINT_FILE = "best.pt"     #pesos del modelo final
TEST_FILE = "test.json"       #resultado de test, solo despues de evaluacion final


@dataclass(frozen=True)    #sera todo lo que queda de una corrida terminada
class RunResult:
    """Lo que queda de una corrida. ``train`` y ``val`` se miden con el mejor checkpoint."""

    config: RunConfig    # configuracion usada.
    run_dir: Path       # carpeta donde se guardó
    history: pd.DataFrame    #historial epoca por epoca
    train: Report     # metricas de train con el mejor checkpoint, en modo eval
    val: Report    # metricas de validation con el mejor checkpoint
    best_epoch: int   # epoca del mejor checkpoint
    parameters: int    # numero de parametros entrenables
    device: str     # Device en que se entrenó

    @property
    def gap(self) -> float:
        """Generalization gap: accuracy de train en modo eval menos accuracy de validation."""
        return self.train.accuracy - self.val.accuracy    # que tan mejor le va a la red con imags que ya vio que con nuevas

    def summary(self) -> dict[str, float]:
        return {     #resumen de una linea por corrida. Lo usan la terminal, MLflow y las tablas comparativas
            "best_epoch": self.best_epoch,
            "val_loss": self.val.loss,
            "val_acc": self.val.accuracy,
            "val_macro_f1": self.val.macro_f1,
            "train_acc": self.train.accuracy,
            "gap": self.gap,
            "val_errors": self.val.errors,
            "seconds_per_epoch": float(self.history["seconds"].mean()),    #este promedio de seg por epoca es para comparar costo computacional
            "parameters": self.parameters,
        }


def build_model(config: RunConfig, num_classes: int) -> CloudCNN:
    return CloudCNN(   #aqui es el unico lugar en donde se traduce la configuracion a argumentos de la red
        num_classes=num_classes,
        blocks=config.blocks,
        base_channels=config.base_channels,
        max_channels=config.max_channels,
        kernel_size=config.kernel_size,
        pooling=config.pooling,
        batchnorm=config.batchnorm,
        head_grid=config.head_grid,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
    )


def build_loss(config: RunConfig, device: torch.device) -> nn.CrossEntropyLoss:
    """Cross entropy. Con ``class_weights`` cada clase pesa ``N / (K * n_clase)`` en train."""
    if not config.class_weights:      #no hay pesos, con enteropia curzada todas las imagenes pesan igual
        return nn.CrossEntropyLoss()
    _, stats = load_manifest(config.data_dir)     # Con pesos se necesitan los conteos de train
    counts = np.array([stats["counts"]["train"][code] for code in stats["classes"]], dtype=float)     # imags de train por clase, en orden de etiqueta
    weights = counts.sum() / (len(counts) * counts)   # Las clases pequeñas pesan más, para que la red no las ignora
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))    # entropia cruzada con esos pesos, en el mismo device que la red
  

def _predictions_table(dataset: CloudDataset, result: Pass) -> pd.DataFrame:
    rows = np.arange(len(result.targets))    #indices para la probabilidad
    return pd.DataFrame(
        {    #Tabla con una fila por imagen para el análisis de errores
            "path": dataset.frame["path"],   #  que imagen es
            "real": [dataset.classes[i] for i in result.targets],   #codigo de la clse
            "predicha": [dataset.classes[i] for i in result.predictions],   #clse predicha
            "confianza": result.probabilities.max(axis=1),    #probabilidad de la clase predicha
            "p_real": result.probabilities[rows, result.targets],   #probabiidad que le dio a la clase correcta
         }
    )



def _tracking_uri() -> str:
    # dónde guarda MLflow sus registros. si la variable de entorno existe se usa esa, si no, una base de datos SQLite mlflow.db 
    return (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )


def _log_to_mlflow(result: RunResult) -> None:
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri())   # Le dice a MLflow dónde guardar
    name = result.config.experiment        # Cada experimento de MLflow corresponde a una familia de corridas
    if mlflow.get_experiment_by_name(name) is None:    # Si el experimento no existe, se crea.
        location = None     # Carpeta de archivos adjuntos
        if "MLFLOW_TRACKING_URI" not in os.environ:      # Si se usa el MLflow local, los adjuntos van a mlruns/
            location = (PROJECT_ROOT / "mlruns").as_uri()
        mlflow.create_experiment(name, artifact_location=location)    #crea el experimento
    mlflow.set_experiment(name)     #lo marca como activo
    config = result.config
    with mlflow.start_run(run_name=f"{config.name}-seed{config.seed}"):       # Abre un registro nuevo con nombre "configuración-semilla"
        mlflow.log_params(      # Guarda todos los hiperparámetros, el número de parámetros y el device
            {**config.to_dict(), "parameters": result.parameters, "device_used": result.device}
        )
        for row in result.history.to_dict("records"):     # Guarda las métricas de cada época para poder graficar curvas en MLflow
            step = int(row["epoch"])     # Número de época como eje x
            mlflow.log_metrics(        # Todas las columnas menos epoch y best
                {k: float(v) for k, v in row.items() if k not in ("epoch", "best")}, step=step
            )
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in result.summary().items()})    # Guarda las métricas finales con prefijo final_
        for file in (CONFIG_FILE, HISTORY_FILE, METRICS_FILE, "predictions_val.csv"):      # Adjunta los archivos de la corrida
            mlflow.log_artifact(str(result.run_dir / file))


def run(config: RunConfig, verbose: bool = True, overwrite: bool = False) -> RunResult:
    """Entrena con train, selecciona con validation y guarda todo en ``config.run_dir``.

    Una corrida terminada no se pisa salvo que se pida ``overwrite=True``.
    """
    run_dir = config.run_dir     # Carpeta de esta corrida
    if (run_dir / METRICS_FILE).exists() and not overwrite:    # Si ya terminó antes, no se repite sin permiso para no perder resultados
        raise FileExistsError(f"La corrida {run_dir} ya existe. Usa overwrite para repetirla.")
    for stale in (METRICS_FILE, TEST_FILE):      # Si se va a repetir, se borran las métricas y el test viejos para no mezclarlos
        (run_dir / stale).unlink(missing_ok=True)

    # 1. Elige GPU o CPU
    device = pick_device(config.device)
    # 2. Siembra todo antes de crear datos y red, para qeu los pesos iniciales sean reproducibles
    seed_everything(config.seed, config.deterministic)
    # 3. Crea los loaders de train y validation unicamente
    loaders = make_loaders(config, device)
    classes = loaders.val.dataset.classes      # Nombres de las clases
    # 4. Crea la red y la mueve al device
    model = build_model(config, len(classes)).to(device)
    # 5. Crea la función de pérdida
    loss_fn = build_loss(config, device)
    # 6. Crea el optimizador AdamW con learning rate y weight decay de la configuración
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if verbose:   # Mensaje de inicio
        print(f"{config.experiment} / {config.name} / seed {config.seed} en {device}")
    checkpoint_path = run_dir / CHECKPOINT_FILE     # Ruta del mejor checkpoint
    # 7. Entrena todas las épocas. fit guarda el checkpoint de menor val_loss
    rows = fit(
        model, loaders.train, loaders.val, loss_fn, optimizer, device,
        config.epochs, checkpoint_path, config.to_dict(), verbose,
    )  

    # 8. Al terminar, la red tiene los pesos de la ULIMA época. Se recargan los del MEJOR
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)    # weights_only=True carga solo datos, sin ejecutar código guardado en el archivo
    model.load_state_dict(checkpoint["model_state"])    # Pone esos pesos en la red
    # 9. EvalUA train sin augmentation y en modo eval, para que el gap sea justo
    train_pass = evaluate(model, loaders.train_eval, loss_fn, device)
    val_pass = evaluate(model, loaders.val, loss_fn, device)    # Evalúa validation con el mismo checkpoint
    # 10. Arma el resultado
    result = RunResult(
        config=config,
        run_dir=run_dir,
        history=pd.DataFrame(rows),   # Historial como tabla
        train=build_report(     # Reporte completo de train
            train_pass.targets, train_pass.predictions, len(classes), train_pass.loss
        ),
        val=build_report(val_pass.targets, val_pass.predictions, len(classes), val_pass.loss),         # Reporte completo de validation
        best_epoch=int(checkpoint["epoch"]),   # Época del checkpoint
        parameters=count_parameters(model),    # Tamaño de la red
        device=str(device),      # Device como texto
    )

    # 11. Guarda todo en la carpeta de la corrida
    write_json(run_dir / CONFIG_FILE, config.to_dict())   #configuracion
    result.history.to_csv(run_dir / HISTORY_FILE, index=False)    #historial
    _predictions_table(loaders.val.dataset, val_pass).to_csv(   # Las predicciones de validation, imagen por imagen
        run_dir / "predictions_val.csv", index=False
    )
    # las metricas finales
    write_json(
        run_dir / METRICS_FILE,
        {
            "classes": classes,
            "best_epoch": result.best_epoch,
            "parameters": result.parameters,
            "device": result.device,
            "summary": result.summary(),
            "train": result.train.to_dict(),
            "val": result.val.to_dict(),
        },
    )
    # 12. Registra en MLflow si está activado
    if config.track_mlflow:
        _log_to_mlflow(result)

    return result


def run_seeds(config: RunConfig, seeds: list[int], verbose: bool = True) -> list[RunResult]:
    """La misma configuración con varias semillas.

    Una corrida ya terminada se carga en vez de repetirse. Si la corrida guardada tiene
    otra configuración se lanza un error para no mezclar resultados.
    """
    results = []    #lista de resutlados, uno por semilla
    for seed in seeds:
        seeded = config.with_changes(seed=seed)    # Copia de la configuración con esa semilla
        if not (seeded.run_dir / METRICS_FILE).exists():    # Si no se ha corrido, se entrena y se pasa a la siguiente semilla
            results.append(run(seeded, verbose=verbose))
            continue
        stored = load_result(seeded.run_dir)     # Si ya existe, se carga desde disco en vez de reentrenar
        changed = diff(seeded, stored.config)     # Se compara la configuración guardada con la pedida
        if changed:     # Si son distintas, se cambió el TOML sin cambiar el nombre. Se avisa para no mezclar
            raise ValueError(
                f"{seeded.run_dir} se entrenó con otra configuración: {changed}. "
                "Cambia name en el TOML o borra esa carpeta."
            )
        results.append(stored)      # Si coinciden, se usa la guardada
    return results


def load_result(run_dir: Path) -> RunResult:
    """Reconstruye un ``RunResult`` desde los archivos de la corrida."""
    metrics = read_json(run_dir / METRICS_FILE)    #lee las metricas
    return RunResult(      #arma el resultado sin volver a entrenar
        config=RunConfig.from_dict(read_json(run_dir / CONFIG_FILE)),
        run_dir=run_dir,
        history=pd.read_csv(run_dir / HISTORY_FILE),
        train=Report.from_dict(metrics["train"]),
        val=Report.from_dict(metrics["val"]),
        best_epoch=metrics["best_epoch"],
        parameters=metrics["parameters"],
        device=metrics["device"],
    )


def load_model(run_dir: Path, device: torch.device) -> tuple[CloudCNN, RunConfig, list[str]]:
    """Modelo con los pesos del mejor checkpoint, en modo eval."""
    checkpoint = torch.load(run_dir / CHECKPOINT_FILE, map_location=device, weights_only=True)    #carga el checkpoints
    config = RunConfig.from_dict(checkpoint["config"])    #carga la config guardada en el checkpoint
    classes = read_json(run_dir / METRICS_FILE)["classes"]    #nombre de las clases
    model = build_model(config, len(classes)).to(device)    #reconstruye la misma arquitectura en una red
    model.load_state_dict(checkpoint["model_state"])    #pone los pesos guardados
    model.eval()    #evaluacion
    return model, config, classes    #devuelve la red lista para predecir


def evaluate_test(run_dir: Path) -> Report:
    """Evaluación final en test del mejor checkpoint de una corrida.

    Se ejecuta una sola vez por corrida: si ``test.json`` existe se lanza un error.
    """
    target = run_dir / TEST_FILE    #archivo donde queda el resultado
    if target.exists():     # Si ya existe, se bloquea. Test se mira una vez para que no se vuelva otra validation
        raise FileExistsError(f"{run_dir} ya fue evaluada en test. El resultado está en {target}.")
    device = pick_device("auto")   
    model, config, classes = load_model(run_dir, device)    #carga la red con el mejor checkpoint
    loader = make_loader(config, "test", device)     #unico lugar en donde se crea el loader de test
    result = evaluate(model, loader, build_loss(config, device), device)    #evalua
    report = build_report(result.targets, result.predictions, len(classes), result.loss)    #arma el reporte   
    _predictions_table(loader.dataset, result).to_csv(run_dir / "predictions_test.csv", index=False)    #guarda las predicciones de cada imagen
    write_json(target, {"classes": classes, "test": report.to_dict()})   #guarda el reporte, entonces ya no se puede repetir eval
    return report
