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
from nubes.metricas import Report, build_report
from nubes.modelo import CloudCNN, count_parameters
from nubes.motor import Pass, evaluate, fit
from nubes.utilidades import PROJECT_ROOT, pick_device, read_json, seed_everything, write_json

CONFIG_FILE = "config.json"
HISTORY_FILE = "history.csv"
METRICS_FILE = "metrics.json"
CHECKPOINT_FILE = "best.pt"
TEST_FILE = "test.json"


@dataclass(frozen=True)
class RunResult:
    """Lo que queda de una corrida. ``train`` y ``val`` se miden con el mejor checkpoint."""

    config: RunConfig
    run_dir: Path
    history: pd.DataFrame
    train: Report
    val: Report
    best_epoch: int
    parameters: int
    device: str

    @property
    def gap(self) -> float:
        """Generalization gap: accuracy de train en modo eval menos accuracy de validation."""
        return self.train.accuracy - self.val.accuracy

    def summary(self) -> dict[str, float]:
        return {
            "best_epoch": self.best_epoch,
            "val_loss": self.val.loss,
            "val_acc": self.val.accuracy,
            "val_macro_f1": self.val.macro_f1,
            "train_acc": self.train.accuracy,
            "gap": self.gap,
            "val_errors": self.val.errors,
            "seconds_per_epoch": float(self.history["seconds"].mean()),
            "parameters": self.parameters,
        }


def build_model(config: RunConfig, num_classes: int) -> CloudCNN:
    return CloudCNN(
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
    if not config.class_weights:
        return nn.CrossEntropyLoss()
    _, stats = load_manifest(config.data_dir)
    counts = np.array([stats["counts"]["train"][code] for code in stats["classes"]], dtype=float)
    weights = counts.sum() / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def _predictions_table(dataset: CloudDataset, result: Pass) -> pd.DataFrame:
    rows = np.arange(len(result.targets))
    return pd.DataFrame(
        {
            "path": dataset.frame["path"],
            "real": [dataset.classes[i] for i in result.targets],
            "predicha": [dataset.classes[i] for i in result.predictions],
            "confianza": result.probabilities.max(axis=1),
            "p_real": result.probabilities[rows, result.targets],
        }
    )


def _tracking_uri() -> str:
    return (
        os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )


def _log_to_mlflow(result: RunResult) -> None:
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri())
    name = result.config.experiment
    if mlflow.get_experiment_by_name(name) is None:
        location = None
        if "MLFLOW_TRACKING_URI" not in os.environ:
            location = (PROJECT_ROOT / "mlruns").as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    config = result.config
    with mlflow.start_run(run_name=f"{config.name}-seed{config.seed}"):
        mlflow.log_params(
            {**config.to_dict(), "parameters": result.parameters, "device_used": result.device}
        )
        for row in result.history.to_dict("records"):
            step = int(row["epoch"])
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if k not in ("epoch", "best")}, step=step
            )
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in result.summary().items()})
        for file in (CONFIG_FILE, HISTORY_FILE, METRICS_FILE, "predictions_val.csv"):
            mlflow.log_artifact(str(result.run_dir / file))


def run(config: RunConfig, verbose: bool = True, overwrite: bool = False) -> RunResult:
    """Entrena con train, selecciona con validation y guarda todo en ``config.run_dir``.

    Una corrida terminada no se pisa salvo que se pida ``overwrite=True``.
    """
    run_dir = config.run_dir
    if (run_dir / METRICS_FILE).exists() and not overwrite:
        raise FileExistsError(f"La corrida {run_dir} ya existe. Usa overwrite para repetirla.")
    for stale in (METRICS_FILE, TEST_FILE):
        (run_dir / stale).unlink(missing_ok=True)

    device = pick_device(config.device)
    seed_everything(config.seed, config.deterministic)
    loaders = make_loaders(config, device)
    classes = loaders.val.dataset.classes
    model = build_model(config, len(classes)).to(device)
    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if verbose:
        print(f"{config.experiment} / {config.name} / seed {config.seed} en {device}")
    checkpoint_path = run_dir / CHECKPOINT_FILE
    rows = fit(
        model, loaders.train, loaders.val, loss_fn, optimizer, device,
        config.epochs, checkpoint_path, config.to_dict(), verbose,
    )  # fmt: skip

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    train_pass = evaluate(model, loaders.train_eval, loss_fn, device)
    val_pass = evaluate(model, loaders.val, loss_fn, device)
    result = RunResult(
        config=config,
        run_dir=run_dir,
        history=pd.DataFrame(rows),
        train=build_report(
            train_pass.targets, train_pass.predictions, len(classes), train_pass.loss
        ),
        val=build_report(val_pass.targets, val_pass.predictions, len(classes), val_pass.loss),
        best_epoch=int(checkpoint["epoch"]),
        parameters=count_parameters(model),
        device=str(device),
    )

    write_json(run_dir / CONFIG_FILE, config.to_dict())
    result.history.to_csv(run_dir / HISTORY_FILE, index=False)
    _predictions_table(loaders.val.dataset, val_pass).to_csv(
        run_dir / "predictions_val.csv", index=False
    )
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
    if config.track_mlflow:
        _log_to_mlflow(result)
    return result


def run_seeds(config: RunConfig, seeds: list[int], verbose: bool = True) -> list[RunResult]:
    """La misma configuración con varias semillas.

    Una corrida ya terminada se carga en vez de repetirse. Si la corrida guardada tiene
    otra configuración se lanza un error para no mezclar resultados.
    """
    results = []
    for seed in seeds:
        seeded = config.with_changes(seed=seed)
        if not (seeded.run_dir / METRICS_FILE).exists():
            results.append(run(seeded, verbose=verbose))
            continue
        stored = load_result(seeded.run_dir)
        changed = diff(seeded, stored.config)
        if changed:
            raise ValueError(
                f"{seeded.run_dir} se entrenó con otra configuración: {changed}. "
                "Cambia name en el TOML o borra esa carpeta."
            )
        results.append(stored)
    return results


def load_result(run_dir: Path) -> RunResult:
    """Reconstruye un ``RunResult`` desde los archivos de la corrida."""
    metrics = read_json(run_dir / METRICS_FILE)
    return RunResult(
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
    checkpoint = torch.load(run_dir / CHECKPOINT_FILE, map_location=device, weights_only=True)
    config = RunConfig.from_dict(checkpoint["config"])
    classes = read_json(run_dir / METRICS_FILE)["classes"]
    model = build_model(config, len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, classes


def evaluate_test(run_dir: Path) -> Report:
    """Evaluación final en test del mejor checkpoint de una corrida.

    Se ejecuta una sola vez por corrida: si ``test.json`` existe se lanza un error.
    """
    target = run_dir / TEST_FILE
    if target.exists():
        raise FileExistsError(f"{run_dir} ya fue evaluada en test. El resultado está en {target}.")
    device = pick_device("auto")
    model, config, classes = load_model(run_dir, device)
    loader = make_loader(config, "test", device)
    result = evaluate(model, loader, build_loss(config, device), device)
    report = build_report(result.targets, result.predictions, len(classes), result.loss)
    _predictions_table(loader.dataset, result).to_csv(run_dir / "predictions_test.csv", index=False)
    write_json(target, {"classes": classes, "test": report.to_dict()})
    return report
