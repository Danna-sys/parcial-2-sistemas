"""Línea de comandos: ``uv run nubes <comando>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nubes.config import RunConfig
from nubes.utilidades import PROJECT_ROOT, resolve_path


def _preparar(args: argparse.Namespace) -> None:
    from nubes.preparacion import prepare

    stats = prepare(
        resolve_path(args.source), resolve_path(args.output), args.split_seed, args.overwrite
    )
    print(f"total = {stats['total']}   split_seed = {stats['split_seed']}")
    for split, counts in stats["counts"].items():
        print(f"{split:>5}: {sum(counts.values()):>5}  {counts}")
    print(f"duplicados: {stats['duplicates']}")
    if stats["differs_from_published"]:
        print(f"diferencia con los conteos publicados: {stats['differs_from_published']}")
    print(f"mean_train = {stats['mean_train']}")
    print(f"std_train  = {stats['std_train']}")


def _train(args: argparse.Namespace) -> None:
    from nubes.experimento import run

    config = RunConfig.from_toml(args.config)
    if args.epochs is not None:
        config = config.with_changes(epochs=args.epochs)
    for seed in args.seeds or [config.seed]:
        result = run(config.with_changes(seed=seed), overwrite=args.overwrite)
        print(f"-> {result.run_dir}")
        for key, value in result.summary().items():
            print(
                f"   {key:<18} {value:.4f}" if isinstance(value, float) else f"   {key:<18} {value}"
            )


def _test(args: argparse.Namespace) -> None:
    from nubes.experimento import evaluate_test

    report = evaluate_test(resolve_path(args.run))
    print(
        f"test  loss {report.loss:.4f}  acc {report.accuracy:.4f}  macro_f1 {report.macro_f1:.4f}"
    )


def _resumen(args: argparse.Namespace) -> None:
    from nubes.resumen import aggregate, load_experiment, runs_table, to_markdown

    results = load_experiment(resolve_path(args.runs_dir), args.experiment)
    print(to_markdown(aggregate(runs_table(results))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nubes", description="CNN para géneros de nubes en CCSN")
    commands = parser.add_subparsers(dest="command", required=True)

    preparar = commands.add_parser("preparar", help="inventario, duplicados y split")
    preparar.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw")
    preparar.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "ccsn")
    preparar.add_argument("--split-seed", type=int, default=2026)
    preparar.add_argument("--overwrite", action="store_true")
    preparar.set_defaults(handler=_preparar)

    train = commands.add_parser("train", help="entrena con train y selecciona con validation")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seeds", type=int, nargs="+")
    train.add_argument("--epochs", type=int)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=_train)

    test = commands.add_parser("test", help="evaluación final en test, una sola vez por corrida")
    test.add_argument("--run", type=Path, required=True)
    test.set_defaults(handler=_test)

    resumen = commands.add_parser("resumen", help="tabla Markdown de un experimento")
    resumen.add_argument("--experiment", required=True)
    resumen.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    resumen.set_defaults(handler=_resumen)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
