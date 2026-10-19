"""Línea de comandos: ``uv run nubes <comando>``."""

from __future__ import annotations

import argparse  #esta lee argumentos de la terminal
from collections.abc import Sequence  #anota los aargumentos
from pathlib import Path

from nubes.config import RunConfig  #corridas
from nubes.utilidades import PROJECT_ROOT, resolve_path  #raiz del proyecto y conversión de rutas


def _preparar(args: argparse.Namespace) -> None:
    from nubes.preparacion import (
        prepare,  #importampos a dentro para que no se carguen cosas innecesariamente
    )

    stats = prepare(              #este corre la preparacion como tal con los argumentos de la terminal
        resolve_path(args.source), resolve_path(args.output), args.split_seed, args.overwrite
    )
    print(f"total = {stats['total']}   split_seed = {stats['split_seed']}")    #imprime la semilla a usar en el split y el total
    for split, counts in stats["counts"].items():    #este ciclo cuenta e imprime la cantidad en split, por clase, y en total
        print(f"{split:>5}: {sum(counts.values()):>5}  {counts}")
    print(f"duplicados: {stats['duplicates']}")         #imprime el conteo de los duplicados
    if stats["differs_from_published"]:        #aqui se imprimen los valores que resultaron diferentesa a los publicados
        print(f"diferencia con los conteos publicados: {stats['differs_from_published']}")
    print(f"mean_train = {stats['mean_train']}")        #estas dos lineas son de normalización bajo train
    print(f"std_train  = {stats['std_train']}")


def _train(args: argparse.Namespace) -> None:
    from nubes.experimento import run  # imprime la funcion de corrida

    config = RunConfig.from_toml(args.config)            #lee el toml correspondiente para configurar
    if args.epochs is not None:                  # si --epochs no en vacio, se ntercambia por las epocas del toml
        config = config.with_changes(epochs=args.epochs)
    for seed in args.seeds or [config.seed]:          #este ciclo recorre las semillas
        result = run(config.with_changes(seed=seed), overwrite=args.overwrite)         # corre a partir de las semillas establecidas
        print(f"-> {result.run_dir}")     #mustra donde quedo guardado ese entreno
        for key, value in result.summary().items():      # en este for se muestra el resumen numerico
            print(
                f"   {key:<18} {value:.4f}" if isinstance(value, float) else f"   {key:<18} {value}"
            )


def _test(args: argparse.Namespace) -> None:
    from nubes.experimento import evaluate_test  #importa el test final

    report = evaluate_test(resolve_path(args.run))    # se evalua la corrida en test
    print(
        f"test  loss {report.loss:.4f}  acc {report.accuracy:.4f}  macro_f1 {report.macro_f1:.4f}"      #muestra las metricas en test
    )


def _resumen(args: argparse.Namespace) -> None:
    from nubes.resumen import (  #funciones que resumen
        aggregate,
        load_experiment,
        runs_table,
        to_markdown,
    )

    results = load_experiment(resolve_path(args.runs_dir), args.experiment)   # muestra todas las corridas del experimento
    print(to_markdown(aggregate(runs_table(results))))      #muestra media y desviacion enuna tabla de markdown


def build_parser() -> argparse.ArgumentParser:        #un parser organiza la info de texto en algo que pueda usar
    parser = argparse.ArgumentParser(prog="nubes", description="CNN para géneros de nubes en CCSN")   #el parser principal que se va a usar
    commands = parser.add_subparsers(dest="command", required=True)   #comandos de preparar, train, test y resumen

    #comando para preparacion los datos
    preparar = commands.add_parser("preparar", help="inventario, duplicados y split")    
    preparar.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "raw")     #carpeta de los datos originales
    preparar.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "ccsn")    #carpeta en donde se van a escribir los manifest.csv y stats.json.
    preparar.add_argument("--split-seed", type=int, default=2026)         #la semilla del split
    preparar.add_argument("--overwrite", action="store_true")         # por si se quiere reescribir una preparacion ya corrida
    preparar.set_defaults(handler=_preparar)          #corre


    #comando para entrenar
    train = commands.add_parser("train", help="entrena con train y selecciona con validation")
    train.add_argument("--config", type=Path, required=True)    #lee la config dle toml
    train.add_argument("--seeds", type=int, nargs="+")     #las semillas
    train.add_argument("--epochs", type=int)            #si es necesario establecer un numero de epocas
    train.add_argument("--overwrite", action="store_true")         #por si se quiere repetir un train corrido
    train.set_defaults(handler=_train)           #ejecuta

    #comando para evaluar
    test = commands.add_parser("test", help="evaluación final en test, una sola vez por corrida")
    test.add_argument("--run", type=Path, required=True)    #carpeta de la corrida a evaluar  
    test.set_defaults(handler=_test)           #ejecuta

    #Comando para resumir
    resumen = commands.add_parser("resumen", help="tabla Markdown de un experimento")
    resumen.add_argument("--experiment", required=True)         #nombre del experimento
    resumen.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")         #carpeta donde estan las corridas
    resumen.set_defaults(handler=_resumen)          #ejecuta

    return parser         #arma el parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)         #lee los argumentos de la terminal
    args.handler(args)              #llama a la funcion del comando elegido.


if __name__ == "__main__":    # esto corre solo si el archivo se ejecuta directamente
    main()
