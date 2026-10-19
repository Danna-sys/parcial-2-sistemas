from pathlib import Path

import pytest  #para esperar errores y capturar lo que se imprime

from nubes.cli import main  # funcion principal de la linea de comandos


def test_full_cli_flow(raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture):    # capsys captura lo que se imprime en pantalla
    #carpetas temporales para los datos preparados y las corridas
    data = tmp_path / "ccsn"      
    runs = tmp_path / "runs"
    main(["preparar", "--source", str(raw_dir), "--output", str(data)])     #es como escribir en la terminal: nubes preparar --source ... --output ...
    assert "duplicados" in capsys.readouterr().out     #lo impreso debe incluir el resumen de duplicados

    toml = tmp_path / "mini.toml"     # un TOML minimo escrito en una carpeta temporal
    toml.write_text(
        'name = "mini"\nexperiment = "cli"\nimage_size = 32\nblocks = 2\nbase_channels = 4\n'
        'hidden_units = 8\nepochs = 3\ndevice = "cpu"\ntrack_mlflow = false\n'
        f'data_dir = "{data.as_posix()}"\nruns_dir = "{runs.as_posix()}"\n'
    )
    main(["train", "--config", str(toml), "--seeds", "1", "2", "--epochs", "1"])    # nubes train con dos semillas. --epochs 1 reemplaza las 3 del TOML para ir rapido
    assert (runs / "cli" / "mini" / "seed_2" / "metrics.json").is_file()   # La corrida de la segunda semilla debe haber dejado su metrics.json

    main(["resumen", "--experiment", "cli", "--runs-dir", str(runs)])    # nubes resumen del experimento
    assert "| mini | 2 |" in capsys.readouterr().out      #la tabla impresa debe tener la configuracion "mini" con 2 semillas

    main(["test", "--run", str(runs / "cli" / "mini" / "seed_1")])     # nubes test sobre la primera semilla (evaluacion final)
    with pytest.raises(FileExistsError):    #repetirla debe estar bloqueado, test se evalúa una sola vez
        main(["test", "--run", str(runs / "cli" / "mini" / "seed_1")])
