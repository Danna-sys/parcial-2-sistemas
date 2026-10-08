from pathlib import Path

import pytest

from nubes.cli import main


def test_full_cli_flow(raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture):
    data = tmp_path / "ccsn"
    runs = tmp_path / "runs"
    main(["preparar", "--source", str(raw_dir), "--output", str(data)])
    assert "duplicados" in capsys.readouterr().out

    toml = tmp_path / "mini.toml"
    toml.write_text(
        'name = "mini"\nexperiment = "cli"\nimage_size = 32\nblocks = 2\nbase_channels = 4\n'
        'hidden_units = 8\nepochs = 3\ndevice = "cpu"\ntrack_mlflow = false\n'
        f'data_dir = "{data.as_posix()}"\nruns_dir = "{runs.as_posix()}"\n'
    )
    main(["train", "--config", str(toml), "--seeds", "1", "2", "--epochs", "1"])
    assert (runs / "cli" / "mini" / "seed_2" / "metrics.json").is_file()

    main(["resumen", "--experiment", "cli", "--runs-dir", str(runs)])
    assert "| mini | 2 |" in capsys.readouterr().out

    main(["test", "--run", str(runs / "cli" / "mini" / "seed_1")])
    with pytest.raises(FileExistsError):
        main(["test", "--run", str(runs / "cli" / "mini" / "seed_1")])
