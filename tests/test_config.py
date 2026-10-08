from pathlib import Path

import pytest

from nubes.config import RunConfig, diff
from nubes.utilidades import PROJECT_ROOT


def test_defaults_are_valid_and_paths_absolute():
    config = RunConfig()
    assert config.data_dir.is_absolute()
    assert config.run_dir == PROJECT_ROOT / "runs" / "linea-base" / "base" / "seed_11"


@pytest.mark.parametrize(
    "changes",
    [
        {"augment": "rotar"},
        {"pooling": "min"},
        {"kernel_size": 4},
        {"dropout": 1.0},
        {"epochs": 0},
        {"image_size": 32, "blocks": 5, "head_grid": 2},
    ],
)
def test_invalid_values_raise(changes):
    with pytest.raises(ValueError):
        RunConfig(**changes)


def test_roundtrip_dict():
    config = RunConfig(name="x", dropout=0.2)
    assert RunConfig.from_dict(config.to_dict()) == config


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"lr": 0.1})


def test_toml_extends(tmp_path: Path):
    (tmp_path / "base.toml").write_text('name = "base"\nimage_size = 64\ndropout = 0.1\n')
    (tmp_path / "child.toml").write_text('extends = "base.toml"\nname = "child"\ndropout = 0.5\n')
    child = RunConfig.from_toml(tmp_path / "child.toml")
    assert (child.name, child.image_size, child.dropout) == ("child", 64, 0.5)


def test_toml_circular_extends(tmp_path: Path):
    (tmp_path / "a.toml").write_text('extends = "b.toml"\n')
    (tmp_path / "b.toml").write_text('extends = "a.toml"\n')
    with pytest.raises(ValueError):
        RunConfig.from_toml(tmp_path / "a.toml")


def test_repo_configs_load_and_change_one_thing():
    configs = PROJECT_ROOT / "configs"
    base = RunConfig.from_toml(configs / "base.toml")
    for path in sorted(configs.glob("*.toml")):
        config = RunConfig.from_toml(path)
        changed = set(diff(base, config)) - {"experiment"}
        assert len(changed) <= 2, f"{path.name} cambia demasiadas variables: {changed}"


def test_diff_ignores_name_and_seed():
    base = RunConfig()
    assert diff(base, base.with_changes(name="otro", seed=3, device="cpu")) == {}
    assert diff(base, base.with_changes(blocks=5)) == {"blocks": (4, 5)}
