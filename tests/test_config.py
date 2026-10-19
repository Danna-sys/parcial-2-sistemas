# aqui buscamos que la configuracion se valide, se guarde y se lea bien.
# 


from pathlib import Path

import pytest

from nubes.config import RunConfig, diff      # Lo que se prueba.
from nubes.utilidades import PROJECT_ROOT     # raiz del proyecto, para comparar rutas y encontrar configs/


def test_defaults_are_valid_and_paths_absolute():
    config = RunConfig()         #al crear la config por defecto no debe lanzar ningun error
    assert config.data_dir.is_absolute()      #las rutas deben quedar absolutas, para funcionar desde cualquier carpeta
    assert config.run_dir == PROJECT_ROOT / "runs" / "linea-base" / "base" / "seed_11"    #la carpeta de la corrida sigue el patron runs/<experimento>/<nombre>/seed_<semilla>



@pytest.mark.parametrize(         #corre el mismo test una vez por cada caso de la lista
    "changes",
    [
        {"augment": "rotar"},    # Augmentation que no existe
        {"pooling": "min"},       # Pooling que no existe
        {"kernel_size": 4},       # Kernel par
        {"dropout": 1.0},         # Dropout de 1 apagaria todas las neuronas
        {"epochs": 0},            # Cero épocas
        {"image_size": 32, "blocks": 5, "head_grid": 2},    # 32 / 2^5 = 1 píxel al final, que no alcanza para una rejilla de 2x2
    ],
)


def test_invalid_values_raise(changes):
    with pytest.raises(ValueError):       #afirma que lo de adentro DEBE lanzar ese error
        RunConfig(**changes)


def test_roundtrip_dict():
    config = RunConfig(name="x", dropout=0.2)   # una configuración con algo cambiado
    assert RunConfig.from_dict(config.to_dict()) == config        # Convertirla a diccionario y de vuelta debe dar exactamente la misma


def test_unknown_key_raises():
    # "lr" no es un campo; el correcto es "learning_rate". Debe dar error y no ignorarse.
    with pytest.raises(ValueError):
        RunConfig.from_dict({"lr": 0.1})


def test_toml_extends(tmp_path: Path):   # tmp_path es una fixture de pytest: una carpeta temporal nueva para este test
    (tmp_path / "base.toml").write_text('name = "base"\nimage_size = 64\ndropout = 0.1\n')        # Se escribe un TOML padre
    (tmp_path / "child.toml").write_text('extends = "base.toml"\nname = "child"\ndropout = 0.5\n')         # Y un hijo que hereda del padre y cambia dos cosas
    child = RunConfig.from_toml(tmp_path / "child.toml")         # Se lee el hijo
    assert (child.name, child.image_size, child.dropout) == ("child", 64, 0.5)        # debe tener su nombre y su dropout, y la resolucion heredada del padre


def test_toml_circular_extends(tmp_path: Path):
    # a hereda de b y b hereda de a
    (tmp_path / "a.toml").write_text('extends = "b.toml"\n')
    (tmp_path / "b.toml").write_text('extends = "a.toml"\n')
    with pytest.raises(ValueError):    #debe detectarlo y dar error en vez de quedarse pegado
        RunConfig.from_toml(tmp_path / "a.toml")


def test_repo_configs_load_and_change_one_thing():     # Este test revisa los toml reales del proyecto, no unos inventados
    configs = PROJECT_ROOT / "configs"
    base = RunConfig.from_toml(configs / "base.toml")     # La línea base
    for path in sorted(configs.glob("*.toml")):    # cada TOML de la carpeta
        config = RunConfig.from_toml(path)        # Debe poder leerse sin errores
        changed = set(diff(base, config)) - {"experiment"}         # que cambia frente a la linea base, sin contar el nombre del experimento
        # ningun TOML puede cambiar mas de dos variables
        assert len(changed) <= 2, f"{path.name} cambia demasiadas variables: {changed}"


def test_diff_ignores_name_and_seed():
    base = RunConfig()    # config base
    assert diff(base, base.with_changes(name="otro", seed=3, device="cpu")) == {}    # Cambiar nombre, semilla y device no cuenta como cambiar el experimento
    assert diff(base, base.with_changes(blocks=5)) == {"blocks": (4, 5)}    # diff dice el valor de antes y el de después.

