"""Configuración de una corrida, leída desde archivos TOML."""

from __future__ import annotations

import tomllib           #este es un lector de tomls
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from nubes.utilidades import DEVICES, PROJECT_ROOT, resolve_path    #reutilizamos los devices válidos, la raiz del proyecto y la funcion de rutas

AUGMENTATIONS: tuple[str, ...] = ("none", "flip", "flip_crop", "flip_crop_color", "flip_hv")     #se definen los argumentos válidos
POOLINGS: tuple[str, ...] = ("max", "avg", "stride")     #se definen las opciones validas de pooling
_PATHS: tuple[str, ...] = ("data_dir", "runs_dir")     #se defines los campos que son rutas


@dataclass(frozen=True)    #frozen=True hace que una vez creada la config no se pueda modificar
class RunConfig:
    """Todo lo que define una corrida.

    Es inmutable. Las variantes se crean con ``with_changes`` o con un TOML que
    declara ``extends``.

    Attributes:
        name: Nombre de la configuración dentro del experimento.
        experiment: Familia de corridas que se comparan entre sí.
        seed: Semilla de pesos iniciales, orden de batches, Dropout y augmentation.
        image_size: Lado de la imagen que entra a la red.
        augment: Transformaciones aleatorias de train. Ver ``AUGMENTATIONS``.
        batch_size: Tamaño de batch.
        num_workers: Procesos del ``DataLoader``. Con 0 carga el proceso principal.
        blocks: Número de bloques convolucionales.
        base_channels: Canales del primer bloque. Cada bloque duplica hasta ``max_channels``.
        max_channels: Tope de canales.
        kernel_size: Lado impar del kernel de cada convolución.
        pooling: Cómo reduce la resolución cada bloque. Ver ``POOLINGS``.
        batchnorm: Si hay ``BatchNorm2d`` después de cada convolución.
        head_grid: Lado de la rejilla a la que se promedia el último mapa. Con 1 es
            global average pooling y se pierde la posición.
        hidden_units: Neuronas de la capa densa intermedia.
        dropout: Probabilidad de Dropout antes de la capa de salida.
        epochs: Número fijo de épocas.
        learning_rate: Learning rate de AdamW.
        weight_decay: Weight decay de AdamW.
        class_weights: Si la pérdida pondera cada clase por el inverso de su frecuencia.
        deterministic: Si cuDNN usa algoritmos deterministas.
        device: ``auto``, ``cuda``, ``mps`` o ``cpu``.
        track_mlflow: Si la corrida se registra también en MLflow.
        data_dir: Carpeta con ``manifest.csv`` y ``stats.json``.
        runs_dir: Carpeta raíz de las corridas.
    """
    #para identificar la corrida
    name: str = "base"
    experiment: str = "linea-base"
    seed: int = 11

    #los datos. resolucion de entrada, augmentation, tamaño de batch y procesos de carga
    image_size: int = 128
    augment: str = "none"
    batch_size: int = 32
    num_workers: int = 0

    #para la arquitectura de la red
    blocks: int = 4
    base_channels: int = 32
    max_channels: int = 256
    kernel_size: int = 3
    pooling: str = "max"
    batchnorm: bool = False
    head_grid: int = 1
    hidden_units: int = 128
    dropout: float = 0.0

    #para el entrenamiento.  epocas, optimizacion y perdida
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    class_weights: bool = False

    #para la ejecución. reproducibilidad, hardware y MLflow.
    deterministic: bool = True
    device: str = "auto"
    track_mlflow: bool = True

    #las rutas, que realmente dependen de PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "ccsn")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    def __post_init__(self) -> None:             #una funcionq ue recorre los campos de ruta justo despues de crear la config
        for name in _PATHS:
            object.__setattr__(self, name, resolve_path(getattr(self, name)))      #convierte la ruta en absoluta 
        problems = [message for failed, message in self._checks() if failed]     #Se ejecutan todas las validaciones y se guardan los mensajes de las que fallaron
        if problems:
            raise ValueError("RunConfig inválida: " + "; ".join(problems))    #si hay falla, se detiene antes de usar una configuracion que no sirve

    def _checks(self) -> list[tuple[bool, str]]:
        smallest = self.image_size // 2**self.blocks   #para el ultimo mapa de caracteristicas, cada bloque divide la imagen entre 2.
        return [      # esta es toda la lista de pares (condicion de error, mensaje). Si la condicion es True, hay un problema.
            (not self.name.strip(), "name vacío"),    #no pueden haber nombres vacios
            (not self.experiment.strip(), "experiment vacío"),    
            (self.seed < 0, "seed debe ser >= 0"),        # la semilla debe ser positiva
            (self.image_size < 16, "image_size debe ser >= 16"),          #los tamaños de imagenes menores a 16px no son validos
            (self.augment not in AUGMENTATIONS, f"augment debe estar en {AUGMENTATIONS}"),      #debe ser argumentation valido de acuerdo a cli.py
            (self.batch_size < 1, "batch_size debe ser >= 1"),      #al menos una imagen por batch
            (self.num_workers < 0, "num_workers debe ser >= 0"),        #el numero de trabajos no puede ser negativo
            (self.blocks < 1, "blocks debe ser >= 1"),         # por lo menos un bloque convolucional
            (self.base_channels < 1, "base_channels debe ser >= 1"),    #al menos un canal en el primer bloque
            (self.max_channels < self.base_channels, "max_channels < base_channels"),    # el tope de canales no puede ser menor que el punto de partida
            (self.kernel_size < 1 or self.kernel_size % 2 == 0, "kernel_size debe ser impar"),     #el kernel debe ser impar para tener pixel central
            (self.pooling not in POOLINGS, f"pooling debe estar en {POOLINGS}"),    #debe ser un pooling valido de acuerdo con cli.py
            (self.head_grid < 1, "head_grid debe ser >= 1"),   #la grid de la cabeza mide al menos 1x1
            (
                # no se puede promediar a una grid mas grande que el ultimo mapa
                smallest < self.head_grid,
                f"con image_size={self.image_size} y blocks={self.blocks} el último mapa "
                f"mide {smallest} y no alcanza para head_grid={self.head_grid}",
            ),
            (self.hidden_units < 1, "hidden_units debe ser >= 1"),    # al menso una nurona en las capas ocultas
            (not 0.0 <= self.dropout < 1.0, "dropout debe estar en [0, 1)"),     #la probabilidad dropout
            (self.epochs < 1, "epochs debe ser >= 1"),                #una epoca por lo menos
            (self.learning_rate <= 0, "learning_rate debe ser > 0"),        #el learning rate debe ser positivo
            (self.weight_decay < 0, "weight_decay debe ser >= 0"),         #el decay debe ser no negativo
            (self.device not in DEVICES, f"device debe estar en {DEVICES}"),       #devices debe ser una opcion valida segun cli.py
        ]

    @property          #permite usar config.run_dir como si fuera un campo
    def run_dir(self) -> Path:
        """Carpeta de esta corrida: ``runs/<experiment>/<name>/seed_<seed>``."""
        return self.runs_dir / self.experiment / self.name / f"seed_{self.seed}"  # arma la ruta donde se guardan los resultados, entonces cada semilla tiene su propia carpeta.

    def with_changes(self, **changes: Any) -> RunConfig:
        return replace(self, **changes)         #devuelve una copia con los campos indicados cambiados
        
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)     #convierte la configuracio en diccionario para guardarla en json y en el checkpoint
        for name in _PATHS:     #json no sabe guardar objetos Path, las rutas se pasan a texto
            data[name] = str(data[name]) 
        return data      #dicicionario ya para guardar

    @classmethod        #se llama sobre la clase (RunConfig.from_dict) y devuelve una instancia nueva
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        unknown = set(data) - {f.name for f in fields(cls)}    #busca claves que no correspondan a ningun campo
        if unknown:        #claves desconocidas las avisa. si no, un "lr" mal escrito se ignora
            raise ValueError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:      #lee el toml, resuelve la herencia y crea la config
        """Lee un TOML. Si declara ``extends``, parte del archivo indicado y lo sobrescribe."""
        return cls.from_dict(_read_toml(Path(path), seen=()))


def _read_toml(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()     #convierte la ruta en absoluta para poder compararla con las otras
    if path in seen:       #si el archivo ya se visitó, hay un ciclo
        raise ValueError(f"extends circular en {path}")
    with path.open("rb") as handle:    #abre el archivo en modo binario para que sirva tomllib
        data = tomllib.load(handle)    #convierte el toml en diccionario
    parent = data.pop("extends", None)      #saca la clave "extends" si existe
    if parent is None:      #si no se hereda nada, el diccionario ya está completo. si si, se sobreescribe; el hijo le gana al padre
        return data
    return {**_read_toml(path.parent / parent, (*seen, path)), **data}


def diff(base: RunConfig, other: RunConfig) -> dict[str, tuple[Any, Any]]:
    """Campos en los que ``other`` difiere de ``base``.

    No cuenta nombre, semilla ni los campos que no cambian lo que se entrena.
    """
    ignored = {"name", "seed", "device", "num_workers", "track_mlflow"}   #se ignoran porque no cambian el experimento como tal
    left, right = base.to_dict(), other.to_dict()     #configuraciones como diccionarios para compararlas clave por clave
    return {     #devuelve solo campos distintos, con el valor de cada lado. comprueba que un experimento cambia una sola variable a la vez
        key: (left[key], right[key])
        for key in left
        if key not in ignored and left[key] != right[key]
    }
