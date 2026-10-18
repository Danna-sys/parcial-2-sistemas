import pytest
import torch
from torch import nn

from nubes.inspeccion import feature_maps, receptive_field, shape_trace
from nubes.modelo import CloudCNN, count_parameters, stage_channels


def test_stage_channels_double_until_cap():
    assert stage_channels(5, 32, 256) == [32, 64, 128, 256, 256]  # verifica que los canales se dupliquen hasta alcanzar el máximo permitido


@pytest.mark.parametrize("size", [64, 128, 224])  # prueba diferentes tamaños de imagen
def test_logits_shape_for_any_resolution(size):
    model = CloudCNN(num_classes=11)  # crea una cnn para clasificar once clases de nubes
    assert model(torch.zeros(2, 3, size, size)).shape == (2, 11)  # comprueba que la salida tenga dos muestras y once puntuaciones por imagen


@pytest.mark.parametrize("pooling", ["max", "avg", "stride"])  # prueba los tres métodos de reducción espacial
def test_every_pooling_halves_the_map(pooling):
    model = CloudCNN(blocks=3, base_channels=4, pooling=pooling)  # crea un modelo con tres bloques y el método de reducción elegido
    trace = shape_trace(model, 64)  # registra las dimensiones de salida de cada bloque
    assert trace.loc["stages.s3", "shape"] == (16, 8, 8)  # verifica los canales y las dimensiones espaciales del tercer bloque


def test_batchnorm_and_dropout_are_swapped_for_identity():
    plain = CloudCNN()  # crea un modelo sin regularización adicional
    regularized = CloudCNN(batchnorm=True, dropout=0.3)  # crea un modelo con batch normalization y dropout
    assert isinstance(plain.stages.s1.norm, nn.Identity)  # comprueba que la normalización sea una operación identidad cuando está desactivada
    assert isinstance(plain.head.drop, nn.Identity)  # comprueba que el dropout sea una operación identidad cuando está desactivado
    assert isinstance(regularized.stages.s1.norm, nn.BatchNorm2d)  # verifica que se active batch normalization
    assert regularized.stages.s1.conv.bias is None  # comprueba que la convolución no tenga sesgo cuando se utiliza batch normalization
    assert isinstance(regularized.head.drop, nn.Dropout)  # verifica que se active dropout en el clasificador


def test_head_grid_changes_classifier_input():
    assert CloudCNN(head_grid=1).head.fc1.in_features == 256  # comprueba las entradas de la capa densa con una cuadrícula de tamaño uno
    assert CloudCNN(head_grid=2).head.fc1.in_features == 1024  # comprueba las entradas de la capa densa con una cuadrícula de tamaño dos
    assert count_parameters(CloudCNN(head_grid=2)) > count_parameters(CloudCNN(head_grid=1))  # verifica que la cuadrícula mayor aumente los parámetros entrenables


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        CloudCNN(pooling="min")  # comprueba que se rechace un método de reducción no permitido
    with pytest.raises(ValueError):
        CloudCNN(dropout=1.0)  # comprueba que se rechace una probabilidad de dropout igual a uno


def test_receptive_field_matches_hand_computation():
    table = receptive_field(CloudCNN(blocks=4), 128)  # calcula el campo receptivo para una cnn de cuatro bloques
    assert table["rf"].tolist() == [4, 10, 22, 46]  # verifica los campos receptivos calculados manualmente
    assert table["map_size"].tolist() == [64, 32, 16, 8]  # comprueba las dimensiones espaciales después de cada bloque
    stride = receptive_field(CloudCNN(blocks=2, pooling="stride"), 64)  # calcula el campo receptivo usando convoluciones con stride
    assert stride["rf"].tolist() == [3, 7]  # verifica los campos receptivos de los dos bloques


def test_feature_maps_hook_is_removed_and_mode_restored():
    model = CloudCNN(blocks=2, base_channels=4)  # crea un modelo pequeño para probar los mapas de características
    model.train()  # activa el modo de entrenamiento antes de la prueba
    maps = feature_maps(model, torch.zeros(2, 3, 32, 32), "stages.s1.act")  # obtiene los mapas de características de la activación del primer bloque
    assert maps.shape == (2, 4, 32, 32)  # comprueba las dimensiones de los mapas obtenidos
    assert model.training  # verifica que se restaure el modo de entrenamiento original
    assert not model.stages.s1.act._forward_hooks  # comprueba que se elimine el hook temporal después de la extracción
    with pytest.raises(ValueError):
        feature_maps(model, torch.zeros(1, 3, 32, 32), "no.existe")  # verifica que se informe si la capa solicitada no existe