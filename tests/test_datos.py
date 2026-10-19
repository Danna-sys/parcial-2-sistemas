import pytest
import torch

from nubes.datos import build_transform, make_dataset, make_loaders  #lo que se prueba   


def test_dataset_item_shape_and_label(config):  # config es fixture de conftest.py: configuracion diminuta con datos sinteticos
    dataset = make_dataset(config, "val")    # Dataset de validation
    image, label = dataset[0]    # Primer elemento: imagen y etiqueta
    assert image.shape == (3, config.image_size, config.image_size)    # la img debe ser un tensor de 3 canales x image_size x image_size

    assert label == dataset.frame["label"].iloc[0]    # la etiqueta debe ser la de la primera fila del manifest de ese split
    assert len(dataset.classes) == 11    #el dataset conoce las 11 clases



def test_augmentation_only_on_train(config):
    #pedir augmentation en validation debe dar error
    with pytest.raises(ValueError):
        make_dataset(config, "val", augment=True)


def test_eval_transform_is_deterministic(config):
    dataset = make_dataset(config.with_changes(augment="flip_crop_color"), "train")    #configuracion con la augmentation mas fuerte, pero el dataset se pide SIN augmentation

    #,leer la misma imagen dos veces debe dar exactamente el mismo tensor
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_train_augmentation_changes_images(config):
    # Ahora el dataset sí se pide con augmentation.
    dataset = make_dataset(config.with_changes(augment="flip_crop_color"), "train", augment=True)
    # Semilla fija para que el test dé siempre el mismo resultado.
    torch.manual_seed(0)
    # Leer la misma imagen dos veces debe dar tensores distintos, porque cada lectura
    # aplica un recorte, volteo y color aleatorios.
    assert not torch.equal(dataset[0][0], dataset[0][0])


def test_unknown_augment_raises():
    with pytest.raises(ValueError):
        build_transform(32, [0.5] * 3, [0.2] * 3, "mixup")    # "mixup" no es una opción de augmentation de este proyecto


def test_loaders_never_include_test_and_shuffle_is_seeded(config):
    device = torch.device("cpu")       # CPU
    #dos juegos de loaders con la misma configuracion y semilla
    first = make_loaders(config, device)
    second = make_loaders(config, device)
    assert set(first.train.dataset.frame["split"]) == {"train"}    # loader de train solo tiene imgs de train
    assert set(first.val.dataset.frame["split"]) == {"val"}    # El de validation solo tiene imagenes de validation. No hay test por 
    #etiquetas en el orden en que salen del loader barajado, en ambos juegos
    labels_a = torch.cat([y for _, y in first.train])
    labels_b = torch.cat([y for _, y in second.train])
    labels_eval = torch.cat([y for _, y in first.train_eval])       #etiquetas del loader de train sin barajar
    assert torch.equal(labels_a, labels_b)    #misma semilla, mismo orden barajado
    assert labels_eval.tolist() == first.train_eval.dataset.frame["label"].tolist()    #el loader sin barajar sigue exactamente el orden del manifest
