import pytest
import torch

from nubes.datos import build_transform, make_dataset, make_loaders


def test_dataset_item_shape_and_label(config):
    dataset = make_dataset(config, "val")
    image, label = dataset[0]
    assert image.shape == (3, config.image_size, config.image_size)
    assert label == dataset.frame["label"].iloc[0]
    assert len(dataset.classes) == 11


def test_augmentation_only_on_train(config):
    with pytest.raises(ValueError):
        make_dataset(config, "val", augment=True)


def test_eval_transform_is_deterministic(config):
    dataset = make_dataset(config.with_changes(augment="flip_crop_color"), "train")
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_train_augmentation_changes_images(config):
    dataset = make_dataset(config.with_changes(augment="flip_crop_color"), "train", augment=True)
    torch.manual_seed(0)
    assert not torch.equal(dataset[0][0], dataset[0][0])


def test_unknown_augment_raises():
    with pytest.raises(ValueError):
        build_transform(32, [0.5] * 3, [0.2] * 3, "mixup")


def test_loaders_never_include_test_and_shuffle_is_seeded(config):
    device = torch.device("cpu")
    first = make_loaders(config, device)
    second = make_loaders(config, device)
    assert set(first.train.dataset.frame["split"]) == {"train"}
    assert set(first.val.dataset.frame["split"]) == {"val"}
    labels_a = torch.cat([y for _, y in first.train])
    labels_b = torch.cat([y for _, y in second.train])
    labels_eval = torch.cat([y for _, y in first.train_eval])
    assert torch.equal(labels_a, labels_b)
    assert labels_eval.tolist() == first.train_eval.dataset.frame["label"].tolist()
