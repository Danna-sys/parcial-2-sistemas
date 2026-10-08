import pytest
import torch
from torch import nn

from nubes.inspeccion import feature_maps, receptive_field, shape_trace
from nubes.modelo import CloudCNN, count_parameters, stage_channels


def test_stage_channels_double_until_cap():
    assert stage_channels(5, 32, 256) == [32, 64, 128, 256, 256]


@pytest.mark.parametrize("size", [64, 128, 224])
def test_logits_shape_for_any_resolution(size):
    model = CloudCNN(num_classes=11)
    assert model(torch.zeros(2, 3, size, size)).shape == (2, 11)


@pytest.mark.parametrize("pooling", ["max", "avg", "stride"])
def test_every_pooling_halves_the_map(pooling):
    model = CloudCNN(blocks=3, base_channels=4, pooling=pooling)
    trace = shape_trace(model, 64)
    assert trace.loc["stages.s3", "shape"] == (16, 8, 8)


def test_batchnorm_and_dropout_are_swapped_for_identity():
    plain = CloudCNN()
    regularized = CloudCNN(batchnorm=True, dropout=0.3)
    assert isinstance(plain.stages.s1.norm, nn.Identity)
    assert isinstance(plain.head.drop, nn.Identity)
    assert isinstance(regularized.stages.s1.norm, nn.BatchNorm2d)
    assert regularized.stages.s1.conv.bias is None
    assert isinstance(regularized.head.drop, nn.Dropout)


def test_head_grid_changes_classifier_input():
    assert CloudCNN(head_grid=1).head.fc1.in_features == 256
    assert CloudCNN(head_grid=2).head.fc1.in_features == 1024
    assert count_parameters(CloudCNN(head_grid=2)) > count_parameters(CloudCNN(head_grid=1))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        CloudCNN(pooling="min")
    with pytest.raises(ValueError):
        CloudCNN(dropout=1.0)


def test_receptive_field_matches_hand_computation():
    table = receptive_field(CloudCNN(blocks=4), 128)
    assert table["rf"].tolist() == [4, 10, 22, 46]
    assert table["map_size"].tolist() == [64, 32, 16, 8]
    stride = receptive_field(CloudCNN(blocks=2, pooling="stride"), 64)
    assert stride["rf"].tolist() == [3, 7]


def test_feature_maps_hook_is_removed_and_mode_restored():
    model = CloudCNN(blocks=2, base_channels=4)
    model.train()
    maps = feature_maps(model, torch.zeros(2, 3, 32, 32), "stages.s1.act")
    assert maps.shape == (2, 4, 32, 32)
    assert model.training
    assert not model.stages.s1.act._forward_hooks
    with pytest.raises(ValueError):
        feature_maps(model, torch.zeros(1, 3, 32, 32), "no.existe")
