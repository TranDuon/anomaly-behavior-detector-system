import copy

import numpy as np
import pytest
import torch

from action_engine import InteractionHead, interaction_bag, rgb_inputs
from train_interaction import annotation_at, validate


def test_rgb_matches_training_decoder_not_bgr_demo():
    frames = np.zeros((9, 24, 32, 3), np.uint8)
    frames[..., 2] = 255
    tensor, shape = rgb_inputs(frames)
    assert tensor.shape[:3] == (1, 3, 8)
    assert shape == (256, 341)
    assert float(tensor[0, 0, 0, 0, 0]) == pytest.approx((255 - 123.675) / 58.395)
    assert float(tensor[0, 2, 0, 0, 0]) == pytest.approx(-103.53 / 57.375)
    with pytest.raises(ValueError):
        rgb_inputs(frames[:8])


def test_interaction_bag_is_pair_order_invariant_and_supports_single_people():
    features = torch.randn(2, 2048)
    boxes = [[0, 0, 10, 20], [8, 4, 25, 25]]
    first, regions = interaction_bag(features, boxes)
    second, _ = interaction_bag(features.flip(0), boxes[::-1])
    assert first.shape == (3, 4096)
    assert torch.equal(first[-1], second[-1])
    assert regions[-1] == [0, 0, 25, 25]
    assert interaction_bag(features[:1], boxes[:1])[0].shape == (1, 4096)


def test_unknown_time_and_fight_boundary_are_not_negative_labels():
    source = {
        "intervals": [
            {"start_seconds": 0, "end_seconds": 2, "action": "normal"},
            {"start_seconds": 2, "end_seconds": 4, "action": "fight"},
        ]
    }
    assert annotation_at(source, 1.8)[0] is None
    assert annotation_at(source, 3)[0] == 1
    assert annotation_at(source, 6)[0] is None


def test_train_split_cannot_share_performance_with_test():
    source = {
        "dataset": "airtlab",
        "video": "one",
        "split": "train",
        "group": "performance",
        "duration": 5,
        "intervals": [],
    }
    other = copy.deepcopy(source)
    other.update(video="other_camera", split="test")
    with pytest.raises(ValueError, match="leakage"):
        validate({"sources": [source, other]})


def test_head_trains_without_modifying_cached_original_features():
    head = InteractionHead()
    features = torch.randn(3, 4096)
    original = features.clone()
    loss = head(features).max()
    loss.backward()
    assert head.layers[0].weight.grad is not None
    assert torch.equal(features, original)
