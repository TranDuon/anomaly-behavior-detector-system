import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from action_engine import (
    AUX_NAMES,
    InteractionHead,
    InteractionHeadV2,
    fusion_auxiliary,
    interaction_bag,
    interaction_candidates,
    rgb_inputs,
)
from train_interaction_v2 import operating_point, validate_indoor_corpus
from train_interaction import annotation_at, validate


def test_training_overrides_inherited_colab_matplotlib_backend():
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import train_interaction; import matplotlib.pyplot; "
            "import matplotlib; assert matplotlib.get_backend().lower() == 'agg'",
        ],
        cwd=Path(__file__).resolve().parent,
        env=environment,
        check=True,
        timeout=60,
    )


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


def test_v2_context_marks_pairs_and_preserves_motion_and_base_score():
    frames = np.zeros((9, 24, 32, 3), np.uint8)
    frames[-2:] = 255
    candidates = interaction_candidates(2)
    auxiliary = fusion_auxiliary(
        frames,
        2,
        0.125,
        candidates,
        boxes=[[0, 0, 20, 20], [10, 0, 30, 20]],
    )
    assert auxiliary.shape == (3, len(AUX_NAMES))
    assert auxiliary[:, 0].tolist() == pytest.approx([0.125] * 3)
    assert auxiliary[:, 2].tolist() == [0, 0, 1]
    assert auxiliary[:, 4].min() > 0
    assert auxiliary[-1, 7] > 0
    assert auxiliary[-1, 8] == pytest.approx(1 / 3)


def test_v2_head_and_constrained_operating_point_are_trainable():
    head = InteractionHeadV2()
    features = torch.randn(3, 4096)
    auxiliary = torch.randn(3, len(AUX_NAMES))
    loss = head(features, auxiliary).max()
    loss.backward()
    assert head.layers[0].weight.grad is not None
    rows = [
        {"label": label, "group": str(index), "people": 2}
        for index, label in enumerate([0, 0, 0, 0, 1, 1, 1, 1])
    ]
    selected = operating_point(rows, [0.01, 0.02, 0.03, 0.04, 0.8, 0.85, 0.9, 0.95])
    assert selected["target_feasible"]
    assert 0.04 < selected["threshold"] < 0.8


def test_indoor_v2_rejects_seed_manifest_and_nonfight_positive():
    source = {
        "dataset": "airtlab",
        "video": "airtlab/non-violent/cam1/22.mp4",
        "split": "train",
        "group": "performance-22",
        "duration": 4,
        "environment": "indoor",
        "source_label": "non_fight",
        "intervals": [{"start_seconds": 0, "end_seconds": 4, "action": "fight"}],
    }
    corpus = {
        "schema_version": 2,
        "scope_profile": "indoor_room_fight_v2",
        "readiness": "seed_only",
        "sources": [source],
    }
    with pytest.raises(ValueError, match="not ready"):
        validate_indoor_corpus(corpus)
    corpus["readiness"] = "ready"
    with pytest.raises(ValueError, match="labels"):
        validate_indoor_corpus(corpus)
