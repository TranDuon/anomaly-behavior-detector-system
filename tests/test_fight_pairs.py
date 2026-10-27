from types import SimpleNamespace

import numpy as np
import torch

from fight.vision import ClipConfig, build_clip_inputs, pair_regions
from scripts.fight_regions_ablation import (
    balanced_source_sample,
    compatible_window,
    variant_configs,
)


def test_dense_chain_does_not_merge_all_pairs_into_one_crowd_crop():
    boxes = [[80 + index * 65, 100, 140 + index * 65, 300] for index in range(8)]
    regions = pair_regions(boxes, 1000, 600)
    assert regions[-1] == (0, 0, 1000, 600)
    assert 2 <= len(regions) <= 3
    assert all(region[2] - region[0] < 400 for region in regions[:-1])


def test_pair_mode_keeps_context_when_detection_is_missing_or_incomplete():
    assert pair_regions([], 640, 360) == [(0, 0, 640, 360)]
    assert pair_regions([[10, 10, 50, 200]], 640, 360) == [(0, 0, 640, 360)]
    assert pair_regions([[0, 0, 20, 100], [500, 0, 520, 100]], 640, 360) == [
        (0, 0, 640, 360)
    ]
    assert pair_regions([[float("nan"), 0, 10, 100]], 640, 360) == [(0, 0, 640, 360)]


def test_pair_mode_is_shared_and_regions_are_bounded_on_original_frame():
    class Detector:
        def detect(self, frame):
            return [
                SimpleNamespace(bbox_xyxy=[-10, 5, 60, 160]),
                SimpleNamespace(bbox_xyxy=[55, 5, 110, 160]),
            ]

    frames = [np.full((200, 320, 3), 150, dtype=np.uint8)] * 16
    config = ClipConfig(region_mode="pairs", trim_black_borders=True, version=4)
    clips, regions, people, bounds, counts = build_clip_inputs(
        frames, Detector(), config
    )
    assert clips.shape[1:] == (16, 182, 182, 3)
    assert people == 2
    assert regions[-1] == [0, 0, 320, 200]
    assert all(
        0 <= left < right <= 320 and 0 <= top < bottom <= 200
        for left, top, right, bottom in regions
    )


def test_time_ablation_drops_windows_that_change_label_or_exceed_video():
    row = {"start": 0, "fps": 30, "label": 0}
    labels = np.r_[np.zeros(60), np.ones(40)]
    assert compatible_window(row, labels, [variant_configs()["tubes_legacy"]])
    assert not compatible_window(row, labels, variant_configs().values())
    assert not compatible_window(
        {**row, "start": 70}, labels, variant_configs().values()
    )
    assert compatible_window(
        {**row, "label": 1}, np.ones(100), variant_configs().values()
    )


def test_training_sampling_is_balanced_and_covers_sources_before_repetition():
    rows = [
        {"label": label, "video": f"{label}_{video}", "start": start}
        for label in (0, 1)
        for video in range(10)
        for start in range(10)
    ]
    selected = balanced_source_sample(rows, per_class=12)
    assert len(selected) == 24
    for label in (0, 1):
        matching = [row for row in selected if row["label"] == label]
        assert len(matching) == 12 and len({row["video"] for row in matching}) == 10
    assert selected == balanced_source_sample(rows, per_class=12)


def test_real_x3d_supports_native_temporal_input():
    from fight.model import VideoClassifier

    torch.set_num_threads(4)
    model = VideoClassifier("x3d_s", pretrained=False).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 13, 182, 182))
    assert result.shape == (1,) and torch.isfinite(result).all()


def test_unreviewed_label_ablation_is_blocked_without_explicit_acknowledgment():
    import pytest
    from scripts.fight_regions_ablation import run

    with pytest.raises(ValueError, match="flagged"):
        run(SimpleNamespace())
