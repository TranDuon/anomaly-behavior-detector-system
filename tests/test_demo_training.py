import random

import numpy as np

from fight.demo_training import augment_close_view, expand_training
from fight.vision import ClipConfig


def source(name, split="train", label=0, actions=None):
    return {
        "video": name,
        "split": split,
        "label": label,
        "dataset": "airtlab",
        "actions": actions or ["hug"],
        "fps": 30,
        "frames": 150,
        "performance": name,
    }


def test_expansion_never_moves_holdouts_or_ambiguous_or_conflicting_sources_to_train():
    rows = [
        source("safe"),
        source("dev", "development"),
        source("cal", "calibration"),
        source("test", "test"),
        source("unknown", actions=["unrecognized"]),
        source("conflict"),
    ]
    reviewed = [{**source("conflict", label=1), "start": 0}]
    windows, report = expand_training(reviewed, rows, ClipConfig(fps=12))
    added = windows[1:]
    assert len(added) == report["added_windows"] == 3
    assert {row["video"] for row in added} == {"safe"}
    assert all(row["split"] == "train" and row["label"] == 0 for row in added)
    assert all(
        row["action_mask"][0] == 1 and row["action_targets"][0] == 0 for row in added
    )


def test_ambiguous_greeting_is_not_fabricated_into_handshake_label():
    windows, report = expand_training(
        [], [source("greeting", actions=["greet"])], ClipConfig(fps=12)
    )
    assert report["added_windows"] == 3
    assert all(row["action_mask"][2] == 0 for row in windows)


def test_close_view_aug_keeps_temporal_consistency_and_does_not_cut_unknown_positive():
    picture = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    clips = np.tile(picture, (1, 16, 1, 1, 1))
    result = augment_close_view(clips, {"label": 0}, random.Random(22))
    assert result.shape == clips.shape and result.dtype == clips.dtype
    assert np.array_equal(result[:, 0], result[:, -1])
    assert not np.array_equal(result, clips)
    assert np.array_equal(
        augment_close_view(
            clips, {"label": 1, "publisher_actions": ["kick"]}, random.Random(22)
        ),
        clips,
    )
    assert np.array_equal(
        augment_close_view(clips, {"label": 1}, random.Random(22)), clips
    )
