import copy
import json
from pathlib import Path

import pytest

from scripts.fight_action_compare import (
    action_report,
    calibrate,
    labeled_points,
    validate,
)


def corpus():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "benchmarks/punch_kick_v1.json"
        ).read_text()
    )


def test_public_corpus_has_no_performance_split_leakage():
    data = corpus()
    validate(data)
    changed = copy.deepcopy(data["sources"][0])
    changed.update(video="another-camera.mp4", split="test")
    data["sources"].append(changed)
    with pytest.raises(ValueError, match="leakage"):
        validate(data)


def test_unlabeled_time_is_not_assumed_normal():
    sources = [
        {
            "video": "public",
            "group": "one",
            "intervals": [
                {"start_seconds": 1, "end_seconds": 2, "action": "kick", "label": 1}
            ],
        }
    ]
    traces = [
        {
            "video": "public",
            "predictions": [
                {"timestamp": 1, "score": 0.8},
                {"timestamp": 2, "score": 0.9},
            ],
        }
    ]
    points = labeled_points(traces, sources)
    assert len(points) == 1 and points[0]["action"] == "kick"
    assert action_report(points, 0.5)["kick"]["alarm_fraction"] == 1


def test_calibration_uses_only_supplied_labeled_scores():
    assert (
        0.2
        < calibrate(
            [
                {"label": 0, "score": 0.2, "group": "a"},
                {"label": 1, "score": 0.8, "group": "b"},
            ]
        )
        < 0.8
    )
    with pytest.raises(ValueError, match="positive and normal"):
        calibrate([{"label": 0, "score": 0.2}])
