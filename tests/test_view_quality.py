from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fight.runtime import FightRuntime
from fight.view_quality import assess_view
from fight.vision import ClipConfig


def test_close_cropped_person_reports_missing_partner_without_inventing_one():
    report = assess_view([[0, 30, 519, 477]], (480, 640, 3))
    assert report["limited"] and report["visible_people"] == 1
    assert "interaction_partner_not_visible" in report["reasons"]


def test_small_person_at_edge_does_not_invalidate_clear_group_view():
    assert not assess_view([[0, 70, 30, 200], [160, 60, 260, 300]], (480, 640, 3))[
        "limited"
    ]
    assert not assess_view([[100, 50, 260, 420], [300, 50, 460, 420]], (480, 640, 3))[
        "limited"
    ]
    with pytest.raises(ValueError, match="Finite"):
        assess_view([[0, 0, float("nan"), 2]], (480, 640, 3))


@pytest.mark.parametrize("logit, expected", [(-5.0, "insufficient"), (5.0, "fighting")])
def test_limited_view_does_not_suppress_sustained_positive_evidence(logit, expected):
    class Model(torch.nn.Module):
        def forward(self, clips):
            return clips.mean(dim=(1, 2, 3, 4)) * 0 + logit

    class Detector:
        def detect(self, frame):
            return [SimpleNamespace(bbox_xyxy=[0, 2, 60, 47])]

    runtime = FightRuntime(
        predictor=Model(),
        detector=Detector(),
        config=ClipConfig(frames=4, fps=4, size=16),
    )
    for index in range(16):
        result = runtime.push(np.zeros((48, 64, 3), np.uint8), index / 4)
    assert result["state"] == expected
    assert result["view_quality"]["limited"]
    if expected == "insufficient":
        assert "không phải kết luận an toàn" in result["message"]
