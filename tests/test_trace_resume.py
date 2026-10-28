import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fight.vision import ClipConfig
from scripts import fight_refinement_train as runner


def test_trace_identity_changes_with_weights_source_or_runtime():
    model = torch.nn.Linear(1, 1)
    model.use_interaction = False
    row = dict(
        video="public",
        sha256="source",
        fps=12,
        frames=2,
        split="development",
        dataset="public",
    )
    metadata = {"clip": ClipConfig().to_dict(), "pose": {}}
    original = runner.trace_fingerprint(model, row, metadata, {"code": "v1"})
    assert original != runner.trace_fingerprint(model, row, metadata, {"code": "v2"})
    assert original != runner.trace_fingerprint(
        model, {**row, "sha256": "new"}, metadata, {"code": "v1"}
    )
    with torch.no_grad():
        model.weight.add_(1)
    assert original != runner.trace_fingerprint(model, row, metadata, {"code": "v1"})


def test_completed_source_survives_timeout_and_resume_uses_no_decoder(
    tmp_path, monkeypatch
):
    class Runtime:
        def __init__(self, **kwargs):
            self.result = {}

        def push(self, frame, timestamp):
            self.result = {"prediction_timestamp": timestamp, "score": 0.1, "people": 2}
            return self.result

    captures = []

    class Capture:
        def __init__(self, *args):
            self.frames = 0
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            self.frames += 1
            return (
                (True, np.zeros((8, 8, 3), dtype=np.uint8))
                if self.frames <= 2
                else (False, None)
            )

        def release(self):
            pass

    monkeypatch.setattr(runner, "FightRuntime", Runtime)
    monkeypatch.setattr(runner.cv2, "VideoCapture", Capture)
    model = torch.nn.Linear(1, 1)
    model.use_interaction = False
    row = dict(
        video="one",
        sha256="sha",
        fps=12,
        frames=2,
        split="development",
        dataset="public",
        path="public.mp4",
    )
    metadata = {"clip": ClipConfig().to_dict(), "pose": {}}
    arguments = (model, [row], metadata, SimpleNamespace(), "cpu")
    first = runner.trace_sources(*arguments, float("inf"), tmp_path, {"code": "v1"})
    cached = runner.trace_sources(*arguments, 0, tmp_path, {"code": "v1"})
    assert cached == first and len(captures) == 1
    with pytest.raises(TimeoutError):
        runner.trace_sources(
            model,
            [row, {**row, "video": "two"}],
            metadata,
            SimpleNamespace(),
            "cpu",
            0,
            tmp_path,
            {"code": "v1"},
        )
    assert len(list(tmp_path.glob("*.json"))) == 1
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["trace"]["predictions"][0]["score"] = 0.9
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.trace_sources(*arguments, float("inf"), tmp_path, {"code": "v1"})
