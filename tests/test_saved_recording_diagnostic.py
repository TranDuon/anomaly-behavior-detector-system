import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import fight_saved_recording_diagnostic as diagnostic


def test_saved_replay_reuses_logged_times_without_opening_camera(monkeypatch, tmp_path):
    received, paths = [], []

    class Runtime:
        model_ready = True
        metadata = {"event_gate": {"confirm_seconds": 0.4}}
        gate = SimpleNamespace(threshold=0.8)

        def __init__(self, **options):
            self.result = {}

        def warmup(self):
            pass

        def push(self, frame, timestamp, sampled=False):
            received.append((timestamp, sampled))
            self.result = {
                "prediction_timestamp": timestamp,
                "score": 0.9,
                "event": "started" if len(received) == 2 else None,
            }
            return self.result

    class Capture:
        def __init__(self, path, *options):
            paths.append(path)
            self.count = 0

        def isOpened(self):
            return True

        def get(self, option):
            return 12

        def read(self):
            self.count += 1
            return (
                (True, np.zeros((8, 8, 3), np.uint8))
                if self.count <= 3
                else (False, None)
            )

        def release(self):
            pass

    monkeypatch.setattr(diagnostic, "FightRuntime", Runtime)
    monkeypatch.setattr(diagnostic.cv2, "VideoCapture", Capture)
    metadata = tmp_path / "recording.json"
    metadata.write_text(json.dumps({"frame_source_timestamps": [0.6, 0.69, 0.69]}))
    output = tmp_path / "diagnosis.json"
    video = tmp_path / "saved.mp4"
    report = diagnostic.diagnose(tmp_path, video, metadata, output)
    assert received == [(0.6, True), (0.69, True), (0.69, True)]
    assert paths == [str(video)]
    assert report["alarm_starts"] == [0.69] and report["frames"] == 3
    assert report["approved_for_demo"] is False
    with pytest.raises(ValueError, match="replacing"):
        diagnostic.diagnose(tmp_path, video, metadata, output)
