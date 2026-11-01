import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from fight.runtime import EventGate
from fight.vision import ClipConfig


def test_replay_preview_continues_while_inference_is_busy(tmp_path, monkeypatch):
    import api.fight_server as server

    busy = threading.Event()
    runtime = SimpleNamespace(
        model_ready=True,
        warmup=lambda: None,
        config=ClipConfig(frames=4, fps=4, size=16),
        gate=EventGate(),
        result={"state": "normal", "score": 0.01, "event": None},
    )

    def push(frame, timestamp, **kwargs):
        busy.set()
        time.sleep(0.3)
        busy.clear()
        runtime.result = {"state": "normal", "score": 0.01, "event": None,
                          "prediction_timestamp": timestamp, "person_boxes": []}
        return runtime.result

    runtime.push = push
    monkeypatch.setattr(server, "FightRuntime", lambda *args, **kwargs: runtime)
    processor = server.FightProcessor()
    processor.evidence_dir = tmp_path
    previews = []
    monkeypatch.setattr(processor, "publish_preview", lambda frame, result: previews.append((time.monotonic(), int(frame.mean()), busy.is_set())))
    path = tmp_path / "smooth.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (40, 32))
    assert writer.isOpened()
    for index in range(40):
        writer.write(np.full((32, 40, 3), index * 4, np.uint8))
    writer.release()
    processor.start(str(path), allow_pilot=True, replay=True)
    processor.thread.join(timeout=10)
    if processor.is_running():
        processor.stop()
        raise AssertionError("Replay did not terminate")
    assert processor.phase == "completed", processor.error
    busy_frames = [value for _, value, was_busy in previews if was_busy]
    assert len(set(busy_frames)) >= 12
    assert len(previews) >= 20
    assert processor.dropped_samples == 0


def test_scaled_preview_does_not_mutate_inference_boxes(monkeypatch):
    import api.fight_server as server

    processor = server.FightProcessor()
    monkeypatch.setattr(server.settings, "fight_preview_max_width", 100)
    result = {"state": "normal", "person_boxes": [[20, 40, 80, 90]], "regions": [],
              "actions": [{"box": [20, 40, 80, 90], "labels": []}]}
    processor.publish_preview(np.zeros((100, 200, 3), np.uint8), result)
    assert result["person_boxes"] == [[20, 40, 80, 90]]
    assert result["actions"][0]["box"] == [20, 40, 80, 90]
    assert processor.jpeg and processor.revision > 0
