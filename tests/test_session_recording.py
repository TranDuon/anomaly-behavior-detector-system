import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from fight.runtime import FightRuntime
from fight.session_recording import SessionRecording
from fight.vision import ClipConfig


def test_recording_saves_normal_video_trace_and_source_timing(tmp_path):
    recorder = SessionRecording(tmp_path, 4, {"threshold": 0.95}, limit_seconds=1)
    for timestamp in (0.0, 0.5, 0.75):
        assert recorder.offer(np.full((32, 40, 3), 120, np.uint8), timestamp)
    result = {
        "prediction_timestamp": 0.75,
        "score": 0.01,
        "state": "normal",
        "event": None,
        "visible_people": 2,
        "person_boxes": [[0, 0, 10, 20]],
    }
    recorder.observe(result, 0.9)
    recorder.observe(result, 1.0)
    assert not recorder.offer(np.zeros((32, 40, 3), np.uint8), 1.0)
    recorder.finish(outcome={"phase": "stopped"})
    status = recorder.status()
    assert status["phase"] == "saved" and status["video"] and status["report"]
    assert status["predictions"] == 1
    report = json.loads(recorder.report_path.read_text(encoding="utf-8"))
    assert report["events"] == []
    assert report["maximum_observed_score"] == 0.01
    assert report["frame_source_timestamps"] == [0.0, 0.0, 0.5, 0.75]
    assert report["recording"]["repeated_frames"] == 1
    assert report["metadata"]["session_end"]["phase"] == "stopped"
    assert report["predictions"][0]["visible_people"] == 2
    capture = cv2.VideoCapture(str(recorder.video_path))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert capture.read()[0]
    capture.release()


def test_recording_queue_is_bounded_and_does_not_wait_for_encoder(
    tmp_path, monkeypatch
):
    import fight.session_recording as recording

    entered, released = threading.Event(), threading.Event()

    class Writer:
        def __init__(self, filename, *arguments):
            self.path = recording.Path(filename)
            self.path.write_bytes(b"fake-video-for-unit-test")

        def isOpened(self):
            return True

        def write(self, frame):
            entered.set()
            assert released.wait(timeout=3)

        def release(self):
            pass

    monkeypatch.setattr(recording.cv2, "VideoWriter", Writer)
    recorder = SessionRecording(tmp_path, 4, {}, limit_seconds=1, capacity=1)
    frame = np.zeros((32, 40, 3), np.uint8)
    try:
        assert recorder.offer(frame, 0)
        assert entered.wait(timeout=2)
        assert recorder.offer(frame, 0.25)
        assert not recorder.offer(frame, 0.5)
        assert recorder.status()["recording_dropped_frames"] == 1
        assert recorder.pending.qsize() == 1
    finally:
        released.set()
        recorder.finish()
    assert not recorder.thread.is_alive()


def test_recording_preserves_jittered_samples_without_false_repeat_counts(tmp_path):
    recorder = SessionRecording(tmp_path, 12, {}, limit_seconds=1)
    timestamps = [0.05, 0.084, 0.184, 0.25, 0.35, 0.417, 0.517, 0.584]
    for timestamp in timestamps:
        assert recorder.offer(np.zeros((32, 40, 3), np.uint8), timestamp)
    recorder.finish()
    report = json.loads(recorder.report_path.read_text(encoding="utf-8"))
    assert report["frame_source_timestamps"] == timestamps
    assert report["recording"]["repeated_frames"] == 0


def test_recording_error_is_not_reported_as_a_saved_video(tmp_path, monkeypatch):
    import fight.session_recording as recording

    monkeypatch.setattr(
        recording.cv2,
        "VideoWriter",
        lambda *args: SimpleNamespace(isOpened=lambda: False, release=lambda: None),
    )
    recorder = SessionRecording(tmp_path, 4, {}, limit_seconds=1)
    recorder.offer(np.zeros((32, 40, 3), np.uint8), 0)
    recorder.finish()
    status = recorder.status()
    assert status["phase"] == "failed" and status["error"]
    assert status["video"] is None and status["report"]


def test_recording_without_frames_fails_and_limits_are_enforced(tmp_path):
    with pytest.raises(ValueError):
        SessionRecording(tmp_path, 12, {}, limit_seconds=31)
    recorder = SessionRecording(tmp_path, 4, {}, limit_seconds=1)
    recorder.finish()
    assert recorder.status()["phase"] == "failed"
    assert recorder.status()["video"] is None


def test_recording_logs_insufficient_input_even_without_a_model_score(tmp_path):
    recorder = SessionRecording(tmp_path, 4, {}, limit_seconds=1)
    recorder.offer(np.zeros((32, 40, 3), np.uint8), 0)
    recorder.observe({"state": "warming_up", "prediction_timestamp": None}, 0.1, 0)
    recorder.observe(
        {
            "state": "insufficient",
            "message": "FPS too low",
            "prediction_timestamp": None,
        },
        0.6,
        0.5,
    )
    recorder.finish()
    report = json.loads(recorder.report_path.read_text(encoding="utf-8"))
    assert report["predictions"] == []
    assert report["maximum_observed_score"] is None
    assert [row["state"] for row in report["state_changes"]] == [
        "warming_up",
        "insufficient",
    ]
    assert report["state_changes"][-1]["source_timestamp"] == 0.5


def test_realtime_session_records_without_alert_and_stops_at_limit(
    tmp_path, monkeypatch
):
    import api.fight_server as server

    class LowScore(nn.Module):
        def forward(self, clips):
            return clips.mean(dim=(1, 2, 3, 4)) * 0 - 5

    detector = SimpleNamespace(
        detect=lambda frame: [SimpleNamespace(bbox_xyxy=[2, 2, 30, 30])]
    )
    runtime = FightRuntime(
        predictor=LowScore(),
        config=ClipConfig(frames=4, fps=8, size=16),
        detector=detector,
    )
    monkeypatch.setattr(server, "FightRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(server.settings, "fight_test_record_seconds", 0.5)
    processor = server.FightProcessor()
    processor.evidence_dir = tmp_path
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 32))
    for index in range(20):
        writer.write(np.full((32, 40, 3), 100, np.uint8))
    writer.release()
    processor.start(str(source), allow_pilot=True, replay=True, record_test=True)
    processor.thread.join(timeout=8)
    assert not processor.is_running()
    assert processor.phase == "completed"
    assert processor.frame_count < 20
    assert processor.events == []
    status = processor.status()["recording"]
    assert status["phase"] == "saved" and status["video"] and status["report"]
    report = json.loads(processor.recording.report_path.read_text(encoding="utf-8"))
    assert report["predictions"]
    assert all(row["state"] == "normal" for row in report["predictions"])
    assert report["metadata"]["session_end"]["phase"] == "completed"


def test_preview_cannot_silently_record_or_start_a_camera(monkeypatch):
    import api.fight_server as server

    processor = server.FightProcessor()
    monkeypatch.setattr(
        processor,
        "stop",
        lambda: pytest.fail("Must reject before touching active camera"),
    )
    with pytest.raises(ValueError, match="Ghi test"):
        processor.start(0, preview=True, record_test=True)


def test_new_source_cannot_accumulate_unfinished_recording_workers(monkeypatch):
    import api.fight_server as server

    processor = server.FightProcessor()
    monkeypatch.setattr(processor, "stop", lambda: None)
    processor.recording = SimpleNamespace(thread=SimpleNamespace(is_alive=lambda: True))
    with pytest.raises(RuntimeError, match="đang hoàn tất"):
        processor.start(0)


def test_presampled_camera_jitter_is_not_sampled_a_second_time():
    class LowScore(nn.Module):
        def forward(self, clips):
            return clips.mean(dim=(1, 2, 3, 4)) * 0 - 5

    detector = SimpleNamespace(
        detect=lambda frame: [SimpleNamespace(bbox_xyxy=[2, 2, 30, 30])]
    )
    config = ClipConfig(frames=16, fps=12, size=16)
    runtime = FightRuntime(predictor=LowScore(), config=config, detector=detector)
    frame = np.zeros((32, 40, 3), np.uint8)
    timestamps = [0.095 + index / 30 for index in range(80) if index % 5 in (0, 3)]
    for timestamp in timestamps:
        result = runtime.push(frame, timestamp, sampled=True)
    assert len(runtime.samples) == 16
    assert result["prediction_timestamp"] is not None
    assert result["state"] == "normal"
    assert list(runtime.sample_times) == timestamps[-16:]
    reset = runtime.push(frame, 8, sampled=True)
    assert reset["state"] == "warming_up"
    assert len(runtime.samples) == 1
