"""Checks for reviewed labels, calibration, and quick alert release."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mmaction2_demo"))

from fight.runtime import EventGate  # noqa: E402
from train_action_transition_v2 import anchors, choose_threshold, reviewed_sources  # noqa: E402
from scripts.evaluate_action_trace import evaluate  # noqa: E402


class TransitionTests(unittest.TestCase):
    def test_early_fall_anchors_and_reviewed_after_fight(self):
        fall = {"dataset": "gmdcsa24", "label": 1, "start": 2.0,
                "end": 4.0, "video": "fall"}
        self.assertAlmostEqual(anchors(fall, 6.0)[0], 2.1)
        transition = {"dataset": "reviewed", "action": "after_fight",
                      "start": 10.0, "end": 12.0, "video": "fight"}
        self.assertEqual(anchors(transition, 20.0), [10.25, 10.75])

    def test_reviewed_video_checks_hash_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "sample.mp4"
            video.write_bytes(b"reviewed clip")
            item = {"video": video.name, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "group": "same-scene", "split": "test", "intervals": [
                        {"start_seconds": 0.0, "end_seconds": 2.0,
                         "label": 0, "action": "sports_wrestle"}]}
            manifest = root / "review.json"
            manifest.write_text(json.dumps({"schema_version": 1, "sources": [item]}))
            self.assertEqual(reviewed_sources(manifest, root)[0]["label"], 0)
            item["intervals"][0]["label"] = 1
            manifest.write_text(json.dumps({"schema_version": 1, "sources": [item]}))
            with self.assertRaisesRegex(ValueError, "Invalid reviewed interval"):
                reviewed_sources(manifest, root)

    def test_threshold_ignores_unobservable_positive_windows(self):
        rows = [{"label": 1, "people": 0}, {"label": 1, "people": 1},
                {"label": 0, "people": 1}]
        threshold = choose_threshold(rows, [0.0, 0.8, 0.2])
        self.assertGreater(threshold, 0.2)

    def test_fast_release_needs_two_low_predictions(self):
        gate = EventGate(.836, .585, confirm_seconds=.4, release_seconds=.25)
        self.assertEqual(gate.update(.9, 0.0)[0], "verifying")
        self.assertEqual(gate.update(.9, .53)[0], "fighting")
        self.assertEqual(gate.update(.5, 1.06)[0], "fighting")
        self.assertEqual(gate.update(.5, 1.59), ("normal", "ended"))

    def test_trace_report_uses_reviewed_transition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "sample.mp4"
            video.write_bytes(b"clip")
            checksum = hashlib.sha256(video.read_bytes()).hexdigest()
            review = root / "review.json"
            review.write_text(json.dumps({"sources": [{"sha256": checksum, "intervals": [
                {"start_seconds": 2.0, "end_seconds": 4.0, "action": "after_fight"}]}]}))
            trace = root / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(row) for row in [
                {"kind": "session", "session_id": "one", "source": str(video), "backend": "mmaction_fall_wrestle"},
                {"kind": "prediction", "timestamp": 2.1, "state": "fighting"},
                {"kind": "prediction", "timestamp": 2.6, "state": "normal"},
            ]))
            self.assertEqual(evaluate(trace, review)["intervals"][0]["delay_seconds"], .6)


if __name__ == "__main__":
    unittest.main()


def test_web_writes_one_trace_row_per_prediction(tmp_path, monkeypatch):
    import api.fight_server as server
    import fight.mmaction_runtime as mmaction
    from fight.vision import ClipConfig

    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    runtime = SimpleNamespace(
        model_ready=True, warmup=lambda: None, close=lambda: None,
        config=ClipConfig(frames=4, fps=4, size=16),
        gate=EventGate(.8, .5, confirm_seconds=.4, release_seconds=.25),
        result={"state": "normal", "score": .1, "event": None},
        metadata={"deployment": {}}, input_kind="rgb_actions",
    )

    def push(frame, timestamp, **kwargs):
        if not hasattr(runtime, "last_prediction") or timestamp - runtime.last_prediction >= .5:
            runtime.last_prediction = timestamp
            runtime.result = {"state": "normal", "score": .1, "event": None,
                              "prediction_timestamp": timestamp, "people": 2,
                              "person_boxes": [], "regions": [], "inference_seconds": .01}
        return dict(runtime.result)

    runtime.push = push
    monkeypatch.setattr(mmaction, "MMActionRuntime", lambda **kwargs: runtime)
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8, (40, 32))
    assert writer.isOpened()
    for _ in range(16):
        writer.write(np.zeros((32, 40, 3), dtype=np.uint8))
    writer.release()
    processor = server.FightProcessor()
    processor.start(str(video), allow_pilot=True, backend="mmaction_fall_wrestle")
    processor.thread.join(timeout=10)
    assert processor.phase == "completed", processor.error
    assert processor.trace_path.is_file()
    rows = [json.loads(line) for line in processor.trace_path.read_text().splitlines()]
    assert rows[0]["kind"] == "session"
    predictions = [row for row in rows if row["kind"] == "prediction"]
    assert len(predictions) >= 3
    assert len({row["timestamp"] for row in predictions}) == len(predictions)
