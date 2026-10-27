import random
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from fight.runtime import EventGate, FightRuntime
from fight.vision import ClipConfig, DetectionCache
from scripts.fight_large_train import (
    operating_point,
    people_focused_training,
    validate_manifest,
)
from scripts.fight_public_data import airtlab_items, assign_groups, validate_zip


def public_rows():
    generator = random.Random(17)
    return [
        {
            "video": f"source/{index}",
            "dataset": "rlvs",
            "label": index % 2,
            "sha256": str(index),
            "scene_hashes": [generator.getrandbits(64) for _ in range(3)],
        }
        for index in range(400)
    ]


def test_grouped_four_way_split_keeps_camera_pairs_and_scene_negatives():
    rows = public_rows()
    rows.extend(
        [
            {
                **rows[0],
                "video": "airtlab/cam1/1",
                "dataset": "airtlab",
                "sha256": "cam1",
                "performance": "same_performance",
            },
            {
                **rows[2],
                "video": "airtlab/cam2/1",
                "dataset": "airtlab",
                "sha256": "cam2",
                "performance": "same_performance",
            },
        ]
    )
    rows[1]["scene_hashes"] = rows[0]["scene_hashes"]
    accepted, conflicts = assign_groups(rows)
    assert not conflicts
    validate_manifest({"rows": accepted})
    pair = [row for row in accepted if row.get("performance")]
    assert len({row["scene_group"] for row in pair}) == 1
    scene = [row for row in accepted if row["video"] in {"source/0", "source/1"}]
    assert len({row["scene_group"] for row in scene}) == 1
    assert {row["label"] for row in scene} == {0, 1}
    accepted[0]["scene_group"] = next(
        row["scene_group"] for row in accepted if row["split"] != accepted[0]["split"]
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({"rows": accepted})


def test_airtlab_excludes_weapons_but_keeps_hard_negatives(tmp_path):
    (tmp_path / "violent-action-classes.csv").write_text(
        "FILE;ACTION CLASSES\n1.mp4;punch,fight\n2.mp4;club,fight\n3.mp4;gunshot\n4.mp4;push\n"
    )
    for category in ("violent", "non-violent"):
        for camera in ("cam1", "cam2"):
            directory = tmp_path / category / camera
            directory.mkdir(parents=True)
            for index in range(1, 5):
                (directory / f"{index}.mp4").touch()
    rows = airtlab_items(tmp_path)
    assert sum(row["label"] for row in rows) == 2
    assert sum(row["label"] == 0 for row in rows) == 8
    assert len({row["performance"] for row in rows if row["label"]}) == 1


def test_zip_traversal_and_incomplete_download_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.mp4", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        validate_zip(archive, 1)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("clip.mp4", "video")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_zip(archive, 2)


def test_calibration_never_claims_impossible_precision_recall():
    rows = [
        {"label": label, "video": str(index)} for index, label in enumerate([0, 1] * 10)
    ]
    failed = operating_point(rows, [0.5] * 20)
    assert not failed["target_feasible_on_calibration"]
    passed = operating_point(rows, [0.1, 0.9] * 10)
    assert passed["target_feasible_on_calibration"]
    assert passed["calibration"]["precision"] == 1
    assert passed["calibration"]["recall"] == 1
    assert passed["calibration"]["precision_wilson95"][0] < 0.8


def test_gate_does_not_join_distant_high_predictions():
    gate = EventGate(confirm_seconds=0.8, max_observation_gap=1.5)
    assert gate.update(0.95, 0) == ("verifying", None)
    assert gate.update(0.95, 10) == ("verifying", None)
    assert gate.update(0.95, 11) == ("fighting", "started")
    assert gate.update(0.95, 15) == ("insufficient", "ended")
    assert gate.update(0.95, 15.5) == ("verifying", None)
    with pytest.raises(ValueError):
        gate.update(float("nan"), 16)
    with pytest.raises(ValueError):
        gate.update(0.9, 14)
    assert gate.fresh().max_observation_gap == 1.5


def test_people_selection_only_filters_training_background_negatives(tmp_path):
    rows = []
    for index in range(20):
        row = {"cache": str(index), "sha256": str(index), "label": int(index < 10)}
        np.savez(tmp_path / f"{index}.npz", people=0)
        rows.append(row)
    kept, report = people_focused_training(rows, tmp_path)
    assert sum(row["label"] for row in kept) == 10
    assert report["negative_without_people_excluded"] == 10
    assert len(rows) == 20


class BrightModel(torch.nn.Module):
    def forward(self, values):
        return values.mean(dim=(1, 2, 3, 4)) * 0 + 5


class People:
    def detect(self, frame):
        return [SimpleNamespace(bbox_xyxy=[1, 1, 22, 30])]


def test_detection_cache_reuses_only_identical_frame_crop_and_evicts():
    calls = []
    detector = SimpleNamespace(
        detect=lambda frame: calls.append(frame.shape) or [frame.sum()]
    )
    cache = DetectionCache(detector, capacity=2)
    original = np.zeros((32, 40, 3), np.uint8)
    assert cache.detect(original[:20, :20]) == cache.detect(original[:20, :20])
    assert len(calls) == 1
    cache.detect(original[:20, 1:21])
    assert len(calls) == 2
    cache.detect(original.copy())
    assert len(cache.entries) == 2
    cache.retain([])
    assert len(cache.entries) == 0


def test_runtime_detection_reuse_preserves_uncached_scores():
    frame = np.zeros((32, 40, 3), np.uint8)
    config = ClipConfig(frames=4, fps=4, size=16, stride_seconds=0.25)
    cached = FightRuntime(predictor=BrightModel(), config=config, detector=People())
    reference = FightRuntime(predictor=BrightModel(), config=config, detector=People())
    for index in range(20):
        first = cached.push(frame, index / 4)
        reference.detection_cache = None
        second = reference.push(frame, index / 4)
        assert {
            key: value for key, value in first.items() if key != "inference_seconds"
        } == {key: value for key, value in second.items() if key != "inference_seconds"}


def test_evidence_resamples_missing_frames_in_source_time(tmp_path):
    from api.fight_server import FightProcessor

    processor = FightProcessor()
    processor.runtime = FightRuntime(
        predictor=BrightModel(),
        config=ClipConfig(frames=4, fps=4, size=16),
        detector=People(),
    )
    processor.evidence_dir = tmp_path
    frame = np.zeros((32, 40, 3), np.uint8)
    for timestamp, event in (
        (0.0, None),
        (0.25, "started"),
        (1.5, None),
        (2.0, "ended"),
    ):
        processor.timestamp = timestamp
        processor.record(
            frame, {"event": event, "score": 0.95, "state": "fighting", "regions": []}
        )
    report = processor.events[0]
    assert report["source_samples"] == 4
    assert report["max_sample_gap_seconds"] == 1.25
    capture = cv2.VideoCapture(str(tmp_path / report["clip"].split("/")[-1]))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 9
    assert capture.get(cv2.CAP_PROP_FPS) == 4
    capture.release()


def test_low_fps_reset_preserves_custom_gate_settings():
    runtime = FightRuntime(
        predictor=BrightModel(),
        config=ClipConfig(frames=4, fps=8, size=16),
        detector=People(),
    )
    runtime.gate = EventGate(
        confirm_seconds=2, release_seconds=3, max_observation_gap=4
    )
    for index in range(10):
        runtime.push(np.zeros((32, 40, 3), np.uint8), index / 2)
    assert runtime.result["state"] == "insufficient"
    assert runtime.gate.confirm_seconds == 2
    assert runtime.gate.release_seconds == 3
    assert runtime.gate.max_observation_gap == 4


def test_live_status_hides_stale_red_prediction(monkeypatch):
    import api.fight_server as server

    processor = server.FightProcessor()
    processor.runtime.result.update(
        state="fighting", score=0.95, prediction_timestamp=1.0
    )
    processor.session_started = 10.0
    processor.timestamp = 1.0
    processor.replay = True
    processor.thread = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(server.time, "monotonic", lambda: 15.0)
    status = processor.status()
    assert status["prediction"]["state"] == "insufficient"
    assert status["prediction"]["score_stale"]
    assert status["prediction_age_seconds"] == 4
    assert processor.runtime.result["state"] == "fighting"


def test_missing_people_ends_active_alert_without_claiming_normal():
    detector = People()
    runtime = FightRuntime(
        predictor=BrightModel(),
        config=ClipConfig(frames=4, fps=4, size=16),
        detector=detector,
    )
    frame = np.zeros((32, 40, 3), np.uint8)
    for index in range(12):
        runtime.push(frame, index / 4)
    assert runtime.result["state"] == "fighting"
    detector.detect = lambda frame: []
    runtime.detection_cache = None
    result = runtime.push(frame, 3.25)
    assert result["state"] == "insufficient"
    assert result["event"] == "ended"
    assert not runtime.gate.active


def test_processor_uploaded_file_and_paced_camera_replay(tmp_path):
    from api.fight_server import FightProcessor

    source = tmp_path / "recorded.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (40, 32))
    for index in range(32):
        writer.write(np.full((32, 40, 3), index * 5, np.uint8))
    writer.release()
    outcomes = []
    for replay in (False, True):
        processor = FightProcessor()
        processor.runtime = FightRuntime(
            predictor=BrightModel(),
            config=ClipConfig(frames=4, fps=4, size=16, stride_seconds=0.25),
            detector=People(),
        )
        processor.preview = False
        processor.replay = replay
        processor.evidence_dir = tmp_path
        records = []
        processor.record = lambda frame, result: records.append(
            (processor.timestamp, dict(result))
        )
        processor.run(str(source))
        assert processor.phase == "completed", processor.error
        assert processor.frame_count == 32
        assert len(records) == 16
        outcomes.append(
            [
                (round(timestamp, 2), result["event"], result["state"])
                for timestamp, result in records
            ]
        )
    assert outcomes[0] == outcomes[1]
    assert sum(event == "started" for _, event, _ in outcomes[1]) == 1
