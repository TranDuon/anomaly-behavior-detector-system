import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from anomaly.fight_data import VideoItem
from fight.runtime import EventGate, FightRuntime
from fight.vision import ClipConfig, clip_tensor, prepare_clip, regions_for
from scripts.fight_pilot import build_manifest, cache_key, metrics


class ConstantModel(nn.Module):
    def forward(self, values):
        return values.mean(dim=(1, 2, 3, 4)) * 0 + 5


class Detector:
    def __init__(self, count=2):
        self.count = count

    def detect(self, frame):
        return [
            SimpleNamespace(bbox_xyxy=[index * 5 + 4, 3, index * 5 + 14, 27])
            for index in range(self.count)
        ]


def test_gate_needs_sustained_evidence_and_emits_once():
    gate = EventGate(confirm_seconds=0.8, release_seconds=1)
    assert gate.update(0.9, 0) == ("verifying", None)
    assert gate.update(0.1, 0.5) == ("normal", None)
    assert gate.update(0.9, 1) == ("verifying", None)
    assert gate.update(0.9, 2) == ("fighting", "started")
    assert gate.update(0.9, 3) == ("fighting", None)
    assert gate.update(0.1, 4) == ("fighting", None)
    assert gate.update(0.1, 5.1) == ("normal", "ended")


def test_no_legacy_checkpoint_fallback(tmp_path):
    (tmp_path / "lstm_autoencoder.pt").write_bytes(b"legacy")
    runtime = FightRuntime(tmp_path)
    assert not runtime.model_ready
    assert runtime.push(np.zeros((32, 32, 3), np.uint8), 0)["state"] == "unavailable"


def test_unapproved_pilot_requires_opt_in(tmp_path):
    config = ClipConfig(frames=4, fps=2, size=16)
    torch.jit.trace(ConstantModel(), torch.zeros(1, 3, 4, 16, 16)).save(
        str(tmp_path / "fight_model.ts")
    )
    (tmp_path / "fight_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": config.to_dict(),
                "threshold": 0.8,
                "approved_for_demo": False,
            }
        )
    )
    assert not FightRuntime(tmp_path).model_ready
    assert FightRuntime(tmp_path, allow_pilot=True).model_ready


def test_replay_same_samples_produces_same_predictions():
    config = ClipConfig(frames=4, fps=2, size=16)
    frames = [np.full((32, 40, 3), index, np.uint8) for index in range(20)]
    first = FightRuntime(predictor=ConstantModel(), config=config, detector=Detector())
    second = FightRuntime(predictor=ConstantModel(), config=config, detector=Detector())
    offline = [first.push(frame, index / 4) for index, frame in enumerate(frames)]
    live = [second.push(frame, index / 4) for index, frame in enumerate(frames)]
    assert offline == live
    assert sum(row["event"] == "started" for row in live) == 1
    assert live[-1]["state"] == "fighting"


def test_no_people_does_not_confirm_fight():
    runtime = FightRuntime(
        predictor=ConstantModel(),
        config=ClipConfig(frames=4, fps=2, size=16),
        detector=Detector(0),
    )
    results = [
        runtime.push(np.zeros((32, 32, 3), np.uint8), index / 2) for index in range(20)
    ]
    assert all(result["event"] is None for result in results)
    assert results[-1]["state"] == "insufficient"


def test_stream_gap_resets_stale_alerts():
    runtime = FightRuntime(
        predictor=ConstantModel(),
        config=ClipConfig(frames=4, fps=4, size=16),
        detector=Detector(),
    )
    for index in range(20):
        runtime.push(np.zeros((32, 32, 3), np.uint8), index / 4)
    assert runtime.result["state"] == "fighting"
    assert runtime.push(np.zeros((32, 32, 3), np.uint8), 20)["state"] == "warming_up"
    with pytest.raises(ValueError):
        runtime.push(np.zeros((32, 32, 3), np.uint8), 19)


def test_low_fps_is_insufficient_instead_of_stretching_the_action():
    runtime = FightRuntime(
        predictor=ConstantModel(),
        config=ClipConfig(frames=4, fps=8, size=16),
        detector=Detector(),
    )
    results = [
        runtime.push(np.zeros((32, 32, 3), np.uint8), index / 2) for index in range(20)
    ]
    assert results[-1]["state"] == "insufficient"
    assert all(result["event"] is None for result in results)


def test_regions_preserve_both_people_and_do_not_distort():
    region = regions_for([[1, 3, 12, 28], [14, 4, 26, 27]], 40, 32)[0]
    assert region[0] <= 1 and region[2] >= 26
    assert prepare_clip([np.zeros((32, 40, 3), np.uint8)] * 4, region, 16).shape == (
        4,
        16,
        16,
        3,
    )
    assert clip_tensor(np.zeros((2, 4, 16, 16, 3), np.uint8)).shape == (2, 3, 4, 16, 16)


def test_manifest_never_uses_official_test_and_keeps_groups_together(tmp_path):
    items = []
    rng = np.random.default_rng(42)
    for index in range(10):
        path = tmp_path / f"{index}.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32)
        )
        frame = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        for frame_index in range(80):
            writer.write(frame)
        writer.release()
        labels = np.zeros(80, np.int32)
        if index % 2:
            labels[25:55] = 1
        items.append(
            VideoItem(
                "ubi",
                str(index),
                path,
                "test" if index == 9 else "train",
                frame_labels=labels,
            )
        )
    manifest = build_manifest(items, 8, 8, ClipConfig(), 42)
    assert manifest["official_test_used"] is False
    assert all(row["video"] != "ubi/9" for row in manifest["rows"])
    partitions = {}
    for row in manifest["rows"]:
        partitions.setdefault(row["scene_group"], set()).add(row["split"])
    assert all(len(values) == 1 for values in partitions.values())


def test_metrics_detect_scene_shortcut_and_cache_is_versioned():
    rows = [
        {"label": label, "video": video}
        for video in ("first", "second")
        for label in (0, 1)
    ]
    result = metrics(rows, [0.8, 0.8, 0.1, 0.1], 0.5)
    assert result["in_video_auc"] == 0.5
    row = {"video": "ubi/1", "start": 0, "fps": 30, "label": 0}
    assert cache_key(row, ClipConfig(), "s3d") != cache_key(row, ClipConfig(), "x3d_s")


def test_api_missing_model_returns_conflict_and_upload_cannot_traverse(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient
    from api.fight_server import app, processor

    monkeypatch.setattr(
        processor,
        "start",
        lambda *args: (_ for _ in ()).throw(ValueError("missing model")),
    )
    client = TestClient(app)
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/stream/camera").status_code == 409
    assert client.post("/api/stream/url?url=file:///secret").status_code == 400
    assert (
        client.post(
            "/api/upload", files={"file": ("bad.txt", b"data", "text/plain")}
        ).status_code
        == 400
    )
    assert client.get("/api/evidence/unknown.mp4").status_code == 404


def test_uploaded_video_finishes_once_and_saves_one_event(tmp_path, monkeypatch):
    import api.fight_server as server

    runtime = FightRuntime(
        predictor=ConstantModel(),
        config=ClipConfig(frames=4, fps=4, size=16),
        detector=Detector(),
    )
    monkeypatch.setattr(server, "FightRuntime", lambda *args, **kwargs: runtime)
    processor = server.FightProcessor()
    processor.evidence_dir = tmp_path / "evidence"
    processor.evidence_dir.mkdir()
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 32))
    for frame_index in range(40):
        writer.write(np.full((32, 40, 3), frame_index, np.uint8))
    writer.release()
    processor.start(str(path), allow_pilot=True)
    processor.thread.join(timeout=10)
    assert not processor.is_running()
    assert processor.phase == "completed"
    assert processor.frame_count == 40
    assert len(processor.events) == 1
    assert processor.events[0]["ended_at"] is not None
    assert (
        processor.output
        and (processor.evidence_dir / Path(processor.output).name).exists()
    )


def test_partial_finetuning_does_not_unfreeze_the_entire_backbone():
    from fight.model import VideoClassifier

    model = VideoClassifier.__new__(VideoClassifier)
    nn.Module.__init__(model)
    model.backbone = nn.Module()
    model.backbone.blocks = nn.ModuleList([nn.Linear(2, 2) for block_index in range(5)])
    model.head = nn.Linear(2, 1)
    model.set_stage("finetune")
    assert not any(
        parameter.requires_grad for parameter in model.backbone.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.backbone.blocks[-1].parameters()
    )
    assert all(parameter.requires_grad for parameter in model.head.parameters())
