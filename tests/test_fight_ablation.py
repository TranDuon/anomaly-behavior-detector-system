import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fight.vision import ClipConfig, DetectorConfig, clip_regions
from scripts.fight_ablation import (
    extra_negative_candidates,
    fixed_manifest,
    stable_train,
    threshold_report,
)


def test_full_frame_mode_and_legacy_clip_metadata():
    legacy = ClipConfig(**{"frames": 16, "fps": 8, "size": 182})
    assert legacy.region_mode == "interaction"
    assert clip_regions([[1, 1, 10, 10]], 40, 32, ClipConfig(region_mode="full")) == [
        (0, 0, 40, 32)
    ]
    with pytest.raises(ValueError):
        ClipConfig(region_mode="unsupported")


def test_runtime_detector_uses_explicit_bundle_configuration():
    from detection.detector import PersonDetector

    received = {}
    detector = PersonDetector(**DetectorConfig().to_dict(), device="cpu")

    class FakeModel:
        def predict(self, frame, **kwargs):
            received.update(kwargs)
            return []

    detector._model = FakeModel()
    assert detector.detect(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    assert received["iou"] == 0.45
    assert received["conf"] == 0.4
    assert received["imgsz"] == 640
    assert received["device"] == "cpu"


def test_fixed_manifest_rejects_official_test_and_scene_leakage(tmp_path, monkeypatch):
    import scripts.fight_ablation as ablation

    source = tmp_path / "manifest.json"
    rows = [
        {
            "video": "ubi/one",
            "start": 0,
            "label": 0,
            "scene_group": "same",
            "split": "train",
        },
        {
            "video": "ubi/two",
            "start": 0,
            "label": 1,
            "scene_group": "same",
            "split": "validation",
        },
    ]
    source.write_text(json.dumps({"official_test_used": False, "rows": rows}))
    items = [
        SimpleNamespace(uid="ubi/one", split="train", path=tmp_path / "one.mp4"),
        SimpleNamespace(uid="ubi/two", split="train", path=tmp_path / "two.mp4"),
    ]
    monkeypatch.setattr(ablation, "ubi_fights_items", lambda root: items)
    with pytest.raises(ValueError, match="Scene group"):
        fixed_manifest(source, tmp_path)
    rows[1]["scene_group"] = "other"
    source.write_text(json.dumps({"official_test_used": False, "rows": rows}))
    result = fixed_manifest(source, tmp_path)
    assert [
        (row["video"], row["start"], row["label"], row["split"])
        for row in result["rows"]
    ] == [(row["video"], row["start"], row["label"], row["split"]) for row in rows]
    items[1].split = "test"
    with pytest.raises(ValueError, match="Official test"):
        fixed_manifest(source, tmp_path)


def test_threshold_report_does_not_hide_precision_recall_tradeoff():
    rows = [{"video": "one", "label": label} for label in (0, 1, 0, 1)]
    report = threshold_report(rows, np.array([0.9, 0.8, 0.7, 0.6]))
    assert report["best_recall_at_precision75"] == 0
    assert report["best_precision_at_recall60"] == 0.5


def test_hard_negative_mining_never_reads_validation_video(tmp_path, monkeypatch):
    import scripts.fight_ablation as ablation

    class Capture:
        def get(self, property_id):
            return 200

        def release(self):
            pass

    class Item:
        uid = "ubi/train"
        split = "train"
        path = tmp_path / "train.mp4"

        def labels_for(self, count):
            return np.zeros(count, dtype=int)

    items = [
        Item(),
        SimpleNamespace(
            uid="ubi/validation", split="train", path=tmp_path / "validation.mp4"
        ),
    ]
    monkeypatch.setattr(ablation, "ubi_fights_items", lambda root: items)
    monkeypatch.setattr(ablation.cv2, "VideoCapture", lambda path: Capture())
    rows = [
        {
            "video": "ubi/train",
            "start": 0,
            "fps": 10,
            "label": 0,
            "split": "train",
            "scene_group": "train",
        },
        {
            "video": "ubi/validation",
            "start": 0,
            "fps": 10,
            "label": 0,
            "split": "validation",
            "scene_group": "validation",
        },
    ]
    candidates = extra_negative_candidates(rows, tmp_path, ClipConfig(frames=4, fps=2))
    assert candidates
    assert all(
        row["video"] == "ubi/train"
        and row["split"] == "train"
        and row["label"] == 0
        and row["start"] >= 16
        for row in candidates
    )


def test_accumulated_training_keeps_best_checkpoint(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Identity()
            self.head = nn.Linear(2, 1)

        def set_stage(self, stage):
            pass

    rows = []
    for index in range(8):
        label = index % 2
        np.savez(
            tmp_path / f"{index}.npz",
            features=np.array([[label * 2 - 1, 1]], dtype=np.float32),
        )
        rows.append({"cache": str(index), "label": label, "video": str(index // 2)})
    model = Model()
    history = stable_train(
        model,
        rows,
        rows,
        tmp_path,
        "cpu",
        time.monotonic() + 30,
        tmp_path,
        epochs=(3, 0),
    )
    assert len(history) == 3
    state = torch.load(tmp_path / "best.pt", weights_only=True)["state_dict"]
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, state[name])
