"""Handgun extension skeleton: registry, shared pose pass, weapon output, runtime display state.

No model weights are needed: pose, classifier, veto and locator are stubbed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from extensions import base
from extensions.handgun.extension import HandgunExtension
from extensions.sharp_object import extension as sharp_module
from extensions.sharp_object.extension import SharpObjectExtension
from fight import behaviors as behavior_registry


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _pose_result():
    keypoints = np.zeros((1, 17, 3), np.float32)
    keypoints[0, 8] = (150, 120, 0.9)   # right elbow
    keypoints[0, 10] = (160, 150, 0.9)  # right wrist
    return SimpleNamespace(boxes=_Boxes(xyxy=_Tensor([[100, 40, 220, 300]]), conf=_Tensor([0.9])),
                           keypoints=SimpleNamespace(data=_Tensor(keypoints)))


class _Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy.value)


class _Pose:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [_pose_result()]


def _ready(cls, pose, score=0.9):
    extension = cls()
    extension.pose, extension.pose_key = pose, ("yolo11s-pose.pt", 1, 640, 0.35, 8)
    extension.pose_imgsz, extension.pose_conf, extension.max_people = 640, 0.35, 8
    extension.device, extension.half = "cpu", False
    from extensions.sharp_object.hand_crop import crop_params

    extension.crop, extension.input_size = crop_params(), 32
    extension.locator = extension.veto = extension.tool_veto = None
    extension.threshold = 0.5
    extension._classify = lambda crops: np.full(len(crops), score, np.float32)
    return extension


@pytest.fixture(autouse=True)
def _clear_pose_cache():
    sharp_module._POSE_CACHE.clear()
    yield
    sharp_module._POSE_CACHE.clear()


def test_handgun_is_registered_with_its_own_settings():
    assert "handgun" in base.available()
    gun = base.create("handgun")
    assert isinstance(gun, HandgunExtension)
    assert gun.spec()["display_label"] == "HANDGUN"
    assert gun.method == "hand_crop" and gun.weapon_class == "handgun" and not gun.use_tool_veto
    # Look-alikes of a gun in a hand must not cancel a handgun alert.
    assert not {"cell phone", "remote", "hair drier", "mouse"} & set(gun.veto_classes)
    assert "handgun" in gun.locator_classes


def test_sharp_object_settings_unchanged():
    sharp = SharpObjectExtension()
    assert sharp.veto_classes is sharp_module.VETO_CLASSES
    assert {"cell phone", "remote"} <= sharp.veto_classes
    assert sharp.locator_classes == sharp_module.LOCATOR_CLASSES
    assert sharp.use_tool_veto and sharp.weapon_class == "sharp"


def test_pose_runs_once_per_step_for_both_hand_crop_extensions():
    pose = _Pose()
    sharp, gun = _ready(SharpObjectExtension, pose), _ready(HandgunExtension, pose)
    frames = np.zeros((9, 320, 320, 3), np.uint8)
    first = sharp._pose_people(frames)
    assert gun._pose_people(frames) is first
    assert pose.calls == 1
    gun._pose_people(frames.copy())  # next worker step: a new frames array
    assert pose.calls == 2


def test_pose_not_shared_when_settings_differ():
    pose = _Pose()
    sharp, gun = _ready(SharpObjectExtension, pose), _ready(HandgunExtension, pose)
    gun.pose_key = ("other-pose.pt", 1, 640, 0.35, 8)
    frames = np.zeros((9, 320, 320, 3), np.uint8)
    sharp._pose_people(frames)
    gun._pose_people(frames)
    assert pose.calls == 2


def test_handgun_reports_handgun_weapon_boxes():
    gun = _ready(HandgunExtension, _Pose())
    frames = np.zeros((9, 320, 320, 3), np.uint8)
    output = gun.observe(frames, 1.0, {"person_boxes": [[100, 40, 220, 300]], "frame_size": [320, 320]})
    assert output["score"] == pytest.approx(0.9)
    assert output["boxes"] == [[100.0, 40.0, 220.0, 300.0]]
    weapons = output["details"]["weapons"]
    assert weapons and all(w["class"] == "handgun" for w in weapons)


def test_handgun_below_threshold_has_no_holder():
    gun = _ready(HandgunExtension, _Pose(), score=0.2)
    output = gun.observe(np.zeros((9, 320, 320, 3), np.uint8), 1.0, {"person_boxes": [[100, 40, 220, 300]]})
    assert output["boxes"] == [] and output["details"]["weapons"] == []


def test_runtime_knows_handgun_and_keeps_display_state_apart():
    from fight import mmaction_runtime

    assert "handgun" in mmaction_runtime.KNOWN_EXTENSIONS
    assert "handgun" in mmaction_runtime.EVERY_STEP_EXTENSIONS
    assert mmaction_runtime.WEAPON_EXTENSIONS == behavior_registry.WEAPON_BEHAVIORS == ("sharp_object", "handgun")
    runtime = SimpleNamespace()
    display = mmaction_runtime.MMActionRuntime.sharp_display
    knife = [{"box": [0, 0, 10, 10], "conf": 0.9, "class": "sharp", "holder": 0}]
    gun = [{"box": [50, 50, 60, 60], "conf": 0.9, "class": "handgun", "holder": 0}]
    display(runtime, knife, [[0, 0, 20, 20]], "fighting", 1.0)
    display(runtime, gun, [[40, 40, 70, 70]], "fighting", 1.0, "handgun")
    held_knife, _ = display(runtime, [], [], "fighting", 1.5)
    held_gun, _ = display(runtime, [], [], "fighting", 1.5, "handgun")
    assert held_knife[0]["class"] == "sharp" and held_gun[0]["class"] == "handgun"
    assert "sharp_hold" in runtime.__dict__  # sharp_object keeps its original state key


def test_same_weights_file_is_loaded_once(tmp_path, monkeypatch):
    import sys
    import types

    loaded = []
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: loaded.append(path) or object()
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(sharp_module, "_MODEL_CACHE", {})
    (tmp_path / "sharp").mkdir()
    (tmp_path / "gun").mkdir()
    for folder in ("sharp", "gun"):
        (tmp_path / folder / "yolo11s-pose.pt").write_bytes(b"same pose weights")
    (tmp_path / "gun" / "other.pt").write_bytes(b"different weights")
    first, key = sharp_module.shared_yolo(tmp_path / "sharp" / "yolo11s-pose.pt")
    second, same_key = sharp_module.shared_yolo(tmp_path / "gun" / "yolo11s-pose.pt")
    third, other_key = sharp_module.shared_yolo(tmp_path / "gun" / "other.pt")
    assert first is second and key == same_key
    assert third is not first and other_key != key
    assert len(loaded) == 2
