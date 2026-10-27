import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fight.fusion import InteractionFusion
from fight.pose import PoseConfig, flip_pose, pair_pose_sequence
from fight.runtime import FightRuntime
from fight.vision import ClipConfig
from scripts.fight_fusion_train import (
    events_from_trace,
    progress_allowed,
    selection_key,
)


def pose_record(offset=0, confidence=0.9):
    return {
        "box": np.array([10 + offset, 10, 40 + offset, 100]),
        "points": np.tile(np.array([25 + offset, 40, confidence]), (17, 1)),
    }


def test_pose_normalization_ignores_camera_translation_and_preserves_pair_distance():
    detections = [[pose_record(), pose_record(50)]] * 8
    translated = [[pose_record(100), pose_record(150)]] * 8
    original = pair_pose_sequence(detections, [0, 0, 250, 120])
    moved = pair_pose_sequence(translated, [0, 0, 250, 120])
    np.testing.assert_allclose(original, moved)
    assert original.shape == (8, 2, 17, 3)
    assert original[0, 0, 0, 0] < original[0, 1, 0, 0]


def test_pose_missing_person_and_low_quality_are_masked_not_invented():
    sequence = pair_pose_sequence([[pose_record()]] * 8, [0, 0, 100, 120])
    assert sequence[:, 0, :, 2].min() > 0
    assert not sequence[:, 1].any()
    assert not pair_pose_sequence(
        [[pose_record(confidence=0.1)]], [0, 0, 100, 120]
    ).any()
    invalid = pose_record()
    invalid["points"][0, 0] = np.nan
    assert not pair_pose_sequence([[invalid]], [0, 0, 100, 120]).any()


def test_pose_tracks_do_not_swap_when_detection_order_changes():
    sequence = pair_pose_sequence(
        [[pose_record(), pose_record(50)], [pose_record(50), pose_record()]],
        [0, 0, 150, 120],
    )
    np.testing.assert_allclose(sequence[0], sequence[1])
    np.testing.assert_array_equal(flip_pose(flip_pose(sequence)), sequence)
    with pytest.raises(ValueError):
        PoseConfig(model_name="arbitrary.pt")


def test_sparse_pose_reuses_four_observations_after_half_second_window_shift():
    from fight.pose import PoseDetector, pose_inputs
    from fight.vision import build_clip_inputs

    calls = []

    class Model:
        def predict(self, frame, **options):
            calls.append(1)
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[10, 10, 40, 100]], dtype=torch.float32)
                    ),
                    keypoints=SimpleNamespace(
                        data=torch.from_numpy(np.array([pose_record()["points"]]))
                    ),
                )
            ]

    detector = PoseDetector.__new__(PoseDetector)
    detector.config, detector.device, detector.model, detector.cache = (
        PoseConfig(),
        "cpu",
        Model(),
        {},
    )
    frames = [np.zeros((120, 160, 3), np.uint8) for index in range(22)]
    for offset in (0, 6):
        window = frames[offset : offset + 16]
        detector.retain(window)
        clips, regions, people, bounds, counts = build_clip_inputs(
            window, detector, ClipConfig(fps=12, size=16, region_mode="pairs")
        )
        assert pose_inputs(window, regions, detector).shape[1:] == (8, 2, 17, 3)
    assert len(calls) == 12


def test_cpu_acceptance_installs_pose_weights_into_isolated_candidate(tmp_path):
    import zipfile
    from scripts.fight_cpu_acceptance import install_candidate

    bundle = tmp_path / "runtime.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in ("fight_model.ts", "fight_runtime.json", "yolov8n-pose.pt"):
            archive.writestr("models/" + name, name.encode())
    models = install_candidate(bundle, tmp_path / "qa")
    assert (models / "yolov8n-pose.pt").read_bytes() == b"yolov8n-pose.pt"


def test_hard_negative_report_does_not_count_paired_views_as_independent():
    from scripts.fight_fusion_report import grouped_negatives

    rows = [
        {
            "video": "cam1",
            "performance": "hug1",
            "label": 0,
            "actions": ["hug", "greet"],
            "score": 0.9,
        },
        {
            "video": "cam2",
            "performance": "hug1",
            "label": 0,
            "actions": ["hug", "greet"],
            "score": 0.1,
        },
        {"video": "fight", "label": 1, "actions": ["hug"], "score": 0.9},
    ]
    result = grouped_negatives(rows, 0.5)["hug"]
    assert result["clips"] == 2 and result["performances"] == 1
    assert (
        result["false_positive_clips"]
        == result["performances_with_false_positive"]
        == 1
    )


def test_residual_fusion_starts_at_rgb_baseline_and_has_finite_missing_pose():
    torch.set_num_threads(2)
    model = InteractionFusion().eval()
    features = torch.randn(2, 2048)
    poses = torch.zeros(2, 8, 2, 17, 3)
    logits, actions = model.classify_features(features, poses)
    torch.testing.assert_close(logits, model.rgb.head(features).flatten())
    assert actions.shape == (2, 6) and torch.isfinite(actions).all()
    model.set_stage("probe")
    assert not any(parameter.requires_grad for parameter in model.rgb.parameters())
    model.set_stage("finetune")
    assert any(parameter.requires_grad for parameter in model.rgb.backbone.parameters())


def test_real_fusion_exports_two_input_cpu_model_with_dynamic_region_count(tmp_path):
    from fight.model import prepare_export

    torch.set_num_threads(2)
    model = prepare_export(InteractionFusion().eval())
    clips = torch.zeros(1, 3, 16, 182, 182)
    poses = torch.zeros(1, 8, 2, 17, 3)
    with torch.inference_mode():
        traced = torch.jit.trace(model, (clips, poses), check_trace=False)
        traced.save(str(tmp_path / "fusion.ts"))
        restored = torch.jit.load(str(tmp_path / "fusion.ts"))
        torch.testing.assert_close(
            restored(clips.repeat(2, 1, 1, 1, 1), poses.repeat(2, 1, 1, 1, 1)),
            model(clips.repeat(2, 1, 1, 1, 1), poses.repeat(2, 1, 1, 1, 1)),
        )


def test_long_training_requires_progress_and_protects_hard_source():
    initial = {
        "worst_precision_recall": 0.7,
        "macro_ap": 0.8,
        "bce": 0.5,
        "cohorts": {"airtlab": {"ap": 0.8}, "rlvs": {"ap": 0.8}},
    }
    assert not progress_allowed(initial, initial)
    improved = {**initial, "worst_precision_recall": 0.75}
    assert progress_allowed(initial, improved)
    degraded = {**improved, "cohorts": {"airtlab": {"ap": 0.7}, "rlvs": {"ap": 0.95}}}
    assert not progress_allowed(initial, degraded)
    assert selection_key({**initial, "bce": 0.4}) > selection_key(initial)


def test_gate_trace_uses_only_observed_people_and_causal_scores():
    trace = {
        "duration": 3,
        "predictions": [
            {"timestamp": 0.5, "score": 0.9, "people": 2},
            {"timestamp": 1, "score": 0.9, "people": 2},
            {"timestamp": 1.5, "score": 0.9, "people": 0},
        ],
    }
    assert events_from_trace(trace, 0.8, 0.4) == [{"start": 1, "end": 1.5}]
    assert events_from_trace(trace, 0.8, 0.8) == []


class FusionPredictor(nn.Module):
    def forward(self, clips, poses):
        return clips.mean(dim=(1, 2, 3, 4)) * 0 + poses.mean(dim=(1, 2, 3, 4)) * 0 + 5


class MockPose:
    config = PoseConfig()
    cache = {}

    def retain(self, frames):
        pass

    def poses(self, frame):
        return [pose_record(), pose_record(50)]

    def detect(self, frame):
        return [
            SimpleNamespace(bbox_xyxy=record["box"]) for record in self.poses(frame)
        ]


def test_fusion_runtime_same_frames_same_scores_and_missing_bundle_blocked(tmp_path):
    config = ClipConfig(frames=4, fps=4, size=16)
    frames = [np.zeros((120, 160, 3), np.uint8) for index in range(20)]
    runtimes = [
        FightRuntime(
            predictor=FusionPredictor(),
            config=config,
            pose_detector=MockPose(),
            input_kind="rgb_pose",
        )
        for index in range(2)
    ]
    traces = [
        [runtime.push(frame, index / 8) for index, frame in enumerate(frames)]
        for runtime in runtimes
    ]
    assert [row["score"] for row in traces[0]] == [row["score"] for row in traces[1]]
    assert traces[0][-1]["state"] == "fighting"
    model = torch.jit.trace(
        FusionPredictor(), (torch.zeros(1, 3, 4, 16, 16), torch.zeros(1, 8, 2, 17, 3))
    )
    model.save(str(tmp_path / "fight_model.ts"))
    (tmp_path / "fight_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_kind": "rgb_pose",
                "pose": asdict(PoseConfig()),
                "clip": config.to_dict(),
                "threshold": 0.8,
            }
        )
    )
    runtime = FightRuntime(tmp_path, allow_pilot=True)
    assert not runtime.model_ready and "Pose weights missing" in runtime.message


def test_qa_selection_does_not_depend_on_model_scores():
    from scripts.fight_fusion_qa import select_cases

    rows = [
        {
            "video": "hug1/cam1",
            "performance": "hug1",
            "label": 0,
            "dataset": "airtlab",
            "actions": ["hug", "highfive"],
            "sha256": "a",
            "score": 0.9,
        },
        {
            "video": "hug1/cam2",
            "performance": "hug1",
            "label": 0,
            "dataset": "airtlab",
            "actions": ["hug", "highfive"],
            "sha256": "b",
            "score": 0.1,
        },
        {
            "video": "highfive2",
            "performance": "highfive2",
            "label": 0,
            "dataset": "airtlab",
            "actions": ["highfive"],
            "sha256": "c",
            "score": 0.1,
        },
        {
            "video": "fight",
            "label": 1,
            "dataset": "rlvs",
            "actions": [],
            "sha256": "d",
            "score": 0.1,
        },
    ]
    selected = [row["video"] for row in select_cases(rows)]
    for row in rows:
        row["score"] = 1 - row["score"]
    assert (
        selected
        == [row["video"] for row in select_cases(rows)]
        == ["hug1/cam1", "highfive2", "fight"]
    )


def test_qa_unpack_rejects_traversal_and_preserves_existing_data(tmp_path):
    import zipfile
    from scripts.fight_fusion_qa import unpack

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "qa")
    assert not (tmp_path / "escape.txt").exists()
    (tmp_path / "qa/keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="overwriting"):
        unpack(archive, tmp_path / "qa")
    assert (tmp_path / "qa/keep.txt").read_text() == "keep"
