import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fight.refinement import RefinedFightClassifier, load_deployed_base
from fight.temporal_labels import (
    readiness,
    review_template,
    reviewed_windows,
    validate_annotations,
)
from fight.vision import ClipConfig
from scripts.fight_refinement_train import window_key


def test_cache_key_includes_the_actual_detector_configuration():
    from fight.pose import PoseConfig

    row = {"sha256": "source", "start": 0, "fps": 30}
    normal = window_key(row, ClipConfig(), "base", "pose", PoseConfig())
    changed = window_key(row, ClipConfig(), "base", "pose", PoseConfig(image_size=640))
    assert normal != changed


def source(video="one", split="train", **values):
    return {
        "video": video,
        "sha256": video,
        "split": split,
        "fps": 30,
        "frames": 300,
        "dataset": "airtlab",
        "label": 0,
        "performance": video,
        "path": video,
        **values,
    }


def approved(rows, intervals):
    document = review_template(rows)
    for item in document["sources"]:
        item.update(
            review_status="approved",
            reviewer="reviewer1",
            review_method="human",
            intervals=intervals,
        )
    return document


def test_pending_review_never_becomes_training_ground_truth():
    rows = [source()]
    draft = review_template(rows)
    assert not reviewed_windows(draft, rows)
    assert not readiness([])["ready_for_temporal_pilot"]
    with pytest.raises(ValueError, match="three"):
        readiness([], minimum_groups=1)


def test_annotation_identity_review_and_split_guards():
    rows = [source()]
    document = approved(
        rows, [{"start_seconds": 1, "end_seconds": 2, "action": "fight"}]
    )
    document["sources"][0]["reviewer"] = ""
    with pytest.raises(ValueError, match="reviewer"):
        validate_annotations(document, rows)
    document["sources"][0]["reviewer"] = "reviewer1"
    document["sources"][0]["sha256"] = "other"
    with pytest.raises(ValueError, match="digest"):
        validate_annotations(document, rows)
    with pytest.raises(ValueError, match="leakage"):
        review_template([source(), source("other", split="test", performance="one")])


def test_temporal_windows_respect_reviewed_normal_and_uncertain_boundaries():
    rows = [source()]
    document = approved(
        rows,
        [
            {"start_seconds": 0, "end_seconds": 2, "action": "hug"},
            {"start_seconds": 2, "end_seconds": 2.5, "action": "uncertain"},
            {"start_seconds": 4, "end_seconds": 4.4, "action": "fight"},
        ],
    )
    windows = reviewed_windows(document, rows)
    assert {row["label"] for row in windows} == {0, 1}
    assert all(
        row["start"] / 30 + 38 / 30 <= 2 + 1e-6 for row in windows if row["label"] == 0
    )
    assert all(row["annotation_scope"] == "reviewed_temporal_window" for row in windows)
    assert not any(
        2.5 <= row["start"] / 30 <= 2.7 and row["label"] == 0 for row in windows
    )
    document["sources"][0]["intervals"].append(
        {"start_seconds": 1, "end_seconds": 3, "action": "normal"}
    )
    with pytest.raises(ValueError, match="overlapping"):
        validate_annotations(document, rows)


def test_half_open_end_of_source_and_short_negative_interval():
    rows = [source()]
    document = approved(
        rows,
        [
            {"start_seconds": 0, "end_seconds": 0.2, "action": "normal"},
            {"start_seconds": 9.8, "end_seconds": 10, "action": "fight"},
        ],
    )
    windows = reviewed_windows(document, rows)
    assert windows and all(
        row["label"] == 1 and row["start"] + 39 <= 300 for row in windows
    )


def test_cache_fingerprint_changes_with_classifier_and_window():
    config, row = ClipConfig(fps=12), {**source(), "start": 0}
    first = window_key(row, config, "current", "pose")
    assert first != window_key(row, config, "legacy", "pose")
    assert first != window_key({**row, "start": 1}, config, "current", "pose")


def test_review_selection_counts_camera_pairs_once_and_has_no_model_scores():
    from scripts.fight_temporal_review import select_sources

    rows = [
        source(f"performance{group}/cam{view}", performance=f"group{group}")
        for group in range(8)
        for view in (1, 2)
    ]
    selected = select_sources(rows, per_cohort=3)
    assert len(selected) == len({row["performance"] for row in selected}) == 3
    assert selected == select_sources(list(reversed(rows)), per_cohort=3)


def test_cpu_thread_selection_survives_detector_warmup_reset():
    from fight.runtime import FightRuntime

    class Detector:
        def warmup(self, size):
            torch.set_num_threads(8)

    class Predictor:
        def __call__(self, clips):
            assert torch.get_num_threads() == 2
            return torch.zeros(len(clips))

    runtime = FightRuntime(predictor=Predictor(), detector=Detector(), cpu_threads=2)
    runtime.warmup()
    assert runtime.cpu_threads == 2 and torch.get_num_threads() == 2
    for invalid in (0, 9, True, 2.0):
        with pytest.raises(ValueError, match="threads"):
            FightRuntime(cpu_threads=invalid)


def test_unlabeled_gaps_do_not_become_normal_or_hide_reviewed_false_alarms():
    from fight.reviewed_events import reviewed_event_metrics

    intervals = [
        {"start_seconds": 0, "end_seconds": 2, "action": "hug"},
        {"start_seconds": 5, "end_seconds": 5.3, "action": "fight"},
    ]
    result = reviewed_event_metrics(
        [{"start": 1, "end": 3}, {"start": 3, "end": 4}, {"start": 5.1, "end": 5.8}],
        intervals,
    )
    assert (
        result["true_positive"]
        == result["false_positive"]
        == result["ignored_unlabeled_events"]
        == 1
    )
    assert result["normal_minutes"] == pytest.approx(2 / 60)


def test_event_gate_can_recover_a_brief_punch_but_is_selected_on_development_only():
    from scripts.fight_refinement_train import select_event_gate

    traces = [
        {
            "video": "one",
            "dataset": "airtlab",
            "split": "development",
            "duration": 4,
            "predictions": [
                {"timestamp": 1, "score": 0.1, "people": 2},
                {"timestamp": 2, "score": 0.95, "people": 2},
                {"timestamp": 2.5, "score": 0.1, "people": 2},
                {"timestamp": 3, "score": 0.1, "people": 2},
                {"timestamp": 3.5, "score": 0.1, "people": 2},
            ],
        }
    ]
    approved = {
        "one": [
            {"start_seconds": 0, "end_seconds": 1.5, "action": "normal"},
            {"start_seconds": 2, "end_seconds": 2.2, "action": "fight"},
        ]
    }
    gate, candidates = select_event_gate(traces, approved, 0.8)
    assert gate["confirm_seconds"] == 0.0
    assert candidates[0]["cohorts"]["airtlab"]["recall"] == 1
    assert candidates[1]["cohorts"]["airtlab"]["recall"] == 0
    traces[0]["split"] = "test"
    with pytest.raises(ValueError, match="development-only"):
        select_event_gate(traces, approved, 0.8)


def test_full_scoring_does_not_use_stale_features_after_backbone_finetuning(tmp_path):
    from scripts.fight_refinement_train import score_rows

    np.savez(
        tmp_path / "window.npz",
        clips=np.zeros((1, 16, 8, 8, 3), np.uint8),
        features=np.ones((1, 2048), np.float32),
        poses=np.zeros((1, 8, 2, 17, 3), np.float32),
        motion=np.zeros((1, 2, 15, 32, 32), np.float32),
    )

    class Model(torch.nn.Module):
        def classify_features(self, features, poses, flow):
            return torch.tensor([2.0]), None

        def forward(self, clips, poses, flow):
            assert clips.shape == (1, 3, 16, 8, 8)
            return torch.tensor([-2.0])

    rows = [{"cache": "window"}]
    assert score_rows(Model(), rows, tmp_path, "cpu", features_only=True)[0] > 0.8
    assert score_rows(Model(), rows, tmp_path, "cpu")[0] < 0.2


def test_duplicate_alerts_and_long_normal_overrun_are_not_hidden():
    from fight.reviewed_events import reviewed_event_metrics, summarize_events

    intervals = [
        {"start_seconds": 1, "end_seconds": 2, "action": "fight"},
        {"start_seconds": 2, "end_seconds": 5, "action": "normal"},
    ]
    duplicate = reviewed_event_metrics(
        [{"start": 1.1, "end": 1.5}, {"start": 1.6, "end": 1.9}], intervals
    )
    assert (
        duplicate["true_positive"]
        == duplicate["duplicate_alerts"]
        == duplicate["false_positive"]
        == 1
    )
    assert duplicate["false_alerts_during_normal"] == 0
    overrun = summarize_events(
        [reviewed_event_metrics([{"start": 1.1, "end": 5}], intervals)]
    )
    assert overrun["normal_alert_fraction"] == pytest.approx(1.0)


def test_near_duplicate_sources_are_one_independent_scene_group():
    from fight.temporal_labels import source_group
    from scripts.fight_temporal_review import select_sources

    rows = [
        source(f"clip{index}", performance=None, scene_group="same", dataset="rlvs")
        for index in range(5)
    ]
    assert len(select_sources(rows, per_cohort=3)) == 1
    assert len({source_group(row) for row in rows}) == 1


def test_temporal_review_javascript_helpers():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    result = subprocess.run(
        [
            node,
            "--test",
            str(Path(__file__).with_name("temporal_review_state.test.cjs")),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_interrupted_review_preparation_resumes_without_resampling(
    tmp_path, monkeypatch
):
    from scripts import fight_temporal_review as review

    rows = [source("one"), source("two")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}))
    monkeypatch.setattr(review, "relocate", lambda selected, data: selected)
    calls = []
    interrupted = [False]

    def preview(row, destination):
        calls.append(row["video"])
        if row["video"] == "two" and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("Interrupted")
        destination.write_bytes(row["video"].encode())

    monkeypatch.setattr(review, "make_preview", preview)
    output = tmp_path / "pack"
    with pytest.raises(RuntimeError, match="Interrupted"):
        review.prepare(manifest, output, data=tmp_path)
    assert not (output / "review.json").exists()
    review.prepare(manifest, output, data=tmp_path)
    assert calls == ["one", "two", "two"]
    assert json.loads((output / "preparation.json").read_text())["status"] == "ready"
    assert all(
        item["review_status"] == "pending"
        for item in json.loads((output / "annotations.pending.json").read_text())[
            "sources"
        ]
    )
    with pytest.raises(ValueError, match="existing review"):
        review.prepare(manifest, output, data=tmp_path)


def test_refinement_preserves_complete_classifier_not_just_rgb_head():
    torch.set_num_threads(2)
    model = RefinedFightClassifier().eval()
    torch.nn.init.normal_(model.base.delta.weight)
    features = torch.randn(2, 2048)
    poses = torch.zeros(2, 8, 2, 17, 3)
    flow = torch.zeros(2, 2, 15, 32, 32)
    expected, unused = model.base.classify_features(features, poses)
    actual, actions = model.classify_features(features, poses, flow)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, model.base.rgb.head(features).flatten())
    assert actions.shape == (2, 6)
    model.train()
    assert model.training and not model.base.training
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    model.set_stage("finetune")
    assert any(
        parameter.requires_grad
        for parameter in model.base.rgb.backbone.blocks[-2:].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.base.rgb.head.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.base.delta.parameters()
    )
    with pytest.raises(ValueError):
        model.set_stage("unknown")


def test_review_preview_uses_browser_conversion_before_publishing(
    tmp_path, monkeypatch
):
    import cv2
    from scripts import fight_temporal_review as review

    source_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48)
    )
    assert writer.isOpened()
    for frame_index in range(12):
        writer.write(np.full((48, 64, 3), frame_index, dtype=np.uint8))
    writer.release()
    destination = tmp_path / "preview.mp4"
    converted = []

    def conversion(path):
        assert path.is_file() and not destination.exists()
        converted.append(path.name)

    monkeypatch.setattr(review, "browser_video", conversion)
    review.make_preview({"path": source_path, "frames": 12, "fps": 12}, destination)
    assert converted == ["preview.partial.mp4"]
    assert destination.is_file()
    assert not (tmp_path / converted[0]).exists()


@pytest.mark.parametrize("use_interaction", [False, True])
def test_refined_model_exports_dynamic_three_input_batch(tmp_path, use_interaction):
    from fight.model import prepare_export

    torch.set_num_threads(2)
    model = prepare_export(RefinedFightClassifier(use_interaction).eval())
    inputs = (
        torch.zeros(1, 3, 16, 182, 182),
        torch.zeros(1, 8, 2, 17, 3),
        torch.zeros(1, 2, 15, 32, 32),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, inputs, check_trace=False)
        traced.save(str(tmp_path / "model.ts"))
        restored = torch.jit.load(str(tmp_path / "model.ts"))
        doubled = tuple(value.repeat(2, *([1] * (value.ndim - 1))) for value in inputs)
        torch.testing.assert_close(restored(*doubled), model(*doubled))


def test_actual_deployed_classifier_is_preserved_before_training():
    models = Path("models")
    example = Path("logs/fusion-v1/qa/rgb_aux/e152ff7e4caa97cb58c80f31.npz")
    if not example.exists() or not (models / "fight_model.ts").exists():
        pytest.skip("Optional installed source and public QA example")
    from fight.vision import clip_tensor

    torch.set_num_threads(2)
    model, metadata, digest = load_deployed_base(models)
    script = torch.jit.load(str(models / "fight_model.ts")).eval()
    with np.load(example) as data:
        clips = clip_tensor(data["clips"])
        poses = torch.from_numpy(data["poses"])
    with torch.inference_mode():
        torch.testing.assert_close(
            model(clips, poses, torch.zeros(len(clips), 2, 15, 32, 32)),
            script(clips, poses),
            atol=1e-5,
            rtol=1e-5,
        )
    assert digest == "7cd20e23e1590689a1d08bf63a9aa967c831a71dd985fed1a09fae5bd32b7d09"
