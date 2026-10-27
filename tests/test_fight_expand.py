from types import SimpleNamespace
import io
import zipfile

import numpy as np
import pytest

from fight.vision import (
    ClipConfig,
    augment_handheld,
    build_clip_inputs,
    content_bounds,
    person_tubes,
)
from scripts.fight_ablation import threshold_report
from scripts.fight_expand import archive_selection, grouped_partition, sample_windows
from anomaly.fight_data import VideoItem
from scripts.fight_pilot import metrics
from scripts.fight_controls import temporal_variants


def test_black_border_crop_preserves_content_and_black_frames():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[20:80, 30:130] = 150
    assert content_bounds([frame] * 16) == (30, 20, 130, 80)
    assert content_bounds([np.zeros_like(frame)] * 16) == (0, 0, 160, 100)
    frame[10:11, 10:150] = 200
    assert content_bounds([frame] * 16)[1] == 10


def test_tubes_do_not_count_same_person_three_times_and_keep_motion():
    assert person_tubes(
        [[[10, 10, 30, 70]], [[15, 10, 35, 70]], [[20, 10, 40, 70]]]
    ) == [[10.0, 10.0, 40.0, 70.0]]
    tubes = person_tubes([[[10, 10, 30, 70], [100, 10, 120, 70]]] * 3)
    assert len(tubes) == 2


def test_training_and_runtime_inputs_have_original_image_coordinates():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[20:80, 30:130] = 150

    class Detector:
        def detect(self, image):
            assert image.shape[:2] == (60, 100)
            return [SimpleNamespace(bbox_xyxy=[10, 5, 40, 55])]

    config = ClipConfig(region_mode="tubes", trim_black_borders=True, size=32)
    clips, regions, people, bounds, counts = build_clip_inputs(
        [frame] * 16, Detector(), config
    )
    assert clips.shape == (1, 16, 32, 32, 3)
    assert people == 1 and counts == [1, 1, 1]
    assert bounds == (30, 20, 130, 80)
    assert regions[0][0] >= 30 and regions[0][1] >= 20


def test_handheld_augmentation_keeps_same_camera_transform_across_regions():
    clips = np.tile(
        np.arange(32, dtype=np.uint8)[None, None, None, :, None], (2, 16, 32, 1, 3)
    )
    transformed = augment_handheld(clips, np.random.default_rng(42))
    assert transformed.dtype == np.uint8 and transformed.shape == clips.shape
    np.testing.assert_array_equal(transformed[0], transformed[1])


def test_temporal_controls_remove_or_shuffle_time_without_changing_regions():
    clips = np.broadcast_to(
        np.arange(16, dtype=np.uint8)[None, :, None, None, None], (2, 16, 4, 4, 3)
    ).copy()
    original = clips.copy()
    variants = temporal_variants(clips, np.random.default_rng(42))
    assert np.all(variants["static_middle"] == 8)
    assert variants["shuffled"].shape == clips.shape
    assert not np.array_equal(variants["shuffled"], clips)
    np.testing.assert_array_equal(np.sort(variants["shuffled"], axis=1), clips)
    np.testing.assert_array_equal(variants["shuffled"][0], variants["shuffled"][1])
    np.testing.assert_array_equal(clips, original)


def test_flagged_original_labels_block_new_training_but_preserve_completed_report(
    tmp_path,
):
    import json
    from scripts.fight_expand import expanded_config, train

    source = tmp_path / "artifacts/source"
    source.mkdir(parents=True)
    (source.parent / "ubi_label_review_gate.json").write_text(
        json.dumps({"status": "requires_review"})
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"clip": expanded_config().to_dict(), "rows": []})
    )
    args = SimpleNamespace(source_run=str(source))
    with pytest.raises(ValueError, match="flagged"):
        train(args, output)
    (output / "report.json").write_text(json.dumps({"decision": "old_report"}))
    train(args, output)
    assert json.loads((output / "report.json").read_text()) == {
        "decision": "old_report"
    }


def test_fast_threshold_search_matches_brute_force():
    rows = [{"video": str(index // 4), "label": index % 2} for index in range(32)]
    scores = np.random.default_rng(42).choice(
        [0, 0.1, 0.3, 0.5, 0.8, 1.0], size=len(rows)
    )
    values = [
        (metrics(rows, scores, float(threshold)), float(threshold))
        for threshold in np.unique(np.r_[0.0, scores, 1.0])
    ]
    feasible = [entry for entry in values if entry[0]["recall"] >= 0.7]
    expected = max(
        feasible or values,
        key=lambda entry: (entry[0]["f1"], entry[0]["precision"], entry[1]),
    )
    actual = threshold_report(rows, scores)
    assert actual["threshold"] == expected[1]
    assert actual["validation"] == expected[0]


def test_expansion_never_moves_old_validation_to_train_or_old_train_to_validation():
    items = [
        SimpleNamespace(uid=f"ubi/{index}", has_positive=index % 2 == 0)
        for index in range(10)
    ]
    original = {
        "rows": [
            {"video": "ubi/0", "scene_group": "oldval", "split": "validation"},
            {"video": "ubi/1", "scene_group": "oldtrain", "split": "train"},
        ]
    }
    fingerprints = {
        item.uid: f"{np.random.default_rng(index).integers(0, 2**63):016x}"
        for index, item in enumerate(items)
    }
    partition, quarantine = grouped_partition(
        items, original, fingerprints, validation_per_class=2
    )
    assert partition["ubi/0"][0] == "validation"
    assert partition["ubi/1"][0] == "train"
    assert not quarantine
    fingerprints["ubi/1"] = fingerprints["ubi/0"]
    partition, quarantine = grouped_partition(items, original, fingerprints)
    assert {"ubi/0", "ubi/1"} <= set(quarantine)
    assert partition["ubi/0"][0] == "quarantine"


def test_archive_selection_excludes_official_test_and_dedup_videos():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("UBI_FIGHTS/test_videos.csv", "F_test\n")
        for key in ("F_anchor", "F_new", "F_test", "F_duplicate", "N_one", "N_two"):
            archive.writestr(f"UBI_FIGHTS/videos/fight/{key}.mp4", b"video")
            archive.writestr(
                f"UBI_FIGHTS/annotation/{key}.csv",
                "1\n" if key.startswith("F_") else "0\n",
            )
    original = {"rows": [{"key": "F_anchor"}]}
    with zipfile.ZipFile(memory) as archive:
        selected, *_ = archive_selection(archive, original, 4, {"ubi/F_duplicate"})
        assert set(selected) == {"F_anchor", "F_new", "N_one", "N_two"}
        with pytest.raises(ValueError, match="Historical"):
            archive_selection(archive, {"rows": [{"key": "F_test"}]}, 4)


def test_sampled_windows_obey_labels_and_do_not_overlap(tmp_path, monkeypatch):
    import scripts.fight_expand as expansion

    labels = np.r_[np.zeros(200), np.ones(200), np.zeros(200)].astype(np.int32)
    item = VideoItem("ubi", "F_example", tmp_path / "video.mp4", "train", labels)

    class Capture:
        def get(self, property_id):
            return 10 if property_id == expansion.cv2.CAP_PROP_FPS else len(labels)

        def release(self):
            pass

    monkeypatch.setattr(expansion.cv2, "VideoCapture", lambda path: Capture())
    config = ClipConfig()
    rows = sample_windows(item, "train", "scene", config, 24)
    span = round((config.frames - 1) / config.fps * 10) + 1
    assert {row["label"] for row in rows} == {0, 1}
    for row in rows:
        fraction = labels[row["start"] : row["start"] + span].mean()
        assert fraction >= 0.7 if row["label"] else fraction == 0
    starts = sorted(row["start"] for row in rows)
    assert all(second - first >= span for first, second in zip(starts, starts[1:]))
