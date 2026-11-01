import shutil

import pytest

from scripts.prepare_gmdcsa24_v2_1 import ANNOTATIONS, FALL_TIME, build
from scripts.prepare_general_fall_step2 import build as merge
from scripts.prepare_general_fall_step2 import read
from scripts.prepare_gmdcsa24_v2_1 import ROOT


def test_publisher_manifest_keeps_subject_splits_and_unknown_timing():
    data = build()
    assert len(data["sources"]) == 160
    assert data["counts"] == {
        "publisher_temporal": 78,
        "publisher_clip_negative": 81,
        "needs_temporal_review": 1,
    }
    assert {row["split"] for row in data["sources"] if row["subject"] in (1, 2)} == {"train"}
    assert {row["split"] for row in data["sources"] if row["subject"] == 3} == {"development"}
    assert {row["split"] for row in data["sources"] if row["subject"] == 4} == {"test"}
    assert [row["video"] for row in data["sources"] if row["annotation_status"] == "needs_temporal_review"] == ["Subject 4/Fall/15.mp4"]
    assert all(row["intervals"][0]["label"] == 0 for row in data["sources"] if row["source_category"] == "ADL")
    assert all(row["intervals"][0]["label"] == 1 for row in data["sources"] if row["annotation_status"] == "publisher_temporal")
    assert not data["video_files_verified"]


def test_fall_time_does_not_borrow_sitting_timestamp():
    assert FALL_TIME.search("Falling (FW); Sitting[0 to 1.7]") is None
    assert FALL_TIME.search("Falling (BW[5.9 to 10]; Walking[0 to 1.3]").groups() == ("5.9", "10")


def test_changed_publisher_csv_is_rejected(tmp_path):
    shutil.copytree(ANNOTATIONS, tmp_path / "annotations")
    path = tmp_path / "annotations/Subject 1/Fall.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Publisher CSV changed"):
        build(tmp_path / "annotations")


def test_merge_preserves_parent_splits_and_binary_labels():
    parent = read(ROOT / "benchmarks/mmaction_interaction_public_v1.json")
    pool = read(ROOT / "benchmarks/mmaction_interaction_indoor_v2.json")
    data = merge(parent, pool, build())
    assert len(data["sources"]) == 704
    assert data["counts"]["parent_v1_existing"] == 64
    assert data["counts"]["needs_temporal_review"] == 481
    assert not data["ready_for_training"]
    original = {row["video"]: row for row in parent["sources"]}
    for row in data["sources"]:
        if row["video"] in original:
            assert row["split"] == original[row["video"]]["split"]
            assert all(item["label"] == (1 if item["action"] == "fight" else
                                         None if item["action"] == "uncertain" else 0)
                       for item in row["intervals"])
