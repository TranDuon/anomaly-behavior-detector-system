import hashlib
import json

import pytest

from fight.temporal_labels import review_template
from fight.via_bridge import ACTION_OPTIONS, import_via, via_project
from scripts import fight_via


def fixture():
    rows = [
        {
            "video": f"video{index}",
            "sha256": str(index),
            "frames": 150,
            "fps": 30,
            "split": "train",
            "performance": str(index),
            "dataset": "airtlab",
        }
        for index in (1, 2)
    ]
    pack = {
        "annotations": review_template(rows),
        "items": [
            {
                "video": row["video"],
                "id": f"R{index:03}",
                "media": f"media/{index:024x}.mp4",
                "duration": 5,
            }
            for index, row in enumerate(rows, start=1)
        ],
    }
    return rows, pack


def annotated_project(pack):
    project = via_project(pack)
    option = next(key for key, action in ACTION_OPTIONS.items() if action == "hug")
    project["metadata"]["1_segment"] = {
        "vid": "1",
        "z": [0, 2],
        "xy": [],
        "av": {"1": option},
        "flg": 0,
    }
    return project


def test_via_schema_uses_official_temporal_segments_and_no_default_fight():
    rows, pack = fixture()
    project = via_project(pack)
    assert project["attribute"]["1"]["anchor_id"] == "FILE1_Z2_XY0"
    assert project["attribute"]["1"]["default_option_id"] == ""
    assert not project["metadata"]
    assert project["file"]["1"]["loc"] == 2
    assert project["view"]["1"] == {"fid_list": ["1"]}


def test_import_never_approves_without_explicit_assertion_and_retains_other_sources():
    rows, pack = fixture()
    project = annotated_project(pack)
    pending = import_via(project, pack, rows, "qa", "ai_assisted")
    assert pending["sources"][0]["review_status"] == "pending"
    reviewed = import_via(project, pack, rows, "qa", "human", approve=True)
    assert reviewed["sources"][0]["review_status"] == "approved"
    assert reviewed["sources"][0]["intervals"][0]["action"] == "hug"
    assert reviewed["sources"][1]["review_status"] == "pending"
    assert pack["annotations"]["sources"][0]["intervals"] == []


def test_import_rejects_remapped_video_or_unlabeled_segments():
    rows, pack = fixture()
    project = annotated_project(pack)
    project["file"]["1"]["src"] = "media/other.mp4"
    with pytest.raises(ValueError, match="mapping"):
        import_via(project, pack, rows, "qa", "human", True)
    project = annotated_project(pack)
    project["metadata"]["1_segment"]["av"] = {}
    with pytest.raises(ValueError, match="explicit"):
        import_via(project, pack, rows, "qa", "human", True)
    project = annotated_project(pack)
    project["metadata"]["second"] = {
        "vid": "1",
        "z": [1, 3],
        "xy": [],
        "av": {"1": "1"},
    }
    with pytest.raises(ValueError, match="overlapping"):
        import_via(project, pack, rows, "qa", "human", True)
    with pytest.raises(ValueError, match="overlapping"):
        import_via(project, pack, rows, "qa", "ai_assisted", False)


def test_upstream_app_is_checksum_locked_and_retained_verbatim(tmp_path, monkeypatch):
    data = b"<!doctype html>upstream BSD notices retained"
    monkeypatch.setattr(fight_via, "VIA_SHA256", hashlib.sha256(data).hexdigest())
    (tmp_path / "via_video_annotator.html").write_bytes(data)
    destination = fight_via.install_app(tmp_path)
    assert destination.read_bytes() == data
    assert (
        json.loads((tmp_path / "via_provenance.json").read_text())["sha256"]
        == hashlib.sha256(data).hexdigest()
    )
    destination.write_bytes(b"unreviewed version")
    with pytest.raises(ValueError, match="upstream bytes changed"):
        fight_via.install_app(tmp_path)


def test_prepare_copies_launcher_without_replacing_existing_saved_project(tmp_path):
    rows, pack = fixture()
    (tmp_path / "review.json").write_text(json.dumps(pack))
    fight_via.prepare(tmp_path)
    original = (tmp_path / "via_project.json").read_bytes()
    assert b"via_video_annotator.html" in (tmp_path / "via_review.html").read_bytes()
    (tmp_path / "via_project.json").write_bytes(original + b" ")
    fight_via.prepare(tmp_path)
    assert (tmp_path / "via_project.json").read_bytes() == original + b" "
