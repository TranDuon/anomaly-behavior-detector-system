import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import zipfile

import pytest

from fight.assault_extension import (
    POSITIVE, SPLITS, audit, binary_corpus, build_manifest, pending_annotations,
)
from scripts.prepare_assault_extension import prepare


def source(video, split="train", dataset="airtlab", digest=None):
    return {
        "video": video, "dataset": dataset, "split": split,
        "group": video, "source_sha256": digest or hashlib.sha256(video.encode()).hexdigest(),
        "duration": 5.0, "media": "media/" + hashlib.sha256(video.encode()).hexdigest()[:24] + ".mp4",
        "preview_sha256": hashlib.sha256(b"preview").hexdigest(),
        "intervals": [{"start_seconds": 0, "end_seconds": 5, "action": "fight"}],
    }


def test_parent_splits_propagate_to_other_cameras_and_exact_duplicates():
    parent = source("airtlab/violent/cam1/1.mp4", "test")
    duplicate = source("airtlab/violent/cam1/1.mp4", "train")
    other_camera = source("airtlab/violent/cam2/1.mp4", "train")
    same_bytes = source("rlvs/copy", "development", "rlvs", digest=other_camera["source_sha256"])
    manifest = build_manifest({"sources": [parent]}, {"sources": [duplicate, other_camera, same_bytes]})
    assert {row["split"] for row in manifest["sources"]} == {"test"}
    annotations = pending_annotations(manifest)
    assert all(not row["intervals"] and row["review_status"] == "pending" for row in annotations["sources"])
    assert not audit(manifest, annotations)[0]["ready_for_training"]


def test_conflicting_parent_camera_splits_are_rejected():
    with pytest.raises(ValueError, match="leakage"):
        build_manifest({"sources": [source("airtlab/violent/cam1/1.mp4"), source("airtlab/violent/cam2/1.mp4", "test")]}, {"sources": []})


def test_private_sources_and_changed_original_hash_are_rejected():
    original = source("airtlab/violent/cam1/1.mp4")
    for changed in [dict(original, dataset="local_camera"), dict(original, source_sha256="f" * 64)]:
        with pytest.raises(ValueError):
            build_manifest({"sources": [original]}, {"sources": [changed]})


def complete_review():
    rows = [source(f"rlvs/{split}/{action}", split, "rlvs") for split in SPLITS for action in (*sorted(POSITIVE), "sit_lie")]
    manifest = build_manifest({"sources": []}, {"sources": rows})
    labels = pending_annotations(manifest)
    for entry in labels["sources"]:
        action = entry["video"].rsplit("/", 1)[-1]
        entry.update(review_status="reviewed", reviewer="test reviewer", intervals=[{
            "start_seconds": 1, "end_seconds": 3, "action": action,
            "evidence": "",
        }])
    return manifest, labels


def test_binary_export_keeps_subtypes_and_includes_general_falls_as_positive():
    manifest, labels = complete_review()
    corpus = binary_corpus(manifest, labels)
    assert len(corpus["sources"]) == 16
    for row in corpus["sources"]:
        interval = row["intervals"][0]
        assert interval["action"] == ("fight" if interval["behavior"] in POSITIVE else "normal")
        assert interval["start_seconds"] == 1  # No full-clip pseudo-labels.
    assert all(row["intervals"][0]["action"] == "fight" for row in corpus["sources"] if row["intervals"][0]["behavior"] == "fall")


def test_missing_wrestle_test_coverage_blocks_export():
    manifest, labels = complete_review()
    next(row for row in labels["sources"] if row["split"] == "test" and row["intervals"][0]["action"] == "wrestle")["review_status"] = "pending"
    assert "test:wrestle" in audit(manifest, labels)[0]["missing_coverage"]
    with pytest.raises(ValueError, match="coverage"):
        binary_corpus(manifest, labels)


@pytest.mark.parametrize("change", ["overlap", "duration", "reviewer", "split", "fingerprint"])
def test_invalid_reviews_cannot_pass_audit(change):
    manifest, labels = complete_review()
    entry = next(row for row in labels["sources"] if row["intervals"][0]["action"] == "fall")
    if change == "overlap":
        entry["intervals"].append(copy.deepcopy(entry["intervals"][0]))
    elif change == "duration":
        entry["intervals"][0]["end_seconds"] = 20
    elif change == "reviewer":
        entry["reviewer"] = ""
    elif change == "split":
        entry["split"] = "test" if entry["split"] != "test" else "train"
    else:
        labels["manifest_sha256"] = "changed"
    with pytest.raises(ValueError):
        audit(manifest, labels)


def test_prepare_hash_checks_preview_and_refuses_to_overwrite_reviews(tmp_path):
    row = source("airtlab/violent/cam1/1.mp4")
    parent, pool = tmp_path / "parent.json", tmp_path / "pool.json"
    parent.write_text(json.dumps({"sources": [row]}))
    pool.write_text(json.dumps({"sources": []}))
    archive = tmp_path / "media.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(row["media"], b"preview")
        package.writestr("../../unsafe", b"never extracted")
    output = tmp_path / "pack"
    prepare(parent, pool, [archive], output)
    assert (output / row["media"]).read_bytes() == b"preview"
    assert json.loads((output / "readiness.json").read_text())["reviewed_sources"] == 0
    assert not (tmp_path / "unsafe").exists()
    with pytest.raises(ValueError, match="overwritten"):
        prepare(parent, pool, [archive], output)


def test_browser_review_validation():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    path = Path(__file__).with_name("assault_review.test.cjs")
    result = subprocess.run([node, "--test", str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
