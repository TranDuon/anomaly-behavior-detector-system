import json

import pytest

from scripts.fight_curated_pilot import reviewed_manifest


def fixture_manifest(tmp_path):
    source = tmp_path / "source"
    review = tmp_path / "review"
    source.mkdir()
    review.mkdir()
    rows, judgments = [], []
    for split in ("train", "validation"):
        for index in range(8):
            row = {
                "video": f"ubi/{split}",
                "scene_group": split,
                "start": index * 60,
                "label": index % 2,
                "split": split,
                "audit_id": f"{split}_{index}",
            }
            rows.append(row)
            judgments.append(
                {
                    "audit_id": row["audit_id"],
                    "judgment": "fight" if row["label"] == 0 else "nonfight",
                    "confidence": "high",
                }
            )
    (source / "cached_manifest.json").write_text(
        json.dumps({"official_test_used": False, "rows": rows})
    )
    (review / "private_mapping.json").write_text(json.dumps({"rows": rows}))
    path = tmp_path / "judgments.json"
    path.write_text(
        json.dumps(
            {
                "rater_type": "AI_assisted_not_independent_human_gold",
                "judgments": judgments,
            }
        )
    )
    return source, review, path


def test_curated_labels_are_versioned_without_modifying_source(tmp_path):
    source, review, judgments = fixture_manifest(tmp_path)
    before = (source / "cached_manifest.json").read_bytes()
    result = reviewed_manifest(source, review, judgments)
    assert (source / "cached_manifest.json").read_bytes() == before
    assert result["human_verified"] is False
    assert result["counts"]["train"]["changed_labels_in_new_manifest"] == 8
    assert all(row["label"] != row["original_dataset_label"] for row in result["rows"])


def test_curated_manifest_rejects_missing_or_duplicate_judgments(tmp_path):
    source, review, path = fixture_manifest(tmp_path)
    values = json.loads(path.read_text())
    values["judgments"].append(values["judgments"][0])
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="complete unique"):
        reviewed_manifest(source, review, path)
