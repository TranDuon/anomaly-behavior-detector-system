from collections import Counter

from scripts.fight_review import select_curated_rows, select_review_rows


def test_blind_review_sampling_has_random_sources_and_separate_error_cohort():
    rows = []
    predictions = []
    for split in ("train", "validation"):
        for label in (0, 1):
            for index in range(24):
                row = {
                    "video": f"ubi/{split}_{label}_{index}",
                    "start": 0,
                    "fps": 30,
                    "label": label,
                    "split": split,
                }
                rows.append(row)
                if split == "validation":
                    predictions.append(
                        {
                            "video": row["video"],
                            "start_seconds": 0,
                            "score": float(1 - label),
                        }
                    )
    selected = select_review_rows(rows, predictions)
    assert len(selected) == 60
    assert len({row["audit_id"] for row in selected}) == 60
    random_rows = [
        row for row in selected if row["audit_sampling"] == "stratified_random"
    ]
    assert len({row["video"] for row in random_rows}) == 48
    assert Counter((row["split"], row["label"]) for row in random_rows) == {
        ("train", 0): 12,
        ("train", 1): 12,
        ("validation", 0): 12,
        ("validation", 1): 12,
    }
    assert all(
        row["split"] == "validation"
        for row in selected
        if row["audit_sampling"] == "error_enriched"
    )
    assert selected == select_review_rows(rows, predictions)


def test_curated_review_never_changes_source_partition_or_overflows_quota():
    rows = [
        {
            "video": "ubi/one",
            "split": "train",
            "label": label,
            "start": index * 60 + label * 30,
        }
        for label in (0, 1)
        for index in range(8)
    ]
    selected = select_curated_rows(
        rows, [{"video": "ubi/one", "split": "train", "start": 30}]
    )
    assert len(selected) == 8
    assert all(row["split"] == "train" for row in selected)
    assert Counter(row["label"] for row in selected) == {0: 4, 1: 4}
