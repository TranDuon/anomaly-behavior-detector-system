import random
import time
import zipfile

import numpy as np
import pytest
import torch

from fight.interaction_data import action_catalog, parse_intervals, temporal_windows
from fight.interaction_eval import balanced_calibration, event_metrics, macro_ap
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import cached_rows
from scripts.fight_interaction_pilot import balanced_epoch, public_subset


def test_catalog_reads_numeric_action_ids_and_rejects_wrong_version(tmp_path):
    path = tmp_path / "Actions.xlsx"
    namespace = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f"<sst {namespace}><si><t>kicking other person</t></si><si><t>punching/slapping other person</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet {namespace}><sheetData><row><c r="A2"><v>18</v></c><c r="B2" t="s"><v>0</v></c></row><row><c r="A3"><v>26</v></c><c r="B3" t="s"><v>1</v></c></row></sheetData></worksheet>',
        )
    assert action_catalog(path)[18] == "kicking other person"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f"<sst {namespace}><si><t>kicking something</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet {namespace}><sheetData><row><c r="A2"><v>18</v></c><c r="B2" t="s"><v>0</v></c></row></sheetData></worksheet>',
        )
    with pytest.raises(ValueError, match="Unexpected"):
        action_catalog(path)


def test_label_policy_and_frame_conversion():
    catalog = {18: "kick person", 19: "kick object", 26: "punch", 27: "push", 16: "hug"}
    rows = parse_intervals("18,1,10,2\n19,12,20,2\n27,25,30,2\n26,35,40,0", 50, catalog)
    assert [row["label"] for row in rows] == [1, 0, None, None]
    assert rows[0]["start"] == 0 and rows[0]["end"] == 9
    with pytest.raises(ValueError):
        parse_intervals("18,1,100,2", 50, catalog)
    with pytest.raises(ValueError):
        parse_intervals("18,0,10,2", 50, catalog)


def test_temporal_windows_do_not_label_ignored_push_as_normal():
    reference = {"fps": 10, "frames": 100, "video": "pku/sample"}
    intervals = [
        {
            "start": 10,
            "end": 20,
            "label": 0,
            "action_id": 16,
            "action": "hug",
            "confidence": 2,
        },
        {
            "start": 25,
            "end": 35,
            "label": None,
            "action_id": 27,
            "action": "push",
            "confidence": 2,
        },
        {
            "start": 40,
            "end": 45,
            "label": 0,
            "action_id": 14,
            "action": "handshake",
            "confidence": 2,
        },
        {
            "start": 65,
            "end": 72,
            "label": 1,
            "action_id": 18,
            "action": "kick",
            "confidence": 2,
        },
    ]
    config = ClipConfig(frames=4, fps=4)
    train = temporal_windows(reference, intervals, config, training=True)
    diagnostic = temporal_windows(reference, intervals, config)
    assert all(row["action_id"] != 27 for row in train)
    assert all(row["action_id"] != 14 for row in train)
    assert any(row["label"] == 1 for row in train)
    assert len({row["start"] for row in train}) == len(train)
    assert len(diagnostic) < len(train)


def test_calibration_exposes_small_cohort_failure_hidden_by_pooled_scores():
    rows = [
        {"dataset": "rlvs", "video": str(index), "label": index % 2}
        for index in range(100)
    ]
    scores = [0.1 if row["label"] == 0 else 0.3 for row in rows]
    rows.extend(
        [
            {"dataset": "airtlab", "video": "hard-normal", "label": 0},
            {"dataset": "airtlab", "video": "fight", "label": 1},
        ]
    )
    scores.extend([0.4, 0.9])
    result = balanced_calibration(rows, scores)
    assert not result["all_calibration_cohorts_meet_target"]
    assert set(result["cohorts"]) == {"airtlab", "rlvs"}
    assert macro_ap(rows, scores) == 1
    with pytest.raises(ValueError):
        balanced_calibration(rows, scores[:-1])


def test_balanced_sampling_is_exposure_not_independent_examples():
    rows = [
        {"dataset": source, "label": label, "video": str(index)}
        for source in ("pku", "airtlab", "rlvs")
        for label in (0, 1)
        for index in range(1 if source == "pku" else 20)
    ]
    selected = balanced_epoch(rows, random.Random(1))
    assert len(selected) == 144
    assert all(
        sum(row["dataset"] == source and row["label"] == label for row in selected)
        == 24
        for source in ("pku", "airtlab", "rlvs")
        for label in (0, 1)
    )
    manifest = {
        "rows": [
            {**row, "split": split, "sha256": row["video"]}
            for row in rows
            if row["dataset"] != "pku"
            for split in ("train", "development", "calibration", "test")
        ]
    }
    assert all(row["split"] != "test" for row in public_subset(manifest))


def test_cache_fingerprints_actual_finetuned_backbone(tmp_path, monkeypatch):
    import scripts.fight_ablation as ablation

    calls = []
    model = type(
        "Features",
        (),
        {
            "features": lambda self, values: calls.append(1)
            or torch.zeros(len(values), 2)
        },
    )()
    monkeypatch.setattr(ablation, "read_frames", lambda *args: [])
    monkeypatch.setattr(
        ablation,
        "build_clip_inputs",
        lambda *args: (
            np.zeros((1, 4, 16, 16, 3), np.uint8),
            [[0, 0, 16, 16]],
            2,
            [0, 0, 16, 16],
            [2, 2, 2],
        ),
    )
    row = {"video": "pku/a", "start": 0, "label": 1, "fps": 4}
    config = ClipConfig(frames=4, fps=4, size=16)
    keys = [
        cached_rows(
            [row],
            model,
            None,
            config,
            DetectorConfig(),
            tmp_path,
            "cpu",
            time.monotonic() + 10,
            backbone_id=name,
        )[0]["cache"]
        for name in ("weights1", "weights1", "weights2")
    ]
    assert keys[0] == keys[1] and keys[0] != keys[2]
    assert len(calls) == 2


def test_event_matching_is_one_to_one_and_rejects_always_on_alarm():
    intervals = [
        {"start": 100, "end": 119, "label": 1},
        {"start": 200, "end": 219, "label": None},
    ]
    result = event_metrics(
        [
            {"start": 10.5, "end": 12.5},
            {"start": 10.7, "end": 12.4},
            {"start": 20.2, "end": 21.0},
        ],
        intervals,
        10,
        30,
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["ignored_events"] == 1
    always = event_metrics([{"start": 0, "end": 30}], [intervals[0]], 10, 30)
    assert always["true_positive"] == 0 and always["false_positive"] == 1


def test_saved_report_aggregates_counts_not_per_video_percentages(tmp_path):
    import json
    from scripts.fight_interaction_report import (
        aggregate_events,
        render_timelines,
        summarize,
    )

    metrics = {
        "ground_truth_events": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "ignored_events": 0,
        "normal_minutes": 0.5,
    }
    records = [
        {
            "video": "pku/a",
            "metrics": metrics,
            "predictions": [{"timestamp": 1, "score": 0.7}],
            "events": [{"start": 1, "end": 2}],
        },
        {
            "video": "pku/b",
            "metrics": {
                **metrics,
                "true_positive": 2,
                "false_negative": 0,
                "false_positive": 0,
                "normal_minutes": 1.5,
            },
            "predictions": [],
            "events": [],
        },
    ]
    total = aggregate_events(records)
    assert total["precision"] == total["recall"] == 0.75
    assert total["false_alerts_per_normal_minute"] == 0.5
    condition = {
        "threshold": 0.5,
        "cohorts": {},
        "all_calibration_cohorts_meet_target": False,
        "pku_diagnostic": {},
        "continuous": records,
    }
    comparison = {
        "status": "completed",
        "scope": "small diagnostic",
        "variants": {"sample": condition},
    }
    assert not summarize(comparison)["approved_for_demo"]
    destination = tmp_path / "sample"
    destination.mkdir()
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "video": "pku/a",
                        "fps": 10,
                        "intervals": [{"start": 10, "end": 20, "label": 1}],
                    }
                ]
            }
        )
    )
    paths = render_timelines(comparison, tmp_path)
    assert len(paths) == 1 and paths[0].stat().st_size > 100
