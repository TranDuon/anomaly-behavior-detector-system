"""Explicit reviewed intervals; unlabeled time is never silently normal."""

import hashlib
import json
import math

from fight.vision import ClipConfig

ACTIONS = {
    "fight": 0,
    "hug": 1,
    "handshake_greet": 2,
    "highfive_clap": 3,
    "gesture": 4,
    "solo_motion": 5,
    "normal": None,
    "uncertain": None,
}
SPLITS = ("train", "development", "calibration", "test")


def source_group(row):
    return row.get("performance") or row.get("scene_group") or row["video"]


def source_digest(rows):
    records = [
        {
            key: row.get(key)
            for key in (
                "video",
                "dataset",
                "label",
                "sha256",
                "split",
                "frames",
                "fps",
                "performance",
                "scene_group",
            )
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            sorted(records, key=lambda row: row["video"]), sort_keys=True
        ).encode()
    ).hexdigest()


def validate_sources(rows):
    if len({row["video"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source video")
    for key in ("sha256", "performance", "scene_group"):
        groups = {}
        for row in rows:
            if row.get(key):
                groups.setdefault(row[key], set()).add(row["split"])
        if any(len(partitions) > 1 for partitions in groups.values()):
            raise ValueError("Source split leakage: " + key)
    for row in rows:
        if (
            row["split"] not in SPLITS
            or not math.isfinite(row["fps"])
            or row["fps"] <= 0
            or row["frames"] <= 0
        ):
            raise ValueError("Invalid source split/timebase")


def review_template(rows):
    validate_sources(rows)
    return {
        "schema_version": 1,
        "source_manifest_sha256": source_digest(rows),
        "policy": "fight = clear punch/kick/wrestling/push-down including safe simulation; hugging/greeting/high-five/solo gestures are not fight. Unknown time stays unlabeled.",
        "scope": "Pending review template, not temporal ground truth or approval",
        "sources": [
            {
                "video": row["video"],
                "sha256": row["sha256"],
                "split": row["split"],
                "review_status": "pending",
                "reviewer": "",
                "review_method": "",
                "intervals": [],
            }
            for row in rows
        ],
    }


def validate_annotations(document, rows):
    validate_sources(rows)
    if document.get("schema_version") != 1 or document.get(
        "source_manifest_sha256"
    ) != source_digest(rows):
        raise ValueError("Temporal annotations do not match the frozen source manifest")
    sources, seen, approved = {row["video"]: row for row in rows}, set(), {}
    for item in document["sources"]:
        video = item["video"]
        if video not in sources or video in seen:
            raise ValueError("Unknown or duplicate reviewed source")
        seen.add(video)
        row = sources[video]
        if item.get("sha256") != row["sha256"] or item.get("split") != row["split"]:
            raise ValueError("Reviewed source digest/split changed")
        if item.get("review_status") not in {"pending", "approved"}:
            raise ValueError("Invalid review status")
        is_approved = item["review_status"] == "approved"
        if is_approved and (
            not item.get("reviewer", "").strip()
            or item.get("review_method") not in {"human", "ai_assisted"}
        ):
            raise ValueError(
                "Approved intervals need an explicit reviewer and review method"
            )
        intervals = sorted(
            item["intervals"], key=lambda interval: interval["start_seconds"]
        )
        previous_end = 0.0
        for interval in intervals:
            start, end = interval["start_seconds"], interval["end_seconds"]
            if (
                interval["action"] not in ACTIONS
                or not all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in (start, end)
                )
                or not 0 <= start < end <= row["frames"] / row["fps"] + 1e-6
                or start < previous_end
            ):
                raise ValueError("Invalid/overlapping temporal intervals")
            previous_end = end
        if is_approved and not intervals:
            raise ValueError("Approved source has no reviewed intervals")
        if is_approved:
            approved[video] = intervals
    return approved


def reviewed_windows(document, rows, config=None):
    config = config or ClipConfig(
        fps=12, region_mode="pairs", trim_black_borders=True, version=8
    )
    approved = validate_annotations(document, rows)
    windows = []
    for row in rows:
        intervals = approved.get(row["video"], [])
        span = round((config.frames - 1) * row["fps"] / config.fps) + 1
        if row["frames"] < span:
            continue
        seen = set()
        for occurrence, interval in enumerate(intervals):
            action = interval["action"]
            if action == "uncertain":
                continue
            start_seconds, end_seconds = (
                interval["start_seconds"],
                interval["end_seconds"],
            )
            fractions = (0.5, 0.75, 1.0) if row["split"] == "train" else (1.0,)
            for fraction in fractions:
                endpoint = start_seconds + (end_seconds - start_seconds) * fraction
                start = min(
                    row["frames"] - span, max(0, round(endpoint * row["fps"]) - span)
                )
                window_start, window_end = (
                    start / row["fps"],
                    (start + span) / row["fps"],
                )
                overlap = max(
                    0.0, min(window_end, end_seconds) - max(window_start, start_seconds)
                )
                if action == "fight":
                    if overlap + 1e-6 < min(0.2, end_seconds - start_seconds):
                        continue
                    if any(
                        other["action"] == "uncertain"
                        and window_start < other["end_seconds"]
                        and window_end > other["start_seconds"]
                        for other in intervals
                    ):
                        continue
                else:
                    if (
                        window_start < start_seconds - 1e-6
                        or window_end > end_seconds + 1e-6
                    ):
                        continue
                    if any(
                        other["action"] in {"fight", "uncertain"}
                        and window_start < other["end_seconds"] + 0.25
                        and window_end > other["start_seconds"] - 0.25
                        for other in intervals
                    ):
                        continue
                if start in seen:
                    continue
                seen.add(start)
                targets, mask = [0] * 6, [1] * 6
                if ACTIONS[action] is not None:
                    targets[ACTIONS[action]] = 1
                windows.append(
                    {
                        **row,
                        "start": start,
                        "label": int(action == "fight"),
                        "actions": [action],
                        "action_targets": targets,
                        "action_mask": mask,
                        "annotation_occurrence": occurrence,
                        "annotation_scope": "reviewed_temporal_window",
                        "sample_weight": 1.0,
                    }
                )
    return windows


def readiness(windows, datasets=("airtlab", "rlvs"), minimum_groups=3):
    if minimum_groups < 3:
        raise ValueError(
            "At least three source performances per cohort/class are required for this pilot"
        )
    counts, missing = {}, []
    for split in SPLITS:
        for dataset in datasets:
            for label in (0, 1):
                groups = {
                    source_group(row)
                    for row in windows
                    if (row["split"], row["dataset"], row["label"])
                    == (split, dataset, label)
                }
                key = f"{split}/{dataset}/{label}"
                counts[key] = len(groups)
                if len(groups) < minimum_groups:
                    missing.append(key)
    return {
        "ready_for_temporal_pilot": not missing,
        "minimum_groups_per_class_per_cohort": minimum_groups,
        "independent_source_groups": counts,
        "missing": missing,
        "scope": "Small repeated-public pilot readiness only; not statistical acceptance or camera accuracy",
    }
