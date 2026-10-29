"""Build and audit the indoor-only interaction manifest used by MMAction fusion v2."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


ALLOWED_DATASETS = {"airtlab", "sbu", "ntu_rgbd120", "vid_classroom"}
SPLITS = {"train", "development", "calibration", "test"}
NEGATIVE_ACTIONS = {
    "normal",
    "gesture",
    "solo_motion",
    "approach",
    "depart",
    "stand_close",
    "walk_together",
    "hug",
    "handshake_greet",
    "highfive_clap",
    "pat_back",
    "point",
    "give_exchange",
    "follow",
    "whisper",
    "cheers",
}
ALLOWED_ACTIONS = NEGATIVE_ACTIONS | {"fight", "uncertain"}
MINIMUM_SOURCES = 200
PRIMARY_DATASET = "airtlab"
SECONDARY_DATASETS = {"sbu", "ntu_rgbd120", "vid_classroom"}
CLOSE_PERSON_NEGATIVES = {
    "approach",
    "stand_close",
    "walk_together",
    "hug",
    "handshake_greet",
    "highfive_clap",
}


def source_label(source):
    explicit = source.get("source_label")
    if explicit:
        return explicit
    video = source["video"].replace("\\", "/").lower()
    if "/non-violent/" in video or "/nonfight/" in video:
        return "non_fight"
    if "/violent/" in video or "/fight/" in video:
        return "fight"
    return "mixed_or_unknown"


def normalize_source(source):
    result = copy.deepcopy(source)
    if result["dataset"] == "airtlab":
        result.setdefault("environment", "indoor")
        result.setdefault("camera_context", "fixed_room")
    result["source_label"] = source_label(result)
    return result


def audit_source(source):
    reasons = []
    if source.get("dataset") not in ALLOWED_DATASETS:
        reasons.append("dataset_not_indoor_profile")
    if source.get("environment") != "indoor":
        reasons.append("environment_not_indoor")
    if source.get("split") not in SPLITS:
        reasons.append("invalid_split")
    actions = [item.get("action") for item in source.get("intervals", [])]
    if any(action not in ALLOWED_ACTIONS for action in actions):
        reasons.append("unsupported_or_weapon_action")
    if source.get("source_label") == "non_fight" and "fight" in actions:
        reasons.append("fight_inside_non_fight_source")
    return reasons


def build(corpus):
    accepted, quarantine = [], []
    videos = set()
    groups = {}
    for original in corpus.get("sources", []):
        source = normalize_source(original)
        reasons = audit_source(source)
        if source.get("video") in videos:
            reasons.append("duplicate_video_id")
        videos.add(source.get("video"))
        if reasons:
            quarantine.append({"video": source.get("video"), "reasons": sorted(set(reasons))})
            continue
        accepted.append(source)
        groups.setdefault(source["group"], set()).add(source["split"])
    leaking = {group for group, splits in groups.items() if len(splits) != 1}
    if leaking:
        kept = []
        for source in accepted:
            if source["group"] in leaking:
                quarantine.append(
                    {"video": source["video"], "reasons": ["performance_group_leakage"]}
                )
            else:
                kept.append(source)
        accepted = kept

    dataset_counts = Counter(source["dataset"] for source in accepted)
    split_counts = Counter(source["split"] for source in accepted)
    actions = Counter(
        interval["action"]
        for source in accepted
        for interval in source["intervals"]
    )
    readiness_reasons = []
    positive_sources = [
        source
        for source in accepted
        if any(item["action"] == "fight" for item in source["intervals"])
    ]
    contextual_positives = 0
    for source in positive_sources:
        first_fight = next(
            item for item in source["intervals"] if item["action"] == "fight"
        )
        if any(
            item["end_seconds"] <= first_fight["start_seconds"]
            and item["action"] in NEGATIVE_ACTIONS
            for item in source["intervals"]
        ):
            contextual_positives += 1
    if len(accepted) < MINIMUM_SOURCES:
        readiness_reasons.append(
            f"need_at_least_{MINIMUM_SOURCES}_sources_have_{len(accepted)}"
        )
    if PRIMARY_DATASET not in dataset_counts:
        readiness_reasons.append("missing_primary_dataset:airtlab")
    if not (SECONDARY_DATASETS & set(dataset_counts)):
        readiness_reasons.append(
            "missing_secondary_indoor_dataset:sbu_or_ntu_rgbd120_or_vid_classroom"
        )
    required_context = max(20, round(len(positive_sources) * 0.25))
    if contextual_positives < required_context:
        readiness_reasons.append(
            f"need_prefight_context_{required_context}_have_{contextual_positives}"
        )
    near_person_negatives = sum(actions[action] for action in CLOSE_PERSON_NEGATIVES)
    if near_person_negatives < 30:
        readiness_reasons.append(
            f"need_30_close_person_hard_negative_intervals_have_{near_person_negatives}"
        )
    for split in SPLITS:
        selected = [source for source in accepted if source["split"] == split]
        selected_actions = {
            interval["action"]
            for source in selected
            for interval in source["intervals"]
        }
        if "fight" not in selected_actions or not (selected_actions & NEGATIVE_ACTIONS):
            readiness_reasons.append(f"split_without_both_classes:{split}")

    manifest = {
        "schema_version": 2,
        "scope_profile": "indoor_room_fight_v2",
        "scope": (
            "Indoor room/classroom, unarmed person-to-person fight detection. "
            "CCTV, outdoor, weapons and local acceptance videos are excluded."
        ),
        "readiness": "ready" if not readiness_reasons else "seed_only",
        "readiness_reasons": readiness_reasons,
        "sources": accepted,
        "summary": {
            "sources": len(accepted),
            "datasets": dict(sorted(dataset_counts.items())),
            "splits": dict(sorted(split_counts.items())),
            "actions": dict(sorted(actions.items())),
            "positive_sources_with_prefight_context": contextual_positives,
            "quarantined": len(quarantine),
        },
    }
    return manifest, quarantine


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.input.read_text(encoding="utf-8"))
    manifest, quarantine = build(corpus)
    write(args.output, manifest)
    write(args.quarantine, {"items": quarantine})
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    print("READINESS", manifest["readiness"], *manifest["readiness_reasons"])


if __name__ == "__main__":
    main()
