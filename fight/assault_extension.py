"""Public-only temporal review for the fight, wrestle, or general fall extension."""

from collections import defaultdict
import copy
import hashlib
import json
import math
import re

PROFILE = "assault_extension_v1"
PARENT_SHA = "8c5808db106aacd0f43b72a0e76178b2dec039e0a1766d996bb8374b7e4dddb7"
SPLITS = ("train", "development", "calibration", "test")
PUBLIC_DATASETS = {"airtlab", "rlvs", "sbu", "cctv_fights"}
POSITIVE = {"fight", "wrestle", "fall"}
NEGATIVE = {"normal", "hug", "greet", "sit_lie", "sport_play"}
ACTIONS = POSITIVE | NEGATIVE | {"uncertain"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def performance(source):
    if source["dataset"] == "airtlab":
        return re.sub(r"/cam[12]/", "/", source["video"])
    return source.get("group") or source["video"]


def _source(source):
    if source["dataset"] not in PUBLIC_DATASETS:
        raise ValueError("Only allowlisted public datasets are permitted")
    if source["split"] not in SPLITS:
        raise ValueError("Invalid source split")
    if not re.fullmatch(r"[a-f0-9]{64}", source["source_sha256"]):
        raise ValueError("Source SHA-256 is required")
    if type(source["duration"]) not in (float, int) or not math.isfinite(source["duration"]) or source["duration"] <= 0:
        raise ValueError("Invalid source duration")
    return {
        key: copy.deepcopy(source[key])
        for key in ("video", "dataset", "split", "source_sha256", "duration", "media", "preview_sha256")
    } | {"group": performance(source)}


def build_manifest(parent, pool):
    """Freeze v1 splits and propagate them across duplicate hashes/performance groups."""
    sources, parent_splits = {}, {}
    for original in parent["sources"]:
        row = _source(original)
        if row["video"] in sources:
            raise ValueError("Duplicate parent video")
        sources[row["video"]] = row | {"from_parent_v1": True}
        parent_splits[row["video"]] = row["split"]
    for original in pool["sources"]:
        row = _source(original)
        previous = sources.get(row["video"])
        if previous:
            if previous["source_sha256"] != row["source_sha256"]:
                raise ValueError("Conflicting source hash: " + row["video"])
            continue
        sources[row["video"]] = row | {"from_parent_v1": False}

    # Union exact duplicates and synchronized camera performances before assigning splits.
    roots = {video: video for video in sources}

    def root(video):
        while roots[video] != video:
            roots[video] = roots[roots[video]]
            video = roots[video]
        return video

    for key in ("source_sha256", "group"):
        seen = {}
        for video, row in sources.items():
            if row[key] in seen:
                roots[root(video)] = root(seen[row[key]])
            else:
                seen[row[key]] = video
    components = defaultdict(list)
    for video in sources:
        components[root(video)].append(video)
    for videos in components.values():
        fixed = {parent_splits[video] for video in videos if video in parent_splits}
        if len(fixed) > 1:
            raise ValueError("Parent v1 already contains source/performance leakage")
        split = next(iter(fixed)) if fixed else sources[sorted(videos)[0]]["split"]
        for video in videos:
            sources[video]["split"] = split
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "parent_checkpoint_sha256": PARENT_SHA,
        "parent_manifest_sha256": fingerprint(parent),
        "selection": "general_fall-public-temporal",
        "sources": sorted(sources.values(), key=lambda row: row["video"]),
    }
    manifest["manifest_sha256"] = fingerprint(manifest)
    return manifest


def check_manifest(manifest):
    value = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if manifest.get("profile") != PROFILE or manifest.get("parent_checkpoint_sha256") != PARENT_SHA or fingerprint(value) != manifest.get("manifest_sha256"):
        raise ValueError("Manifest fingerprint/profile/parent changed")
    seen, groups = set(), {}
    for source in manifest["sources"]:
        _source(source)
        if source["video"] in seen:
            raise ValueError("Duplicate source")
        seen.add(source["video"])
        for key in ("group", "source_sha256"):
            identity = (key, source[key])
            previous = groups.setdefault(identity, source["split"])
            if previous != source["split"]:
                raise ValueError("Split leakage")


def pending_annotations(manifest):
    check_manifest(manifest)
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "manifest_sha256": manifest["manifest_sha256"],
        "sources": [
            {"video": row["video"], "source_sha256": row["source_sha256"],
             "split": row["split"], "review_status": "pending", "reviewer": "",
             "intervals": []}
            for row in manifest["sources"]
        ],
    }


def audit(manifest, annotations):
    check_manifest(manifest)
    if annotations.get("profile") != PROFILE or annotations.get("schema_version") != 1 or annotations.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Labels do not match the frozen manifest")
    originals = {row["video"]: row for row in manifest["sources"]}
    seen, reviewed, coverage = set(), [], defaultdict(set)
    for item in annotations["sources"]:
        video = item["video"]
        if video not in originals or video in seen:
            raise ValueError("Unknown/duplicate annotation source")
        seen.add(video)
        row = originals[video]
        if item.get("source_sha256") != row["source_sha256"] or item.get("split") != row["split"]:
            raise ValueError("Annotation source identity/split changed")
        if item.get("review_status") not in {"pending", "reviewed"}:
            raise ValueError("Invalid review status")
        previous = 0.0
        for interval in item["intervals"]:
            start, end = interval["start_seconds"], interval["end_seconds"]
            action = interval["action"]
            if action not in ACTIONS or any(type(t) not in (float, int) or not math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= row["duration"] + 1e-6 or start < previous:
                raise ValueError("Invalid/overlapping interval")
            previous = end
        if item["review_status"] == "reviewed":
            if not item.get("reviewer", "").strip() or not item["intervals"]:
                raise ValueError("Reviewed source requires reviewer and intervals")
            reviewed.append((row, item))
            for interval in item["intervals"]:
                category = interval["action"] if interval["action"] in POSITIVE else "negative" if interval["action"] in NEGATIVE else None
                if category:
                    coverage[(row["split"], category)].add(row["group"])
    missing = [f"{split}:{action}" for split in SPLITS for action in (*sorted(POSITIVE), "negative") if not coverage[(split, action)]]
    report = {
        "sources": len(originals), "reviewed_sources": len(reviewed),
        "pending_sources": len(originals) - len(reviewed),
        "coverage_groups": {split: {action: len(coverage[(split, action)]) for action in (*sorted(POSITIVE), "negative")} for split in SPLITS},
        "missing_coverage": missing,
        "ready_for_training": not missing,
        "scope": "Structural coverage only; one group per category/split is not sufficient evidence of model quality",
        "training_started": False,
    }
    return report, reviewed


def binary_corpus(manifest, annotations):
    report, reviewed = audit(manifest, annotations)
    if not report["ready_for_training"]:
        raise ValueError("Missing reviewed behavior/split coverage; export blocked")
    sources = []
    for row, item in reviewed:
        intervals = []
        for interval in item["intervals"]:
            action = interval["action"]
            intervals.append({**interval, "behavior": action,
                              "action": "fight" if action in POSITIVE else "uncertain" if action == "uncertain" else "normal"})
        sources.append({**row, "review_method": "human", "reviewer": item["reviewer"], "intervals": intervals})
    return {"schema_version": 1, "scope_profile": PROFILE,
            "parent_checkpoint_sha256": PARENT_SHA, "sources": sources}
