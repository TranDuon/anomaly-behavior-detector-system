"""Combine frozen interaction-v1 source identities with GMDCSA24 v2.1 labels.

This is a metadata preflight.  It never downloads videos or starts training.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(parent, pool, fall):
    # The 544-source pack already freezes interaction-v1 splits and groups
    # different AIRTLab cameras from one performance together.
    from fight.assault_extension import build_manifest

    frozen = build_manifest(parent, pool)
    originals = {row["video"]: row for row in parent["sources"]}
    weak = {row["video"]: row for row in pool["sources"]}
    sources = []
    for row in frozen["sources"]:
        video = row["video"]
        if video in originals:
            original = originals[video]
            # These are the labels already associated with the parent model.
            intervals = []
            for item in original["intervals"]:
                action = item["action"]
                if action == "fight":
                    label = 1
                elif action == "uncertain":
                    label = None
                elif action in {"normal", "gesture", "hug", "solo_motion",
                                "handshake_greet", "highfive_clap"}:
                    label = 0
                else:
                    raise ValueError("Unknown parent action: " + action)
                intervals.append({**item, "label": label})
            status = "parent_v1_existing"
        else:
            # Clip tags in the v2 pool are useful for triage, but they have
            # not been reviewed as precise intervals for this new task.
            intervals = []
            status = "needs_temporal_review"
        sources.append({**row, "annotation_status": status,
                        "intervals": intervals,
                        "publisher_clip_tags": weak[video]["intervals"] if video in weak else []})
    for row in fall["sources"]:
        sources.append(row)
    names = [row["video"] for row in sources]
    if len(names) != len(set(names)) or len(sources) != 704:
        raise ValueError("Expected 704 unique public sources")
    splits = {}
    for row in sources:
        key = (row["dataset"], row["group"])
        previous = splits.setdefault(key, row["split"])
        if previous != row["split"]:
            raise ValueError("Performance/subject leakage across splits")
    counts = Counter(row["annotation_status"] for row in sources)
    return {
        "schema_version": 1,
        "scope": "one positive class: fight, wrestle, or general fall",
        "source_checkpoints_unchanged": True,
        "source_manifest_sha256": frozen["manifest_sha256"],
        "fall_publisher_tree_sha": fall["publisher_tree_sha"],
        "video_files_verified": fall["video_files_verified"],
        "counts": dict(sorted(counts.items())),
        "ready_for_training": False,
        "remaining": [
            "Verify the GMDCSA24 archive and video files on Colab",
            "Review Subject 4/Fall/15.mp4 temporal interval",
            "Review the new interaction pool and confirm wrestle examples",
            "Reserve fall examples for threshold calibration without subject leakage",
        ],
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path,
                        default=ROOT / "benchmarks/mmaction_interaction_public_v1.json")
    parser.add_argument("--pool", type=Path,
                        default=ROOT / "benchmarks/mmaction_interaction_indoor_v2.json")
    parser.add_argument("--fall", type=Path,
                        default=ROOT / "benchmarks/gmdcsa24_v2_1_manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/general_fall_step2_manifest.json")
    args = parser.parse_args()
    result = build(read(args.parent), read(args.pool), read(args.fall))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sources": len(result["sources"]), "counts": result["counts"],
                      "ready_for_training": result["ready_for_training"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
