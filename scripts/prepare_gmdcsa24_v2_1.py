"""Build a source-checked, subject-split fall manifest without downloading videos.

The publisher's Fall.csv files contain temporal fall labels.  ADL.csv files
identify negative clips.  Ambiguous fall timing is left for review.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "benchmarks/gmdcsa24_v2_1"
OUTPUT = ROOT / "benchmarks/gmdcsa24_v2_1_manifest.json"
FALL_TIME = re.compile(
    r"\bfall(?:ing)?\b[^;\[]*\[\s*(\d+(?:\.\d+)?)\s*(?:to\s+)?"
    r"(\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)
SPLIT = {1: "train", 2: "train", 3: "development", 4: "test"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(annotations=ANNOTATIONS, video_root=None):
    annotations = Path(annotations)
    provenance = load_json(annotations / "SOURCE.json")
    inventory = load_json(annotations / "VIDEO_INVENTORY.json")
    if provenance["tag"] != "v2.1" or provenance["tree_sha"] != inventory["tree_sha"]:
        raise ValueError("Publisher tag/tree mismatch")
    expected_csv = {row["path"]: row["sha256"] for row in provenance["files"]}
    expected_video = {row["path"]: row for row in inventory["videos"]}
    if len(expected_csv) != 8 or len(expected_video) != 160:
        raise ValueError("Incomplete official CSV/video inventory")
    sources = []
    for subject in range(1, 5):
        for category in ("Fall", "ADL"):
            csv_name = f"Subject {subject}/{category}.csv"
            csv_data = (annotations / csv_name).read_bytes()
            if hashlib.sha256(csv_data).hexdigest() != expected_csv[csv_name]:
                raise ValueError("Publisher CSV changed: " + csv_name)
            rows = list(csv.DictReader(csv_data.decode("utf-8-sig").splitlines()))
            for row in rows:
                name = row["File Name"].strip()
                if not re.fullmatch(r"\d{2}\.mp4", name):
                    raise ValueError("Unexpected video name: " + name)
                relative = f"Subject {subject}/{category}/{name}"
                if relative not in expected_video:
                    raise ValueError("CSV video absent from publisher inventory: " + relative)
                duration = float(row["Length (seconds)"])
                if not 0 < duration <= 120:
                    raise ValueError("Invalid publisher duration: " + relative)
                source_label = row[" Classes"].strip()
                if category == "Fall":
                    match = FALL_TIME.search(source_label)
                    if match:
                        start, end = map(float, match.groups())
                        if not 0 <= start < end <= duration + 0.1:
                            raise ValueError("Invalid fall interval: " + relative)
                        intervals = [{"start_seconds": start, "end_seconds": end,
                                      "action": "fall", "label": 1}]
                        status = "publisher_temporal"
                    else:
                        intervals, status = [], "needs_temporal_review"
                else:
                    if "fall" in source_label.lower():
                        raise ValueError("Fall text in ADL clip: " + relative)
                    intervals = [{"start_seconds": 0.0, "end_seconds": duration,
                                  "action": "normal", "label": 0}]
                    status = "publisher_clip_negative"
                if video_root is not None:
                    video = Path(video_root) / relative
                    if not video.is_file() or video.stat().st_size != expected_video[relative]["size"]:
                        raise ValueError("Video absent or size changed: " + str(video))
                sources.append({
                    "video": relative, "dataset": "gmdcsa24_v2_1",
                    "group": f"gmdcsa24/subject_{subject}", "subject": subject,
                    "split": SPLIT[subject], "source_category": category,
                    "publisher_duration_seconds": duration,
                    "publisher_class_text": source_label,
                    "publisher_git_blob": expected_video[relative]["sha"],
                    "publisher_bytes": expected_video[relative]["size"],
                    "annotation_status": status, "intervals": intervals,
                })
    if len(sources) != 160 or len({row["video"] for row in sources}) != 160:
        raise ValueError("Incomplete or duplicate publisher rows")
    counts = {status: sum(row["annotation_status"] == status for row in sources)
              for status in ("publisher_temporal", "publisher_clip_negative", "needs_temporal_review")}
    return {
        "schema_version": 1,
        "source": provenance["zenodo"], "version": provenance["tag"],
        "publisher_tree_sha": provenance["tree_sha"],
        "archive_md5": provenance["archive_md5"],
        "split_policy": "subjects 1-2 train; 3 development; 4 test; no fall calibration set yet",
        "positive_actions": ["fight", "wrestle", "fall"],
        "negative_actions": ["normal", "hug", "greet", "sit_lie", "sport_play"],
        "counts": counts, "video_files_verified": video_root is not None,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--video-root", type=Path,
                        help="Extracted dataset root containing Subject 1..4; checks all video sizes")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.annotations, args.video_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sources": len(result["sources"]), **result["counts"],
                      "video_files_verified": result["video_files_verified"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
