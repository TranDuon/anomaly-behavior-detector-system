"""Download a small, fixed evaluation sample for the V1 baseline (local).

Writes datasets/vandalism/raw/eval_v1/manifest.jsonl, one record per item:
{path, label, binary (1 = vandalism), source, group, [segment], [intervals_frames]}.
Only test/val splits are used, so nothing here leaks into training.

    .venv\\Scripts\\python.exe -m extensions.vandalism.fetch_eval_sample
"""

import argparse
import json
import random
from pathlib import Path

from extensions.vandalism import data

DEST = data.ROOT / "datasets" / "vandalism" / "raw" / "eval_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives-per-class", type=int, default=12)
    parser.add_argument("--ashen-normal", type=int, default=30)
    parser.add_argument("--charades-throw", type=int, default=40)
    parser.add_argument("--charades-other", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    meta = data.sources()
    records = []

    kinetics = meta["kinetics700_2020"]
    for label, index in kinetics["positive"].items():
        for item in data.fetch_kinetics_class("val", label, index, DEST, seed=args.seed):
            records.append({**item, "binary": 1})
        print("kinetics +", label)
    for label, index in kinetics["hard_negative"].items():
        for item in data.fetch_kinetics_class("val", label, index, DEST, limit=args.negatives_per_class, seed=args.seed):
            records.append({**item, "binary": 0})
        print("kinetics -", label)

    vandal = data.ashen_split("test_anomaly_Vandalism.txt")
    normal = data.ashen_split("test_normal.txt")
    normal = rng.sample(normal, min(args.ashen_normal, len(normal)))
    annotations = data.ashen_temporal_annotations()
    for item in data.fetch_ashen(vandal + normal, DEST):
        name = Path(item["ashen_path"]).name
        item["binary"] = int(item["label"] == "vandalism")
        if item["binary"]:
            item["intervals_frames"] = annotations[name]["intervals"]
        records.append(item)
    print("ashen", len(vandal), "vandalism +", len(normal), "normal")

    charades = meta["charades"]
    throw_classes = set(charades["hard_negative_classes"])
    rows = [row for row in data.charades_annotations(DEST) if row["split"] == "test" and row["segments"]]
    rng.shuffle(rows)
    picked, throw_count, other_count = [], 0, 0
    for row in rows:
        throws = [s for s in row["segments"] if s["class"] in throw_classes and s["end"] - s["start"] >= 2]
        others = [s for s in row["segments"] if s["class"] not in throw_classes and s["end"] - s["start"] >= 3]
        if throws and throw_count < args.charades_throw:
            picked.append((row, throws[0], "charades_throw"))
            throw_count += 1
        elif others and not throws and other_count < args.charades_other:
            picked.append((row, rng.choice(others), "charades_other"))
            other_count += 1
        if throw_count >= args.charades_throw and other_count >= args.charades_other:
            break
    paths = data.fetch_charades_videos([row["id"] for row, _, _ in picked], DEST)
    for row, segment, label in picked:
        if row["id"] in paths:
            records.append({"path": paths[row["id"]], "label": label, "binary": 0, "source": "charades",
                            "group": "charades:" + row["subject"], "segment": [segment["start"], segment["end"]],
                            "charades_class": segment["class"]})
    print("charades", throw_count, "throw +", other_count, "other")

    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    positives = sum(r["binary"] for r in records)
    print(f"wrote {len(records)} items ({positives} vandalism) -> {DEST / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
