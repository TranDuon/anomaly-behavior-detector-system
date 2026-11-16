"""V2: download the vandalism training data and write a leakage-safe manifest.

Runs inside the Colab session (or locally with small limits). Everything comes
from sources usable without registration (datasets/vandalism/sources.json):

  positives : Kinetics-700 smashing + breaking glass, ASHEN Vandalism (weak, video-level)
  negatives : 17 Kinetics hard-negative classes, Charades indoor throwing + daily
              activities, ASHEN normal CCTV (same domain as the ASHEN positives)
  test      : Kinetics val, Charades test split, ASHEN test (frame intervals)

Splits never share a source video: Kinetics by youtube_id, Charades by
subject, ASHEN by its official train/test split (dev carved from train).

    python -m extensions.vandalism.prepare_vandalism --root /content/vandalism_data
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from extensions.vandalism import data


def dev_bucket(group, fraction):
    value = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < fraction


def assign(record, split, binary, weight=1.0, weak=False, **extra):
    return {**record, "split": split, "binary": int(binary), "weight": weight, "weak": weak, **extra}


def kinetics(args, records):
    meta = data.sources()["kinetics700_2020"]
    for label, index in meta["positive"].items():
        for item in data.fetch_kinetics_class("train", label, index, args.root, limit=args.pos_per_class, seed=args.seed):
            records.append(assign(item, "dev" if dev_bucket(item["group"], args.dev_fraction) else "train", 1))
        for item in data.fetch_kinetics_class("val", label, index, args.root, seed=args.seed):
            records.append(assign(item, "test", 1))
        print(f"kinetics + {label}", flush=True)
    for label, index in meta["hard_negative"].items():
        for item in data.fetch_kinetics_class("train", label, index, args.root, limit=args.neg_per_class, seed=args.seed):
            records.append(assign(item, "dev" if dev_bucket(item["group"], args.dev_fraction) else "train", 0))
        for item in data.fetch_kinetics_class("val", label, index, args.root, limit=args.neg_test_per_class, seed=args.seed):
            records.append(assign(item, "test", 0))
        print(f"kinetics - {label}", flush=True)


def ashen(args, records, rng):
    if args.ashen_weight <= 0:
        return
    annotations = data.ashen_temporal_annotations()
    train_v = data.ashen_split("train_anomaly_Vandalism.txt")[:args.ashen_vandal_limit]
    train_n = rng.sample(data.ashen_split("train_normal.txt"), args.ashen_normal_train)
    test_v = data.ashen_split("test_anomaly_Vandalism.txt")[:args.ashen_vandal_limit]
    test_n = rng.sample(data.ashen_split("test_normal.txt"), args.ashen_normal_test)
    train_f = data.ashen_split("train_anomaly_Fighting.txt") if args.ashen_fighting else []
    for item in data.fetch_ashen(train_v + train_n + train_f, args.root):
        split = "dev" if dev_bucket(item["group"], args.dev_fraction) else "train"
        positive = item["label"] == "vandalism"
        records.append(assign(item, split, positive, weight=args.ashen_weight if positive else 1.0, weak=positive))
    for item in data.fetch_ashen(test_v + test_n, args.root):
        positive = item["label"] == "vandalism"
        extra = {"intervals_frames": annotations[Path(item["ashen_path"]).name]["intervals"]} if positive else {}
        records.append(assign(item, "test", positive, **extra))
    print(f"ashen train {len(train_v)}+{len(train_n)}+{len(train_f)}, test {len(test_v)}+{len(test_n)}", flush=True)


def charades(args, records, rng):
    meta = data.sources()["charades"]
    throw_classes = set(meta["hard_negative_classes"])
    rows = data.charades_annotations(args.root)
    picked = []
    for split_name, other_limit in (("train", args.charades_other_train), ("test", args.charades_other_test)):
        candidates = [row for row in rows if row["split"] == split_name and row["segments"]]
        rng.shuffle(candidates)
        others = throws_taken = 0
        for row in candidates:
            throws = [s for s in row["segments"] if s["class"] in throw_classes and s["end"] - s["start"] >= 1.5]
            if throws and (args.charades_throw_limit is None or throws_taken < args.charades_throw_limit):
                throws_taken += 1
                picked.append((row, throws[0], "charades_throw", split_name))
                continue
            if throws:
                continue
            plain = [s for s in row["segments"] if s["end"] - s["start"] >= 3]
            if plain and others < other_limit:
                picked.append((row, rng.choice(plain), "charades_other", split_name))
                others += 1
    started = time.perf_counter()
    paths = data.fetch_charades_videos(sorted({row["id"] for row, *_ in picked}), args.root)
    print(f"charades: {len(paths)} videos in {time.perf_counter() - started:.0f}s", flush=True)
    for row, segment, label, split_name in picked:
        if row["id"] not in paths:
            continue
        group = "charades:" + row["subject"]
        split = "test" if split_name == "test" else ("dev" if dev_bucket(group, args.dev_fraction) else "train")
        records.append({"path": paths[row["id"]], "label": label, "source": "charades", "group": group,
                        "segment": [segment["start"], segment["end"]], "charades_class": segment["class"],
                        "split": split, "binary": 0, "weight": 1.0, "weak": False})


def check_leakage(records):
    by_group = {}
    for record in records:
        by_group.setdefault(record["group"], set()).add(record["split"])
    leaks = {group: splits for group, splits in by_group.items() if len(splits) > 1}
    # Kinetics val and train never share clips, but a YouTube video can appear in both
    # Kinetics splits; drop such test clips instead of trusting the upstream split.
    return leaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pos-per-class", type=int, default=None, help="cap positives per Kinetics class (default all)")
    parser.add_argument("--neg-per-class", type=int, default=150)
    parser.add_argument("--neg-test-per-class", type=int, default=30)
    parser.add_argument("--ashen-weight", type=float, default=0.5, help="0 disables ASHEN")
    parser.add_argument("--ashen-normal-train", type=int, default=60)
    parser.add_argument("--ashen-normal-test", type=int, default=60)
    parser.add_argument("--ashen-fighting", action="store_true")
    parser.add_argument("--ashen-vandal-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--charades-throw-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--charades-other-train", type=int, default=1500)
    parser.add_argument("--charades-other-test", type=int, default=200)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip", nargs="*", default=[], choices=["kinetics", "ashen", "charades"])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records = []
    started = time.perf_counter()
    if "kinetics" not in args.skip:
        kinetics(args, records)
    if "ashen" not in args.skip:
        ashen(args, records, rng)
    if "charades" not in args.skip:
        charades(args, records, rng)
    leaks = check_leakage(records)
    if leaks:
        before = len(records)
        records = [r for r in records if not (r["group"] in leaks and r["split"] == "test")]
        records = [r for r in records if not (r["group"] in leaks and r["split"] == "dev"
                                             and {"train", "dev"} <= leaks[r["group"]])]
        print(f"removed {before - len(records)} records whose source video crossed splits", flush=True)
    assert not check_leakage(records), "split leakage remains"
    manifest = args.root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter((r["split"], r["source"], r["binary"]) for r in records)
    summary = {"manifest": str(manifest), "manifest_sha256": data.sha256(manifest), "items": len(records),
               "seconds": round(time.perf_counter() - started, 1), "args": {k: str(v) for k, v in vars(args).items()},
               "counts": {f"{s}/{src}/{'pos' if b else 'neg'}": n for (s, src, b), n in sorted(counts.items())},
               "kinetics_archives": sorted({(r["archive"], r["archive_sha256"]) for r in records
                                            if r["source"] == "kinetics700_2020" and r.get("archive_sha256")})}
    (args.root / "prepare_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=1))


if __name__ == "__main__":
    main()
