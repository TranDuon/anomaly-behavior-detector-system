"""H1: turn the merged detection dataset (build_dataset.py) into hand crops for the classifier.

    python -m extensions.sharp_object.build_hand_crops --data DATA --out CROPS --pose yolo11s-pose.pt

For every image the pose model finds people; each visible hand gets the runtime crop
(hand_crop.hand_crops). A crop is labelled from the weapon boxes already in the dataset:

  overlap = max over weapon boxes of max(inter / weapon area, inter / crop area)
  positive  overlap >= --pos-overlap      negative  no weapon touches the crop
  in between: ambiguous, not used

Close-up weapon photos often have no detectable person. For those (train split only) a
"pseudo" crop is cut around each weapon box that no positive hand crop covers, and person-less
negative images give random pseudo crops. Val/test keep only real hand crops, so their scores
reflect the runtime path.

Output: CROPS/{train,val,test}/*.jpg (saved at --save-size), index.csv, summary.json and
preview_<split>.jpg contact sheets (positives on top, negatives below) for a visual check.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from extensions.sharp_object.hand_crop import cut, hand_crops, people_from_pose

SPLITS = ("train", "val", "test")


def weapon_boxes(label_path, width, height):
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, x, y, w, h = map(float, parts)
                boxes.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
    return boxes


def overlap(crop, box):
    inter_w = min(crop[2], box[2]) - max(crop[0], box[0])
    inter_h = min(crop[3], box[3]) - max(crop[1], box[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    box_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    crop_area = max(1e-6, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    return max(inter / box_area, inter / crop_area)


def pseudo_positive(box, rng, min_side):
    """Square crop around a weapon box, jittered, 1.1-1.8x its long side."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = max(min_side, max(bw, bh) * rng.uniform(1.1, 1.8))
    cx = (box[0] + box[2]) / 2 + rng.uniform(-0.2, 0.2) * bw
    cy = (box[1] + box[3]) / 2 + rng.uniform(-0.2, 0.2) * bh
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


def pseudo_negative(width, height, rng, min_side):
    side = max(min_side, min(width, height) * rng.uniform(0.3, 0.7))
    x = rng.uniform(0, max(0.0, width - side))
    y = rng.uniform(0, max(0.0, height - side))
    return [x, y, x + side, y + side]


def contact_sheet(rows, out_root, target, per_row=12, rows_each=3, tile=96):
    import cv2

    lines = []
    for label in (1, 0):
        picked = [r for r in rows if r["label"] == label and r["kind"] == "hand"] or [r for r in rows if r["label"] == label]
        picked = random.Random(label).sample(picked, min(len(picked), per_row * rows_each))
        tiles = [cv2.resize(cv2.imread(str(out_root / r["crop"])), (tile, tile)) for r in picked]
        tiles += [np.full((tile, tile, 3), 255, np.uint8)] * (per_row * rows_each - len(tiles))
        for start in range(0, len(tiles), per_row):
            lines.append(np.hstack(tiles[start:start + per_row]))
        lines.append(np.full((8, tile * per_row, 3), (0, 0, 255) if label else (0, 160, 0), np.uint8))
    cv2.imwrite(str(target), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="output of build_dataset.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pose", default="yolo11s-pose.pt")
    parser.add_argument("--pose-imgsz", type=int, default=960, help="larger than runtime to find small people")
    parser.add_argument("--pose-conf", type=float, default=0.35)
    parser.add_argument("--max-people", type=int, default=8)
    parser.add_argument("--pos-overlap", type=float, default=0.25)
    parser.add_argument("--save-size", type=int, default=256, help="crops are stored at this size")
    parser.add_argument("--pseudo-negatives", type=int, default=1, help="random crops per person-less negative image")
    parser.add_argument("--no-pseudo", action="store_true", help="real hand crops only")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="images per split (smoke tests)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from extensions.sharp_object.hand_crop import CROP_DEFAULTS

    params = dict(CROP_DEFAULTS)
    with (args.data / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if args.limit:
        by_split = defaultdict(list)
        for row in manifest:
            by_split[row["split"]].append(row)
        manifest = [row for split in SPLITS for row in by_split[split][:args.limit]]
    for split in SPLITS:
        (args.out / split).mkdir(parents=True, exist_ok=True)
    pose = YOLO(args.pose)
    rng = random.Random(3407)
    rows, counts = [], defaultdict(Counter)
    for begin in range(0, len(manifest), args.batch):
        chunk = manifest[begin:begin + args.batch]
        frames = [cv2.imread(str(args.data / row["image"])) for row in chunk]
        valid = [(row, frame) for row, frame in zip(chunk, frames) if frame is not None]
        if not valid:
            continue
        results = pose.predict([frame for _, frame in valid], imgsz=args.pose_imgsz, conf=args.pose_conf,
                               verbose=False, device=args.device)
        for (row, frame), result in zip(valid, results):
            split, height, width = row["split"], frame.shape[0], frame.shape[1]
            image_rel = Path(row["image"])
            label = args.data / "labels" / image_rel.parts[1] / (image_rel.stem + ".txt")
            weapons = weapon_boxes(label, width, height)
            people = people_from_pose(result, args.pose_conf, args.max_people)
            stat = counts[split]
            stat["images"] += 1
            stat["images_with_people"] += bool(people)
            stat["weapon_images"] += bool(weapons)
            covered = set()
            candidates = []
            for person_index, (box, keypoints) in enumerate(people):
                for crop in hand_crops(box, keypoints, params):
                    scores = [overlap(crop["box"], w) for w in weapons]
                    best = max(scores, default=0.0)
                    if best >= args.pos_overlap:
                        covered.update(i for i, s in enumerate(scores) if s >= args.pos_overlap)
                        candidates.append((crop["box"], 1, "hand", person_index, crop["arm"], best))
                    elif best == 0.0:
                        candidates.append((crop["box"], 0, "hand", person_index, crop["arm"], best))
                    else:
                        stat["ambiguous"] += 1
            if split == "train" and not args.no_pseudo:
                for index, weapon in enumerate(weapons):
                    if index not in covered:
                        candidates.append((pseudo_positive(weapon, rng, params["min_side"]), 1, "pseudo", -1, "", 1.0))
                if not weapons and not people:
                    for _ in range(args.pseudo_negatives):
                        candidates.append((pseudo_negative(width, height, rng, params["min_side"]), 0, "pseudo", -1, "", 0.0))
            stat["weapon_images_with_positive_hand"] += any(c[1] == 1 and c[2] == "hand" for c in candidates)
            for number, (box, positive, kind, person_index, arm, best) in enumerate(candidates):
                name = f"{image_rel.stem}_{number:02d}.jpg"
                cv2.imwrite(str(args.out / split / name), cut(frame, box, args.save_size), [cv2.IMWRITE_JPEG_QUALITY, 92])
                stat[f"{kind}_{'pos' if positive else 'neg'}"] += 1
                rows.append({"crop": f"{split}/{name}", "split": split, "label": positive, "kind": kind,
                             "source": row["source"], "group": row["group"], "image": row["image"],
                             "image_has_weapon": int(bool(weapons)), "person": person_index, "arm": arm,
                             "overlap": round(best, 3)})
        if (begin // args.batch) % 50 == 0:
            print(f"{begin + len(chunk)}/{len(manifest)} images, {len(rows)} crops", flush=True)

    with (args.out / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]) if rows else ["crop"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"pose": args.pose, "pose_imgsz": args.pose_imgsz, "crop": params, "pos_overlap": args.pos_overlap,
               "save_size": args.save_size, "splits": {s: dict(counts[s]) for s in SPLITS}}
    for split in SPLITS:
        stat = counts[split]
        summary["splits"][split]["hand_coverage_of_weapon_images"] = round(
            stat["weapon_images_with_positive_hand"] / stat["weapon_images"], 3) if stat["weapon_images"] else None
        summary["splits"][split]["vn_hand_pos"] = sum(1 for r in rows if r["split"] == split and r["source"] == "vn"
                                                      and r["kind"] == "hand" and r["label"] == 1)
        split_rows = [r for r in rows if r["split"] == split]
        if split_rows:
            contact_sheet(split_rows, args.out, args.out / f"preview_{split}.jpg")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if not any(r["label"] == 1 for r in rows if r["split"] == "train"):
        raise SystemExit("no positive train crops: check the dataset and the pose model")


if __name__ == "__main__":
    main()
