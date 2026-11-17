"""Turn crawled Vietnamese news media into a reviewed, YOLO-format labelling pool.

    python -m extensions.sharp_object.build_vn_pool ingest   # dedup images, video frames -> pool
    python -m extensions.sharp_object.collect prelabel --root <pool>   # Grounding DINO boxes
    python -m extensions.sharp_object.build_vn_pool sheets   # contact sheets for visual review
    python -m extensions.sharp_object.build_vn_pool apply --edits <csv>   # apply review decisions
    python -m extensions.sharp_object.build_vn_pool package  # zips for Drive / CVAT

Review edits CSV columns: image, action (keep|reject|class), class (for action=class).
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from extensions.sharp_object.collect import CLASS_ID, CLASSES, Pool, label_of, video_frames

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "sharp_object" / "vn" / "raw"
POOL = ROOT / "datasets" / "sharp_object" / "vn" / "pool"


def dhash(image, size=8):
    import cv2

    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near(hash_value, seen, distance):
    return any(bin(hash_value ^ other).count("1") <= distance for other in seen)


def ingest(args):
    import cv2

    pool = Pool(args.pool)
    media = [json.loads(l) for l in (args.raw / "media.jsonl").read_text(encoding="utf-8").splitlines() if l]
    seen = []
    for row in pool.rows.values():  # never re-add near-duplicates of media already in the pool
        existing = cv2.imread(str(pool.root / row["image"]))
        if existing is not None:
            seen.append(dhash(existing))
    images = videos = 0
    for item in sorted((m for m in media if m["kind"] == "image"), key=lambda m: -(args.raw / m["path"]).stat().st_size):
        image = cv2.imread(str(args.raw / item["path"]))
        if image is None:
            continue
        h = dhash(image)
        if near(h, seen, args.distance):
            continue
        seen.append(h)
        ident = Path(item["path"]).stem
        if not pool.has("vn_news_images", ident):
            pool.add("vn_news_images", ident, image, [], "news photo, research use only (publisher copyright)",
                     "case:" + item["case_id"], status="needs_prelabel", origin=item["article"])
            images += 1
    for item in (m for m in media if m["kind"] == "video"):
        stem = Path(item["path"]).stem
        local = []
        for t, frame in video_frames(args.raw / item["path"], every_seconds=args.every):
            h = dhash(frame)
            if near(h, local, args.distance + 4) or near(h, seen, args.distance):
                continue  # static camera: keep only frames that changed
            local.append(h)
            ident = f"{stem}_{t:07.1f}".replace(".", "p")
            if not pool.has("vn_news_frames", ident):
                pool.add("vn_news_frames", ident, frame, [], "news footage, research use only (publisher copyright)",
                         "case:" + item["case_id"], video=True, status="needs_prelabel", origin=item["article"])
                videos += 1
            if len(local) >= args.max_frames_per_video:
                break
        seen.extend(local)
    pool.save()
    print(f"ingested {images} images and {videos} video frames into {args.pool}")


def draw(pool, row, cell=320):
    import cv2

    image = cv2.imread(str(pool.root / row["image"]))
    height, width = image.shape[:2]
    label = pool.root / label_of(row["image"])
    for line in label.read_text().splitlines() if label.is_file() else []:
        cls, x, y, w, h = line.split()
        x, y, w, h = float(x) * width, float(y) * height, float(w) * width, float(h) * height
        cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), max(2, width // 200))
        cv2.putText(image, CLASSES[int(cls)], (int(x - w / 2), max(20, int(y - h / 2) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, width / 900), (0, 0, 255), 2)
    scale = cell / max(height, width)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))
    canvas = np.full((cell + 22, cell, 3), 255, np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    cv2.putText(canvas, Path(row["image"]).stem[-28:], (2, cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return canvas


def sheets(args):
    import cv2

    pool = Pool(args.pool)
    out = args.pool / "review_sheets"
    out.mkdir(exist_ok=True)
    for status in ("needs_review", "no_candidate"):
        rows = sorted((r for r in pool.rows.values() if r["status"] == status), key=lambda r: r["image"])
        index_path = out / f"{status}_index.csv"
        with index_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sheet", "position", "image"])
            for number in range(0, len(rows), args.per_sheet):
                chunk = rows[number:number + args.per_sheet]
                tiles = [draw(pool, r) for r in chunk]
                while len(tiles) % args.columns:
                    tiles.append(np.full_like(tiles[0], 255))
                grid = np.vstack([np.hstack(tiles[i:i + args.columns]) for i in range(0, len(tiles), args.columns)])
                name = f"{status}_{number // args.per_sheet + 1:03d}.jpg"
                cv2.imwrite(str(out / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
                for position, row in enumerate(chunk, 1):
                    writer.writerow([name, position, row["image"]])
        print(f"{status}: {len(rows)} images -> {out}")


def apply(args):
    pool = Pool(args.pool)
    with open(args.edits, encoding="utf-8") as stream:
        edits = list(csv.DictReader(stream))
    for edit in edits:
        row = pool.rows.get(edit["image"])
        if row is None:
            continue
        label = pool.root / label_of(row["image"])
        if edit["action"] == "reject":
            row["status"] = "rejected"
        elif edit["action"] == "class":
            lines = [l.split() for l in label.read_text().splitlines() if l.strip()]
            label.write_text("".join(f"{CLASS_ID[edit['class']]} {' '.join(p[1:])}\n" for p in lines), encoding="utf-8")
            row.update(status="reviewed", classes=edit["class"])
        elif edit["action"] == "keep":
            row["status"] = "reviewed"
        elif edit["action"] == "needs_box":
            row["status"] = "needs_human_box"  # weapon visible but no usable box: fix in CVAT
    pool.save()
    print(f"applied {len(edits)} edits")


def package(args):
    pool = Pool(args.pool)
    out = args.pool / "export"
    out.mkdir(exist_ok=True)
    keep = [r for r in pool.rows.values() if r["status"] in {"reviewed", "needs_review", "needs_human_box"}]
    archive = out / "vn_weapons_pool.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("classes.txt", "\n".join(CLASSES) + "\n")
        bundle.write(pool.manifest, "manifest.csv")
        for row in keep:
            bundle.write(pool.root / row["image"], row["image"])
            label = pool.root / label_of(row["image"])
            if label.is_file():
                bundle.write(label, label_of(row["image"]))
    print(f"{archive}: {len(keep)} images, {archive.stat().st_size / 1e6:.1f} MB")
    # CVAT / Label Studio import (YOLO 1.1): images that a person must still box precisely.
    todo = sorted((r for r in keep if r["status"] == "needs_human_box"), key=lambda r: (r["group"], r["image"]))
    for number in range(0, len(todo), args.batch_size):
        chunk = todo[number:number + args.batch_size]
        batch = out / f"vn_weapons_cvat_batch_{number // args.batch_size + 1:03d}.zip"
        with zipfile.ZipFile(batch, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("obj.names", "\n".join(CLASSES) + "\n")
            bundle.writestr("obj.data", f"classes = {len(CLASSES)}\nnames = obj.names\ntrain = train.txt\n")
            names = []
            for row in chunk:
                flat = row["image"][len("images/"):].replace("/", "__")
                bundle.write(pool.root / row["image"], f"obj_train_data/{flat}")
                label = pool.root / label_of(row["image"])
                bundle.write(label, f"obj_train_data/{flat[:-4]}.txt")
                names.append(f"data/obj_train_data/{flat}")
            bundle.writestr("train.txt", "\n".join(names) + "\n")
        print(f"{batch}: {len(chunk)} images to correct")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["ingest", "sheets", "apply", "package"])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--every", type=float, default=1.0, help="seconds between video frames")
    parser.add_argument("--max-frames-per-video", type=int, default=60)
    parser.add_argument("--distance", type=int, default=6, help="dHash Hamming distance treated as duplicate")
    parser.add_argument("--per-sheet", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--edits", type=Path)
    parser.add_argument("--batch-size", type=int, default=300, help="images per CVAT batch zip")
    args = parser.parse_args()
    {"ingest": ingest, "sheets": sheets, "apply": apply, "package": package}[args.step](args)


if __name__ == "__main__":
    main()
