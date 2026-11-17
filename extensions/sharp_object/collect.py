"""S1: collect sharp-object data (no registration) into one YOLO-format pool.

Designed to run in Colab with the output on Google Drive, so the pool can be
reviewed/labelled from anywhere. Steps (each skips work already done):

  labeled   public sets that already have boxes -> mapped to our 7 classes
            Zenodo Dangerous Items, Granada knife (HF), Open Images, COCO,
            Simuletic CCTV samples; plus Open Images hard negatives (bat,
            umbrella, phone, bottle) as images without boxes
  frames    frames from videos (>= 20 % of the pool must come from video):
            Kinetics knife/sword/axe classes, ASHEN Robbery CCTV, and any
            videos the user drops into <root>/user_videos (e.g. Vietnamese news)
  prelabel  Grounding DINO boxes for video frames (to be corrected by a person)
  batches   CVAT / Label Studio import zips (YOLO 1.1) of N frames each
  stats     counts per source/class and the video-frame share

Layout under --root:
  images/<source>/<id>.jpg   labels/<source>/<id>.txt   (YOLO, class ids per class_map.json)
  manifest.csv               one row per image (source, license, group, video, review state)
"""

import argparse
import csv
import io
import json
import random
import re
import time
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from extensions.vandalism import data as fetch

ROOT = Path(__file__).resolve().parents[2]
CLASS_MAP = json.loads((ROOT / "datasets" / "sharp_object" / "class_map.json").read_text(encoding="utf-8"))
CLASSES = CLASS_MAP["classes"]
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
FIELDS = ["image", "source", "license", "group", "is_video_frame", "boxes", "classes", "status", "origin"]
MAX_SIDE = 1024

ZENODO_URL = "https://zenodo.org/records/16422779/files/Dangerous%20Items.zip?download=1"
OI_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BOXES = {"train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
            "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"}
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE = "http://images.cocodataset.org/{split}/{name}"
GRANADA_REPO = "shravya11/weapon-detection-dataset"
SIMULETIC = [("Simuletic/cctv-knife-detection-dataset", "CC BY 4.0"),
             ("Simuletic/CCTV_ATM_Robbery_Detection_Dataset_Gun_Knife", "CC BY-NC 4.0")]
KINETICS_CLASSES = {"sword fighting": 603, "throwing knife": 623, "sharpening knives": 517, "chopping meat": 101,
                    "carving wood with a knife": 84, "throwing axe": 620, "chopping wood": 102}


# ------------------------------------------------------------------ pool helpers
def label_of(image):
    """images/<source>/<id>.jpg -> labels/<source>/<id>.txt"""
    return "labels/" + image[len("images/"):-len(".jpg")] + ".txt"


class Pool:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.root / "manifest.csv"
        self.rows = {}
        if self.manifest.is_file():
            with self.manifest.open(encoding="utf-8", newline="") as stream:
                self.rows = {row["image"]: row for row in csv.DictReader(stream)}

    def has(self, source, ident):
        return f"images/{source}/{ident}.jpg" in self.rows

    def add(self, source, ident, image_bgr, boxes, license_, group, video=False, status="labeled", origin=""):
        """boxes: [(class_name, x1, y1, x2, y2) in pixels]."""
        import cv2

        height, width = image_bgr.shape[:2]
        scale = min(1.0, MAX_SIDE / max(height, width))
        if scale < 1:
            image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        image_path = self.root / "images" / source / f"{ident}.jpg"
        label_path = self.root / "labels" / source / f"{ident}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = []
        for name, x1, y1, x2, y2 in boxes:
            x1, x2 = sorted((max(0.0, x1), min(width, x2)))
            y1, y2 = sorted((max(0.0, y1), min(height, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{CLASS_ID[name]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        key = f"images/{source}/{ident}.jpg"
        self.rows[key] = {"image": key, "source": source, "license": license_, "group": group,
                          "is_video_frame": int(video), "boxes": len(lines),
                          "classes": ";".join(sorted({b[0] for b in boxes})), "status": status, "origin": origin}

    def save(self):
        partial = self.manifest.with_suffix(".partial.csv")
        with partial.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(self.rows.values())
        partial.replace(self.manifest)


def decode(data_bytes):
    import cv2

    image = cv2.imdecode(np.frombuffer(data_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("undecodable image")
    return image


def http_bytes(url, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            with fetch._open(url) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)


def parallel(function, items, workers=24):
    results = []
    with ThreadPoolExecutor(workers) as executor:
        for item, value in zip(items, executor.map(_safe(function), items)):
            results.append((item, value))
    return results


def _safe(function):
    def wrapped(item):
        try:
            return function(item)
        except Exception as error:  # one broken image must not stop the collection
            return error
    return wrapped


# ------------------------------------------------------------------ labeled sources
def zenodo(pool, work, limit):
    archive = Path(work) / "zenodo_dangerous_items.zip"
    fetch.download(ZENODO_URL, archive)
    mapping = CLASS_MAP["map"]["zenodo_dangerous_items"]
    negative_names = set(CLASS_MAP["map"]["zenodo_negative_only"])
    added = 0
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        yaml_name = next(n for n in names if n.lower().endswith((".yaml", ".yml")))
        text = bundle.read(yaml_name).decode("utf-8", "replace")
        class_names = _yaml_names(text)
        images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = {Path(n).stem: n for n in names if n.endswith(".txt") and "/labels/" in n.replace("\\", "/")}
        for name in images:
            ident = re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem)
            if pool.has("zenodo", ident):
                continue
            rows = []
            if Path(name).stem in labels:
                for line in bundle.read(labels[Path(name).stem]).decode().split("\n"):
                    parts = line.split()
                    if len(parts) == 5:
                        rows.append((class_names[int(parts[0])], *map(float, parts[1:])))
            kept = [r for r in rows if r[0] in mapping]
            others = {r[0] for r in rows} - set(mapping)
            if not kept and others - negative_names:
                continue  # gun / rifle only: out of scope
            image = decode(bundle.read(name))
            height, width = image.shape[:2]
            boxes = [(mapping[c], (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
                     for c, x, y, w, h in kept]
            pool.add("zenodo", ident, image, boxes, "CC BY-NC 4.0", "zenodo:" + ident, origin=name)
            added += 1
            if limit and added >= limit:
                break
    archive.unlink(missing_ok=True)
    return added


def _yaml_names(text):
    match = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if match:
        return [n.strip().strip("'\"") for n in match.group(1).split(",")]
    names = {}
    block = re.search(r"names:\s*\n((?:\s+.+\n?)+)", text)
    for line in (block.group(1) if block else "").splitlines():
        found = re.match(r"\s*(\d+)\s*:\s*(.+)", line) or re.match(r"\s*-\s*(.+)", line)
        if found and len(found.groups()) == 2:
            names[int(found.group(1))] = found.group(2).strip().strip("'\"")
        elif found:
            names[len(names)] = found.group(1).strip().strip("'\"")
    return [names[i] for i in sorted(names)]


def granada(pool, work, limit):
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(GRANADA_REPO, repo_type="dataset", allow_patterns=["Knife_detection/**"],
                                   local_dir=Path(work) / "granada"))
    mapping = CLASS_MAP["map"]["granada_knife"]
    added = 0
    for xml_path in sorted(local.rglob("*.xml")):
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", xml_path.stem)
        if pool.has("granada_knife", ident):
            continue
        root = ElementTree.parse(xml_path).getroot()
        image_path = next((p for p in (xml_path.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG")) if p.is_file()), None)
        if image_path is None:
            filename = root.findtext("filename") or ""
            image_path = next(iter(xml_path.parent.parent.rglob(filename)), None) if filename else None
        if image_path is None:
            continue
        boxes = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if name in mapping and box is not None:
                boxes.append((mapping[name], *(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        pool.add("granada_knife", ident, decode(image_path.read_bytes()), boxes, "CC BY 4.0", "granada:" + ident,
                 origin=str(xml_path.relative_to(local)))
        added += 1
        if limit and added >= limit:
            break
    return added


def openimages(pool, work, per_class, negatives_per_class, splits=("train", "validation", "test")):
    descriptions = http_bytes(OI_DESCRIPTIONS).decode("utf-8").splitlines()
    mid_name = {line.split(",", 1)[0]: line.split(",", 1)[1] for line in descriptions if "," in line}
    mapping = CLASS_MAP["map"]["openimages"]
    negatives = set(CLASS_MAP["hard_negative_labels"])
    wanted = {mid for mid, name in mid_name.items() if name in mapping or name in negatives}
    per_image = defaultdict(list)
    for split in splits:
        with fetch._open(OI_BOXES[split]) as response:
            reader = csv.reader(io.TextIOWrapper(response, encoding="utf-8"))
            header = next(reader)
            col = {k: i for i, k in enumerate(header)}
            for row in reader:
                mid = row[col["LabelName"]]
                if mid in wanted and row[col["IsDepiction"]] != "1":
                    per_image[(split, row[col["ImageID"]])].append(
                        (mid_name[mid], *(float(row[col[k]]) for k in ("XMin", "YMin", "XMax", "YMax"))))
        print(f"  open images {split}: {len(per_image)} candidate images so far", flush=True)
    rng = random.Random(3407)
    by_class = defaultdict(list)
    for key, boxes in per_image.items():
        positive = sorted({b[0] for b in boxes if b[0] in mapping})
        if positive:
            by_class[positive[0]].append(key)
        else:
            by_class["neg:" + sorted({b[0] for b in boxes})[0]].append(key)
    chosen = []
    for name, keys in sorted(by_class.items()):
        rng.shuffle(keys)
        chosen += keys[: (negatives_per_class if name.startswith("neg:") else per_class)]

    def load(key):
        split, image_id = key
        if pool.has("openimages", image_id):
            return None
        return decode(http_bytes(OI_IMAGE.format(split=split, image_id=image_id)))

    added = 0
    for (split, image_id), image in parallel(load, chosen):
        if image is None or isinstance(image, Exception):
            continue
        height, width = image.shape[:2]
        boxes = [(mapping[name], x1 * width, y1 * height, x2 * width, y2 * height)
                 for name, x1, y1, x2, y2 in per_image[(split, image_id)] if name in mapping]
        pool.add("openimages", image_id, image, boxes, "CC BY 2.0 (images, Flickr) / CC BY 4.0 (boxes)",
                 "openimages:" + image_id, origin=f"{split}")
        added += 1
    return added


def coco(pool, work, limit, min_area=0.002):
    archive = Path(work) / "coco_annotations_2017.zip"
    fetch.download(COCO_ANNOTATIONS, archive)
    selected = []
    with zipfile.ZipFile(archive) as bundle:
        for split in ("train2017", "val2017"):
            content = json.loads(bundle.read(f"annotations/instances_{split}.json"))
            knife = next(c["id"] for c in content["categories"] if c["name"] == "knife")
            images = {i["id"]: i for i in content["images"]}
            boxes = defaultdict(list)
            for ann in content["annotations"]:
                if ann["category_id"] == knife and not ann.get("iscrowd"):
                    boxes[ann["image_id"]].append(ann["bbox"])
            for image_id, items in boxes.items():
                info = images[image_id]
                largest = max(w * h for _, _, w, h in items) / (info["width"] * info["height"])
                if largest >= min_area:
                    selected.append((split, info["file_name"], items, largest))
    archive.unlink(missing_ok=True)
    random.Random(3407).shuffle(selected)
    selected = selected[:limit]

    def load(item):
        split, name, _, _ = item
        if pool.has("coco", Path(name).stem):
            return None
        return decode(http_bytes(COCO_IMAGE.format(split=split, name=name)))

    added = 0
    for (split, name, items, _), image in parallel(load, selected):
        if image is None or isinstance(image, Exception):
            continue
        boxes = [("dao_ngan", x, y, x + w, y + h) for x, y, w, h in items]
        pool.add("coco", Path(name).stem, image, boxes, "CC BY 4.0 (annotations); images per Flickr terms",
                 "coco:" + Path(name).stem, origin=split)
        added += 1
    return added


# Hard negatives for the hand-crop classifier: people holding long non-weapon objects.
COCO_HARD_NEGATIVES = ("baseball bat", "tennis racket", "umbrella", "skis")
COCO_EXCLUDE = ("knife", "scissors")  # anything sharp in the image -> not a clean negative


def coco_negatives(pool, work, limit, min_area=0.002):
    """COCO images with a person and a bat / racket / umbrella / skis and nothing sharp; no boxes (negatives)."""
    archive = Path(work) / "coco_annotations_2017.zip"
    fetch.download(COCO_ANNOTATIONS, archive)
    selected = []
    with zipfile.ZipFile(archive) as bundle:
        for split in ("train2017", "val2017"):
            content = json.loads(bundle.read(f"annotations/instances_{split}.json"))
            ids = {c["name"]: c["id"] for c in content["categories"]}
            wanted = {ids[n] for n in COCO_HARD_NEGATIVES}
            excluded = {ids[n] for n in COCO_EXCLUDE}
            images = {i["id"]: i for i in content["images"]}
            seen = defaultdict(set)
            largest = defaultdict(float)
            for ann in content["annotations"]:
                seen[ann["image_id"]].add(ann["category_id"])
                if ann["category_id"] in wanted:
                    info = images[ann["image_id"]]
                    largest[ann["image_id"]] = max(largest[ann["image_id"]],
                                                   ann["area"] / (info["width"] * info["height"]))
            for image_id, area in largest.items():
                if area >= min_area and ids["person"] in seen[image_id] and not seen[image_id] & excluded:
                    selected.append((split, images[image_id]["file_name"]))
    archive.unlink(missing_ok=True)
    random.Random(3407).shuffle(selected)
    selected = selected[:limit]

    def load(item):
        split, name = item
        if pool.has("coco_neg", Path(name).stem):
            return None
        return decode(http_bytes(COCO_IMAGE.format(split=split, name=name)))

    added = 0
    for (split, name), image in parallel(load, selected):
        if image is None or isinstance(image, Exception):
            continue
        pool.add("coco_neg", Path(name).stem, image, [], "CC BY 4.0 (annotations); images per Flickr terms",
                 "coco_neg:" + Path(name).stem, origin=split)
        added += 1
    return added


def simuletic(pool, work):
    from huggingface_hub import snapshot_download

    added = 0
    for repo, license_ in SIMULETIC:
        local = Path(snapshot_download(repo, repo_type="dataset", local_dir=Path(work) / repo.replace("/", "__")))
        yaml_files = list(local.rglob("*.yaml"))
        names = _yaml_names(yaml_files[0].read_text(encoding="utf-8")) if yaml_files else ["person", "knife"]
        for image_path in sorted(local.rglob("*.png")) + sorted(local.rglob("*.jpg")):
            if "annotated_example" in image_path.name:
                continue
            label_path = next(iter(local.rglob(image_path.stem + ".txt")), None)
            ident = re.sub(r"[^A-Za-z0-9_-]", "_", image_path.stem)
            if label_path is None or pool.has("simuletic", ident):
                continue
            image = decode(image_path.read_bytes())
            height, width = image.shape[:2]
            boxes = []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) < len(names) and names[int(parts[0])].lower() == "knife":
                    x, y, w, h = map(float, parts[1:])
                    boxes.append(("dao_ngan", (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
            pool.add("simuletic", ident, image, boxes, license_ + " (synthetic)", "simuletic:" + ident, origin=repo)
            added += 1
    return added


# ------------------------------------------------------------------ video frames
def video_frames(path, every_seconds=None, count=None):
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if count and total:
        targets = {int(i) for i in np.linspace(total * 0.1, total * 0.9, count)}
    else:
        step = max(1, round((every_seconds or 1.0) * fps))
        targets = set(range(0, total or 10 ** 7, step))
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets:
            frames.append((index / fps, frame))
        index += 1
    capture.release()
    return frames


def frames_stage(pool, work, kinetics_per_class, frames_per_clip, ashen_every, user_every, ashen_limit=None, kinetics_classes=None):
    added = Counter()
    for label, index in list(KINETICS_CLASSES.items())[:kinetics_classes]:
        for item in fetch.fetch_kinetics_class("train", label, index, Path(work), limit=kinetics_per_class, seed=3407):
            stem = Path(item["path"]).stem
            for t, frame in video_frames(item["path"], count=frames_per_clip):
                ident = f"{stem}_{t:06.2f}".replace(".", "p")
                if not pool.has("kinetics_frames", ident):
                    pool.add("kinetics_frames", ident, frame, [], "Kinetics (CC BY 4.0 labels; YouTube video)",
                             item["group"], video=True, status="needs_prelabel", origin=label)
                    added["kinetics"] += 1
        print(f"  kinetics frames: {label}", flush=True)
        pool.save()
    robbery = (fetch.ashen_split("train_anomaly_Robbery.txt") + fetch.ashen_split("test_anomaly_Robbery.txt"))[:ashen_limit]
    for item in fetch.fetch_ashen(robbery, Path(work)):
        stem = Path(item["path"]).stem
        for t, frame in video_frames(item["path"], every_seconds=ashen_every):
            ident = f"{stem}_{t:07.1f}".replace(".", "p")
            if not pool.has("ashen_robbery", ident):
                pool.add("ashen_robbery", ident, frame, [], "CC BY-NC 4.0", "ashen:" + stem, video=True,
                         status="needs_prelabel", origin=item["ashen_path"])
                added["ashen_robbery"] += 1
    pool.save()
    user_dir = pool.root / "user_videos"
    user_dir.mkdir(exist_ok=True)
    for video in sorted(p for p in user_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}):
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", video.stem)
        for t, frame in video_frames(video, every_seconds=user_every):
            ident = f"{stem}_{t:07.1f}".replace(".", "p")
            if not pool.has("user_videos", ident):
                pool.add("user_videos", ident, frame, [], "user-collected (news footage, research use)", "user:" + stem,
                         video=True, status="needs_prelabel", origin=video.name)
                added["user_videos"] += 1
    pool.save()
    return dict(added)


# ------------------------------------------------------------------ pre-labelling
PROMPTS = ["a blade on a long pole", "a machete", "a knife", "a sword", "a sickle", "an axe"]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def refine_boxes(raw, shape, iou=0.5):
    """Rules for Vietnamese weapons on top of open-vocabulary boxes (see class_map prelabel_rules)."""
    height, width = shape
    area = float(height * width)
    kept = []
    for name, x1, y1, x2, y2, score in sorted(raw, key=lambda b: -b[5]):
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or not (0.0005 <= w * h / area <= 0.5):
            continue
        if max(w, h) >= 5 * min(w, h) and (w >= 0.35 * width or h >= 0.35 * height):
            name = "dao_phong_lon"
        box = (name, x1, y1, x2, y2, score)
        if all(_iou(box[1:5], other[1:5]) < iou for other in kept):
            kept.append(box)
    return kept


def prelabel(pool, model_id, box_threshold, text_threshold, keep_empty, batch, status="needs_prelabel", source=None):
    import cv2
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    mapping = CLASS_MAP["map"]["grounding_dino"]
    text = ". ".join(PROMPTS) + "."
    rows = [r for r in pool.rows.values() if r["status"] == status and (source is None or r["source"] == source)]
    rng = random.Random(3407)
    for begin in range(0, len(rows), batch):
        chunk = rows[begin:begin + batch]
        images = [cv2.imread(str(pool.root / r["image"]))[:, :, ::-1] for r in chunk]
        inputs = processor(images=list(images), text=[text] * len(images), return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[img.shape[:2] for img in images])
        for row, image, result in zip(chunk, images, results):
            labels = result.get("text_labels", result.get("labels"))
            raw = []
            for score, label, box in zip(result["scores"].tolist(), labels, result["boxes"].tolist()):
                name = next((mapping[k] for k in sorted(mapping, key=len, reverse=True) if k in str(label)), None)
                if name:
                    raw.append((name, *box, score))
            boxes = refine_boxes(raw, image.shape[:2])
            if not boxes and row["source"].startswith("vn_"):
                row["status"] = "no_candidate_2" if status == "no_candidate" else "no_candidate"  # Vietnamese weapons are rare: a person checks these too
                continue
            if not boxes and not (rng.random() < keep_empty or row["source"] == "kinetics_frames"):
                row["status"] = "dropped_no_candidate"
                continue
            height, width = image.shape[:2]
            label_path = pool.root / label_of(row["image"])
            label_path.write_text("".join(
                f"{CLASS_ID[n]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} "
                f"{(y2 - y1) / height:.6f}\n" for n, x1, y1, x2, y2, _ in boxes), encoding="utf-8")
            row.update(status="needs_review", boxes=len(boxes), classes=";".join(sorted({b[0] for b in boxes})))
        if (begin // batch) % 20 == 0:
            print(f"  prelabel {begin + len(chunk)}/{len(rows)}", flush=True)
            pool.save()
    pool.save()


# ------------------------------------------------------------------ CVAT batches
def batches(pool, size):
    out = pool.root / "batches"
    out.mkdir(exist_ok=True)
    pending = [r for r in pool.rows.values() if r["status"] == "needs_review"]
    pending.sort(key=lambda r: (r["source"], r["group"], r["image"]))
    existing = sorted(out.glob("batch_*.zip"))
    already = set()
    for archive in existing:
        with zipfile.ZipFile(archive) as bundle:
            already |= {n.split("/", 1)[1] for n in bundle.namelist() if n.startswith("obj_train_data/") and n.endswith(".jpg")}
    pending = [r for r in pending if r["image"].replace("/", "__") not in already]
    number = len(existing)
    for begin in range(0, len(pending), size):
        number += 1
        chunk = pending[begin:begin + size]
        archive = out / f"batch_{number:03d}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("obj.names", "\n".join(CLASSES) + "\n")
            bundle.writestr("obj.data", f"classes = {len(CLASSES)}\nnames = obj.names\ntrain = train.txt\n")
            names = []
            for row in chunk:
                flat = row["image"].replace("/", "__")
                bundle.write(pool.root / row["image"], f"obj_train_data/{flat}")
                label = pool.root / label_of(row["image"])
                bundle.write(label, f"obj_train_data/{Path(flat).with_suffix('.txt')}")
                names.append(f"data/obj_train_data/{flat}")
            bundle.writestr("train.txt", "\n".join(names) + "\n")
        print(f"  {archive.name}: {len(chunk)} frames", flush=True)


# ------------------------------------------------------------------ stats
def stats(pool):
    rows = list(pool.rows.values())
    usable = [r for r in rows if r["status"] in {"labeled", "needs_review", "reviewed"}]
    by_class = Counter(c for r in usable for c in r["classes"].split(";") if c)
    video = sum(int(r["is_video_frame"]) for r in usable)
    summary = {"images": len(usable), "video_frames": video,
               "video_share": round(video / len(usable), 3) if usable else 0,
               "by_source": dict(Counter(r["source"] for r in usable)),
               "by_status": dict(Counter(r["status"] for r in rows)),
               "images_with_class": dict(by_class),
               "negatives_without_boxes": sum(1 for r in usable if r["status"] == "labeled" and r["boxes"] in ("0", 0))}
    (pool.root / "stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["labeled", "frames", "prelabel", "batches", "stats"])
    parser.add_argument("--root", type=Path, required=True, help="pool directory (on Drive in Colab)")
    parser.add_argument("--work", type=Path, default=Path("/content/sharp_work"), help="scratch for archives/videos")
    parser.add_argument("--sources", nargs="*", default=["zenodo", "granada", "openimages", "coco", "simuletic"])
    parser.add_argument("--limit", type=int, default=None, help="cap per source (smoke tests)")
    parser.add_argument("--oi-per-class", type=int, default=1500)
    parser.add_argument("--oi-negatives-per-class", type=int, default=300)
    parser.add_argument("--oi-splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("--coco-limit", type=int, default=1500)
    parser.add_argument("--coco-neg-limit", type=int, default=3000, help="coco_neg: people holding bats/rackets/...")
    parser.add_argument("--kinetics-per-class", type=int, default=120)
    parser.add_argument("--kinetics-classes", type=int, default=None, help="smoke tests only")
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--ashen-every", type=float, default=3.0)
    parser.add_argument("--user-every", type=float, default=1.0)
    parser.add_argument("--prelabel-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--keep-empty", type=float, default=0.05, help="share of empty CCTV frames kept as negatives")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--status", default="needs_prelabel", help="pool rows to pre-label")
    parser.add_argument("--source", default=None)
    parser.add_argument("--batch-size", type=int, default=300, help="frames per labelling batch zip")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.root)
    if args.step == "labeled":
        for source in args.sources:
            started = time.perf_counter()
            if source == "zenodo":
                count = zenodo(pool, args.work, args.limit)
            elif source == "granada":
                count = granada(pool, args.work, args.limit)
            elif source == "openimages":
                count = openimages(pool, args.work, args.limit or args.oi_per_class,
                                   min(args.limit or 10 ** 9, args.oi_negatives_per_class), args.oi_splits)
            elif source == "coco":
                count = coco(pool, args.work, args.limit or args.coco_limit)
            elif source == "coco_neg":
                count = coco_negatives(pool, args.work, args.limit or args.coco_neg_limit)
            else:
                count = simuletic(pool, args.work)
            pool.save()
            print(f"{source}: +{count} images in {time.perf_counter() - started:.0f}s", flush=True)
    elif args.step == "frames":
        print(frames_stage(pool, args.work, args.limit or args.kinetics_per_class, args.frames_per_clip,
                           args.ashen_every, args.user_every, args.limit, args.kinetics_classes))
    elif args.step == "prelabel":
        prelabel(pool, args.prelabel_model, args.box_threshold, args.text_threshold, args.keep_empty, args.batch,
                 args.status, args.source)
    elif args.step == "batches":
        batches(pool, args.batch_size)
    stats(pool)


if __name__ == "__main__":
    main()
