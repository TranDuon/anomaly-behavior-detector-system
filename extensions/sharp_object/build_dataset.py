"""S2: merge the public pool and reviewed Vietnamese labels into one Ultralytics dataset.

Inputs
  --pool   pool built by collect.py (only rows with status "labeled" are used: public sets with
           boxes and Open Images hard negatives without boxes)
  --vn     reviewed Vietnamese data, any number of zips or folders. Accepted layouts:
             * Ultralytics:   images/<split>/x.jpg + labels/<split>/x.txt (+ data.yaml with names)
             * CVAT YOLO 1.1: obj.names + obj_train_data/x.jpg + obj_train_data/x.txt
           Only images that have a label file are used, so folders such as unlabeled/ are ignored.
           Empty label files are skipped (an unreviewed frame is not a reliable negative) unless
           --vn-keep-empty is given. Class names are mapped by name onto class_map.json; boxes of
           unknown classes are dropped. A later --vn input replaces an identical image of an earlier one.

Splits are made by source group (same video / article / case never crosses train/val/test).
Vietnamese groups get their own val/test share so the Vietnamese score is measured separately.
Groups come from a manifest.csv inside the input when it has one, else from the file name.

Output (--out):
  images/{train,val,test}/  labels/{train,val,test}/
  train.txt (Vietnamese train images repeated --vn-repeat times)  val.txt  test.txt
  data.yaml        all splits
  data_vn.yaml     val/test restricted to Vietnamese images
  manifest.csv     one row per image: split, source, group, boxes, classes
  summary.json
"""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
CLASS_MAP = json.loads((ROOT / "datasets" / "sharp_object" / "class_map.json").read_text(encoding="utf-8"))
CLASSES = CLASS_MAP["classes"]
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ reading zips and folders alike
class Tree:
    """Read-only view of a zip or a folder with POSIX relative paths."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_file():
            self.zip = zipfile.ZipFile(self.path)
            self.names = [n for n in self.zip.namelist() if not n.endswith("/")]
        else:
            self.zip = None
            self.names = [p.relative_to(self.path).as_posix() for p in sorted(self.path.rglob("*")) if p.is_file()]
        self.name_set = set(self.names)

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()

    def text(self, name):
        return self.read(name).decode("utf-8-sig", "replace")


def yaml_names(text):
    """names from a data.yaml (dict or list form) without needing PyYAML."""
    match = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if match:
        return [n.strip().strip("'\"") for n in match.group(1).split(",") if n.strip()]
    block = re.search(r"names:\s*\n((?:[ \t]+.+\n?)+)", text)
    names = {}
    for line in (block.group(1) if block else "").splitlines():
        keyed = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
        listed = re.match(r"\s*-\s*(.+)", line)
        if keyed:
            names[int(keyed.group(1))] = keyed.group(2).strip().strip("'\"")
        elif listed:
            names[len(names)] = listed.group(1).strip().strip("'\"")
    return [names[i] for i in sorted(names)]


def class_names_for(tree, image_name):
    """Nearest data.yaml / obj.names / classes.txt at or above the image's folder."""
    folder = PurePosixPath(image_name).parent
    while True:
        prefix = "" if str(folder) == "." else str(folder) + "/"
        for candidate in ("data.yaml", "dataset.yaml"):
            if prefix + candidate in tree.name_set:
                return yaml_names(tree.text(prefix + candidate))
        for candidate in ("obj.names", "classes.txt"):
            if prefix + candidate in tree.name_set:
                return [line.strip() for line in tree.text(prefix + candidate).splitlines() if line.strip()]
        if str(folder) == ".":
            return list(CLASSES)
        folder = folder.parent


def label_name_for(tree, image_name):
    path = PurePosixPath(image_name)
    candidates = [str(path.with_suffix(".txt"))]
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            swapped = parts[:index] + ["labels"] + parts[index + 1:]
            candidates.insert(0, str(PurePosixPath(*swapped).with_suffix(".txt")))
            break
    return next((c for c in candidates if c in tree.name_set), None)


def manifest_groups(tree):
    """{basename stem: group} from any csv in the input that names images and their source group."""
    groups = {}
    for name in tree.names:
        if not name.lower().endswith(".csv"):
            continue
        try:
            rows = list(csv.DictReader(io.StringIO(tree.text(name))))
        except csv.Error:
            continue
        if not rows:
            continue
        group_key = next((k for k in ("source_group", "group", "case") if k in rows[0]), None)
        image_keys = [k for k in ("package_image", "image", "original_image", "file", "filename") if k in rows[0]]
        if not group_key or not image_keys:
            continue
        for row in rows:
            for key in image_keys:
                if row.get(key) and row.get(group_key):
                    groups.setdefault(PurePosixPath(row[key]).stem, row[group_key])
    return groups


def group_from_name(stem):
    """Fallback when no manifest: drop packaging prefixes and frame/photo counters."""
    name = re.sub(r"^(pool|batch\w*?|b\d+|mixed|video|telegram_\d+)_\d+_", "", stem)
    name = re.sub(r"^.*__", "", name)  # CVAT flattened path: images__vn_news__x.jpg
    base = re.sub(r"(_\d+(p\d+)?)+$", "", name)
    if "_" not in base:  # e.g. news_017: one article per number, keep it whole
        return "name:" + name
    return "name:" + base


def parse_label(text, names):
    boxes, dropped = [], Counter()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        index = int(float(parts[0]))
        name = names[index] if index < len(names) else str(index)
        if name not in CLASS_ID:
            dropped[name] += 1
            continue
        x, y, w, h = (min(1.0, max(0.0, float(v))) for v in parts[1:])
        if w > 0 and h > 0:
            boxes.append((CLASS_ID[name], x, y, w, h))
    return boxes, dropped


# ------------------------------------------------------------------ collecting items
def vn_items(inputs, keep_empty):
    """[(key, read(), boxes, group, origin)] newest input wins for identical images."""
    by_hash, report = {}, []
    for order, path in enumerate(inputs):
        tree = Tree(path)
        groups = manifest_groups(tree)
        counts = Counter()
        dropped = Counter()
        for name in tree.names:
            if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES or "/qc/" in "/" + name:
                continue
            label = label_name_for(tree, name)
            if label is None:
                counts["no_label"] += 1
                continue
            boxes, lost = parse_label(tree.text(label), class_names_for(tree, name))
            dropped += lost
            if not boxes and not keep_empty:
                counts["empty_skipped"] += 1
                continue
            data = tree.read(name)
            digest = hashlib.md5(data).hexdigest()
            stem = PurePosixPath(name).stem
            group = groups.get(stem) or group_from_name(stem)
            if digest in by_hash:
                counts["replaced_duplicate"] += 1
            by_hash[digest] = {"data": data, "boxes": boxes, "group": "vn:" + group, "origin": f"{Path(path).name}:{name}",
                               "stem": re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:80], "suffix": PurePosixPath(name).suffix.lower()}
            counts["used"] += 1
        report.append({"input": str(path), **counts, "dropped_unknown_classes": dict(dropped),
                       "groups_from_manifest": len(groups)})
    return list(by_hash.values()), report


def pool_items(pool_root):
    pool_root = Path(pool_root)
    items = []
    with (pool_root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "labeled":
                continue
            image = pool_root / row["image"]
            label = pool_root / ("labels/" + row["image"][len("images/"):-len(".jpg")] + ".txt")
            if not image.is_file() or not label.is_file():
                continue
            boxes, _ = parse_label(label.read_text(encoding="utf-8"), CLASSES)
            items.append({"path": image, "boxes": boxes, "group": row["group"] or row["image"], "source": row["source"],
                          "stem": f"{row['source']}__{image.stem}"[:100], "suffix": ".jpg"})
    return items


def split_of(group, val, test):
    """Deterministic split by group hash."""
    point = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if point < test else "val" if point < test + val else "train"


def vn_splits(groups, val, test):
    """Hash order over Vietnamese groups, but at least one group each in val and test once there are >= 3."""
    ordered = sorted(groups, key=lambda g: hashlib.sha1(g.encode("utf-8")).hexdigest())
    if len(ordered) < 3:
        return {g: "train" for g in ordered}
    n_test = max(1, round(len(ordered) * test))
    n_val = max(1, round(len(ordered) * val))
    return {g: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, g in enumerate(ordered)}


# ------------------------------------------------------------------ writing
def write_dataset(out, public, vietnamese, vn_assignment, args):
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    rows, lists = [], defaultdict(list)
    used_names = set()

    def place(item, split, source, payload):
        name = item["stem"]
        while name in used_names:
            name += "_x"
        used_names.add(name)
        image_path = out / "images" / split / (name + item["suffix"])
        payload(image_path)
        (out / "labels" / split / (name + ".txt")).write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in item["boxes"]), encoding="utf-8")
        repeat = args.vn_repeat if (source == "vn" and split == "train") else 1
        lists[split] += [str(image_path.resolve())] * repeat
        if source == "vn" and split != "train":
            lists["vn_" + split].append(str(image_path.resolve()))
        rows.append({"image": image_path.relative_to(out).as_posix(), "split": split, "source": source,
                     "group": item["group"], "boxes": len(item["boxes"]),
                     "classes": ";".join(sorted({CLASSES[b[0]] for b in item["boxes"]})),
                     "origin": item.get("origin", "")})

    for item in public:
        split = split_of(item["group"], args.public_val, args.public_test)
        place(item, split, item["source"], lambda target, item=item: _link_or_copy(item["path"], target))
    for item in vietnamese:
        place(item, vn_assignment[item["group"]], "vn", lambda target, item=item: target.write_bytes(item["data"]))

    for split in ("train", "val", "test", "vn_val", "vn_test"):
        (out / f"{split}.txt").write_text("\n".join(lists[split]) + ("\n" if lists[split] else ""), encoding="utf-8")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
                                   f"names:\n{names}", encoding="utf-8")
    vn_val = "vn_val.txt" if lists["vn_val"] else "val.txt"
    vn_test = "vn_test.txt" if lists["vn_test"] else vn_val
    (out / "data_vn.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: train.txt\nval: {vn_val}\n"
                                      f"test: {vn_test}\nnames:\n{names}", encoding="utf-8")
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["image", "split", "source", "group", "boxes", "classes", "origin"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, lists


def _link_or_copy(source, target):
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=None, help="public pool from collect.py")
    parser.add_argument("--vn", type=Path, nargs="*", default=[], help="reviewed Vietnamese zips/folders, oldest first")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vn-keep-empty", action="store_true", help="use empty Vietnamese label files as negatives")
    parser.add_argument("--vn-repeat", type=int, default=6, help="times each Vietnamese train image is listed")
    parser.add_argument("--vn-val", type=float, default=0.2)
    parser.add_argument("--vn-test", type=float, default=0.2)
    parser.add_argument("--public-val", type=float, default=0.08)
    parser.add_argument("--public-test", type=float, default=0.07)
    parser.add_argument("--vn-train-groups", nargs="*", default=[],
                        help="Vietnamese groups containing any of these substrings always go to train "
                             "(e.g. self-filmed videos of one room that must not be used for evaluation)")
    args = parser.parse_args()

    public = pool_items(args.pool) if args.pool else []
    vietnamese, report = vn_items(args.vn, args.vn_keep_empty)
    forced = {item["group"] for item in vietnamese if any(k in item["group"] for k in args.vn_train_groups)}
    assignment = vn_splits({item["group"] for item in vietnamese} - forced, args.vn_val, args.vn_test)
    assignment.update({group: "train" for group in forced})
    if forced:
        print(f"forced to train: {sorted(forced)}", flush=True)
    if not public and not vietnamese:
        raise SystemExit("no labelled images found")
    rows, lists = write_dataset(args.out, public, vietnamese, assignment, args)

    summary = {"classes": CLASSES, "vn_inputs": report, "vn_repeat": args.vn_repeat,
               "images": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
               "train_list_length": len(lists["train"])}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        summary[split] = {
            "by_source": dict(Counter(r["source"] for r in part)),
            "negatives": sum(r["boxes"] == 0 for r in part),
            "boxes_by_class": dict(Counter(c for r in part for c in r["classes"].split(";") if c)),
            "vn_images": sum(r["source"] == "vn" for r in part),
            "vn_groups": len({r["group"] for r in part if r["source"] == "vn"}),
        }
    warnings = []
    if summary["val"]["vn_images"] < 10 or summary["test"]["vn_images"] < 10:
        warnings.append("Vietnamese val/test have < 10 images: their scores are indicative only.")
    if not any(r["boxes"] == 0 for r in rows if r["split"] == "val"):
        warnings.append("val has no negatives: the alert threshold cannot be calibrated on false alarms.")
    summary["warnings"] = warnings
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
