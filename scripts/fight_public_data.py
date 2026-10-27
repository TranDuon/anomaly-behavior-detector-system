"""Public clip-labelled sources, filtered actions and leakage-aware partitions."""

import argparse
import csv
import hashlib
import io
import json
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw

from anomaly.fight_data import rlvs_items
from fight.vision import ClipConfig
from scripts.fight_pilot import write_json
from scripts.fight_pilot import read_frames

SOURCES = {
    "rlvs": "https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset",
    "airtlab": "https://codeload.github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/zip/refs/heads/master",
}
SPLITS = ("train", "development", "calibration", "test")


def validate_zip(path, minimum_videos):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                name.is_absolute()
                or ".." in name.parts
                or any(":" in part for part in name.parts)
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member")
        if sum(member.file_size for member in members) > 40 * 1024**3:
            raise ValueError("Archive exceeds 40 GiB extraction limit")
        videos = sum(
            Path(member.filename).suffix.lower() in {".mp4", ".avi", ".mov"}
            for member in members
        )
        if videos < minimum_videos or archive.testzip() is not None:
            raise ValueError(f"Incomplete/corrupt dataset: {videos} videos")
    return videos


def acquire(archives, destination):
    archives.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = archives / f"{name}_public_v1.zip"
        minimum = 1900 if name == "rlvs" else 350
        if not path.exists():
            partial = path.with_suffix(".zip.partial")
            subprocess.run(
                [
                    "wget",
                    "-c",
                    "--timeout=60",
                    "--tries=2",
                    "--progress=dot:giga",
                    url,
                    "-O",
                    str(partial),
                ],
                check=True,
                timeout=1800,
            )
            validate_zip(partial, minimum)
            partial.replace(path)
        count = validate_zip(path, minimum)
        target = destination / name
        marker = target / ".extracted.json"
        signature = {"archive_bytes": path.stat().st_size, "videos": count}
        if not marker.exists() or json.loads(marker.read_text()) != signature:
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            write_json(marker, signature)
        print(f"SOURCE_READY {name}: {count} videos", flush=True)


def airtlab_items(root):
    annotations = list(root.rglob("violent-action-classes.csv"))
    if len(annotations) != 1:
        raise ValueError("Expected one AIRTLab action CSV")
    base = annotations[0].parent
    action_rows = csv.reader(
        io.StringIO(annotations[0].read_text(encoding="utf-8-sig")), delimiter=";"
    )
    next(action_rows)
    actions = {
        parts[0].strip(): {action.strip() for action in parts[1].split(",")}
        for parts in action_rows
        if len(parts) >= 2
    }
    selected = []
    for category, label in (("non-violent", 0), ("violent", 1)):
        for path in sorted((base / category).rglob("*.mp4")):
            tags = actions.get(path.name, set()) if label else set()
            if label and (
                tags & {"club", "stab", "gunshot"}
                or not tags & {"fight", "punch", "kick", "slap"}
            ):
                continue
            selected.append(
                {
                    "path": str(path),
                    "video": "airtlab/" + path.relative_to(base).as_posix(),
                    "dataset": "airtlab",
                    "label": label,
                    "performance": f"airtlab/{category}/{path.name}",
                    "actions": sorted(tags),
                }
            )
    return selected


def fingerprint(row, config):
    path = Path(row["path"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    capture = cv2.VideoCapture(str(path))
    hashes = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError(f"Invalid FPS: {row['video']}")
        span = round((config.frames - 1) * fps / config.fps) + 1
        if count < span:
            return None
        for fraction in (0.2, 0.5, 0.8):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round((count - 1) * fraction))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {row['video']}")
            tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
            hashes.append(
                int.from_bytes(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes(), "big")
            )
    finally:
        capture.release()
    return {
        **row,
        "sha256": digest.hexdigest(),
        "scene_hashes": hashes,
        "fps": fps,
        "start": (count - span) // 2,
        "key": hashlib.sha256(row["video"].encode()).hexdigest()[:16],
        "frames": count,
    }


def assign_groups(rows, seed=73):
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        for previous in range(index):
            other = rows[previous]
            paired = row.get("performance") and row.get("performance") == other.get(
                "performance"
            )
            near = (
                row["dataset"] == other["dataset"] == "rlvs"
                and sum(
                    (first ^ second).bit_count() <= 4
                    for first, second in zip(row["scene_hashes"], other["scene_hashes"])
                )
                >= 2
            )
            if paired or row["sha256"] == other["sha256"] or near:
                parents[root(index)] = root(previous)
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    accepted, conflicts = [], []
    for members in groups.values():
        group = hashlib.sha256(
            "\n".join(sorted(row["video"] for row in members)).encode()
        ).hexdigest()[:24]
        labels_by_hash = {}
        for row in members:
            labels_by_hash.setdefault(row["sha256"], set()).add(row["label"])
        if any(len(labels) > 1 for labels in labels_by_hash.values()):
            conflicts.extend(
                {**row, "reason": "duplicate_group_label_conflict"} for row in members
            )
            continue
        fraction = (
            int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16)
            / 16**12
        )
        split = (
            "train"
            if fraction < 0.7
            else (
                "development"
                if fraction < 0.8
                else "calibration" if fraction < 0.9 else "test"
            )
        )
        accepted.extend(
            {**row, "scene_group": group, "split": split} for row in members
        )
    for split in SPLITS:
        if any(
            sum(row["split"] == split and row["label"] == label for row in accepted) < 4
            for label in (0, 1)
        ):
            raise ValueError(f"Insufficient grouped classes in {split}")
    return accepted, conflicts


def build_manifest(data, output):
    config = ClipConfig(region_mode="tubes", trim_black_borders=True, version=5)
    originals = [
        {
            "path": str(item.path),
            "video": item.uid,
            "dataset": "rlvs",
            "label": item.clip_label,
        }
        for item in rlvs_items(data / "rlvs")
    ]
    originals.extend(airtlab_items(data / "airtlab"))
    usable, excluded = [], []
    for index, row in enumerate(originals):
        prepared = fingerprint(row, config)
        if prepared is None:
            excluded.append({**row, "reason": "shorter_than_model_window"})
        else:
            usable.append(prepared)
        if (index + 1) % 100 == 0:
            print(f"FINGERPRINT {index + 1}/{len(originals)}", flush=True)
    rows, conflicts = assign_groups(usable)
    manifest = {
        "schema_version": 1,
        "rows": rows,
        "excluded": excluded + conflicts,
        "clip": config.to_dict(),
        "seed": 73,
        "label_provenance": "Publisher clip labels; AIRTLab weapons excluded. Not fully human-reviewed active-fight temporal gold.",
        "grouping_note": "Exact SHA256, RLVS >=2 of 3 dHashes within 4 bits, paired AIRTLab camera performance. AIRTLab shares actors/room across performances; this is not actor/room-independent. Heuristic cannot prove absence of source overlap.",
        "phone_acceptance": False,
        "approved_for_demo": False,
        "counts": {
            split: {
                "clips": sum(row["split"] == split for row in rows),
                "positive": sum(
                    row["split"] == split and row["label"] == 1 for row in rows
                ),
                "groups": len(
                    {row["scene_group"] for row in rows if row["split"] == split}
                ),
            }
            for split in SPLITS
        },
    }
    write_json(output / "manifest.json", manifest)
    audit_manifest(manifest, output)
    print("MANIFEST " + json.dumps(manifest["counts"]), flush=True)
    return manifest


def audit_manifest(manifest, output):
    import random

    generator = random.Random(73)
    chosen = []
    for dataset in ("rlvs", "airtlab"):
        for label in (0, 1):
            pool = [
                row
                for row in manifest["rows"]
                if row["dataset"] == dataset
                and row["label"] == label
                and row["split"] in {"train", "development"}
            ]
            chosen.extend(generator.sample(pool, min(10, len(pool))))
    generator.shuffle(chosen)
    directory = output / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    mapping = []
    config = ClipConfig(**manifest["clip"])
    for page_start in range(0, len(chosen), 4):
        page = Image.new("RGB", (1080, 4 * 145), "white")
        draw = ImageDraw.Draw(page)
        for row_index, row in enumerate(chosen[page_start : page_start + 4]):
            audit_id = f"P{page_start + row_index + 1:03d}"
            frames = read_frames(row, config)
            draw.text((8, row_index * 145 + 4), audit_id, fill="black")
            for column, frame_index in enumerate(
                np.linspace(0, len(frames) - 1, 6).astype(int)
            ):
                image = Image.fromarray(
                    cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)
                )
                image.thumbnail((175, 116))
                page.paste(image, (column * 180, row_index * 145 + 24))
            mapping.append({"audit_id": audit_id, **row})
        page.save(directory / f"page{page_start // 4 + 1:02d}.jpg")
    write_json(directory / "private_mapping.json", mapping)
    with zipfile.ZipFile(
        output / "blinded_training_audit.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(directory.glob("page*.jpg")):
            archive.write(path, path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.archives, args.data)
    if not (args.output / "manifest.json").exists():
        build_manifest(args.data, args.output)


if __name__ == "__main__":
    main()
