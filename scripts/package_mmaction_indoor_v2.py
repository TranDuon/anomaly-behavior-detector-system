"""Build the reproducible indoor MMAction v2 media pack from AIRTLab and SBU.

AIRTLab publisher clip tags provide broad labels. A small, explicit table below adds
AI-assisted temporal boundaries for staged unarmed clips that visibly contain a
non-violent lead-in. These labels are intentionally not described as human gold.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2

from scripts.build_mmaction_indoor_manifest import build, write


WEAPON_TAGS = {"club", "stab", "gunshot"}
SBU_ACTIONS = {
    "01": "approach",
    "02": "depart",
    "03": "fight",  # pushing
    "04": "fight",  # kicking
    "05": "fight",  # punching
    "06": "give_exchange",
    "07": "hug",
    "08": "handshake_greet",
}
NONVIOLENT_PRIORITY = (
    ("hug", "hug"),
    ("highfive", "highfive_clap"),
    ("handshake", "handshake_greet"),
    ("greet", "handshake_greet"),
    ("walk", "walk_together"),
    ("handgestures", "gesture"),
    ("jump", "solo_motion"),
)

# Contact sheets sampled at 0.5 s were inspected on 2026-10-29. Boundaries are
# conservative approximations and are transferred to the synchronized second view.
AIRTLAB_PREFIGHT = {
    "11": 2.25,
    "12": 2.0,
    "1": 1.5,
    "2": 1.3,
    "3": 0.5,
    "7": 1.5,
    "22": 2.25,
    "26": 2.25,
    "13": 1.0,
    "15": 0.8,
    "37": 1.4,
    "38": 1.8,
    "39": 0.8,
    "41": 0.4,
    "46": 1.5,
    "47": 2.8,
    "58": 2.0,
    "73": 2.0,
    "87": 2.0,
    "99": 2.0,
    "96": 2.0,
    "102": 2.5,
    "103": 2.5,
    "107": 2.0,
    "109": 2.5,
    "112": 2.5,
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def video_duration(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Cannot read video timebase: {path}")
    return frames / fps


def assign_stratified_splits(records):
    """Keep camera/performance groups together while giving every split both labels.

    AIRTLab performances are either violent or non-violent, so assigning them with
    one hash stream can accidentally leave a small evaluation split with one class.
    SBU performance pairs contain several action classes and are assigned by pair.
    """
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    strata = {}
    for group, members in groups.items():
        dataset = members[0]["dataset"]
        labels = {member["source_label"] for member in members}
        label = next(iter(labels)) if dataset == "airtlab" and len(labels) == 1 else "mixed"
        strata.setdefault((dataset, label), []).append(group)
    for _, names in strata.items():
        ordered = sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
        count = len(ordered)
        train = round(count * 0.60)
        development = round(count * 0.15)
        calibration = round(count * 0.15)
        # Every stratum in this package has at least four performance groups.
        cuts = (train, train + development, train + development + calibration)
        for index, group in enumerate(ordered):
            split = (
                "train"
                if index < cuts[0]
                else "development"
                if index < cuts[1]
                else "calibration"
                if index < cuts[2]
                else "test"
            )
            for record in groups[group]:
                record["split"] = split
    return records


def read_tags(path):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter=";"):
            if not row or row[0].strip().lower() == "file":
                continue
            result[Path(row[0].strip()).stem] = {
                item.strip().lower() for item in row[1].split(",")
            }
    return result


def transcode(source, target, fps=15):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    partial = target.with_suffix(".partial.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"fps={fps},scale=640:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        ],
        check=True,
    )
    partial.replace(target)


def encode_rgb_sequence(frames, target, fps=15):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    if not frames:
        raise ValueError(f"No RGB frames for {target}")
    list_path = target.with_suffix(".frames.txt")
    list_path.write_text(
        "".join(f"file '{path.as_posix()}'\nduration {1 / fps:.9f}\n" for path in frames)
        + f"file '{frames[-1].as_posix()}'\n",
        encoding="utf-8",
    )
    partial = target.with_suffix(".partial.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-an",
                "-vf",
                "scale=640:-2:flags=lanczos",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                str(partial),
            ],
            check=True,
        )
        partial.replace(target)
    finally:
        list_path.unlink(missing_ok=True)


def media_name(video):
    return "media/" + hashlib.sha256(video.encode()).hexdigest()[:24] + ".mp4"


def source_record(video, dataset, group, source, preview, split, action, method, intervals=None):
    duration = video_duration(preview)
    intervals = intervals or [{"start_seconds": 0, "end_seconds": duration, "action": action}]
    cleaned = []
    for item in intervals:
        start = max(0.0, min(float(item["start_seconds"]), duration))
        end = max(start, min(float(item["end_seconds"]), duration))
        if end - start >= 0.05:
            cleaned.append({"start_seconds": start, "end_seconds": end, "action": item["action"]})
    return {
        "video": video,
        "dataset": dataset,
        "split": split,
        "group": group,
        "source_sha256": digest(source),
        "media": media_name(video),
        "preview_sha256": digest(preview),
        "duration": duration,
        "review_method": method,
        "intervals": cleaned,
        "environment": "indoor",
        "camera_context": "fixed_room",
        "source_label": "fight" if action == "fight" else "non_fight",
    }


def airtlab_sources(root, output):
    base = root / "violence-detection-dataset"
    negative = read_tags(base / "nonviolent-action-classes.csv")
    positive = read_tags(base / "violent-action-classes.csv")
    records = []
    for label, tags_by_id in (("non-violent", negative), ("violent", positive)):
        for clip_id, tags in sorted(tags_by_id.items(), key=lambda item: int(item[0])):
            if tags & WEAPON_TAGS:
                continue
            action = "fight"
            if label == "non-violent":
                action = next(
                    (mapped for raw, mapped in NONVIOLENT_PRIORITY if raw in tags),
                    "normal",
                )
            group = f"airtlab/{label}/{clip_id}.mp4"
            for camera in ("cam1", "cam2"):
                source = base / label / camera / f"{clip_id}.mp4"
                if not source.is_file():
                    raise FileNotFoundError(source)
                video = f"airtlab/{label}/{camera}/{clip_id}.mp4"
                target = output / media_name(video)
                transcode(source, target)
                intervals = None
                method = "publisher_clip_tags"
                if label == "violent" and clip_id in AIRTLAB_PREFIGHT:
                    duration = video_duration(target)
                    onset = min(AIRTLAB_PREFIGHT[clip_id], max(0.05, duration - 0.1))
                    intervals = [
                        {"start_seconds": 0, "end_seconds": onset, "action": "stand_close" if clip_id != "37" else "approach"},
                        {"start_seconds": onset, "end_seconds": duration, "action": "fight"},
                    ]
                    method = "ai_assisted_contact_sheet"
                records.append(
                    source_record(video, "airtlab", group, source, target, "train", action, method, intervals)
                )
    return records


def sbu_sources(root, output):
    records = []
    pairs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("s"))
    for pair in pairs:
        for action_dir in sorted(path for path in pair.iterdir() if path.is_dir()):
            action = SBU_ACTIONS.get(action_dir.name)
            if action is None:
                continue
            for take in sorted(path for path in action_dir.iterdir() if path.is_dir()):
                frames = sorted(take.glob("rgb_*.png"))
                if not frames:
                    continue
                video = f"sbu/{pair.name}/{action_dir.name}/{take.name}.mp4"
                target = output / media_name(video)
                encode_rgb_sequence(frames, target)
                records.append(
                    source_record(
                        video,
                        "sbu",
                        "sbu/" + pair.name,
                        take / "skeleton_pos.txt",
                        target,
                        "train",
                        action,
                        "publisher_action_class",
                    )
                )
    return records


def make_archive(pack, archive):
    temporary = archive.with_suffix(".partial.zip")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(item for item in pack.rglob("*") if item.is_file()):
            bundle.write(path, path.relative_to(pack).as_posix())
    temporary.replace(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airtlab", type=Path, required=True)
    parser.add_argument("--sbu", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    (args.pack / "media").mkdir(parents=True, exist_ok=True)
    sources = assign_stratified_splits(
        airtlab_sources(args.airtlab, args.pack) + sbu_sources(args.sbu, args.pack)
    )
    manifest, quarantine = build({"sources": sources})
    write(args.manifest, manifest)
    write(args.quarantine, {"items": quarantine})
    shutil.copyfile(args.manifest, args.pack / "manifest.json")
    write(
        args.pack / "PACKAGE.json",
        {
            "scope_profile": "indoor_room_fight_v2",
            "manifest_sha256": digest(args.manifest),
            "readiness": manifest["readiness"],
            "summary": manifest["summary"],
            "label_note": "Publisher clip/action labels plus explicitly marked AI-assisted AIRTLab temporal lead-ins; not human gold.",
        },
    )
    (args.pack / "DATASET_SOURCES.md").write_text(
        "# Dataset sources and use notes\n\n"
        "This research pilot contains derived indoor video previews from:\n\n"
        "- AIRTLab: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos\n"
        "- SBU Kinect Interaction: https://www3.cs.stonybrook.edu/~cvl/dataset.html\n\n"
        "The archive is for research and model development. Retain the original dataset attribution and follow each publisher's terms. "
        "Do not redistribute this archive publicly without checking the source terms.\n\n"
        "AIRTLab labels use publisher clip tags. Fight lead-in intervals were estimated with AI-assisted contact-sheet review and are not human-gold annotations. "
        "Review these intervals before using model scores as evidence of deployment quality.\n",
        encoding="utf-8",
    )
    if manifest["readiness"] != "ready":
        raise ValueError("Pack is not ready: " + ", ".join(manifest["readiness_reasons"]))
    make_archive(args.pack, args.archive)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print("ARCHIVE", args.archive, digest(args.archive))


if __name__ == "__main__":
    main()
