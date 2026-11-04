"""Derive short fight-to-calm review clips from Re-DID's frame annotations.

Download the two original archives from https://loki.disi.unitn.it/~ReDID/ .
Only derived clips in the local review directory are written; source archives
and videos are never committed to the repository. A fight's end is considered
safe for a negative label only after every overlapping fight/separating event
has ended, and only until the next annotated event begins.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import cv2


VIDEO_SHA = "b8874d3cc07b8310398e27438d64f399505d2a85daec229d9c8d2cd91773b346"
ACTION_SHA = "0b7571357b01c29def9009b3335b73c7251aa818fc24eb15c910d2f40b8d062b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotations(data):
    events = []
    for line in data.decode("utf-8").splitlines():
        start, end, first, action, second = line.split()
        if action not in {"fight", "separating"} or int(start) < 0 or int(end) <= int(start):
            raise ValueError("Invalid Re-DID annotation: " + line)
        events.append((int(start), int(end), action))
    return sorted(events)


def active_components(events):
    """Merge overlapping activity, retaining the fights inside each component."""
    result = []
    for start, end, action in events:
        if not result or start > result[-1][1]:
            result.append([start, end, []])
        else:
            result[-1][1] = max(result[-1][1], end)
        if action == "fight":
            result[-1][2].append((start, end))
    return result


def transitions(events, frame_count, fps):
    """Return safe (clip start, clip end, fight, after-fight) times in seconds."""
    components = active_components(events)
    result = []
    for index, (_, boundary, fights) in enumerate(components):
        if not fights:
            continue
        next_start = components[index + 1][0] if index + 1 < len(components) else frame_count
        gap = (min(next_start, frame_count) - boundary) / fps
        if gap < 2.25:
            continue
        fight_start, fight_end = max(fights, key=lambda item: item[1])
        # Long separating-only tails are uncertain, so leave them for review.
        if (boundary - fight_end) / fps > 0.75:
            continue
        positive_start = max(fight_start / fps, boundary / fps - 3.0)
        positive_end = fight_end / fps - 0.10
        if positive_end - positive_start < 0.8:
            continue
        clip_start = max(0.0, positive_start - 0.20)
        after_start = boundary / fps + 0.15
        after_end = boundary / fps + min(3.0, gap - 0.20)
        if after_end - after_start < 1.5:
            continue
        clip_end = min(frame_count / fps, after_end + 0.10)
        result.append((clip_start, clip_end, positive_start, positive_end,
                       after_start, after_end))
    return result


def split_for_video(video_id):
    # One source video, with all its fight instances, stays in one split.
    rank = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 10
    return "development" if rank == 0 else "test" if rank in {1, 3} else "train"


def build(video_archive, action_archive, review_root, *, check_hash=True):
    if check_hash and (sha256(video_archive) != VIDEO_SHA or sha256(action_archive) != ACTION_SHA):
        raise ValueError("Re-DID archive checksum mismatch")
    review_root.mkdir(parents=True, exist_ok=True)
    manifest_path = review_root / "review.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"schema_version": 1, "sources": []})
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported review schema")
    existing = [item for item in manifest["sources"]
                if not item["group"].startswith("redid/")]
    created = []
    with zipfile.ZipFile(video_archive) as videos, zipfile.ZipFile(action_archive) as labels:
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for source in sorted(item for item in videos.namelist() if item.endswith(".avi")):
                video_id = Path(source).stem
                label_name = f"actions_annotations/{video_id}_act.txt"
                if label_name not in labels.namelist():
                    raise ValueError("Missing Re-DID annotation: " + video_id)
                if video_id.startswith("pos_") is False:
                    continue
                local = temp / (video_id + ".avi")
                local.write_bytes(videos.read(source))
                capture = cv2.VideoCapture(str(local))
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                if not 1 <= fps <= 120 or frame_count < 3:
                    raise ValueError("Cannot decode Re-DID video: " + video_id)
                events = annotations(labels.read(label_name))
                if any(end > frame_count for _, end, _ in events):
                    raise ValueError("Annotation exceeds video: " + video_id)
                for ordinal, values in enumerate(transitions(events, frame_count, fps), 1):
                    clip_start, clip_end, pos_start, pos_end, neg_start, neg_end = values
                    relative = Path("videos") / "redid" / f"{video_id}_transition_{ordinal:02d}.mp4"
                    destination = review_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        partial = destination.with_suffix(".partial.mp4")
                        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                   "-i", str(local), "-ss", f"{clip_start:.6f}",
                                   "-t", f"{clip_end - clip_start:.6f}",
                                   "-vf", "scale=640:-2", "-an", "-c:v", "libx264",
                                   "-preset", "fast", "-crf", "23", str(partial)]
                        subprocess.run(command, check=True)
                        partial.replace(destination)
                    clip = cv2.VideoCapture(str(destination))
                    actual_fps = clip.get(cv2.CAP_PROP_FPS)
                    actual_count = clip.get(cv2.CAP_PROP_FRAME_COUNT)
                    clip.release()
                    actual_duration = actual_count / actual_fps
                    if actual_duration + 0.1 < neg_end - clip_start:
                        raise ValueError("Encoded clip ends before label: " + str(destination))
                    created.append({
                        "video": relative.as_posix(), "sha256": sha256(destination),
                        "group": "redid/" + video_id, "split": split_for_video(video_id),
                        "duration_seconds": round(actual_duration, 3),
                        "source": {"dataset": "Re-DID", "video": video_id,
                                   "annotation": label_name, "fps": fps,
                                   "clip_start_seconds": round(clip_start, 6)},
                        "intervals": [
                            {"start_seconds": round(pos_start - clip_start, 6),
                             "end_seconds": round(pos_end - clip_start, 6),
                             "label": 1, "action": "fight"},
                            {"start_seconds": round(neg_start - clip_start, 6),
                             "end_seconds": round(neg_end - clip_start, 6),
                             "label": 0, "action": "after_fight"}],
                    })
                local.unlink()
    manifest["sources"] = existing + created
    temporary = manifest_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(manifest_path)
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-archive", type=Path, required=True)
    parser.add_argument("--action-archive", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args()
    clips = build(args.video_archive, args.action_archive, args.review_root)
    from collections import Counter
    print("Re-DID transitions:", len(clips), dict(Counter(x["split"] for x in clips)))
    print("Review manifest:", args.review_root / "review.json")


if __name__ == "__main__":
    main()
