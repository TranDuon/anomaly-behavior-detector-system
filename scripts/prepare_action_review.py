"""Make a local review pack from uploaded videos without guessing action labels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(paths, output):
    output.mkdir(parents=True, exist_ok=True)
    videos = output / "videos"
    videos.mkdir(exist_ok=True)
    entries, seen = [], set()
    for path in paths:
        checksum = sha256(path)
        if checksum in seen:
            continue
        seen.add(checksum)
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not 1 <= fps <= 120 or count < 1:
            raise ValueError("Cannot decode video: " + str(path))
        duration = count / fps
        tiles = []
        for second in np.linspace(0, max(0, duration - 1 / fps), 12):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            thumbnail = cv2.resize(frame, (320, 180))
            cv2.rectangle(thumbnail, (0, 0), (120, 26), (0, 0, 0), -1)
            cv2.putText(thumbnail, f"{second:.1f}s", (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(thumbnail)
        capture.release()
        if not tiles:
            raise ValueError("Cannot sample video: " + str(path))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        name = checksum[:16] + path.suffix.lower()
        destination = videos / name
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
        cv2.imwrite(str(output / (checksum[:16] + ".jpg")), grid)
        entries.append({
            "video": "videos/" + name, "sha256": checksum,
            "group": checksum[:16], "split": "test", "duration_seconds": round(duration, 3),
            "intervals": [],
        })
    (output / "review.json").write_text(
        json.dumps({"schema_version": 1, "sources": entries}, indent=2,
                   ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print("Unique videos:", prepare(args.videos, args.output))
    print("Review manifest:", args.output / "review.json")


if __name__ == "__main__":
    main()
