"""Render timestamp contact sheets for conservative review of AIRTLab lead-ins."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


WEAPON_TAGS = {"club", "stab", "gunshot"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--times", type=float, nargs="+", default=[0, 0.5, 1, 1.5])
    args = parser.parse_args()
    base = args.dataset / "violence-detection-dataset"
    tags_path = base / "violent-action-classes.csv"
    tags = {}
    with tags_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter=";"):
            if row and row[0].strip().lower() != "file":
                tags[Path(row[0]).stem] = {
                    item.strip().lower() for item in row[1].split(",")
                }
    clips = [
        clip
        for clip, actions in sorted(tags.items(), key=lambda item: int(item[0]))
        if not (actions & WEAPON_TAGS)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for sheet_index, start in enumerate(range(0, len(clips), args.per_sheet)):
        rows = []
        for clip in clips[start : start + args.per_sheet]:
            cells = []
            path = base / "violent" / args.camera / f"{clip}.mp4"
            capture = cv2.VideoCapture(str(path))
            for timestamp in args.times:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = capture.read()
                if not success:
                    frame = np.zeros((135, 240, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (240, 135))
                cv2.putText(
                    frame,
                    f"{clip} @ {timestamp:.1f}s",
                    (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cells.append(frame)
            capture.release()
            rows.append(np.hstack(cells))
        target = args.output / f"sheet_{sheet_index:02}.jpg"
        if rows:
            cv2.imwrite(str(target), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(target)


if __name__ == "__main__":
    main()
