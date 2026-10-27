"""Check archive integrity and frame seeking without changing data or labels."""

import argparse
import json
import re
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np


def check_case(archive, dataset, key, second):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or second < 0:
        raise ValueError("Invalid source case")
    matches = [
        path
        for path in (dataset / "videos").rglob(key + ".*")
        if path.suffix.lower() in {".mp4", ".avi"}
    ]
    if len(matches) != 1:
        raise ValueError("Source video missing or ambiguous")
    path = matches[0]
    videos = [
        name
        for name in archive.namelist()
        if "/videos/" in "/" + name and Path(name).name == path.name
    ]
    annotations = [
        name
        for name in archive.namelist()
        if "/annotation/" in "/" + name and Path(name).name == key + ".csv"
    ]
    if len(videos) != 1 or len(annotations) != 1:
        raise ValueError("Archive source/annotation missing or ambiguous")
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    sequential = cv2.VideoCapture(str(path))
    fps = sequential.get(cv2.CAP_PROP_FPS)
    target = round(second * fps)
    try:
        if fps <= 0:
            raise ValueError("Invalid video FPS")
        for index in range(target):
            if not sequential.grab():
                raise ValueError("Video ends before diagnostic frame")
        ok, expected = sequential.read()
        if not ok:
            raise ValueError("Cannot read sequential frame")
    finally:
        sequential.release()
    seek = cv2.VideoCapture(str(path))
    try:
        seek.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, actual = seek.read()
        if not ok:
            raise ValueError("Cannot read seek frame")
        timestamp = seek.get(cv2.CAP_PROP_POS_MSEC) / 1000
    finally:
        seek.release()
    return {
        "video": key,
        "second": second,
        "archive_video_crc_matches": crc == archive.getinfo(videos[0]).CRC,
        "archive_annotation_matches": (
            dataset / "annotation" / (key + ".csv")
        ).read_bytes()
        == archive.read(annotations[0]),
        "seek_equals_sequential": bool(np.array_equal(expected, actual)),
        "mean_pixel_difference": float(cv2.absdiff(expected, actual).mean()),
        "seek_timestamp_seconds": timestamp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        values = [
            check_case(
                archive,
                args.dataset_root,
                case.rsplit(":", 1)[0],
                float(case.rsplit(":", 1)[1]),
            )
            for case in args.case
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
