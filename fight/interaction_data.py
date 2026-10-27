"""Temporal PKU action windows and source-preserving feasibility manifests."""

import hashlib
import re
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import cv2
import numpy as np

POSITIVE_ACTIONS = {18, 26}
NEGATIVE_ACTIONS = {5, 6, 7, 12, 13, 14, 15, 16, 17, 19, 21, 24, 25}


def action_catalog(path):
    namespace = {"sheet": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(item.itertext()) for item in shared.findall("sheet:si", namespace)
        ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    catalog = {}
    for row in sheet.findall(".//sheet:row", namespace):
        cells = {}
        for cell in row:
            value = cell.find("sheet:v", namespace)
            if value is not None:
                cells[re.sub(r"\d", "", cell.attrib["r"])] = (
                    strings[int(value.text)]
                    if cell.attrib.get("t") == "s"
                    else value.text
                )
        if cells.get("A", "").isdigit() and cells.get("B"):
            catalog[int(cells["A"])] = cells["B"].strip()
    if (
        catalog.get(18) != "kicking other person"
        or catalog.get(26) != "punching/slapping other person"
    ):
        raise ValueError(
            "Unexpected PKU action IDs; do not infer labels from filenames"
        )
    return catalog


def parse_intervals(text, frame_count, catalog):
    intervals = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [int(value.strip()) for value in line.split(",")]
        if len(parts) != 4:
            raise ValueError("PKU labels require action,start,end,confidence")
        action, start, end, confidence = parts
        if (
            action not in catalog
            or start < 1
            or end < start
            or end > frame_count
            or confidence not in {0, 1, 2}
        ):
            raise ValueError("Invalid PKU timing/action/confidence")
        label = (
            1
            if action in POSITIVE_ACTIONS
            else 0 if action in NEGATIVE_ACTIONS else None
        )
        if confidence == 0:
            label = None
        intervals.append(
            {
                "action_id": action,
                "action": catalog[action],
                "start": start - 1,
                "end": end - 1,
                "confidence": confidence,
                "label": label,
            }
        )
    return intervals


def temporal_windows(reference, intervals, config, training=False):
    fps, count = reference["fps"], reference["frames"]
    span = round((config.frames - 1) * fps / config.fps) + 1
    if count < span:
        raise ValueError("Source shorter than model window")
    forbidden = [interval for interval in intervals if interval["label"] in {1, None}]
    records = {}
    for occurrence, interval in enumerate(intervals):
        if interval["label"] is None:
            continue
        anchors = (0.5, 0.75, 1.0) if training else (1.0,)
        for fraction in anchors:
            endpoint = interval["start"] + round(
                (interval["end"] - interval["start"]) * fraction
            )
            start = min(count - span, max(0, endpoint - span + 1))
            end = start + span - 1
            overlap = min(end, interval["end"]) - max(start, interval["start"]) + 1
            if overlap < min(round(fps * 0.2), interval["end"] - interval["start"] + 1):
                continue
            if interval["label"] == 0 and any(
                start <= other["end"] + fps * 0.25
                and end >= other["start"] - fps * 0.25
                for other in forbidden
            ):
                continue
            records[(start, interval["label"])] = {
                **reference,
                "start": start,
                "label": interval["label"],
                "action_id": interval["action_id"],
                "action": interval["action"],
                "occurrence": occurrence,
                "annotation_confidence": interval["confidence"],
            }
    return list(records.values())


def pku_sources(root, config):
    root = Path(root)
    catalog = action_catalog(root / "Actions.xlsx")
    sources, training, diagnostic = [], [], []
    for path in sorted((root / "rgb").glob("*.avi")):
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not np.isfinite(fps) or not 0 < fps <= 240 or count <= 0:
            raise ValueError("Invalid source video timing")
        matches = list((root / "labels").rglob(path.stem + ".txt"))
        if len(matches) != 1:
            raise ValueError(f"Expected one matching label file: {path.stem}")
        intervals = parse_intervals(matches[0].read_text(), count, catalog)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        reference = {
            "path": str(path),
            "video": "pku/" + path.stem,
            "key": path.stem,
            "dataset": "pku",
            "fps": fps,
            "frames": count,
            "scene_group": "pku/" + path.stem.rsplit("-", 1)[0],
            "sha256": digest,
            "annotation_sha256": hashlib.sha256(matches[0].read_bytes()).hexdigest(),
        }
        sources.append({**reference, "intervals": intervals})
        training.extend(temporal_windows(reference, intervals, config, training=True))
        diagnostic.extend(temporal_windows(reference, intervals, config))
    if len(sources) < 3 or not any(row["label"] == 1 for row in training):
        raise ValueError("Need at least three source recordings and positive actions")
    return sources, training, diagnostic
