"""
anomaly/fight_data.py — Dataset adapters for the fight (skeleton) stream.

Every adapter returns VideoItem objects with one label convention:
labels_for(frame_count) gives a 0/1 label per decoded frame, whatever the
dataset's own granularity (per-frame CSV, temporal intervals, or one label
per clip). Keys are unique inside a dataset and name the pose cache file.

Supported layouts:
  ubi   UBI_FIGHTS/{annotation/*.csv, videos/{fight,normal}/*.mp4, test_videos.csv}
  scf   fight-detection-surv-dataset/{fight,noFight}/*.mp4
  rlvs  .../Violence/*.mp4 and .../NonViolence/*.mp4 (Kaggle layout)
  ucf   UCF-Crime videos (any depth) + Temporal_Anomaly_Annotation*.txt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
DATASET_NAMES = ("ubi", "scf", "rlvs", "ucf")


@dataclass
class VideoItem:
    dataset: str
    key: str
    path: Path
    split: str                                   # "train" or "test"
    frame_labels: Optional[np.ndarray] = None    # per-frame 0/1
    intervals: List[Tuple[int, int]] = field(default_factory=list)  # inclusive
    clip_label: Optional[int] = None

    @property
    def uid(self) -> str:
        return f"{self.dataset}/{self.key}"

    @property
    def has_positive(self) -> bool:
        if self.frame_labels is not None:
            return bool(np.any(self.frame_labels > 0))
        if self.intervals:
            return True
        return bool(self.clip_label)

    def labels_for(self, frame_count: int) -> np.ndarray:
        """0/1 label for each of frame_count decoded frames."""
        labels = np.zeros(frame_count, dtype=np.int32)
        if self.frame_labels is not None:
            source = np.asarray(self.frame_labels, dtype=np.int32)
            n = min(len(source), frame_count)
            labels[:n] = source[:n]
            if n and n < frame_count:
                # Annotation a few frames shorter than the decoded video.
                labels[n:] = source[n - 1]
        elif self.intervals:
            for start, end in self.intervals:
                labels[max(start, 0) : min(end + 1, frame_count)] = 1
        elif self.clip_label:
            labels[:] = 1
        return labels


def _videos_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _find_dir(root: Path, name: str) -> Optional[Path]:
    """First directory below root (or root itself) named `name`, case-insensitive."""
    if root.name.lower() == name.lower() and root.is_dir():
        return root
    for path in sorted(root.rglob("*")):
        if path.is_dir() and path.name.lower() == name.lower():
            return path
    return None


def parse_ubi_annotation(text: str) -> np.ndarray:
    values = [float(token) for token in text.split()]
    return (np.asarray(values, dtype=np.float32) > 0.5).astype(np.int32)


def ubi_fights_items(root: Path) -> list[VideoItem]:
    base = _find_dir(root, "UBI_FIGHTS") or root
    annotation_dir = base / "annotation"
    videos_dir = base / "videos"
    if not annotation_dir.is_dir() or not videos_dir.is_dir():
        raise FileNotFoundError(f"Not a UBI-Fights directory: {base}")
    test_file = base / "test_videos.csv"
    test_keys = set()
    if test_file.exists():
        test_keys = {
            line.strip()
            for line in test_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    items = []
    for video in _videos_in(videos_dir):
        annotation = annotation_dir / f"{video.stem}.csv"
        if not annotation.exists():
            raise FileNotFoundError(f"Missing UBI-Fights annotation: {annotation}")
        items.append(
            VideoItem(
                dataset="ubi",
                key=video.stem,
                path=video,
                split="test" if video.stem in test_keys else "train",
                frame_labels=parse_ubi_annotation(
                    annotation.read_text(encoding="utf-8")
                ),
            )
        )
    return items


def _clip_label_items(
    dataset: str,
    positive_dir: Optional[Path],
    negative_dir: Optional[Path],
    split: str,
) -> list[VideoItem]:
    if positive_dir is None or negative_dir is None:
        raise FileNotFoundError(f"{dataset}: positive/negative folders not found")
    items = []
    for label, directory in ((1, positive_dir), (0, negative_dir)):
        for video in _videos_in(directory):
            items.append(
                VideoItem(
                    dataset=dataset,
                    key=video.stem,
                    path=video,
                    split=split,
                    clip_label=label,
                )
            )
    return items


def scf_items(root: Path, split: str = "test") -> list[VideoItem]:
    """Surveillance Camera Fight: held out for cross-dataset testing by default."""
    return _clip_label_items(
        "scf", _find_dir(root, "fight"), _find_dir(root, "noFight"), split
    )


def rlvs_items(root: Path, split: str = "train") -> list[VideoItem]:
    """Real Life Violence Situations: extra clip-labelled training data."""
    return _clip_label_items(
        "rlvs", _find_dir(root, "Violence"), _find_dir(root, "NonViolence"), split
    )


def parse_ucf_annotation(text: str) -> dict[str, tuple[str, list[tuple[int, int]]]]:
    """Map video name -> (class, inclusive anomaly intervals); -1 marks no interval."""
    parsed = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        name, category = parts[0], parts[1]
        bounds = [int(value) for value in parts[2:6]]
        intervals = [
            (start, end)
            for start, end in (bounds[0:2], bounds[2:4])
            if start >= 0 and end >= 0
        ]
        parsed[Path(name).stem] = (category, intervals)
    return parsed


def ucf_crime_items(
    root: Path,
    annotation_file: Optional[Path] = None,
    classes: Iterable[str] = ("Fighting",),
    include_normal: bool = True,
) -> list[VideoItem]:
    """UCF-Crime test videos with temporal labels (frame-level evaluation)."""
    if annotation_file is None:
        candidates = sorted(root.rglob("Temporal_Anomaly_Annotation*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No UCF-Crime temporal annotation below {root}")
        annotation_file = candidates[0]
    annotations = parse_ucf_annotation(annotation_file.read_text(encoding="utf-8"))
    wanted = set(classes) | ({"Normal"} if include_normal else set())
    videos = {video.stem: video for video in _videos_in(root)}

    items = []
    for name, (category, intervals) in sorted(annotations.items()):
        if category not in wanted or name not in videos:
            continue
        items.append(
            VideoItem(
                dataset="ucf",
                key=name,
                path=videos[name],
                split="test",
                intervals=intervals,
            )
        )
    return items


def load_items(spec: str) -> list[VideoItem]:
    """Parse 'name:root' or 'name:root:split' and return that dataset's items."""
    name, _, rest = spec.partition(":")
    if name not in DATASET_NAMES or not rest:
        raise ValueError(
            f"Dataset spec must be name:root[:split] with name in {DATASET_NAMES}: {spec}"
        )
    split_override = None
    root_text = rest
    head, _, tail = rest.rpartition(":")
    if tail in ("train", "test") and head:
        root_text, split_override = head, tail
    root = Path(root_text).expanduser().resolve()

    if name == "ubi":
        items = ubi_fights_items(root)
    elif name == "scf":
        items = scf_items(root)
    elif name == "rlvs":
        items = rlvs_items(root)
    else:
        items = ucf_crime_items(root)
    if split_override:
        for item in items:
            item.split = split_override
    return items
