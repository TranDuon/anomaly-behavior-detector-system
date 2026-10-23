"""Find near-duplicate videos across (and inside) the fight datasets.

UBI-Fights, Surveillance Camera Fight, RLVS and UCF-Crime were all collected
from YouTube-style sources, so one clip can appear in two of them. A copy in
the training data of a video used for testing inflates every metric.

Each video gets 64-bit difference hashes (dHash) of `--frames` frames spread
over its length. Two videos are duplicates when at least `--min-match` of the
shorter one's hashes have a counterpart within `--max-distance` bits.

Frame hashes also match two different moments filmed by the same fixed
camera (the background dominates a 9x8 hash), so only pairs from different
datasets are excluded automatically: independently collected datasets that
share frames share source footage. Pairs inside one dataset are reported but
kept, so the authors' official split (and comparability with published
results) is preserved.

Output exclude.json lists the uids to drop from training: when one copy is a
test video the training copy goes; when both are training videos the second
one goes. Test/test pairs are only reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import VideoItem, load_items


NO_HASH = np.uint64(0)        # sentinel for unreadable or flat frames
_POPCOUNT16 = np.array([bin(value).count("1") for value in range(1 << 16)], dtype=np.uint8)


def dhash(frame: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if float(gray.std()) < 5.0:
        return NO_HASH
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = np.uint64(0)
    for bit in bits:
        value = (value << np.uint64(1)) | np.uint64(int(bit))
    return value if value != NO_HASH else np.uint64(1)


def video_hashes(path: Path, count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    hashes = np.full(count, NO_HASH, dtype=np.uint64)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return hashes
        positions = np.linspace(0, total - 1, count + 2)[1:-1].astype(int)
        for slot, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            success, frame = capture.read()
            if success:
                hashes[slot] = dhash(frame)
    finally:
        capture.release()
    return hashes


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise bit distance between uint64 arrays a (n,) and b (m,) -> (n, m)."""
    x = a[:, None] ^ b[None, :]
    distance = np.zeros(x.shape, dtype=np.uint8)
    for shift in (0, 16, 32, 48):
        distance += _POPCOUNT16[((x >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.int64)]
    return distance


def match_fraction(hashes: np.ndarray, max_distance: int, chunk: int = 16) -> np.ndarray:
    """(N, F) hashes -> (N, N) fraction of row video's frames matched in column video."""
    n, frames = hashes.shape
    flat = hashes.reshape(-1)
    valid = flat != NO_HASH
    result = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, chunk):
        rows = hashes[start : start + chunk].reshape(-1)
        distance = hamming(rows, flat).astype(np.int16)
        distance[:, ~valid] = 99
        distance[rows == NO_HASH, :] = 99
        nearest = distance.reshape(len(rows), n, frames).min(axis=2)
        matched = (nearest <= max_distance).reshape(-1, frames, n)
        row_valid = (rows != NO_HASH).reshape(-1, frames)
        counts = np.maximum(row_valid.sum(axis=1), 1)
        result[start : start + chunk] = matched.sum(axis=1) / counts[:, None]
    np.fill_diagonal(result, 0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate videos across datasets")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--min-match", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[VideoItem] = [item for spec in args.dataset for item in load_items(spec)]

    cache_path = output_dir / "hashes.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as data:
            cached = {
                uid: row
                for uid, row in zip(data["uids"].tolist(), data["hashes"])
                if row.shape[0] == args.frames
            }
    for index, item in enumerate(items, start=1):
        if item.uid not in cached:
            cached[item.uid] = video_hashes(item.path, args.frames)
            if index % 50 == 0:
                print(f"[dedup] hashed {index}/{len(items)}", flush=True)
    uids = [item.uid for item in items]
    hashes = np.stack([cached[uid] for uid in uids])
    np.savez(cache_path, uids=np.asarray(uids), hashes=hashes)

    fraction = match_fraction(hashes, args.max_distance)
    # A pair counts when most of the shorter video's frames are found in the other.
    pair_score = np.maximum(fraction, fraction.T)
    duplicates, same_dataset, exclude = [], [], set()
    for i, j in zip(*np.nonzero(np.triu(pair_score >= args.min_match, k=1))):
        a, b = items[i], items[j]
        pair = {"a": a.uid, "a_split": a.split, "b": b.uid, "b_split": b.split,
                "match": round(float(pair_score[i, j]), 3)}
        if a.dataset == b.dataset:
            same_dataset.append(pair)
            continue
        duplicates.append(pair)
        if a.split == "test" and b.split == "test":
            continue
        if a.split == "test":
            exclude.add(b.uid)
        elif b.split == "test":
            exclude.add(a.uid)
        else:
            exclude.add(b.uid)

    report = {
        "videos": len(items),
        "frames_per_video": args.frames,
        "max_distance": args.max_distance,
        "min_match": args.min_match,
        "duplicate_pairs": duplicates,
        "same_dataset_pairs_kept": same_dataset,
        "exclude_from_training": sorted(exclude),
    }
    (output_dir / "exclude.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[dedup] {len(items)} videos, {len(duplicates)} cross-dataset duplicate pair(s), "
        f"{len(same_dataset)} same-dataset pair(s) reported, "
        f"{len(exclude)} excluded from training -> {output_dir / 'exclude.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
