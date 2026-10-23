"""
anomaly/skeleton.py — Pose records → person-pair tensors for the ST-GCN.

A video's pose cache is a flat table of tracked people (one row per person
per sampled frame). For a window of T sampled frames we:
  1. keep tracks present in at least `min_presence` samples,
  2. rank pairs within `radius` person heights by interaction — limb motion
     relative to the torso, divided by how close the two get (centre to
     centre, or a wrist/ankle to the other's head/torso) — so in a crowd a
     fighting pair outranks people merely walking side by side,
  3. turn each kept pair into a (C, T, V) tensor on a 34-joint two-person graph.

A window with one candidate and no partner yields a pair whose second person
is all zeros, so single-person actions use the same model input.

Channels (C = 5): x, y, keypoint confidence, dx, dy — coordinates centred on
the pair and divided by the pair's mean bbox height, so values do not depend
on camera resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


NUM_JOINTS = 17                      # COCO keypoints from YOLOv8-pose
NUM_NODES = 2 * NUM_JOINTS           # two-person graph
NUM_CHANNELS = 5

# COCO skeleton (0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles)
COCO_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
LIMB_ENDS = [9, 10, 15, 16]          # wrists, ankles: what strikes
BODY_TARGETS = [0, 5, 6, 11, 12]     # head, shoulders, hips: what gets hit
TORSO = [5, 6, 11, 12]
MIN_KEYPOINT_CONF = 0.1


# ── Graph ──────────────────────────────────────────────────────────────────

def _normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    return (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]).astype(np.float32)


def two_person_adjacency() -> np.ndarray:
    """(3, 34, 34) adjacency subsets: self loops, intra-person, inter-person."""
    intra = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    inter = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for offset in (0, NUM_JOINTS):
        for a, b in COCO_EDGES:
            intra[a + offset, b + offset] = intra[b + offset, a + offset] = 1.0
    for joint in range(NUM_JOINTS):
        inter[joint, joint + NUM_JOINTS] = inter[joint + NUM_JOINTS, joint] = 1.0
    for attacker, target in ((0, NUM_JOINTS), (NUM_JOINTS, 0)):
        for limb in LIMB_ENDS:
            for body in BODY_TARGETS:
                inter[limb + attacker, body + target] = 1.0
                inter[body + target, limb + attacker] = 1.0
    identity = np.eye(NUM_NODES, dtype=np.float32)
    return np.stack([identity, _normalize(intra), _normalize(inter)])


# ── Pose cache ─────────────────────────────────────────────────────────────

@dataclass
class PoseTrack:
    """Flat pose table of one video, as written by scripts/extract_pose.py."""
    sample_frames: np.ndarray    # (S,) decoded frame index of each sample
    sample_index: np.ndarray     # (N,) sample position of each row, sorted
    track_ids: np.ndarray        # (N,)
    boxes: np.ndarray            # (N, 4) xyxy pixels
    keypoints: np.ndarray        # (N, 17, 3) x, y pixels + confidence
    frame_count: int
    fps: float

    @classmethod
    def load(cls, path) -> "PoseTrack":
        with np.load(path) as data:
            sample_frames = np.asarray(data["sample_frames"], dtype=np.int64)
            frame_index = np.asarray(data["frame_index"], dtype=np.int64)
            return cls(
                sample_frames=sample_frames,
                sample_index=np.searchsorted(sample_frames, frame_index),
                track_ids=np.asarray(data["track_ids"], dtype=np.int64),
                boxes=np.asarray(data["boxes"], dtype=np.float32),
                keypoints=np.asarray(data["keypoints"], dtype=np.float32),
                frame_count=int(data["frame_count"]),
                fps=float(data["fps"]),
            )

    @property
    def num_samples(self) -> int:
        return len(self.sample_frames)

    def window_tracks(
        self, start: int, length: int, min_presence: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Tracks in samples [start, start+length): (keypoints (T,17,3), heights (T,))."""
        return [(kp, h) for _, kp, h in self.window_tracks_with_ids(start, length, min_presence)]

    def window_tracks_with_ids(
        self, start: int, length: int, min_presence: int
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        lo = np.searchsorted(self.sample_index, start, side="left")
        hi = np.searchsorted(self.sample_index, start + length, side="left")
        if hi <= lo:
            return []
        rows = np.arange(lo, hi)
        ids = self.track_ids[rows]
        tracks = []
        for track_id in np.unique(ids):
            selected = rows[ids == track_id]
            if len(selected) < min_presence:
                continue
            positions = self.sample_index[selected] - start
            keypoints = np.zeros((length, NUM_JOINTS, 3), dtype=np.float32)
            heights = np.zeros(length, dtype=np.float32)
            keypoints[positions] = self.keypoints[selected]
            boxes = self.boxes[selected]
            heights[positions] = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
            tracks.append((int(track_id), keypoints, heights))
        return tracks


# ── Pairing ────────────────────────────────────────────────────────────────

@dataclass
class _TrackStats:
    center: np.ndarray       # (T, 2) torso centre, NaN where absent
    height: float            # mean bbox height over present samples
    limb_energy: float       # mean limb speed relative to torso, in heights/sample


def _track_stats(keypoints: np.ndarray, heights: np.ndarray) -> Optional[_TrackStats]:
    present = heights > 0
    valid = (keypoints[:, :, 2] >= MIN_KEYPOINT_CONF) & present[:, None]
    torso_valid = valid[:, TORSO]
    use_torso = torso_valid.any(axis=1)
    use_any = valid.any(axis=1)
    if not use_any.any():
        return None
    center = np.full((len(keypoints), 2), np.nan, dtype=np.float32)
    torso_xy = keypoints[:, TORSO, :2]
    center[use_torso] = (
        (torso_xy * torso_valid[..., None]).sum(1)[use_torso]
        / torso_valid.sum(1)[use_torso, None]
    )
    fallback = use_any & ~use_torso
    center[fallback] = (
        (keypoints[:, :, :2] * valid[..., None]).sum(1)[fallback]
        / valid.sum(1)[fallback, None]
    )
    height = float(heights[present].mean())

    relative = keypoints[:, LIMB_ENDS, :2] - center[:, None, :]
    limb_valid = valid[:, LIMB_ENDS] & ~np.isnan(center[:, :1])
    both = limb_valid[1:] & limb_valid[:-1]
    if both.any():
        step = np.linalg.norm(relative[1:] - relative[:-1], axis=-1)
        energy = float(step[both].mean() / height)
    else:
        energy = 0.0
    return _TrackStats(center, height, energy)


def proximity_matrix(keypoints: np.ndarray, stats: List[_TrackStats]) -> np.ndarray:
    """(n, n) closeness in person heights for every track pair, inf when never together.

    Closeness is the smaller of the 20th-percentile centre distance and the
    closest a wrist/ankle of one person gets to the other's head/torso, over
    the samples where both are present. All pairs are computed at once: a
    crowd of 19 people is 171 pairs per window.
    """
    centers = np.stack([item.center for item in stats])                   # (n, T, 2)
    heights = np.asarray([item.height for item in stats], dtype=np.float32)
    present = ~np.isnan(centers[..., 0])
    together = present[:, None, :] & present[None, :, :]                  # (n, n, T)
    any_together = together.any(axis=-1)
    scale = (heights[:, None] + heights[None, :]) / 2

    # 20th percentile over the shared samples, linear interpolation as in
    # np.quantile; done with one sort because np.nanquantile loops per row.
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    ordered = np.sort(np.where(together, distance, np.inf), axis=-1)
    count = together.sum(axis=-1)
    position = 0.2 * np.maximum(count - 1, 0)
    low = np.floor(position).astype(np.int64)[..., None]
    high = np.ceil(position).astype(np.int64)[..., None]
    low_value = np.take_along_axis(ordered, low, axis=-1)[..., 0]
    high_value = np.take_along_axis(ordered, high, axis=-1)[..., 0]
    with np.errstate(invalid="ignore"):
        quantile = low_value + (position - low[..., 0]) * (high_value - low_value)
    closeness = np.where(any_together, quantile / scale, np.inf)

    limbs = keypoints[:, :, LIMB_ENDS]                                    # (n, T, 4, 3)
    bodies = keypoints[:, :, BODY_TARGETS]                                # (n, T, 5, 3)
    gaps = np.linalg.norm(
        limbs[:, None, :, :, None, :2] - bodies[None, :, :, None, :, :2], axis=-1
    )                                                                      # (n, n, T, 4, 5)
    ok = (
        (limbs[:, None, :, :, None, 2] >= MIN_KEYPOINT_CONF)
        & (bodies[None, :, :, None, :, 2] >= MIN_KEYPOINT_CONF)
        & together[:, :, :, None, None]
    )
    reach_one_way = np.where(ok, gaps, np.inf).min(axis=(2, 3, 4))      # i's limbs -> j's body
    reach = np.minimum(reach_one_way, reach_one_way.T) / scale
    proximity = np.minimum(closeness, reach)
    proximity[~any_together] = np.inf
    return proximity


def select_pairs(
    tracks: List[Tuple[np.ndarray, np.ndarray]],
    max_pairs: int,
    radius: float,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, Optional[int]]]:
    """Most interacting pairs within `radius` person heights; singles when nobody is close.

    With `rng` (training) a quarter of the slots go to random pairs outside
    the top ranks when there are more candidates than slots, so a fight the
    heuristic ranks low can still reach the bag and the model learns to find
    it (the bag score is the max over pairs).
    """
    stats = [_track_stats(kp, h) for kp, h in tracks]
    valid = [index for index, item in enumerate(stats) if item is not None]
    candidates = []
    if len(valid) >= 2:
        valid_stats = [stats[index] for index in valid]
        proximity = proximity_matrix(np.stack([tracks[index][0] for index in valid]), valid_stats)
        energy = np.asarray([item.limb_energy for item in valid_stats])
        interaction = (energy[:, None] + energy[None, :] + 1e-3) / (0.25 + proximity)
        rows, columns = np.nonzero(np.triu(proximity <= radius, k=1))
        candidates = [
            (-float(interaction[a, b]), float(proximity[a, b]), valid[a], valid[b])
            for a, b in zip(rows, columns)
        ]
    if candidates:
        candidates.sort()
        ranked = [(i, j) for _, _, i, j in candidates]
        if rng is None or len(ranked) <= max_pairs:
            return ranked[:max_pairs]
        explore = max_pairs // 4
        keep = max_pairs - explore
        extra = rng.choice(len(ranked) - keep, size=explore, replace=False) + keep
        return ranked[:keep] + [ranked[index] for index in sorted(extra)]
    ranked = sorted(valid, key=lambda index: -stats[index].limb_energy)
    return [(index, None) for index in ranked[:max_pairs]]


# ── Pair tensor ────────────────────────────────────────────────────────────

def pair_tensor(
    first: Tuple[np.ndarray, np.ndarray],
    second: Optional[Tuple[np.ndarray, np.ndarray]],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a (C, T, 34) tensor; `rng` turns on training augmentation."""
    people = [first[0].copy()]
    heights = [first[1]]
    if second is not None:
        people.append(second[0].copy())
        heights.append(second[1])
    length = people[0].shape[0]

    valid = [person[:, :, 2] >= MIN_KEYPOINT_CONF for person in people]
    all_valid = np.concatenate([v[..., None] for v in valid], axis=-1)
    all_xy = np.concatenate([p[:, :, None, :2] for p in people], axis=2)
    if all_valid.any():
        center = all_xy[all_valid].mean(axis=0)
    else:
        center = np.zeros(2, dtype=np.float32)
    present_heights = np.concatenate([h[h > 0] for h in heights])
    scale = float(present_heights.mean()) if len(present_heights) else 1.0

    if rng is not None and rng.random() < 0.5:
        people.reverse()
        valid.reverse()

    angle = scale_jitter = 0.0
    flip = False
    if rng is not None:
        angle = np.deg2rad(rng.uniform(-10, 10))
        scale_jitter = rng.uniform(-0.1, 0.1)
        flip = bool(rng.random() < 0.5)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )

    tensor = np.zeros((NUM_CHANNELS, length, NUM_NODES), dtype=np.float32)
    for slot, (person, person_valid) in enumerate(zip(people, valid)):
        xy = (person[:, :, :2] - center) / scale
        confidence = person[:, :, 2]
        if rng is not None:
            xy = xy @ rotation.T * (1.0 + scale_jitter)
            xy = xy + rng.normal(0.0, 0.01, xy.shape)
            dropped = rng.random(person_valid.shape) < 0.05
            person_valid = person_valid & ~dropped
        if flip:
            xy = xy[:, FLIP_INDEX] * np.array([-1.0, 1.0], dtype=np.float32)
            confidence = confidence[:, FLIP_INDEX]
            person_valid = person_valid[:, FLIP_INDEX]
        xy = np.where(person_valid[..., None], xy, 0.0)
        confidence = np.where(person_valid, confidence, 0.0)
        velocity = np.zeros_like(xy)
        moving = person_valid[1:] & person_valid[:-1]
        velocity[1:] = np.where(moving[..., None], xy[1:] - xy[:-1], 0.0)

        nodes = slice(slot * NUM_JOINTS, (slot + 1) * NUM_JOINTS)
        tensor[0, :, nodes] = xy[..., 0]
        tensor[1, :, nodes] = xy[..., 1]
        tensor[2, :, nodes] = confidence
        tensor[3, :, nodes] = velocity[..., 0]
        tensor[4, :, nodes] = velocity[..., 1]
    return tensor


def window_pairs(
    pose: PoseTrack,
    start: int,
    length: int,
    min_presence: int,
    max_pairs: int,
    radius: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[Tuple[int, Optional[int]]]]:
    """(K, C, T, V) pair tensors for one window (K may be 0) and each pair's track ids."""
    tracks = pose.window_tracks_with_ids(start, length, min_presence)
    pairs = select_pairs([(kp, h) for _, kp, h in tracks], max_pairs, radius, rng)
    if not pairs:
        return np.zeros((0, NUM_CHANNELS, length, NUM_NODES), dtype=np.float32), []
    tensors = np.stack(
        [
            pair_tensor(tracks[i][1:], tracks[j][1:] if j is not None else None, rng)
            for i, j in pairs
        ]
    )
    ids = [(tracks[i][0], tracks[j][0] if j is not None else None) for i, j in pairs]
    return tensors, ids


def window_bag(
    pose: PoseTrack,
    start: int,
    length: int,
    min_presence: int,
    max_pairs: int,
    radius: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """(K, C, T, V) pair tensors for one window; K may be 0."""
    return window_pairs(pose, start, length, min_presence, max_pairs, radius, rng)[0]


def window_starts(num_samples: int, length: int, stride: int) -> np.ndarray:
    """Start positions of full windows; one window for clips shorter than `length`."""
    if num_samples <= length:
        return np.zeros(1, dtype=np.int64)
    return np.arange(0, num_samples - length + 1, stride, dtype=np.int64)
