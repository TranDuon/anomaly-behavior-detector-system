"""
anomaly/visual_fight.py — Fight detection from video appearance (stream A).

Two stages, so the expensive part runs once per video:
  1. A frozen S3D (Kinetics-400) turns each clip of CLIP_FRAMES frames sampled
     at CLIP_FPS into a 2048-d feature: the 1024-channel map averaged over
     time and space, and averaged over time then max-pooled over space (a
     fight filling a small part of a wide CCTV view is not averaged away).
     One clip starts every CLIP_STRIDE samples (one second).
  2. FightTemporalHead, a small causal temporal convolution over the clip
     features of a video, gives a fight logit per clip. Causal: the score of
     a clip uses only that clip and earlier ones, as a live camera would.

Each clip also carries a motion energy (mean absolute difference between
consecutive sampled frames, on small grey frames). It is the baseline the
model has to beat: a model that only reacts to "much movement" is not
detecting fights.

The same preprocessing (sampling grid, resize, normalisation) is used for
training-time extraction and by StreamingFightScorer at runtime.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_FRAMES = 16          # S3D needs >= 14 frames
CLIP_FPS = 15.0           # sampling rate of the torchvision Kinetics recipe
CLIP_STRIDE = 15          # samples between clip starts (one second)
INPUT_SIZE = 224          # whole frame resized (not cropped): fights at the edges stay in view
MOTION_SIZE = (64, 36)    # grey frames for the motion-energy baseline
FEATURE_DIM = 2048
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)

EXTRACTION_CONFIG = {
    "backbone": "torchvision s3d KINETICS400_V1",
    "clip_frames": CLIP_FRAMES,
    "clip_fps": CLIP_FPS,
    "clip_stride": CLIP_STRIDE,
    "input_size": INPUT_SIZE,
    "pooling": "mean(T,H,W) + max(H,W) of mean(T)",
}


# ── Frame sampling ─────────────────────────────────────────────────────────

def prepare_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR frame -> (INPUT_SIZE x INPUT_SIZE RGB uint8, small grey float32 for motion)."""
    rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    grey = cv2.cvtColor(
        cv2.resize(frame_bgr, MOTION_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0
    return rgb, grey


def sample_video(path: Path, max_seconds: Optional[float] = None) -> Iterator[tuple]:
    """Yield (frame_index, rgb, grey) on a CLIP_FPS grid, then a final ("end", frame_count, fps)."""
    for sample in grid_frames(path, max_seconds):
        yield sample if sample[0] == "end" else (sample[0], *prepare_frame(sample[1]))


def grid_frames(path: Path, max_seconds: Optional[float] = None) -> Iterator[tuple]:
    """Yield (frame_index, full-resolution BGR frame) on a CLIP_FPS grid, then ("end", frame_count, fps).

    Frames off the grid are only grabbed (no colour conversion).
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    step = max(source_fps / CLIP_FPS, 1.0)
    max_frames = int(max_seconds * source_fps) if max_seconds else None
    decoded, next_sample = 0, 0.0
    try:
        while max_frames is None or decoded < max_frames:
            wanted = decoded + 1e-6 >= next_sample
            if wanted:
                success, frame = capture.read()
            else:
                success, frame = capture.grab(), None
            if not success:
                break
            if wanted:
                next_sample += step
                yield decoded, frame
            decoded += 1
    finally:
        capture.release()
    yield ("end", decoded, float(source_fps))


def clip_starts(num_samples: int) -> np.ndarray:
    """Clip start samples; a video shorter than one clip still gives one (padded) clip."""
    if num_samples <= CLIP_FRAMES:
        return np.zeros(1 if num_samples else 0, dtype=np.int64)
    return np.arange(0, num_samples - CLIP_FRAMES + 1, CLIP_STRIDE, dtype=np.int64)


def motion_energy(greys: np.ndarray) -> float:
    """Mean absolute difference between consecutive grey frames of one clip."""
    if len(greys) < 2:
        return 0.0
    return float(np.abs(np.diff(greys, axis=0)).mean())


# ── Backbone ───────────────────────────────────────────────────────────────

class ClipFeatureExtractor(nn.Module):
    """Frozen S3D: (B, T, H, W, 3) uint8 RGB clips -> (B, FEATURE_DIM) features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import S3D_Weights, s3d

        backbone = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
        self.features = backbone.features
        self.register_buffer("mean", torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = clips.permute(0, 4, 1, 2, 3).float().div_(255.0)        # (B, 3, T, H, W)
        x = (x - self.mean) / self.std
        maps = self.features(x)                                      # (B, 1024, T', H', W')
        temporal = maps.mean(dim=2)                                  # (B, 1024, H', W')
        mean = temporal.mean(dim=(2, 3))
        peak = temporal.amax(dim=(2, 3))
        return torch.cat([mean, peak], dim=1).float()


def pad_clip(frames: list[np.ndarray]) -> np.ndarray:
    """Stack up to CLIP_FRAMES frames, repeating the last one for short videos."""
    clip = list(frames[:CLIP_FRAMES])
    while len(clip) < CLIP_FRAMES:
        clip.append(clip[-1])
    return np.stack(clip)


def extract_clips(
    samples: Iterable[tuple],
    extractor: ClipFeatureExtractor,
    device: torch.device,
    batch_size: int = 16,
    half: bool = False,
) -> dict:
    """Features of every clip of one video, from sample_video()'s stream.

    Streaming: only the samples of the clips in flight are kept in memory,
    so hour-long videos fit.
    """
    buffer: deque = deque()           # (sample position, frame index, rgb, grey)
    pending, features, motions, first_frames, last_frames = [], [], [], [], []
    next_start, position, frame_count, source_fps = 0, 0, 0, CLIP_FPS

    def flush() -> None:
        clips = torch.from_numpy(np.stack(pending)).to(device)
        with torch.autocast(device_type=device.type, enabled=half):
            output = extractor(clips)
        features.append(output.cpu().numpy().astype(np.float16))
        pending.clear()

    def emit(window: list) -> None:
        pending.append(pad_clip([rgb for _, _, rgb, _ in window]))
        motions.append(motion_energy(np.stack([grey for _, _, _, grey in window])))
        first_frames.append(window[0][1])
        last_frames.append(window[-1][1])
        if len(pending) >= batch_size:
            flush()

    for sample in samples:
        if sample[0] == "end":
            _, frame_count, source_fps = sample
            break
        frame_index, rgb, grey = sample
        buffer.append((position, frame_index, rgb, grey))
        position += 1
        if position == next_start + CLIP_FRAMES:
            emit([entry for entry in buffer if entry[0] >= next_start])
            next_start += CLIP_STRIDE
            while buffer and buffer[0][0] < next_start:
                buffer.popleft()
    if next_start == 0 and buffer:
        emit(list(buffer))                                   # shorter than one clip
    if pending:
        flush()
    return {
        "features": np.concatenate(features) if features else np.zeros((0, FEATURE_DIM), np.float16),
        "motion": np.asarray(motions, dtype=np.float32),
        "first_frames": np.asarray(first_frames, dtype=np.int32),
        "last_frames": np.asarray(last_frames, dtype=np.int32),
        "frame_count": np.int32(frame_count),
        "fps": np.float32(source_fps),
    }


# ── Person-group crops ─────────────────────────────────────────────────────
# Whole-frame S3D at 224 px sees the scene, not the fight (visual_fight_v1:
# zero-shot Kinetics in-video AUC 0.51; removing the scene drops frame AUC
# from 0.82 to 0.52). Crops give the people the backbone's full resolution.

MAX_GROUPS = 3            # crops per clip: the largest groups of people
GROUP_GAP = 0.5           # boxes join a group when they come within this many person heights
CROP_MARGIN = 0.25        # extra context around a group, as a share of its size
MIN_CROP = 96             # pixels: very small groups are upsampled from at least this


CROP_EXTRACTION_CONFIG = {
    **EXTRACTION_CONFIG,
    "crops": f"up to {MAX_GROUPS} person-group crops per clip (gap {GROUP_GAP} heights, margin {CROP_MARGIN}, "
             f"min {MIN_CROP} px); features = max over crops",
    "person_boxes": "skeleton-stream pose caches (YOLOv8n-pose 1280, BoT-SORT, 10 FPS)",
}


def person_groups(boxes: np.ndarray, max_groups: int = MAX_GROUPS, gap: float = GROUP_GAP) -> list[np.ndarray]:
    """Union boxes of people close together, most people first.

    `boxes` (N, 4) xyxy may hold the same person at several moments of a
    clip: a group is the region where its people were during that second.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    pad = gap * (boxes[:, 3] - boxes[:, 1])[:, None] * np.array([-1, -1, 1, 1], dtype=np.float32)
    grown = boxes + pad
    touching = (
        (grown[:, None, 0] <= grown[None, :, 2]) & (grown[None, :, 0] <= grown[:, None, 2])
        & (grown[:, None, 1] <= grown[None, :, 3]) & (grown[None, :, 1] <= grown[:, None, 3])
    )
    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in zip(*np.nonzero(np.triu(touching, k=1))):
        parent[find(int(a))] = find(int(b))
    members: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        members.setdefault(find(index), []).append(index)
    groups = sorted(members.values(), key=len, reverse=True)[:max_groups]
    return [
        np.array([boxes[group, 0].min(), boxes[group, 1].min(), boxes[group, 2].max(), boxes[group, 3].max()])
        for group in groups
    ]


def square_crop_box(box: np.ndarray, frame_w: int, frame_h: int, margin: float = CROP_MARGIN) -> np.ndarray:
    """Square xyxy region around `box` with margin, at least MIN_CROP, kept inside the frame when possible."""
    x1, y1, x2, y2 = (float(value) for value in box)
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(max(side, MIN_CROP), max(frame_w, frame_h))
    cx = min(max((x1 + x2) / 2, side / 2), frame_w - side / 2) if side <= frame_w else frame_w / 2
    cy = min(max((y1 + y2) / 2, side / 2), frame_h - side / 2) if side <= frame_h else frame_h / 2
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)


def crop_frame(frame_bgr: np.ndarray, region: np.ndarray) -> np.ndarray:
    """INPUT_SIZE x INPUT_SIZE RGB crop of `region`; parts outside the frame are black."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in region)
    top, left = max(-y1, 0), max(-x1, 0)
    bottom, right = max(y2 - height, 0), max(x2 - width, 0)
    patch = frame_bgr[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def extract_crop_clips(
    frames: Iterable[tuple],
    boxes_for: Callable[[int, int], np.ndarray],
    extractor: ClipFeatureExtractor,
    device: torch.device,
    batch_size: int = 16,
    half: bool = False,
) -> dict:
    """Like extract_clips, on crops around the largest groups of people of each clip.

    `frames` is grid_frames()'s stream; `boxes_for(first_frame, last_frame)`
    returns the person boxes (xyxy, source pixels) detected in that span. A
    clip without people uses the whole frame. `features` is the max over a
    clip's crops; `crop_features` (N, MAX_GROUPS, D) and `crop_boxes` keep
    each crop (unused slots are zero, `crop_count` says how many are used).
    """
    buffer: deque = deque()
    pending, crop_features = [], []
    motions, first_frames, last_frames, counts, regions = [], [], [], [], []
    next_start, position, frame_count, source_fps = 0, 0, 0, CLIP_FPS

    def flush() -> None:
        clips = torch.from_numpy(np.stack(pending)).to(device)
        with torch.autocast(device_type=device.type, enabled=half):
            output = extractor(clips)
        crop_features.append(output.cpu().numpy().astype(np.float16))
        pending.clear()

    def emit(window: list) -> None:
        frames_bgr = [frame for _, _, frame, _ in window]
        height, width = frames_bgr[0].shape[:2]
        first, last = window[0][1], window[-1][1]
        groups = person_groups(boxes_for(first, last))
        crops = [square_crop_box(group, width, height) for group in groups] or [
            np.array([0, 0, width, height], dtype=np.float32)
        ]
        for region in crops:
            pending.append(pad_clip([crop_frame(frame, region) for frame in frames_bgr]))
            if len(pending) >= batch_size:
                flush()
        slots = np.zeros((MAX_GROUPS, 4), dtype=np.float32)
        slots[: len(crops)] = crops
        regions.append(slots)
        counts.append(len(crops))
        motions.append(motion_energy(np.stack([grey for _, _, _, grey in window])))
        first_frames.append(first)
        last_frames.append(last)

    for sample in frames:
        if sample[0] == "end":
            _, frame_count, source_fps = sample
            break
        frame_index, frame = sample
        grey = cv2.cvtColor(cv2.resize(frame, MOTION_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        buffer.append((position, frame_index, frame, grey.astype(np.float32) / 255.0))
        position += 1
        if position == next_start + CLIP_FRAMES:
            emit([entry for entry in buffer if entry[0] >= next_start])
            next_start += CLIP_STRIDE
            while buffer and buffer[0][0] < next_start:
                buffer.popleft()
    if next_start == 0 and buffer:
        emit(list(buffer))
    if pending:
        flush()

    flat = np.concatenate(crop_features) if crop_features else np.zeros((0, FEATURE_DIM), np.float16)
    per_clip = np.zeros((len(counts), MAX_GROUPS, FEATURE_DIM), dtype=np.float16)
    offset = 0
    for index, count in enumerate(counts):
        per_clip[index, :count] = flat[offset: offset + count]
        offset += count
    pooled = np.stack([per_clip[index, :count].max(axis=0) for index, count in enumerate(counts)]) \
        if counts else np.zeros((0, FEATURE_DIM), np.float16)
    return {
        "features": pooled.astype(np.float16),
        "crop_features": per_clip,
        "crop_count": np.asarray(counts, dtype=np.int8),
        "crop_boxes": np.asarray(regions, dtype=np.float32).reshape(-1, MAX_GROUPS, 4),
        "motion": np.asarray(motions, dtype=np.float32),
        "first_frames": np.asarray(first_frames, dtype=np.int32),
        "last_frames": np.asarray(last_frames, dtype=np.int32),
        "frame_count": np.int32(frame_count),
        "fps": np.float32(source_fps),
    }


# ── Feature cache ──────────────────────────────────────────────────────────

class ClipFeatures:
    """Clip features of one video as written by scripts/extract_visual.py."""

    def __init__(self, features, motion, first_frames, last_frames, frame_count, fps):
        self.features = np.asarray(features, dtype=np.float16)    # ~80 h of video: 1.2 GB in RAM
        self.motion = np.asarray(motion, dtype=np.float32)
        self.first_frames = np.asarray(first_frames, dtype=np.int64)
        self.last_frames = np.asarray(last_frames, dtype=np.int64)
        self.frame_count = int(frame_count)
        self.fps = float(fps)

    @classmethod
    def load(cls, path) -> "ClipFeatures":
        with np.load(path) as data:
            return cls(
                data["features"], data["motion"], data["first_frames"], data["last_frames"],
                data["frame_count"], data["fps"],
            )

    def __len__(self) -> int:
        return len(self.features)

    def clip_labels(self, frame_labels: np.ndarray, guard_frames: int = 0) -> np.ndarray:
        """1 = fight (>= half the clip), 0 = normal (none), -1 = ambiguous.

        Normal clips within guard_frames of a fight frame are ambiguous too:
        annotated fight boundaries are approximate.
        """
        fight = np.asarray(frame_labels) > 0
        cumulative = np.concatenate([[0], np.cumsum(fight)])
        near = cumulative
        if guard_frames > 0 and fight.any():
            near_mask = np.convolve(fight.astype(np.int32), np.ones(2 * guard_frames + 1, dtype=np.int32), "same") > 0
            near = np.concatenate([[0], np.cumsum(near_mask)])
        labels = np.empty(len(self), dtype=np.int64)
        for index, (first, last) in enumerate(zip(self.first_frames, self.last_frames)):
            first, last = int(first), min(int(last), len(fight) - 1)
            span = max(last - first + 1, 1)
            ratio = (cumulative[last + 1] - cumulative[first]) / span
            close = near[last + 1] - near[first] > 0
            labels[index] = 1 if ratio >= 0.5 else (0 if ratio == 0 and not close else -1)
        return labels


# ── Temporal head ──────────────────────────────────────────────────────────

class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        y = self.conv(F.pad(x, (self.left_pad, 0)))
        y = self.dropout(F.gelu(y))
        return self.norm((x + y).transpose(1, 2)).transpose(1, 2)


class FightTemporalHead(nn.Module):
    """(B, T, FEATURE_DIM) clip features -> (B, T) fight logits, causal in T.

    LayerNorm instead of BatchNorm: batches pad sequences at the end, and
    padding must not change the statistics (the BatchNorm lesson of stream B).
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden: int = 256,
        layers: int = 3,
        kernel: int = 3,
        dropout: float = 0.3,
        input_dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "in_dim": in_dim, "hidden": hidden, "layers": layers, "kernel": kernel,
            "dropout": dropout, "input_dropout": input_dropout,
        }
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.project = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            CausalBlock(hidden, kernel, 2 ** level, dropout) for level in range(layers)
        )
        self.output = nn.Linear(hidden, 1)

    @property
    def receptive_field(self) -> int:
        """Clips (seconds) of history that reach one output, the current clip included."""
        kernel, layers = self.config["kernel"], self.config["layers"]
        return 1 + (kernel - 1) * (2 ** layers - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.project(self.input_dropout(self.input_norm(features)))   # (B, T, H)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.output(x.transpose(1, 2))[..., 0]

    def save(self, path: Path, metadata: dict | None = None) -> None:
        torch.save(
            {"config": self.config, "state_dict": self.state_dict(), "metadata": metadata or {}},
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "FightTemporalHead":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


def predict_video(model: FightTemporalHead, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Fight probability per clip of one whole video."""
    if len(features) == 0:
        return np.zeros(0, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device))[0]
    return torch.sigmoid(logits).double().cpu().numpy()


# ── Runtime ────────────────────────────────────────────────────────────────

class StreamingFightScorer:
    """Live scoring of one camera: feed every decoded frame, get a probability per second.

    Samples frames on the same CLIP_FPS grid as extraction, runs the backbone
    once per CLIP_STRIDE samples and the head over the last receptive-field
    clips, so a live score equals the offline score of the same clip.
    """

    def __init__(self, extractor: ClipFeatureExtractor, head: FightTemporalHead,
                 source_fps: float, device: torch.device | str = "cpu"):
        self.extractor = extractor
        self.head = head.eval()
        self.device = torch.device(device)
        self.step = max(float(source_fps) / CLIP_FPS, 1.0)
        self.frame_index = 0
        self.next_sample = 0.0
        self.samples: deque = deque(maxlen=CLIP_FRAMES)
        self.sample_count = 0
        self.history: deque = deque(maxlen=head.receptive_field)

    def push(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Fight probability when this frame completes a clip, else None."""
        current = self.frame_index
        self.frame_index += 1
        if current + 1e-6 < self.next_sample:
            return None
        self.next_sample += self.step
        rgb, _ = prepare_frame(frame_bgr)
        self.samples.append(rgb)
        self.sample_count += 1
        if self.sample_count < CLIP_FRAMES or (self.sample_count - CLIP_FRAMES) % CLIP_STRIDE:
            return None
        clip = torch.from_numpy(np.stack(self.samples))[None].to(self.device)
        feature = self.extractor(clip)[0].cpu().numpy().astype(np.float16).astype(np.float32)
        self.history.append(feature)
        return float(predict_video(self.head, np.stack(self.history), self.device)[-1])
