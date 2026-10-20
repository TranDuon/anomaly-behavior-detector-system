"""
scripts/generate_test_video.py

Generates a synthetic test video with simulated person blobs performing:
  - Normal walking (most of the time)
  - Running (fast-moving blob)
  - Loitering (blob staying in same zone)
  - Fighting (two blobs converging with jitter)

Output: test_video.mp4 in the project root.
No external dataset required.

Usage:
  python scripts/generate_test_video.py [--out test_video.mp4] [--duration 60]
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


# ── Blob actor ────────────────────────────────────────────────────────────────

@dataclass
class Actor:
    x: float          # centre x (0–1 normalised)
    y: float          # centre y (0–1 normalised)
    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.06   # normalised width
    h: float = 0.16   # normalised height
    color: Tuple = field(default_factory=lambda: (
        random.randint(60, 180),
        random.randint(60, 180),
        random.randint(60, 200),
    ))
    mode: str = "walk"
    mode_timer: int = 0
    loiter_cx: float = 0.0
    loiter_cy: float = 0.0

    def update(self, frame_w: int, frame_h: int, other_actors: List["Actor"]):
        diag = math.sqrt(frame_w ** 2 + frame_h ** 2)

        # Randomly switch behaviour
        self.mode_timer -= 1
        if self.mode_timer <= 0:
            roll = random.random()
            if roll < 0.70:
                self.mode = "walk"
                self.mode_timer = random.randint(60, 180)
                self.vx = random.uniform(-0.004, 0.004)
                self.vy = random.uniform(-0.001, 0.001)
            elif roll < 0.82:
                self.mode = "run"
                self.mode_timer = random.randint(40, 100)
                self.vx = random.choice([-1, 1]) * random.uniform(0.010, 0.018)
                self.vy = random.uniform(-0.002, 0.002)
            elif roll < 0.92:
                self.mode = "loiter"
                self.mode_timer = random.randint(100, 200)
                self.loiter_cx = self.x
                self.loiter_cy = self.y
                self.vx = 0.0
                self.vy = 0.0
            else:
                self.mode = "fight"
                self.mode_timer = random.randint(60, 120)

        if self.mode == "loiter":
            self.x = self.loiter_cx + random.gauss(0, 0.003)
            self.y = self.loiter_cy + random.gauss(0, 0.001)
        elif self.mode == "fight" and other_actors:
            target = random.choice(other_actors)
            dx = target.x - self.x
            dy = target.y - self.y
            dist = math.sqrt(dx ** 2 + dy ** 2) + 1e-6
            spd  = 0.009
            self.x += (dx / dist) * spd + random.gauss(0, 0.004)
            self.y += (dy / dist) * spd * 0.5 + random.gauss(0, 0.002)
        else:
            self.x += self.vx + random.gauss(0, 0.0005)
            self.y += self.vy + random.gauss(0, 0.0002)

        # Bounce off walls
        if self.x < 0.05 or self.x > 0.95:
            self.vx *= -1
            self.x = max(0.05, min(0.95, self.x))
        if self.y < 0.05 or self.y > 0.95:
            self.vy *= -1
            self.y = max(0.05, min(0.95, self.y))

    def draw(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        cx  = int(self.x * W)
        cy  = int(self.y * H)
        bw  = int(self.w * W)
        bh  = int(self.h * H)
        x1  = cx - bw // 2
        y1  = cy - bh // 2
        x2  = x1 + bw
        y2  = y1 + bh

        # Body blob (filled ellipse)
        cv2.ellipse(frame, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, self.color, -1)
        # Head
        head_r = max(bw // 4, 8)
        cv2.circle(frame, (cx, y1 + head_r), head_r, self.color, -1)

        # Mode label
        labels = {"walk": "", "run": "RUN", "loiter": "LOIT", "fight": "FIGHT"}
        lbl = labels.get(self.mode, "")
        if lbl:
            cv2.putText(frame, lbl, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main generator ────────────────────────────────────────────────────────────

def generate(out_path: str, duration: int, fps: int, width: int, height: int, n_actors: int):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    actors = [
        Actor(
            x=random.uniform(0.1, 0.9),
            y=random.uniform(0.2, 0.8),
            vx=random.uniform(-0.003, 0.003),
            vy=random.uniform(-0.001, 0.001),
        )
        for _ in range(n_actors)
    ]

    total_frames = duration * fps
    bg_color = (30, 32, 36)  # dark grey background (fake floor)

    # Grid lines — simulate overhead CCTV perspective
    for f in range(total_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Floor grid
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (38, 42, 48), 1)
        for gy in range(0, height, 80):
            cv2.line(frame, (0, gy), (width, gy), (38, 42, 48), 1)

        # Update and draw actors
        for i, actor in enumerate(actors):
            others = [a for j, a in enumerate(actors) if j != i]
            actor.update(width, height, others)
            actor.draw(frame)

        # Frame info overlay
        elapsed = f / fps
        cv2.putText(frame, f"SYNTHETIC  frame={f:05d}  t={elapsed:.1f}s",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 100), 1)

        writer.write(frame)

        if f % fps == 0:
            pct = f / total_frames * 100
            print(f"  {pct:5.1f}%  {elapsed:.0f}s / {duration}s", end="\r", flush=True)

    writer.release()
    print(f"\n[generate] Saved {total_frames} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test video")
    parser.add_argument("--out",      default="test_video.mp4")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--fps",      type=int, default=25)
    parser.add_argument("--width",    type=int, default=960)
    parser.add_argument("--height",   type=int, default=540)
    parser.add_argument("--actors",   type=int, default=5, help="Number of person blobs")
    args = parser.parse_args()

    print(f"[generate] {args.width}×{args.height} @ {args.fps}fps  "
          f"{args.duration}s  {args.actors} actors → {args.out}")
    generate(args.out, args.duration, args.fps, args.width, args.height, args.actors)


if __name__ == "__main__":
    main()
