"""Publisher action tags and immutable public split preparation for fusion."""

import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fight.pose import pose_inputs
from fight.vision import build_clip_inputs, clip_tensor
from scripts.fight_large_train import validate_manifest
from scripts.fight_pilot import read_frames, write_json

ACTION_TAGS = (
    {"fight", "punch", "kick", "slap", "choke"},
    {"hug"},
    {"handshake", "greet"},
    {"highfive", "clap"},
    {"handgestures"},
    {"jump", "walk"},
)


def read_action_tags(data):
    mappings = {}
    for category, filename in (
        ("non-violent", "nonviolent-action-classes.csv"),
        ("violent", "violent-action-classes.csv"),
    ):
        paths = list(Path(data).rglob(filename))
        if len(paths) != 1:
            raise ValueError("Expected one AIRTLab action catalog: " + filename)
        with paths[0].open(encoding="utf-8-sig") as handle:
            records = csv.reader(handle, delimiter=";")
            next(records)
            for fields in records:
                if len(fields) >= 2:
                    mappings[(category, fields[0].strip())] = {
                        tag.strip() for tag in fields[1].split(",")
                    }
    return mappings


def prepare_rows(baseline, data, config):
    manifest = json.loads((Path(baseline) / "manifest.json").read_text())
    validate_manifest(manifest)
    tags = read_action_tags(data)
    rows, excluded = [], []
    for original in manifest["rows"]:
        row = dict(original)
        if row["dataset"] == "airtlab":
            suffix = row["video"].split("/", 1)[1]
            category = suffix.split("/")[0]
            actions = tags[(category, Path(suffix).name)]
            if "friendly punch" in actions:
                excluded.append(
                    {
                        "video": row["video"],
                        "reason": "Policy ambiguous: friendly punch; no automatic relabeling",
                    }
                )
                continue
            matches = list((Path(data) / "airtlab").rglob(suffix))
            if len(matches) != 1:
                raise ValueError("Cannot uniquely relocate " + row["video"])
            row["path"] = str(matches[0])
            row["actions"] = sorted(actions)
            row["action_targets"] = [
                int(bool(actions & group)) for group in ACTION_TAGS
            ]
            row["action_mask"] = [1] * len(ACTION_TAGS)
        else:
            relative = row["path"].split("/rlvs/", 1)[-1]
            path = Path(data) / "rlvs" / relative
            if not path.is_file():
                raise ValueError("Cannot relocate " + row["video"])
            row["path"] = str(path)
            row["actions"] = []
            row["action_targets"] = [int(row["label"])] + [0] * 5
            row["action_mask"] = [1] + [0] * 5
        span = round((config.frames - 1) * row["fps"] / config.fps) + 1
        if row["frames"] < span:
            excluded.append(
                {"video": row["video"], "reason": "Too short for configured history"}
            )
            continue
        row["start"] = (row["frames"] - span) // 2
        rows.append(row)
    return rows, excluded


def cache_rows(
    rows, model, detector, config, cache, weight_id, pose_weight_id, deadline
):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ready = []
    for index, original in enumerate(rows):
        if time.monotonic() > deadline:
            raise TimeoutError("Fusion cache budget exhausted")
        identity = {
            "video": original["video"],
            "sha256": original["sha256"],
            "start": original["start"],
            "fps": original["fps"],
            "clip": config.to_dict(),
            "pose": asdict(detector.config),
            "rgb_weights": weight_id,
            "pose_weights": pose_weight_id,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        row = {**original, "cache": key}
        path = cache / (key + ".npz")
        if not path.exists():
            frames = read_frames(row, config)
            detector.retain(frames)
            clips, regions, people, bounds, counts = build_clip_inputs(
                frames, detector, config
            )
            poses = pose_inputs(frames, regions, detector)
            with torch.inference_mode():
                features = model.rgb.features(clip_tensor(clips, "cuda")).cpu().numpy()
            partial = path.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                clips=clips,
                poses=poses,
                features=features,
                regions=np.array(regions),
                people=people,
            )
            partial.replace(path)
        ready.append(row)
        if (index + 1) % 64 == 0:
            print(f"FUSION_CACHE {index + 1}/{len(rows)}", flush=True)
    return ready
