"""Training-only hard negatives and temporally consistent close-view augmentation."""

import cv2
import numpy as np

from fight.temporal_labels import source_group

SAFE_NEGATIVE_TAGS = {
    "greet",
    "hug",
    "handshake",
    "highfive",
    "handgestures",
    "jump",
    "walk",
}
ACTION_TAGS = {"hug": 1, "handshake": 2, "highfive": 3, "handgestures": 4, "jump": 5}
ACTION_NAMES = {
    "hug": "hug",
    "handshake": "handshake_greet",
    "highfive": "highfive_clap",
    "handgestures": "gesture",
    "jump": "solo_motion",
}


def expand_training(windows, manifest_rows, config):
    result = [dict(row) for row in windows]
    seen = {(row["video"], row["start"]) for row in result}
    rejected = {row["video"] for row in result if row["label"] == 1}
    by_video = {row["video"]: row for row in manifest_rows}
    for row in result:
        row["publisher_actions"] = by_video[row["video"]].get("actions", [])
    added = []
    for source in manifest_rows:
        tags = set(source.get("actions", []))
        if (
            source["split"] != "train"
            or source["dataset"] != "airtlab"
            or source["label"] != 0
            or source["video"] in rejected
            or not tags
            or not tags.issubset(SAFE_NEGATIVE_TAGS)
        ):
            continue
        span = round((config.frames - 1) * source["fps"] / config.fps) + 1
        if source["frames"] < span:
            continue
        for fraction in (0.0, 0.5, 1.0):
            start = int((source["frames"] - span) * fraction)
            if (source["video"], start) in seen:
                continue
            seen.add((source["video"], start))
            targets, mask = [0] * 6, [0] * 6
            mask[0] = 1
            for tag in tags & ACTION_TAGS.keys():
                targets[ACTION_TAGS[tag]] = mask[ACTION_TAGS[tag]] = 1
            row = {
                **source,
                "start": start,
                "actions": [
                    ACTION_NAMES[tag] for tag in sorted(tags & ACTION_NAMES.keys())
                ]
                or ["normal"],
                "publisher_actions": sorted(tags),
                "action_targets": targets,
                "action_mask": mask,
                "annotation_scope": "publisher_nonviolent_train_only_not_temporal_gold",
                "sample_weight": 1.0,
            }
            result.append(row)
            added.append(row)
    return result, {
        "added_windows": len(added),
        "added_source_groups": len({source_group(row) for row in added}),
        "scope": "Only publisher nonviolent AIRTLab train sources with explicit safe tags; no private, development, calibration or test source promoted into train",
    }


def augment_close_view(clips, row, generator):
    clips = np.asarray(clips)
    if clips.ndim != 5 or clips.dtype != np.uint8:
        raise ValueError("Expected uint8 region/video RGB clips")
    result = clips.copy()
    for region in range(len(result)):
        height, width = result.shape[2:4]
        tags = set(row.get("publisher_actions", []))
        upper_body_safe = bool(tags & {"punch", "slap"}) and not bool(
            tags & {"kick", "choke", "fight", "push"}
        )
        if row["label"] == 0:
            crop_height = max(2, round(height * generator.uniform(0.6, 1.0)))
            crop_width = max(2, round(width * generator.uniform(0.65, 1.0)))
            top = generator.randrange(height - crop_height + 1)
            left = generator.randrange(width - crop_width + 1)
        elif upper_body_safe:
            crop_height, crop_width = (
                max(2, round(height * generator.uniform(0.85, 1.0))),
                width,
            )
            top, left = 0, 0
        else:
            continue
        for frame_index, frame in enumerate(result[region]):
            result[region, frame_index] = cv2.resize(
                frame[top : top + crop_height, left : left + crop_width],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
    return result
