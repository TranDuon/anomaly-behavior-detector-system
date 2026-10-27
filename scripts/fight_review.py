"""Build a blinded visual audit; never edits original dataset labels."""

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fight.vision import ClipConfig
from scripts.fight_ablation import row_id
from scripts.fight_pilot import read_frames, write_json


def select_review_rows(rows, predictions, random_per_stratum=12, errors_per_class=6):
    rng = random.Random(20261027)
    chosen, seen = [], set()
    for split in ("train", "validation"):
        for label in (1, 0):
            candidates = [
                row for row in rows if row["split"] == split and row["label"] == label
            ]
            rng.shuffle(candidates)
            count = 0
            for row in candidates:
                if row["video"] in seen:
                    continue
                chosen.append({**row, "audit_sampling": "stratified_random"})
                seen.add(row["video"])
                count += 1
                if count == random_per_stratum:
                    break
            if count != random_per_stratum:
                raise ValueError("Not enough independent source videos for blind audit")
    scores = {
        (entry["video"], round(entry["start_seconds"], 6)): entry["score"]
        for entry in predictions
    }
    remaining = [
        row
        for row in rows
        if row["split"] == "validation"
        and row_id(row) not in {row_id(item) for item in chosen}
    ]
    for label in (0, 1):
        ranked = sorted(
            remaining,
            key=lambda row: abs(
                scores[(row["video"], round(row["start"] / row["fps"], 6))]
                - row["label"]
            ),
            reverse=True,
        )
        count, per_video = 0, Counter()
        for row in ranked:
            if row["label"] != label or per_video[row["video"]] >= 2:
                continue
            chosen.append({**row, "audit_sampling": "error_enriched"})
            per_video[row["video"]] += 1
            count += 1
            if count == errors_per_class:
                break
        if count != errors_per_class:
            raise ValueError("Not enough validation error candidates")
    rng.shuffle(chosen)
    return [
        {**row, "audit_id": f"R{index + 1:03d}"} for index, row in enumerate(chosen)
    ]


def build(source, output):
    if (output / "private_mapping.json").exists():
        raise ValueError(
            "Audit already exists; reuse its pages, do not resample after reviewing"
        )
    manifest = json.loads((source / "cached_manifest.json").read_text())
    if manifest.get("official_test_used") is not False:
        raise ValueError("Blind audit is development-only, not official test")
    rows = select_review_rows(
        manifest["rows"], json.loads((source / "predictions.json").read_text())
    )
    output.mkdir(parents=True, exist_ok=True)
    pages = output / "blind_pages"
    pages.mkdir(exist_ok=True)
    write_json(
        output / "private_mapping.json",
        {"rows": rows, "source_run": str(source), "labels_changed": False},
    )
    write_json(
        output / "blind_inventory.json",
        {
            "audit_ids": [row["audit_id"] for row in rows],
            "instructions": "Review only visible action inside the 1.875s source window: fight, nonfight, uncertain. Also note visibility and crowd/scene. Do not inspect private_mapping or scores before freezing judgments. This is an AI-assisted audit, not independent human gold labels.",
        },
    )
    panels = []
    for index, row in enumerate(rows):
        frames = read_frames(row, ClipConfig())
        panel = np.full((165, 1440, 3), 245, dtype=np.uint8)
        cv2.putText(
            panel,
            row["audit_id"],
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )
        for column, frame_index in enumerate((0, 3, 6, 9, 12, 15)):
            frame = cv2.resize(frames[frame_index], (240, 135))
            panel[30:165, column * 240 : (column + 1) * 240] = frame
            cv2.putText(
                panel,
                f"{frame_index / 8:.3f}s",
                (column * 240 + 75, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
            )
        panels.append(panel)
        if len(panels) == 4 or index == len(rows) - 1:
            cv2.imwrite(
                str(pages / f"page_{index // 4 + 1:02d}.jpg"),
                np.concatenate(panels, axis=0),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            panels = []
            print(f"BLIND_AUDIT {index + 1}/{len(rows)}", flush=True)
    with zipfile.ZipFile(
        output / "blind_pages.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(pages.glob("*.jpg")):
            archive.write(path, path.name)
        archive.write(output / "blind_inventory.json", "blind_inventory.json")
    print("Audit pages ready; private labels were not included in the ZIP.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_run, args.output_dir)


if __name__ == "__main__":
    main()
