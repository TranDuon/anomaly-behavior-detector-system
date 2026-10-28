"""Prepare score-blinded public interval review; never auto-approve annotations."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from fight.temporal_labels import (
    review_template,
    source_digest,
    source_group,
    validate_sources,
)
from scripts.fight_ablation import browser_video
from scripts.fight_pilot import write_json
from scripts.fight_refinement_train import preflight, relocate


def refresh_ui(output):
    if not (output / "review.json").is_file():
        raise ValueError(
            "Existing review.json required; do not generate replacement labels"
        )
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    for source, target in (
        ("temporal_review.html", "index.html"),
        ("temporal_review_state.js", "temporal_review_state.js"),
    ):
        destination = output / target
        data = (frontend / source).read_bytes()
        if not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)


def select_sources(rows, per_cohort=6):
    if per_cohort < 3:
        raise ValueError("Review at least three independent performances per cohort")
    validate_sources(rows)
    selected, groups = [], {}
    for row in sorted(rows, key=lambda row: (row["sha256"], row["video"])):
        cohort = (row["split"], row["dataset"], row["label"])
        seen = groups.setdefault(cohort, set())
        group = source_group(row)
        if group in seen or len(seen) >= per_cohort:
            continue
        seen.add(group)
        selected.append(row)
    return selected


def make_preview(row, destination):
    capture = cv2.VideoCapture(
        str(row["path"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    partial = destination.with_name(destination.stem + ".partial.mp4")
    writer = None
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open reviewed public source")
        while True:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            factor = min(1.0, 640 / width)
            dimensions = (
                max(2, int(width * factor) // 2 * 2),
                max(2, int(height * factor) // 2 * 2),
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(partial),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    row["fps"],
                    dimensions,
                )
                if not writer.isOpened():
                    raise ValueError("Cannot encode review preview")
            writer.write(cv2.resize(frame, dimensions))
            count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if count != row["frames"]:
        raise ValueError("Preview timing differs from source manifest")
    browser_video(partial)
    preview = cv2.VideoCapture(str(partial))
    try:
        if (
            not preview.isOpened()
            or round(preview.get(cv2.CAP_PROP_FRAME_COUNT)) != row["frames"]
            or abs(preview.get(cv2.CAP_PROP_FPS) - row["fps"]) > 1e-3
        ):
            raise ValueError("Browser conversion changed reviewed source timing")
    finally:
        preview.release()
    partial.replace(destination)


def prepare(manifest_path, output, data=None, per_cohort=6):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    selected = select_sources(rows, per_cohort)
    signature = {
        "source_manifest_sha256": source_digest(rows),
        "selected_videos": [row["video"] for row in selected],
        "with_media": bool(data),
        "per_cohort": per_cohort,
        "preview_codec": "h264-yuv420p-v1",
    }
    state_path = output / "preparation.json"
    completed = {}
    if output.exists() and any(output.iterdir()):
        if (output / "review.json").exists() or not state_path.exists():
            raise ValueError(
                "Keep the existing review pack; no silent resampling/overwrite"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "preparing" or state.get("signature") != signature:
            raise ValueError(
                "Cannot resume preparation with a different source/selection"
            )
        completed = state.get("completed_previews", {})
    if data:
        selected = relocate(selected, data)
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir(exist_ok=True)

    def save_state(status):
        partial_state = state_path.with_suffix(".partial.json")
        write_json(
            partial_state,
            {
                "status": status,
                "signature": signature,
                "completed_previews": completed,
                "review_sources": len(selected),
                "with_media": bool(data),
                "training_started": False,
                "approved_for_demo": False,
            },
        )
        partial_state.replace(state_path)

    save_state("preparing")
    template = review_template(rows)
    chosen = {row["video"] for row in selected}
    template["sources"] = [row for row in template["sources"] if row["video"] in chosen]
    items = []
    for index, row in enumerate(selected):
        filename = hashlib.sha256(row["video"].encode()).hexdigest()[:24] + ".mp4"
        media = None
        if data:
            destination = output / "media" / filename
            if (
                filename not in completed
                or not destination.exists()
                or hashlib.sha256(destination.read_bytes()).hexdigest()
                != completed[filename]
            ):
                make_preview(row, destination)
                completed[filename] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()
                save_state("preparing")
            media = "media/" + filename
        items.append(
            {
                "id": f"R{index+1:03}",
                "video": row["video"],
                "media": media,
                "duration": row["frames"] / row["fps"],
            }
        )
    write_json(output / "annotations.pending.json", template)
    write_json(
        output / "review.json",
        {
            "annotations": template,
            "items": items,
            "scope": "Source selection uses frozen SHA ordering and grouped splits, not predictions. Pending intervals are never training labels. Previews retain source FPS/frame count; model training reads original source files.",
        },
    )
    refresh_ui(output)
    save_state("ready")
    print(
        f"Prepared {len(selected)} PENDING public sources; media={bool(data)}. No training or approval. Serve locally: python -m http.server 8002 --bind 127.0.0.1 --directory {output}"
    )
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build = subparsers.add_parser("prepare")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--data", type=Path)
    build.add_argument("--per-cohort", type=int, default=6)
    check = subparsers.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--annotations", type=Path, required=True)
    refresh = subparsers.add_parser("refresh-ui")
    refresh.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.manifest, args.output, args.data, args.per_cohort)
    elif args.mode == "refresh-ui":
        refresh_ui(args.output)
    else:
        manifest, windows, report = preflight(args.manifest, args.annotations)
        print(json.dumps(report, indent=2))
        if not report["ready_for_temporal_pilot"]:
            raise SystemExit(
                "Temporal labels pending/insufficient; GPU training blocked"
            )


if __name__ == "__main__":
    main()
