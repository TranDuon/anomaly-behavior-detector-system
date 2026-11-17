"""H3: temporal stability of a hand-crop package on video clips (every clip shows a held weapon).

    python -m extensions.sharp_object.eval_hand_clips --package PACKAGE_DIR --clips CLIPS_DIR --out clips.json

PACKAGE_DIR is the exported folder (sharp_object_spec.json + classifier + pose weights). Each
clip is sampled at --fps; the step score is the highest hand score, as at runtime. Reported per
clip: share of steps at/above the alert threshold, the number of drops (a step below threshold
right after one above it - the overlay flicker), and score min/median. --extra-package scores the
same steps with a second package (e.g. the previous run) for a side-by-side comparison.

Clips that also gave training frames only measure stability in that room, not generalisation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_package(folder, device):
    import torch
    from ultralytics import YOLO

    from extensions.sharp_object.hand_crop import crop_params, load_classifier

    spec = json.loads((folder / "sharp_object_spec.json").read_text(encoding="utf-8"))
    model, _ = load_classifier(folder / spec["classifier"], device)
    return {"spec": spec, "classifier": model.float().eval(), "pose": YOLO(str(folder / spec["pose_weights"])),
            "crop": crop_params(spec), "threshold": float(spec["alert_threshold"]), "torch": torch}


def step_score(package, frame, device):
    from extensions.sharp_object.hand_crop import cut, hand_crops, people_from_pose, to_tensor

    spec = package["spec"]
    result = package["pose"].predict(frame, imgsz=int(spec.get("pose_imgsz", 640)), conf=float(spec.get("pose_conf", 0.35)),
                                     verbose=False, device=device)[0]
    crops = [cut(frame, hand["box"], int(spec.get("input_size", 224)))
             for box, keypoints in people_from_pose(result, float(spec.get("pose_conf", 0.35)), int(spec.get("max_people", 8)))
             for hand in hand_crops(box, keypoints, package["crop"])]
    if not crops:
        return 0.0
    with package["torch"].inference_mode():
        logits = package["classifier"](to_tensor(crops).to(device))
    return float(package["torch"].sigmoid(logits.squeeze(1)).max())


def summarise(scores, threshold):
    above = [s >= threshold for s in scores]
    drops = sum(1 for previous, current in zip(above, above[1:]) if previous and not current)
    return {"steps": len(scores), "alert_share": round(float(np.mean(above)), 3) if scores else None, "drops": drops,
            "min": round(min(scores), 3) if scores else None, "median": round(float(np.median(scores)), 3) if scores else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--extra-package", type=Path, default=None, help="second package to compare (e.g. hand_01)")
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import cv2
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    packages = {"new": load_package(args.package, device)}
    if args.extra_package:
        packages["previous"] = load_package(args.extra_package, device)
    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    report = {"fps": args.fps, "thresholds": {k: p["threshold"] for k, p in packages.items()}, "clips": {}}
    totals = {k: [] for k in packages}
    for clip in clips:
        video = cv2.VideoCapture(str(clip))
        native = video.get(cv2.CAP_PROP_FPS) or 30.0
        stride, index, scores = max(1, round(native / args.fps)), 0, {k: [] for k in packages}
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if index % stride == 0:
                for name, package in packages.items():
                    scores[name].append(step_score(package, frame, device))
            index += 1
        video.release()
        report["clips"][clip.name] = {k: summarise(v, packages[k]["threshold"]) for k, v in scores.items()}
        for name in packages:
            totals[name] += scores[name]
        print(clip.name, json.dumps(report["clips"][clip.name]), flush=True)
    report["overall"] = {k: summarise(v, packages[k]["threshold"]) for k, v in totals.items()}
    report["overall_drops"] = {k: sum(c[k]["drops"] for c in report["clips"].values()) for k in packages}
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "drops": report["overall_drops"]}, indent=1))


if __name__ == "__main__":
    main()
