"""Recover non-test checkpoint diagnostics without another optimizer step."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from fight.pose import PoseConfig
from fight.refinement import load_deployed_base
from fight.vision import ClipConfig
from scripts.fight_pilot import write_json
from scripts.fight_refinement_train import (
    action_window_report,
    cache_windows,
    preflight,
    score_rows,
    window_calibration,
    window_development,
    window_key,
)


def load_checkpoint(model, path, interaction):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("use_interaction") is not interaction:
        raise ValueError("Checkpoint interaction mode differs from the named variant")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing checkpoint state dictionary")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Invalid or non-finite checkpoint weights")
    model.load_state_dict(state, strict=True)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source(run, ready, source_sha):
    if run.get("source_classifier_sha256") != source_sha:
        raise ValueError("Source classifier differs from the interrupted run")
    for key in ("annotations_sha256", "source_manifest_sha256"):
        if run.get("readiness", {}).get(key) != ready[key]:
            raise ValueError("Frozen review identity differs: " + key)
    if not ready["ready_for_temporal_pilot"]:
        raise ValueError("Reviewed labels do not pass preflight")


def non_test_rows(windows):
    selected = [
        row for row in windows if row["split"] in {"development", "calibration"}
    ]
    if {row["split"] for row in selected} != {"development", "calibration"}:
        raise ValueError("Both non-test diagnostic partitions are required")
    return selected


def run(args):
    if args.output.exists():
        raise ValueError("Refuse replacing an earlier checkpoint audit")
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    torch.set_num_threads(4)
    original = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    manifest, windows, ready = preflight(args.manifest, args.annotations)
    base, metadata, source_sha = load_deployed_base(args.models, False)
    validate_source(original, ready, source_sha)
    pose_weights = args.models / "yolov8n-pose.pt"
    pose_sha = hashlib.sha256(pose_weights.read_bytes()).hexdigest()
    if pose_sha != manifest["pose_sha256"] or metadata["clip"] != manifest["clip"]:
        raise ValueError("Frozen preprocessing or detector weights differ")
    selected = non_test_rows(windows)
    clip = ClipConfig(**metadata["clip"])
    detector = SimpleNamespace(config=PoseConfig(**metadata["pose"]))
    for row in selected:
        key = window_key(row, clip, source_sha, pose_sha, detector.config)
        if not (args.cache / (key + ".npz")).is_file():
            raise ValueError(
                "Missing reviewed cache; no silent source substitution: " + key
            )
    selected = cache_windows(
        selected,
        base,
        detector,
        clip,
        args.cache,
        source_sha,
        pose_sha,
        "cpu",
        deadline,
    )
    development = [row for row in selected if row["split"] == "development"]
    calibration = [row for row in selected if row["split"] == "calibration"]
    report = {
        "status": "non_test_window_audit_only",
        "scope": "CPU checkpoint recovery on frozen reviewed public windows; not continuous-event, test, upload or physical-phone acceptance",
        "source_classifier_sha256": source_sha,
        "readiness": ready,
        "optimizer_steps": 0,
        "test_windows_scored": 0,
        "approved_for_demo": False,
        "variants": {},
    }
    for name, interaction in (
        ("baseline", False),
        ("rgb_control", False),
        ("rgb_interaction", True),
    ):
        model = copy.deepcopy(base)
        model.use_interaction = interaction
        checkpoint_sha = None
        if name != "baseline":
            checkpoint_sha = load_checkpoint(
                model, args.run / name / "best.pt", interaction
            )
        model.eval()
        development_scores = score_rows(
            model, development, args.cache, "cpu", deadline=deadline
        )
        calibration_scores = score_rows(
            model, calibration, args.cache, "cpu", deadline=deadline
        )
        calibrated = window_calibration(calibration, calibration_scores)
        record = {
            "checkpoint_sha256": checkpoint_sha,
            "development": window_development(development, development_scores),
            "calibration_diagnostic": calibrated,
            "calibration_by_action": action_window_report(
                calibration, calibration_scores, calibrated["threshold"]
            ),
            "calibration_predictions": [
                {
                    "video": row["video"],
                    "start": row["start"],
                    "label": row["label"],
                    "score": float(score),
                }
                for row, score in zip(calibration, calibration_scores)
            ],
        }
        if name == "rgb_control":
            frozen = json.loads(
                (args.run / name / "protocol_frozen.json").read_text(encoding="utf-8")
            )
            record["original_frozen_threshold"] = frozen["threshold"]
            record["original_frozen_event_gate"] = frozen["event_gate"]
            record["original_development_events"] = frozen[
                "development_events_by_dataset"
            ]
            record["threshold_cpu_gpu_absolute_difference"] = abs(
                calibrated["threshold"] - frozen["threshold"]
            )
            if record["threshold_cpu_gpu_absolute_difference"] > 1e-4:
                raise ValueError(
                    "Recovered control calibration differs from the frozen GPU protocol"
                )
        report["variants"][name] = record
        print(
            "CHECKPOINT_AUDIT "
            + json.dumps(
                {
                    "variant": name,
                    "checkpoint_sha256": checkpoint_sha,
                    "development_bce": record["development"]["bce"],
                    "threshold": calibrated["threshold"],
                }
            ),
            flush=True,
        )
    report["seconds"] = time.monotonic() - started
    write_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.budget_minutes <= 30:
        parser.error("Audit budget must be between 1 and 30 minutes")
    run(args)


if __name__ == "__main__":
    main()
