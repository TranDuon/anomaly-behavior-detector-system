"""Reviewed-window fine-tuning from the complete deployed RGB-aux classifier."""

import argparse
import copy
import hashlib
import json
import random
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from fight.interaction_eval import balanced_calibration
from fight.model import prepare_export
from fight.pair_motion import MotionConfig, motion_inputs
from fight.pose import PoseConfig, PoseDetector, pose_inputs
from fight.refinement import load_deployed_base
from fight.reviewed_events import reviewed_event_metrics, summarize_events
from fight.runtime import FightRuntime
from fight.temporal_labels import (
    ACTIONS,
    readiness,
    reviewed_windows,
    source_digest,
    source_group,
    validate_annotations,
)
from fight.vision import ClipConfig, build_clip_inputs, clip_tensor
from scripts.fight_fusion_train import (
    development_result,
    events_from_trace,
    progress_allowed,
    selection_key,
)
from scripts.fight_large_train import measured
from scripts.fight_pilot import read_frames, write_json

WINDOW_SCOPE = "Reviewed action-window diagnostic; correlated windows and repeated public splits, not camera accuracy"


def action_window_report(rows, scores, threshold):
    records = {}
    for action in ACTIONS:
        if action == "uncertain":
            continue
        selected = [
            (row, float(score))
            for row, score in zip(rows, scores)
            if action in row["actions"]
        ]
        negatives = sum(not row["label"] for row, score in selected)
        positives = len(selected) - negatives
        false_positive = sum(
            not row["label"] and score >= threshold for row, score in selected
        )
        true_positive = sum(
            bool(row["label"]) and score >= threshold for row, score in selected
        )
        records[action] = {
            "windows": len(selected),
            "independent_source_groups": len(
                {source_group(row) for row, score in selected}
            ),
            "false_positive": false_positive,
            "false_negative": positives - true_positive,
            "false_alarm_fraction": false_positive / negatives if negatives else None,
            "recall": true_positive / positives if positives else None,
            "maximum_score": max((score for row, score in selected), default=None),
            "scope": WINDOW_SCOPE,
        }
    return records


def action_event_report(traces, approved, threshold, confirm):
    records = {}
    for action in ACTIONS:
        if action in {"fight", "uncertain"}:
            continue
        results = []
        for trace in traces:
            intervals = [
                item for item in approved[trace["video"]] if item["action"] == action
            ]
            if intervals:
                results.append(
                    reviewed_event_metrics(
                        events_from_trace(trace, threshold, confirm), intervals
                    )
                )
        records[action] = {
            **summarize_events(results),
            "sources": len(results),
            "scope": "Per-action reviewed normal time; an alarm may touch more than one action category. Not physical camera accuracy.",
        }
    return records


def window_development(rows, scores):
    result = development_result(rows, scores)
    for cohort in result["cohorts"].values():
        cohort["metric_scope"] = WINDOW_SCOPE
    return result


def window_calibration(rows, scores):
    result = balanced_calibration(rows, scores)
    for cohort in result["cohorts"].values():
        cohort["metric_scope"] = WINDOW_SCOPE
    return result


class DevicePredictor(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model, self.device = model, device

    def forward(self, clips, poses, motion):
        return self.model(
            clips.to(self.device), poses.to(self.device), motion.to(self.device)
        )


def trace_fingerprint(model, row, metadata, cache_context):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    identity = {
        "schema": 1,
        "weights": digest.hexdigest(),
        "interaction": model.use_interaction,
        "source": {
            key: row[key]
            for key in ("video", "sha256", "fps", "frames", "split", "dataset")
        },
        "clip": metadata["clip"],
        "pose": metadata["pose"],
        "context": cache_context,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def trace_sources(
    model,
    rows,
    metadata,
    detector,
    device,
    deadline,
    trace_cache=None,
    cache_context=None,
):
    if trace_cache is not None and not cache_context:
        raise ValueError(
            "Persistent traces require explicit detector/runtime provenance"
        )
    model.eval()
    traces = []
    sources = {row["video"]: row for row in rows}
    for row in sources.values():
        saved = None
        if trace_cache is not None:
            key = trace_fingerprint(model, row, metadata, cache_context)
            saved = Path(trace_cache) / (key + ".json")
            if saved.exists():
                record = json.loads(saved.read_text(encoding="utf-8"))
                if record.get("identity") != key:
                    raise ValueError("Continuous trace cache identity mismatch")
                payload = json.dumps(record["trace"], sort_keys=True, allow_nan=False)
                if hashlib.sha256(payload.encode()).hexdigest() != record.get(
                    "checksum"
                ):
                    raise ValueError("Continuous trace cache checksum mismatch")
                traces.append(record["trace"])
                continue
        runtime = FightRuntime(
            predictor=DevicePredictor(model, device),
            config=ClipConfig(**metadata["clip"]),
            pose_detector=detector,
            input_kind="rgb_pose_motion",
        )
        runtime.metadata = {
            "use_pose": model.use_interaction,
            "use_motion": model.use_interaction,
        }
        capture = cv2.VideoCapture(
            str(row["path"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        predictions, frame_index = [], 0
        try:
            if not capture.isOpened():
                raise ValueError("Cannot decode reviewed continuous source")
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Reviewed-event trace budget exhausted")
                success, frame = capture.read()
                if not success:
                    break
                previous = runtime.result.get("prediction_timestamp")
                result = runtime.push(frame, frame_index / row["fps"])
                if (
                    result.get("prediction_timestamp") is not None
                    and result["prediction_timestamp"] != previous
                ):
                    predictions.append(
                        {
                            "timestamp": result["prediction_timestamp"],
                            "score": result["score"],
                            "people": result["people"],
                        }
                    )
                frame_index += 1
        finally:
            capture.release()
        if frame_index != row["frames"]:
            raise ValueError(
                "Continuous source frame count differs from reviewed manifest"
            )
        traces.append(
            {
                "video": row["video"],
                "dataset": row["dataset"],
                "split": row["split"],
                "duration": frame_index / row["fps"],
                "predictions": predictions,
            }
        )
        if saved is not None:
            payload = json.dumps(traces[-1], sort_keys=True, allow_nan=False)
            write_json(
                saved,
                {
                    "identity": key,
                    "trace": traces[-1],
                    "checksum": hashlib.sha256(payload.encode()).hexdigest(),
                },
            )
    return traces


def event_cohorts(traces, approved, threshold, confirm):
    records = {}
    for trace in traces:
        events = events_from_trace(trace, threshold, confirm)
        result = reviewed_event_metrics(events, approved[trace["video"]])
        records.setdefault(trace["dataset"], []).append(result)
    return {dataset: summarize_events(items) for dataset, items in records.items()}


def select_event_gate(traces, approved, threshold):
    if not traces or any(trace.get("split") != "development" for trace in traces):
        raise ValueError(
            "Gate selection is development-only; never use test or phone traces"
        )
    choices = []
    for confirm in (0.0, 0.4, 0.8):
        cohorts = event_cohorts(traces, approved, threshold, confirm)
        if not cohorts or any(
            not row["ground_truth_events"] or not row["normal_minutes"]
            for row in cohorts.values()
        ):
            raise ValueError(
                "Gate selection needs reviewed fight and normal time in every development cohort"
            )
        worst = min(min(row["precision"], row["recall"]) for row in cohorts.values())
        choices.append(
            {
                "confirm_seconds": confirm,
                "worst_precision_recall": worst,
                "cohorts": cohorts,
            }
        )
    selected = max(
        choices,
        key=lambda item: (
            item["worst_precision_recall"],
            -sum(
                row["false_alerts_per_normal_minute"]
                for row in item["cohorts"].values()
            ),
            -max(row["normal_alert_fraction"] for row in item["cohorts"].values()),
            -max(
                (
                    row["p95_start_delay_seconds"]
                    if row["p95_start_delay_seconds"] is not None
                    else 1e6
                )
                for row in item["cohorts"].values()
            ),
        ),
    )
    return {"confirm_seconds": selected["confirm_seconds"]}, choices


def window_key(row, clip, base_sha, pose_sha, pose_config=None):
    identity = {
        "source_sha256": row["sha256"],
        "start": row["start"],
        "fps": row["fps"],
        "clip": clip.to_dict(),
        "pose": asdict(pose_config or PoseConfig()),
        "pose_sha256": pose_sha,
        "base_classifier_sha256": base_sha,
        "motion": asdict(MotionConfig()),
        "schema": 2,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def preflight(manifest_path, annotation_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    clip = ClipConfig(**manifest["clip"])
    if (clip.frames, clip.fps, clip.size) != (16, 12, 182):
        raise ValueError("Source preprocessing differs from the deployed classifier")
    windows = reviewed_windows(annotations, manifest["rows"], clip)
    report = readiness(windows)
    report["windows_by_split"] = {
        split: sum(row["split"] == split for row in windows)
        for split in ("train", "development", "calibration", "test")
    }
    report["source_manifest_sha256"] = source_digest(manifest["rows"])
    report["annotations_sha256"] = hashlib.sha256(
        Path(annotation_path).read_bytes()
    ).hexdigest()
    report["review_methods"] = sorted(
        {
            item["review_method"]
            for item in annotations["sources"]
            if item["review_status"] == "approved"
        }
    )
    report["training_action_source_groups"] = {
        action: len(
            {
                source_group(row)
                for row in windows
                if row["split"] == "train" and action in row["actions"]
            }
        )
        for action in ACTIONS
        if action != "uncertain"
    }
    return manifest, windows, report


def relocate(rows, data):
    relocated = []
    verified = {}
    for row in rows:
        if row["video"] not in verified:
            original = row["path"].replace("\\", "/")
            marker = "/" + row["dataset"] + "/"
            if marker not in original:
                raise ValueError("Cannot identify source-relative video path")
            path = Path(data) / row["dataset"] / original.split(marker, 1)[1]
            if not path.is_file() or not path.resolve().is_relative_to(
                Path(data).resolve()
            ):
                raise ValueError("Missing or unsafe reviewed source: " + row["video"])
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Reviewed source bytes changed: " + row["video"])
            verified[row["video"]] = path
        relocated.append({**row, "path": str(verified[row["video"]])})
    return relocated


def cache_windows(
    rows, model, detector, clip, cache, base_sha, pose_sha, device, deadline
):
    ready = []
    cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("Reviewed-window cache budget exhausted")
        key = window_key(row, clip, base_sha, pose_sha, detector.config)
        path = cache / (key + ".npz")
        if not path.exists():
            frames = read_frames(row, clip)
            detector.retain(frames)
            clips, regions, people, bounds, counts = build_clip_inputs(
                frames, detector, clip
            )
            poses = pose_inputs(frames, regions, detector)
            with torch.inference_mode():
                features = (
                    model.base.rgb.features(clip_tensor(clips, device)).cpu().numpy()
                )
            partial = path.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                clips=clips,
                poses=poses,
                features=features,
                motion=motion_inputs(clips),
                base_sha=np.asarray(base_sha),
                cache_identity=np.asarray(key),
                people=people,
            )
            partial.replace(path)
        with np.load(path, allow_pickle=False) as data:
            if not {
                "base_sha",
                "cache_identity",
                "features",
                "clips",
                "poses",
                "motion",
            }.issubset(data.files):
                raise ValueError("Refuse legacy/missing-provenance feature cache")
            if str(data["base_sha"]) != base_sha or str(data["cache_identity"]) != key:
                raise ValueError("Refuse legacy/mismatched feature cache")
            if data["features"].shape != (len(data["clips"]), 2048):
                raise ValueError("Invalid source feature shape")
            if (
                data["clips"].shape[1:] != (16, 182, 182, 3)
                or data["clips"].dtype != np.uint8
                or data["poses"].shape != (len(data["clips"]), 8, 2, 17, 3)
                or data["motion"].shape != (len(data["clips"]), 2, 15, 32, 32)
                or any(
                    not np.isfinite(data[name]).all()
                    for name in ("features", "poses", "motion")
                )
            ):
                raise ValueError("Invalid reviewed-window tensor schema")
        ready.append({**row, "cache": key})
        if (index + 1) % 16 == 0:
            print(f"REFINEMENT_CACHE {index+1}/{len(rows)}", flush=True)
    return ready


def inputs_for(row, cache, device, features_only):
    with np.load(cache / (row["cache"] + ".npz"), allow_pickle=False) as data:
        appearance = (
            torch.from_numpy(data["features"]).to(device)
            if features_only
            else clip_tensor(data["clips"], device)
        )
        return (
            appearance,
            torch.from_numpy(data["poses"]).to(device),
            torch.from_numpy(data["motion"]).to(device),
        )


def score_rows(model, rows, cache, device, features_only=False, deadline=float("inf")):
    model.eval()
    scores = []
    with torch.inference_mode():
        for row in rows:
            if time.monotonic() >= deadline:
                raise TimeoutError("Scoring budget exhausted")
            values = inputs_for(row, cache, device, features_only)
            logits = (
                model.classify_features(*values)[0] if features_only else model(*values)
            )
            scores.append(float(torch.sigmoid(logits.max()).cpu()))
    return np.asarray(scores)


def train(model, training, development, cache, output, device, deadline):
    generator = random.Random(941)
    initial = window_development(
        development, score_rows(model, development, cache, device, True, deadline)
    )
    best, best_state, history = initial, copy.deepcopy(model.state_dict()), []
    groups = {}
    for row in training:
        groups.setdefault((row["dataset"], row["label"]), []).append(row)
    stop = "maximum_epochs"
    for stage, epochs in (("probe", 3), ("finetune", 12)):
        model.load_state_dict(best_state)
        model.set_stage(stage)
        heads = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("base.")
        ]
        backbone = [
            parameter
            for parameter in model.base.parameters()
            if parameter.requires_grad
        ]
        parameters = [{"params": heads, "lr": 3e-4 if stage == "probe" else 1e-4}]
        if backbone:
            parameters.append({"params": backbone, "lr": 1e-5})
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
        stale = 0
        for epoch in range(epochs):
            if time.monotonic() >= deadline:
                stop = "budget_stop"
                break
            order = [
                row
                for group in groups.values()
                for row in generator.choices(group, k=128)
            ]
            generator.shuffle(order)
            model.train()
            losses = []
            for start in range(0, len(order), 4):
                if time.monotonic() >= deadline:
                    stop = "budget_stop"
                    break
                batch = order[start : start + 4]
                optimizer.zero_grad()
                for row in batch:
                    values = inputs_for(row, cache, device, stage == "probe")
                    features = (
                        values[0]
                        if stage == "probe"
                        else model.base.rgb.features(values[0])
                    )
                    logits, actions = model.classify_features(
                        features, values[1], values[2]
                    )
                    primary = nn.functional.binary_cross_entropy_with_logits(
                        logits.max(), torch.tensor(float(row["label"]), device=device)
                    )
                    targets = torch.tensor(
                        row["action_targets"], dtype=torch.float32, device=device
                    )
                    mask = torch.tensor(
                        row["action_mask"], dtype=torch.float32, device=device
                    )
                    auxiliary = nn.functional.binary_cross_entropy_with_logits(
                        actions.amax(dim=0), targets, reduction="none"
                    )
                    loss = primary + 0.15 * (
                        auxiliary * mask
                    ).sum() / mask.sum().clamp_min(1)
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    1,
                )
                optimizer.step()
            if stop == "budget_stop":
                break
            result = window_development(
                development,
                score_rows(
                    model, development, cache, device, stage == "probe", deadline
                ),
            )
            record = {
                "stage": stage,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "sampled_exposures": len(order),
                **result,
            }
            history.append(record)
            if selection_key(record) > selection_key(best):
                best, best_state, stale = record, copy.deepcopy(model.state_dict()), 0
                torch.save(
                    {
                        "state_dict": best_state,
                        "use_interaction": model.use_interaction,
                    },
                    output / "best.pt",
                )
            else:
                stale += 1
            write_json(output / "history.json", history)
            print(
                "REFINEMENT_EPOCH "
                + json.dumps(
                    {key: value for key, value in record.items() if key != "cohorts"}
                ),
                flush=True,
            )
            if (
                stage == "finetune"
                and epoch >= 2
                and not progress_allowed(initial, best)
            ):
                stop = "no_protected_development_progress"
                break
            if stale >= 4:
                stop = "development_plateau"
                break
        if stop == "budget_stop":
            break
    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": best_state, "use_interaction": model.use_interaction},
        output / "best.pt",
    )
    return {
        "initial": initial,
        "development": best,
        "history_epochs": len(history),
        "stop_reason": stop,
        "protected_progress": progress_allowed(initial, best),
    }


def export(model, metadata, report, output, pose_weights):
    model = prepare_export(model.cpu().eval())
    example = (
        torch.zeros(1, 3, 16, 182, 182),
        torch.zeros(1, 8, 2, 17, 3),
        torch.zeros(1, 2, 15, 32, 32),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        traced.save(str(output / "fight_model.ts"))
        torch.testing.assert_close(
            torch.jit.load(str(output / "fight_model.ts"))(*example), model(*example)
        )
    runtime = {
        "schema_version": 1,
        "architecture": "refined_rgb_aux_x3d_s",
        "input_kind": "rgb_pose_motion",
        "clip": metadata["clip"],
        "pose": metadata["pose"],
        "motion": asdict(MotionConfig()),
        "use_pose": model.use_interaction,
        "use_motion": model.use_interaction,
        "threshold": report["threshold"],
        "event_gate": report["event_gate"],
        "approved_for_demo": False,
        "real_phone_tested": False,
        "test": report["test"],
        "test_by_dataset": report["test_by_dataset"],
        "source_classifier_sha256": report["source_classifier_sha256"],
        "deployment": {
            "display_name": "RGB-aux refinement · unaccepted pilot",
            "selection": "No auto-deployment; reviewed public windows are not camera acceptance",
        },
    }
    write_json(output / "fight_runtime.json", runtime)
    with zipfile.ZipFile(
        output / "sentinel_fight_only_runtime.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for filename in ("fight_model.ts", "fight_runtime.json"):
            archive.write(output / filename, "models/" + filename)
        archive.write(pose_weights, "models/yolov8n-pose.pt")


def run(args):
    manifest, windows, ready = preflight(args.manifest, args.annotations)
    if args.check_only:
        print(json.dumps(ready, indent=2))
        return ready
    if not ready["ready_for_temporal_pilot"]:
        raise ValueError(
            "Reviewed temporal labels not ready; do not allocate GPU: "
            + ", ".join(ready["missing"])
        )
    if (args.output / "run.json").exists():
        raise ValueError("Existing run; no silent retry/overwrite")
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    torch.manual_seed(941)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model, metadata, base_sha = load_deployed_base(args.models)
    pose_weights = args.models / "yolov8n-pose.pt"
    pose_sha = hashlib.sha256(pose_weights.read_bytes()).hexdigest()
    if pose_sha != manifest["pose_sha256"] or metadata["clip"] != manifest["clip"]:
        raise ValueError("Source preprocessing/pose weights mismatch")
    windows = relocate(windows, args.data)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    approved = validate_annotations(annotations, manifest["rows"])
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "running",
        "source_classifier_sha256": base_sha,
        "readiness": ready,
        "variants": {},
        "seed": 941,
        "scope": "Reviewed-window repeated-public pilot, not untouched camera acceptance",
        "approved_for_demo": False,
    }
    write_json(args.output / "run.json", summary)
    try:
        detector = PoseDetector(
            PoseConfig(**metadata["pose"]), device=args.device, model_path=pose_weights
        )
        model = model.to(args.device).eval()
        non_test = [row for row in windows if row["split"] != "test"]
        cached = cache_windows(
            non_test,
            model,
            detector,
            ClipConfig(**metadata["clip"]),
            args.cache,
            base_sha,
            pose_sha,
            args.device,
            deadline - 240,
        )
        partitions = {
            split: [row for row in cached if row["split"] == split]
            for split in ("train", "development", "calibration")
        }
        model.use_interaction = False
        baseline_calibration = window_calibration(
            partitions["calibration"],
            score_rows(
                model,
                partitions["calibration"],
                args.cache,
                args.device,
                True,
                deadline - 240,
            ),
        )
        baseline_traces = trace_sources(
            model,
            partitions["development"],
            metadata,
            detector,
            args.device,
            deadline - 210,
        )
        baseline_gate, baseline_choices = select_event_gate(
            baseline_traces, approved, baseline_calibration["threshold"]
        )
        summary["baseline_frozen"] = {
            "calibration": baseline_calibration,
            "event_gate": baseline_gate,
            "development_gate_candidates": baseline_choices,
            "as_deployed_development_events": event_cohorts(
                baseline_traces,
                approved,
                metadata["threshold"],
                metadata.get("event_gate", {}).get("confirm_seconds", 0.4),
            ),
            "recalibrated_development_events": event_cohorts(
                baseline_traces,
                approved,
                baseline_calibration["threshold"],
                baseline_gate["confirm_seconds"],
            ),
        }
        write_json(args.output / "baseline_development_traces.json", baseline_traces)
        write_json(args.output / "run.json", summary)
        states = {}
        for name, interaction in (("rgb_control", False), ("rgb_interaction", True)):
            torch.manual_seed(941)
            model, metadata, digest = load_deployed_base(args.models, interaction)
            model = model.to(args.device)
            output = args.output / name
            output.mkdir(exist_ok=True)
            report = train(
                model,
                partitions["train"],
                partitions["development"],
                args.cache,
                output,
                args.device,
                min(deadline - 180, time.monotonic() + args.variant_minutes * 60),
            )
            calibration = window_calibration(
                partitions["calibration"],
                score_rows(
                    model,
                    partitions["calibration"],
                    args.cache,
                    args.device,
                    deadline=deadline - 120,
                ),
            )
            report.update(
                calibration=calibration,
                threshold=calibration["threshold"],
                source_classifier_sha256=base_sha,
                use_interaction=interaction,
            )
            development_traces = trace_sources(
                model,
                partitions["development"],
                metadata,
                detector,
                args.device,
                deadline - 150,
            )
            gate, choices = select_event_gate(
                development_traces, approved, calibration["threshold"]
            )
            report["event_gate"] = {**metadata.get("event_gate", {}), **gate}
            report["development_gate_candidates"] = choices
            report["gate_selection_scope"] = (
                "Reviewed development events only, after calibration threshold; not selected on test or private phone recordings"
            )
            report["development_events_by_dataset"] = event_cohorts(
                development_traces,
                approved,
                report["threshold"],
                gate["confirm_seconds"],
            )
            report["public_development_event_target_met"] = all(
                row["precision"] >= 0.8 and row["recall"] >= 0.8
                for row in report["development_events_by_dataset"].values()
            )
            write_json(output / "development_traces.json", development_traces)
            write_json(output / "protocol_frozen.json", report)
            summary["variants"][name] = report
            states[name] = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            write_json(args.output / "run.json", summary)
        summary["recommended_by_development_only"] = max(
            summary["variants"],
            key=lambda name: (
                min(
                    min(cohort["precision"], cohort["recall"])
                    for cohort in summary["variants"][name][
                        "development_events_by_dataset"
                    ].values()
                ),
                -sum(
                    cohort["false_alerts_per_normal_minute"]
                    for cohort in summary["variants"][name][
                        "development_events_by_dataset"
                    ].values()
                ),
                selection_key(summary["variants"][name]["development"]),
            ),
        )
        write_json(args.output / "selection_frozen.json", summary)
        model, metadata, digest = load_deployed_base(args.models)
        model = model.to(args.device).eval()
        testing = cache_windows(
            [row for row in windows if row["split"] == "test"],
            model,
            detector,
            ClipConfig(**metadata["clip"]),
            args.cache,
            base_sha,
            pose_sha,
            args.device,
            deadline - 90,
        )
        model.use_interaction = False
        baseline_test_scores = score_rows(
            model, testing, args.cache, args.device, True, deadline - 75
        )
        summary["baseline_test_windows"] = {
            **measured(
                testing,
                baseline_test_scores,
                summary["baseline_frozen"]["calibration"]["threshold"],
            ),
            "metric_scope": WINDOW_SCOPE,
        }
        baseline_test_traces = trace_sources(
            model, testing, metadata, detector, args.device, deadline - 60
        )
        summary["baseline_test_events"] = event_cohorts(
            baseline_test_traces,
            approved,
            summary["baseline_frozen"]["calibration"]["threshold"],
            summary["baseline_frozen"]["event_gate"]["confirm_seconds"],
        )
        summary["baseline_test_by_action"] = action_window_report(
            testing,
            baseline_test_scores,
            summary["baseline_frozen"]["calibration"]["threshold"],
        )
        summary["baseline_normal_events_by_action"] = action_event_report(
            baseline_test_traces,
            approved,
            summary["baseline_frozen"]["calibration"]["threshold"],
            summary["baseline_frozen"]["event_gate"]["confirm_seconds"],
        )
        write_json(args.output / "baseline_test_traces.json", baseline_test_traces)
        expected = []
        with zipfile.ZipFile(
            args.output / "cpu_validation.zip", "w", zipfile.ZIP_DEFLATED
        ) as qa:
            for name, report in summary["variants"].items():
                model, metadata, digest = load_deployed_base(
                    args.models, report["use_interaction"]
                )
                model = model.to(args.device).eval()
                model.load_state_dict(states[name])
                predictions = score_rows(
                    model, testing, args.cache, args.device, deadline=deadline - 30
                )
                report["test"] = measured(testing, predictions, report["threshold"])
                report["test"]["metric_scope"] = WINDOW_SCOPE
                report["test_by_action"] = action_window_report(
                    testing, predictions, report["threshold"]
                )
                report["test_by_dataset"] = {
                    dataset: measured(
                        [row for row in testing if row["dataset"] == dataset],
                        predictions[[row["dataset"] == dataset for row in testing]],
                        report["threshold"],
                    )
                    for dataset in ("airtlab", "rlvs")
                }
                records = [
                    {**row, "score": float(score)}
                    for row, score in zip(testing, predictions)
                ]
                for cohort in report["test_by_dataset"].values():
                    cohort["metric_scope"] = WINDOW_SCOPE
                test_traces = trace_sources(
                    model, testing, metadata, detector, args.device, deadline - 20
                )
                report["test_events_by_dataset"] = event_cohorts(
                    test_traces,
                    approved,
                    report["threshold"],
                    report["event_gate"]["confirm_seconds"],
                )
                report["normal_events_by_action"] = action_event_report(
                    test_traces,
                    approved,
                    report["threshold"],
                    report["event_gate"]["confirm_seconds"],
                )
                write_json(args.output / name / "test_traces.json", test_traces)
                write_json(args.output / name / "test_predictions.json", records)
                export(model, metadata, report, args.output / name, pose_weights)
                qa.write(
                    args.output / name / "sentinel_fight_only_runtime.zip",
                    name + "/runtime.zip",
                )
                examples = [
                    next(
                        row
                        for row in records
                        if (row["dataset"], row["label"]) == cohort
                    )
                    for cohort in (("airtlab", 0), ("airtlab", 1), ("rlvs", 1))
                ]
                for row in examples:
                    qa.write(
                        args.cache / (row["cache"] + ".npz"),
                        name + "/" + row["cache"] + ".npz",
                    )
                    expected.append(
                        {
                            "variant": name,
                            "cache": row["cache"],
                            "score": row["score"],
                            "label": row["label"],
                            "video": row["video"],
                        }
                    )
                write_json(args.output / name / "report.json", report)
            qa.writestr("expected.json", json.dumps(expected, indent=2))
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(args.output / "run.json", summary)
        print("REFINEMENT_RUN " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-minutes", type=float, default=20)
    parser.add_argument("--variant-minutes", type=float, default=5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and (
        not all((args.models, args.data, args.cache, args.output))
        or not 5 <= args.budget_minutes <= 60
        or not 1 <= args.variant_minutes <= args.budget_minutes / 2
    ):
        parser.error(
            "Training requires models/data/cache/output and a valid bounded GPU budget"
        )
    run(args)


if __name__ == "__main__":
    main()
