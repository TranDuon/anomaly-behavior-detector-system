"""Compare immutable public punch/kick windows without training or private uploads."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from fight.ava import AVA_CHECKPOINTS, AvaPersonActions
from fight.pose import PoseConfig, PoseDetector
from fight.refinement import load_deployed_base
from fight.reviewed_events import reviewed_event_metrics, summarize_events
from fight.vision import ClipConfig, build_clip_inputs, clip_tensor
from scripts.fight_fusion_train import events_from_trace
from scripts.fight_pilot import write_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(corpus):
    if corpus.get("schema_version") != 1:
        raise ValueError("Unsupported public benchmark schema")
    groups, videos = {}, set()
    allowed = {
        "hit",
        "kick",
        "hit_or_kick",
        "normal",
        "gesture",
        "hug",
        "handshake_greet",
        "highfive_clap",
        "solo_motion",
        "giving",
        "pat_back",
    }
    for row in corpus["sources"]:
        if row["video"] in videos or row["split"] not in {
            "calibration",
            "development",
            "test",
        }:
            raise ValueError(
                "Duplicate source or invalid split; never train/private inputs"
            )
        videos.add(row["video"])
        groups.setdefault(row["group"], set()).add(row["split"])
        if row["dataset"] not in {"airtlab", "pku"} or len(row["sha256"]) != 64:
            raise ValueError("Only fingerprinted public sources allowed")
        end = -1
        for interval in row["intervals"]:
            start, stop = interval["start_seconds"], interval["end_seconds"]
            if (
                interval["action"] not in allowed
                or not 0 <= start < stop <= row["frames"] / row["fps"] + 1e-6
                or start < end
            ):
                raise ValueError("Invalid/overlapping reviewed action intervals")
            if interval["label"] != int(
                interval["action"] in {"hit", "kick", "hit_or_kick"}
            ):
                raise ValueError("Action labels conflict with punch/kick policy")
            end = stop
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Performance-group split leakage")


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with (
        urllib.request.urlopen(url, timeout=90) as response,
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    partial.replace(target)


def prepare(corpus, data, pku):
    records = []
    for row in corpus["sources"]:
        relative = row.get("provided_path", row["video"])
        target = (data / relative).resolve()
        if not target.is_relative_to(data.resolve()):
            raise ValueError("Unsafe public source path")
        if not target.exists():
            if row["dataset"] == "pku":
                if not pku or not pku.is_file():
                    raise ValueError(
                        "Missing provided PKU 0360-R source; do not substitute another recording"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(pku, target)
            else:
                if (
                    corpus["source_repository"]
                    != "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos"
                ):
                    raise ValueError("Unexpected public repository")
                download(
                    corpus["source_repository"]
                    + "/"
                    + corpus["source_revision"]
                    + "/"
                    + row["repository_path"],
                    target,
                )
        if sha(target) != row["sha256"]:
            raise ValueError("Public source SHA mismatch: " + row["video"])
        records.append({**row, "path": str(target)})
    return records


def labeled_points(traces, sources):
    result = []
    by_video = {row["video"]: row for row in sources}
    for trace in traces:
        source = by_video[trace["video"]]
        for point in trace["predictions"]:
            anchor = point["timestamp"] - 0.5
            interval = next(
                (
                    item
                    for item in source["intervals"]
                    if item["start_seconds"] <= anchor < item["end_seconds"]
                ),
                None,
            )
            if interval:
                result.append(
                    {
                        **point,
                        "label": interval["label"],
                        "action": interval["action"],
                        "video": source["video"],
                        "group": source["group"],
                    }
                )
    return result


def metrics(points, threshold):
    threshold = float(threshold)
    if not points:
        raise ValueError("No reviewed action observations")
    tp = sum(row["label"] == 1 and row["score"] >= threshold for row in points)
    fp = sum(row["label"] == 0 and row["score"] >= threshold for row in points)
    fn = sum(row["label"] == 1 and row["score"] < threshold for row in points)
    tn = len(points) - tp - fp - fn
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "points": len(points),
        "independent_groups": len({row["group"] for row in points}),
    }


def action_report(points, threshold):
    return {
        action: {
            "observations": len(selected),
            "groups": len({row["group"] for row in selected}),
            "alarm_fraction": sum(row["score"] >= threshold for row in selected)
            / len(selected),
            "positive_action": bool(selected[0]["label"]),
        }
        for action in sorted({row["action"] for row in points})
        for selected in [[row for row in points if row["action"] == action]]
    }


def calibrate(points):
    if {row["label"] for row in points} != {0, 1}:
        raise ValueError("Calibration requires reviewed positive and normal time")
    values = np.unique([row["score"] for row in points])
    choices = np.unique(
        np.clip(np.r_[1e-6, (values[:-1] + values[1:]) / 2, 1 - 1e-6], 1e-6, 1 - 1e-6)
    )
    return float(
        max(
            choices,
            key=lambda threshold: (
                min(
                    metrics(points, threshold)["precision"],
                    metrics(points, threshold)["recall"],
                ),
                metrics(points, threshold)["f1"],
                threshold,
            ),
        )
    )


def event_report(traces, sources, threshold, confirm):
    by_video = {row["video"]: row for row in sources}
    reports = []
    for trace in traces:
        intervals = [
            {**interval, "action": "fight" if interval["label"] else interval["action"]}
            for interval in by_video[trace["video"]]["intervals"]
        ]
        reports.append(
            reviewed_event_metrics(
                events_from_trace(trace, threshold, confirm), intervals
            )
        )
    return summarize_events(reports)


def trace(model, name, source, detector, config, device, deadline):
    capture = cv2.VideoCapture(
        source["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    if not capture.isOpened():
        raise ValueError("Cannot decode " + source["video"])
    sample_fps = 12 if name == "baseline" else 30
    required_frames = 16 if name == "baseline" else 31
    sampled = deque(maxlen=required_frames)
    times = deque(maxlen=required_frames)
    next_sample, previous, index, predictions = 0.0, None, 0, []
    cadence = 0.5
    detector.cache.clear()
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Comparison budget exhausted; completed source traces retained"
                )
            success, frame = capture.read()
            if not success:
                break
            stamp = index / source["fps"]
            index += 1
            if stamp + 1e-6 < next_sample:
                continue
            while next_sample <= stamp + 1e-6:
                next_sample += 1 / sample_fps
            sampled.append(frame)
            times.append(stamp)
            detector.retain(sampled)
            if len(sampled) < required_frames or (
                previous is not None and stamp - previous + 1e-6 < cadence
            ):
                continue
            previous = stamp
            started = time.perf_counter()
            actions = []
            if name == "baseline":
                clips, regions, people, bounds, counts = build_clip_inputs(
                    list(sampled), detector, config
                )
                with torch.inference_mode():
                    logits = model(
                        clip_tensor(clips, device),
                        torch.zeros(len(clips), 8, 2, 17, 3, device=device),
                    )
                    raw_score = float(torch.sigmoid(logits).max().cpu())
            else:
                window = [
                    image
                    for image, timestamp in zip(sampled, times)
                    if timestamp >= stamp - 1.0 - 1e-6
                ]
                boxes = [
                    record["box"] for record in detector.poses(window[len(window) // 2])
                ]
                actions = model.predict(window, boxes)
                people = len(boxes)
                raw_score = max(
                    (
                        max(action["fight_hit_person"], action["kick_person"])
                        for action in actions
                    ),
                    default=0.0,
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            predictions.append(
                {
                    "timestamp": stamp,
                    "score": raw_score if people else 0.0,
                    "raw_score": raw_score,
                    "people": people,
                    "actions": actions,
                    "inference_seconds": time.perf_counter() - started,
                }
            )
    finally:
        capture.release()
    if index != source["frames"]:
        raise ValueError("Source frame count changed: " + source["video"])
    return {
        "video": source["video"],
        "split": source["split"],
        "duration": index / source["fps"],
        "predictions": predictions,
    }


def run(args):
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    validate(corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "corpus_sha256": sha(args.corpus),
        "device": args.device,
        "code_sha256": sha(Path(__file__)),
        "ava_code_sha256": sha(Path(__file__).resolve().parents[1] / "fight/ava.py"),
        "pose_sha256": sha(args.models / "yolov8n-pose.pt"),
        "baseline_sha256": sha(args.models / "fight_model.ts"),
        "variants": args.variants,
        "torch": torch.__version__,
        "training_steps": 0,
    }
    protocol_path = args.output / "input_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(
            "Existing comparison has different inputs/code; use a new output"
        )
    write_json(protocol_path, protocol)
    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = prepare(corpus, args.data, args.pku)
    args.weights.mkdir(parents=True, exist_ok=True)
    base, metadata, digest = load_deployed_base(args.models, False)
    detector = PoseDetector(
        PoseConfig(**metadata["pose"]), model_path=args.models / "yolov8n-pose.pt"
    )
    detector.warmup()
    torch.set_num_threads(4)
    all_traces, summary = {}, {
        "status": "running",
        "training_steps": 0,
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": corpus["scope"],
        "limitations": corpus["protocol"]["scope_limitations"],
        "variants": {},
    }
    try:
        for name in args.variants:
            if name == "baseline":
                model = base.base.to(args.device).eval()
                model_sha = digest
            else:
                checkpoint = args.weights / (name + ".pyth")
                if not checkpoint.exists():
                    download(AVA_CHECKPOINTS[name], checkpoint)
                model = AvaPersonActions(checkpoint, name, args.device)
                model_sha = model.checkpoint_sha256
            collected = []
            for split in ("calibration", "development"):
                for source in [row for row in sources if row["split"] == split]:
                    target = (
                        args.output
                        / name
                        / (
                            hashlib.sha256(source["video"].encode()).hexdigest()[:24]
                            + ".json"
                        )
                    )
                    if target.exists():
                        record = json.loads(target.read_text())
                        if record["model_sha256"] != model_sha:
                            raise ValueError("Cached trace checkpoint changed")
                        current = record["trace"]
                    else:
                        current = trace(
                            model,
                            name,
                            source,
                            detector,
                            ClipConfig(**metadata["clip"]),
                            args.device,
                            deadline,
                        )
                        write_json(
                            target, {"model_sha256": model_sha, "trace": current}
                        )
                    collected.append(current)
            calibration = labeled_points(
                [row for row in collected if row["split"] == "calibration"], sources
            )
            development = [row for row in collected if row["split"] == "development"]
            threshold = calibrate(calibration)
            choices = [
                (confirm, event_report(development, sources, threshold, confirm))
                for confirm in (0.0, 0.4, 0.8)
            ]
            confirm, events = max(
                choices,
                key=lambda item: (
                    min(item[1]["precision"], item[1]["recall"]),
                    -item[1]["false_alerts_per_normal_minute"],
                    -item[1]["normal_alert_fraction"],
                ),
            )
            report = {
                "model_sha256": model_sha,
                "threshold": threshold,
                "confirm_seconds": confirm,
                "calibration": metrics(calibration, threshold),
                "development_events": events,
                "development_windows": metrics(
                    labeled_points(development, sources), threshold
                ),
                "development_actions": action_report(
                    labeled_points(development, sources), threshold
                ),
                "cadence_seconds": 0.5,
                "gpu_timing_note": "Synchronous source-window inference, not CPU or actual webcam throughput",
            }
            all_traces[name] = collected
            summary["variants"][name] = report
            write_json(args.output / "progress.json", summary)
            print("ACTION_COMPARE " + name + " " + json.dumps(report), flush=True)
            if name != "baseline":
                del model
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
        selected = max(
            summary["variants"],
            key=lambda name: (
                min(
                    summary["variants"][name]["development_events"]["precision"],
                    summary["variants"][name]["development_events"]["recall"],
                ),
                -summary["variants"][name]["development_events"][
                    "false_alerts_per_normal_minute"
                ],
                -summary["variants"][name]["development_events"][
                    "normal_alert_fraction"
                ],
            ),
        )
        frozen_path = args.output / "selection_frozen.json"
        frozen = {
            "selected": selected,
            "variants": summary["variants"],
            "protocol": protocol,
            "selection_scope": "Calibration threshold and development events only, no private data or test selection",
        }
        if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
            raise ValueError("Frozen selection changed; no test retuning")
        write_json(frozen_path, frozen)
        if selected == "baseline":
            model = base.base.to(args.device).eval()
        else:
            model = AvaPersonActions(
                args.weights / (selected + ".pyth"), selected, args.device
            )
        testing = [
            trace(
                model,
                selected,
                row,
                detector,
                ClipConfig(**metadata["clip"]),
                args.device,
                deadline,
            )
            for row in sources
            if row["split"] == "test"
        ]
        report = summary["variants"][selected]
        summary.update(
            selected=selected,
            status="completed",
            test_events=event_report(
                testing, sources, report["threshold"], report["confirm_seconds"]
            ),
            test_windows=metrics(labeled_points(testing, sources), report["threshold"]),
            test_actions=action_report(
                labeled_points(testing, sources), report["threshold"]
            ),
        )
        write_json(args.output / "test_traces.json", testing)
    except BaseException as error:
        summary.update(status="failed", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(args.output / "run.json", summary)
    print("ACTION_COMPARE_FINAL " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("corpus", "models", "data", "weights", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--pku", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", *AVA_CHECKPOINTS],
        default=["baseline", *AVA_CHECKPOINTS],
    )
    args = parser.parse_args()
    if not 1 <= args.minutes <= 60 or len(set(args.variants)) != len(args.variants):
        parser.error("Require unique variants and a bounded 1–60 minute run")
    run(args)


if __name__ == "__main__":
    main()
