"""Progress-gated RGB/pose and action-supervision ablation; no auto-deployment."""

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

from fight.fusion import ACTION_NAMES, InteractionFusion
from fight.interaction_data import pku_sources
from fight.interaction_eval import (
    annotated_window_metrics,
    balanced_calibration,
    event_metrics,
    macro_ap,
)
from fight.model import prepare_export
from fight.pose import PoseConfig, PoseDetector, flip_pose
from fight.runtime import EventGate, FightRuntime
from fight.vision import ClipConfig, clip_tensor, augment_handheld
from scripts.fight_fusion_data import cache_rows, prepare_rows
from scripts.fight_large_train import measured, people_focused_training
from scripts.fight_pilot import write_json


def score_rows(model, rows, cache, features_only=False):
    scores = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            with np.load(cache / (row["cache"] + ".npz")) as data:
                poses = torch.from_numpy(data["poses"]).cuda()
                if features_only:
                    logits, auxiliary = model.classify_features(
                        torch.from_numpy(data["features"]).cuda(), poses
                    )
                else:
                    logits = model(clip_tensor(data["clips"], "cuda"), poses)
                scores.append(float(torch.sigmoid(logits.max())))
    return np.asarray(scores)


def development_result(rows, scores):
    calibration = balanced_calibration(rows, scores)
    worst = min(
        min(value["precision"], value["recall"])
        for value in calibration["cohorts"].values()
    )
    labels = np.asarray([row["label"] for row in rows])
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    loss = float(
        np.mean(-labels * np.log(clipped) - (1 - labels) * np.log(1 - clipped))
    )
    return {
        "worst_precision_recall": worst,
        "macro_ap": macro_ap(rows, scores),
        "bce": loss,
        "cohorts": calibration["cohorts"],
    }


def selection_key(record):
    return (record["worst_precision_recall"], record["macro_ap"], -record["bce"])


def progress_allowed(initial, current):
    improved = (
        current["worst_precision_recall"] >= initial["worst_precision_recall"] + 0.01
        or current["macro_ap"] >= initial["macro_ap"] + 0.005
    )
    protected = all(
        current["cohorts"][name]["ap"] >= value["ap"] - 0.02
        for name, value in initial["cohorts"].items()
    )
    return improved and protected


def training_order(rows, generator):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["label"]), []).append(row)
    selected = []
    for group in groups.values():
        selected.extend(generator.choices(group, k=256))
    generator.shuffle(selected)
    return selected


def train_variant(model, training, development, cache, destination, deadline):
    torch.manual_seed(287)
    generator = random.Random(287)
    initial = development_result(
        development, score_rows(model, development, cache, True)
    )
    best, state, history = initial, copy.deepcopy(model.state_dict()), []
    torch.save(
        {
            "state_dict": state,
            "architecture": "rgb_pose_x3d_s",
            "use_pose": model.use_pose,
        },
        destination / "best.pt",
    )
    for stage, epochs in (("probe", 3), ("finetune", 18)):
        model.load_state_dict(state)
        model.set_stage(stage)
        head_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("rgb.")
        ]
        groups = [{"params": head_parameters, "lr": 5e-4 if stage == "probe" else 1e-4}]
        backbone = [
            parameter for parameter in model.rgb.parameters() if parameter.requires_grad
        ]
        if backbone:
            groups.append({"params": backbone, "lr": 1e-5})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        stale = 0
        for epoch in range(epochs):
            model.train()
            model.rgb.backbone.eval()
            order = training_order(training, generator)
            losses = []
            for batch_start in range(0, len(order), 8):
                if time.monotonic() > deadline:
                    model.load_state_dict(state)
                    return history, best, "budget_stop"
                optimizer.zero_grad()
                batch = order[batch_start : batch_start + 8]
                for row in batch:
                    with np.load(cache / (row["cache"] + ".npz")) as data:
                        poses = data["poses"].copy()
                        if generator.random() < 0.5:
                            poses = poses[:, :, ::-1].copy()
                        if stage == "probe":
                            logits, auxiliary = model.classify_features(
                                torch.from_numpy(data["features"]).cuda(),
                                torch.from_numpy(poses).cuda(),
                            )
                        else:
                            clips = data["clips"]
                            if generator.random() < 0.5:
                                clips, poses = clips[:, :, :, ::-1].copy(), flip_pose(
                                    poses
                                )
                            if generator.random() < 0.5:
                                clips = augment_handheld(
                                    clips,
                                    np.random.default_rng(generator.randrange(2**32)),
                                )
                            features = model.rgb.features(clip_tensor(clips, "cuda"))
                            logits, auxiliary = model.classify_features(
                                features, torch.from_numpy(poses).cuda()
                            )
                        primary = nn.functional.binary_cross_entropy_with_logits(
                            logits.max(),
                            torch.tensor(float(row["label"]), device="cuda"),
                        )
                        targets = torch.tensor(
                            row["action_targets"], device="cuda", dtype=torch.float32
                        )
                        mask = torch.tensor(
                            row["action_mask"], device="cuda", dtype=torch.float32
                        )
                        action_loss = nn.functional.binary_cross_entropy_with_logits(
                            auxiliary.max(dim=0).values, targets, reduction="none"
                        )
                        loss = primary + 0.15 * (
                            action_loss * mask
                        ).sum() / mask.sum().clamp_min(1)
                        (loss / len(batch)).backward()
                        losses.append(float(loss.detach()))
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            result = development_result(
                development, score_rows(model, development, cache, stage == "probe")
            )
            record = {
                "stage": stage,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "sampled_exposures": len(order),
                **result,
            }
            history.append(record)
            write_json(destination / "history.json", history)
            print(
                "FUSION_EPOCH "
                + json.dumps(
                    {key: value for key, value in record.items() if key != "cohorts"}
                ),
                flush=True,
            )
            if selection_key(result) > selection_key(best):
                best, stale = result, 0
                state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "state_dict": state,
                        "architecture": "rgb_pose_x3d_s",
                        "use_pose": model.use_pose,
                    },
                    destination / "best.pt",
                )
            else:
                stale += 1
            if (
                stage == "finetune"
                and epoch >= 2
                and not progress_allowed(initial, best)
            ):
                print(
                    "LONG_TRAIN_BLOCKED: no protected development progress", flush=True
                )
                break
            if stale >= 4:
                break
    model.load_state_dict(state)
    model.eval()
    return history, best, "early_stop_or_completed"


class DeviceFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, clips, poses):
        device = next(self.model.parameters()).device
        return self.model(clips.to(device), poses.to(device))


def trace_video(model, source, config, detector, deadline):
    runtime = FightRuntime(
        predictor=DeviceFusion(model.eval()),
        config=config,
        detector=detector,
        pose_detector=detector,
        input_kind="rgb_pose",
    )
    capture = cv2.VideoCapture(source["path"])
    predictions, frame_index, last = [], 0, None
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Continuous fusion evaluation budget exhausted")
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source["fps"]
            frame_index += 1
            result = runtime.push(frame, timestamp)
            if (
                result.get("prediction_timestamp") is not None
                and result["prediction_timestamp"] != last
            ):
                last = result["prediction_timestamp"]
                predictions.append(
                    {
                        "timestamp": last,
                        "score": result["score"],
                        "people": result["people"],
                    }
                )
    finally:
        capture.release()
    if frame_index < source["frames"] - 2:
        raise ValueError("Continuous source decoded incompletely: " + source["video"])
    return {
        "video": source["video"],
        "label": source.get("label"),
        "dataset": source.get("dataset"),
        "duration": frame_index / source["fps"],
        "predictions": predictions,
    }


def events_from_trace(trace, threshold, confirm_seconds):
    gate = EventGate(threshold, threshold * 0.7, confirm_seconds=confirm_seconds)
    events, active = [], None
    for prediction in trace["predictions"]:
        if prediction["people"] == 0:
            if active is not None:
                active["end"] = prediction["timestamp"]
            gate, active = gate.fresh(), None
            continue
        state, event = gate.update(prediction["score"], prediction["timestamp"])
        if event == "started":
            active = {"start": prediction["timestamp"], "end": None}
            events.append(active)
        elif event == "ended" and active is not None:
            active["end"], active = prediction["timestamp"], None
    if active is not None:
        active["end"] = trace["duration"]
    return events


def select_gate(traces, rows, threshold):
    choices = []
    by_video = {trace["video"]: trace for trace in traces}
    for confirm in (0.0, 0.4, 0.8):
        predictions = np.array(
            [
                bool(events_from_trace(by_video[row["video"]], threshold, confirm))
                for row in rows
            ],
            dtype=float,
        )
        cohorts = {
            name: measured(
                [row for row in rows if row["dataset"] == name],
                predictions[[row["dataset"] == name for row in rows]],
                0.5,
            )
            for name in {row["dataset"] for row in rows}
        }
        for cohort in cohorts.values():
            cohort["metric_scope"] = (
                "Development whole-clip any-alarm diagnostic; publisher clip labels, not temporal event ground truth"
            )
        worst = min(min(item["precision"], item["recall"]) for item in cohorts.values())
        choices.append(
            {
                "confirm_seconds": confirm,
                "worst_precision_recall": worst,
                "cohorts": cohorts,
            }
        )
    chosen = max(
        choices,
        key=lambda item: (item["worst_precision_recall"], item["confirm_seconds"]),
    )
    return {"confirm_seconds": chosen["confirm_seconds"]}, choices


def export(model, config, detector, report, destination):
    prepare_export(model.cpu().eval())
    torch.set_num_threads(4)
    target = destination / "runtime" / "models"
    target.mkdir(parents=True, exist_ok=True)
    example = (
        torch.zeros(1, 3, config.frames, config.size, config.size),
        torch.zeros(1, 8, 2, 17, 3),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        traced.save(str(target / "fight_model.ts"))
        torch.testing.assert_close(
            torch.jit.load(str(target / "fight_model.ts"))(*example), model(*example)
        )
    metadata = {
        **report,
        "schema_version": 1,
        "architecture": "rgb_pose_x3d_s",
        "input_kind": "rgb_pose",
        "clip": config.to_dict(),
        "pose": asdict(detector.config),
        "approved_for_demo": False,
        "real_phone_tested": False,
    }
    write_json(target / "fight_runtime.json", metadata)
    weights = Path(detector.config.model_name)
    if not weights.is_file():
        raise ValueError("Cannot bundle official pose weights")
    with zipfile.ZipFile(
        destination / "sentinel_fight_only_runtime.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for filename in ("fight_model.ts", "fight_runtime.json"):
            archive.write(target / filename, "models/" + filename)
        archive.write(weights, "models/" + weights.name)


def run(args):
    if (args.output / "run.json").exists():
        raise ValueError(
            "Existing fusion experiment; read saved files, no automatic retraining"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(287)
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    summary = {
        "status": "running",
        "variants": {},
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Repeated public split and previously inspected PKU diagnostic; not new final acceptance",
        "budget_minutes": args.budget_minutes,
        "seed": 287,
    }
    write_json(args.output / "run.json", summary)
    try:
        config = ClipConfig(
            frames=16,
            fps=12,
            size=182,
            stride_seconds=0.5,
            region_mode="pairs",
            trim_black_borders=True,
            version=8,
        )
        detector = PoseDetector(PoseConfig(), device="cuda")
        pose_sha = hashlib.sha256(
            Path(detector.config.model_name).read_bytes()
        ).hexdigest()
        checkpoint = args.baseline / "best.pt"
        weight_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = InteractionFusion().cuda().eval()
        model.rgb.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True)["state_dict"]
        )
        initial = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        rows, excluded = prepare_rows(args.baseline, args.data, config)
        write_json(
            args.output / "source_manifest.json",
            {
                "rows": rows,
                "excluded": excluded,
                "clip": config.to_dict(),
                "pose": asdict(detector.config),
                "rgb_sha256": weight_id,
                "pose_sha256": pose_sha,
                "action_names": ACTION_NAMES,
                "auxiliary_scope": "Publisher clip-level weak multi-label action tags, not human-verified temporal gold",
            },
        )
        public = cache_rows(
            [row for row in rows if row["split"] != "test"],
            model,
            detector,
            config,
            args.cache,
            weight_id,
            pose_sha,
            deadline - 900,
        )
        partitions = {
            split: [row for row in public if row["split"] == split]
            for split in ("train", "development", "calibration")
        }
        partitions["train"], training_focus = people_focused_training(
            partitions["train"], args.cache
        )
        summary["training_focus"] = training_focus
        summary["counts"] = {name: len(values) for name, values in partitions.items()}
        summary["baseline_reprocessed_development"] = development_result(
            partitions["development"],
            score_rows(model, partitions["development"], args.cache, True),
        )
        summary["baseline_reprocessed_calibration"] = balanced_calibration(
            partitions["calibration"],
            score_rows(model, partitions["calibration"], args.cache, True),
        )
        write_json(args.output / "run.json", summary)
        selected_rows = []
        for dataset in ("airtlab", "rlvs"):
            for label in (0, 1):
                selected_rows.extend(
                    sorted(
                        [
                            row
                            for row in partitions["development"]
                            if row["dataset"] == dataset and row["label"] == label
                        ],
                        key=lambda row: row["sha256"],
                    )[:8]
                )
        for name, use_pose in (("rgb_aux", False), ("rgb_pose_aux", True)):
            destination = args.output / name
            destination.mkdir(exist_ok=True)
            model.load_state_dict(initial)
            model.use_pose = use_pose
            history, development, stop_reason = train_variant(
                model,
                partitions["train"],
                partitions["development"],
                args.cache,
                destination,
                min(deadline - 600, time.monotonic() + args.variant_minutes * 60),
            )
            calibration_scores = score_rows(
                model, partitions["calibration"], args.cache
            )
            calibration = balanced_calibration(
                partitions["calibration"], calibration_scores
            )
            write_json(destination / "calibration_frozen.json", calibration)
            traces = [
                trace_video(model, row, config, detector, deadline - 240)
                for row in selected_rows
            ]
            gate, gate_choices = select_gate(
                traces, selected_rows, calibration["threshold"]
            )
            report = {
                **calibration,
                "development": development,
                "completed_epochs": len(history),
                "stop_reason": stop_reason,
                "use_pose": use_pose,
                "event_gate": gate,
                "gate_selection_scope": "Development whole-clip any-alarm labels only; no claim of frame/event start-delay accuracy",
                "gate_choices": gate_choices,
                "approved_for_demo": False,
                "real_phone_tested": False,
            }
            report["protected_development_progress"] = progress_allowed(
                summary["baseline_reprocessed_development"], development
            )
            write_json(destination / "development_traces.json", traces)
            write_json(destination / "protocol_frozen.json", report)
            trained_rgb = {
                name: value.detach().cpu().clone()
                for name, value in model.rgb.state_dict().items()
            }
            model.rgb.load_state_dict(
                {
                    name.removeprefix("rgb."): value
                    for name, value in initial.items()
                    if name.startswith("rgb.")
                }
            )
            test_rows = cache_rows(
                [row for row in rows if row["split"] == "test"],
                model,
                detector,
                config,
                args.cache,
                weight_id,
                pose_sha,
                deadline - 150,
            )
            model.rgb.load_state_dict(trained_rgb)
            if "baseline_reprocessed_test" not in summary:
                baseline_scores = []
                with torch.inference_mode():
                    for row in test_rows:
                        with np.load(args.cache / (row["cache"] + ".npz")) as data:
                            baseline_scores.append(
                                float(
                                    torch.sigmoid(
                                        model.rgb.head(
                                            torch.from_numpy(data["features"]).cuda()
                                        ).max()
                                    )
                                )
                            )
                baseline_scores = np.asarray(baseline_scores)
                baseline_threshold = summary["baseline_reprocessed_calibration"][
                    "threshold"
                ]
                summary["baseline_reprocessed_test"] = {
                    dataset: measured(
                        [row for row in test_rows if row["dataset"] == dataset],
                        baseline_scores[
                            [row["dataset"] == dataset for row in test_rows]
                        ],
                        baseline_threshold,
                    )
                    for dataset in ("airtlab", "rlvs")
                }
            test_scores = score_rows(model, test_rows, args.cache)
            report["test"] = measured(test_rows, test_scores, report["threshold"])
            report["test_by_dataset"] = {
                dataset: measured(
                    [row for row in test_rows if row["dataset"] == dataset],
                    test_scores[[row["dataset"] == dataset for row in test_rows]],
                    report["threshold"],
                )
                for dataset in ("airtlab", "rlvs")
            }
            report["hard_negative_actions"] = {}
            for action in sorted(
                {
                    tag
                    for row in test_rows
                    if row["label"] == 0
                    for tag in row["actions"]
                }
            ):
                indices = [
                    index
                    for index, row in enumerate(test_rows)
                    if row["label"] == 0 and action in row["actions"]
                ]
                report["hard_negative_actions"][action] = {
                    "clips": len(indices),
                    "false_positive": int(
                        np.sum(test_scores[indices] >= report["threshold"])
                    ),
                }
            write_json(
                destination / "test_predictions.json",
                [
                    {**row, "score": float(score)}
                    for row, score in zip(test_rows, test_scores)
                ],
            )
            if args.pku and args.pku.is_dir():
                sources, train_pku, diagnostic = pku_sources(args.pku, config)
                report["pku_continuous"] = []
                for source in sources:
                    if source["key"] == "0005-M":
                        continue
                    trace = trace_video(model, source, config, detector, deadline - 60)
                    events = events_from_trace(
                        trace, report["threshold"], gate["confirm_seconds"]
                    )
                    report["pku_continuous"].append(
                        {
                            **trace,
                            "events": events,
                            "metrics": event_metrics(
                                events,
                                source["intervals"],
                                source["fps"],
                                trace["duration"],
                            ),
                        }
                    )
            write_json(destination / "report.json", report)
            export(model, config, detector, report, destination)
            summary["variants"][name] = report
            write_json(args.output / "run.json", summary)
            model.cuda()
        summary["recommended_by_development_only"] = max(
            summary["variants"],
            key=lambda name: selection_key(summary["variants"][name]["development"]),
        )
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="incomplete", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(args.output / "run.json", summary)
        print(
            "FUSION_RUN "
            + json.dumps(
                {key: value for key, value in summary.items() if key != "variants"}
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "data", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pku", type=Path)
    parser.add_argument("--budget-minutes", type=float, default=90)
    parser.add_argument("--variant-minutes", type=float, default=25)
    args = parser.parse_args()
    if (
        not torch.cuda.is_available()
        or args.budget_minutes < 20
        or args.variant_minutes < 3
    ):
        parser.error(
            "CUDA, budget >=20 minutes and variant budget >=3 minutes required"
        )
    run(args)


if __name__ == "__main__":
    main()
