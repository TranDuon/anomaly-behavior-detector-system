"""One RGB-only repair, with train-only negatives and resumable event evaluation."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from fight.demo_training import augment_close_view, expand_training
from fight.pose import PoseConfig, PoseDetector
from fight.refinement import load_deployed_base
from fight.temporal_labels import source_group, validate_annotations
from fight.vision import ClipConfig, clip_tensor
from scripts.fight_pilot import write_json
from scripts.fight_refinement_audit import load_checkpoint, validate_source
from scripts.fight_refinement_train import (
    action_event_report,
    action_window_report,
    cache_windows,
    event_cohorts,
    export,
    preflight,
    relocate,
    score_rows,
    select_event_gate,
    trace_sources,
    window_calibration,
    window_development,
)
from scripts.fight_large_train import measured


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_state(path, value):
    temporary = path.with_suffix(".partial.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def event_rank(report):
    cohorts = report["events"].values()
    if not cohorts:
        raise ValueError("Missing continuous development cohorts")
    return (
        min(min(row["precision"], row["recall"]) for row in cohorts),
        -sum(row["false_alerts_per_normal_minute"] for row in cohorts),
        -max(row["normal_alert_fraction"] for row in cohorts),
        report["windows"]["macro_ap"],
    )


def protected(candidate, initial):
    return all(
        candidate["windows"]["cohorts"][name]["ap"] >= row["ap"] - 0.02
        for name, row in initial["windows"]["cohorts"].items()
    )


def balanced_order(rows, generator, exposures=128):
    groups = {}
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Optimizer input must contain training sources only")
        groups.setdefault((row["dataset"], row["label"]), {}).setdefault(
            source_group(row), []
        ).append(row)
    order = []
    for performances in groups.values():
        candidates = list(performances.values())
        for unused in range(exposures):
            order.append(generator.choice(generator.choice(candidates)))
    generator.shuffle(order)
    return order


def train_epoch(model, rows, cache, optimizer, device, generator, deadline):
    if model.use_interaction:
        raise ValueError(
            "Close-view repair is RGB-only; old pose coordinates cannot follow crops"
        )
    model.train()
    losses = []
    order = balanced_order(rows, generator)
    for start in range(0, len(order), 4):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Training budget reached; resume from the last complete epoch"
            )
        batch = order[start : start + 4]
        optimizer.zero_grad(set_to_none=True)
        for row in batch:
            with np.load(cache / (row["cache"] + ".npz"), allow_pickle=False) as data:
                clips = data["clips"]
                if generator.random() < 0.5:
                    clips = augment_close_view(clips, row, generator)
                poses = torch.from_numpy(data["poses"]).to(device)
                motion = torch.from_numpy(data["motion"]).to(device)
            features = model.base.rgb.features(clip_tensor(clips, device))
            logits, actions = model.classify_features(features, poses, motion)
            primary = nn.functional.binary_cross_entropy_with_logits(
                logits.max(), torch.tensor(float(row["label"]), device=device)
            )
            targets = torch.tensor(
                row["action_targets"], dtype=torch.float32, device=device
            )
            mask = torch.tensor(row["action_mask"], dtype=torch.float32, device=device)
            auxiliary = nn.functional.binary_cross_entropy_with_logits(
                actions.amax(dim=0), targets, reduction="none"
            )
            loss = primary + 0.15 * (auxiliary * mask).sum() / mask.sum().clamp_min(1)
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1,
        )
        optimizer.step()
        if start % 64 == 0:
            print(
                f"CLASSROOM_BATCH {start + len(batch)}/{len(order)} loss={np.mean(losses):.6f}",
                flush=True,
            )
    return float(np.mean(losses)), len(order)


def train(
    model,
    partitions,
    cache,
    output,
    device,
    train_deadline,
    evaluation_deadline,
    evaluate,
    identity,
    resume,
):
    generator = random.Random(941)
    model.set_stage("finetune")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not name.startswith("base.")
                ],
                "lr": 1e-4,
            },
            {
                "params": [
                    parameter
                    for parameter in model.base.parameters()
                    if parameter.requires_grad
                ],
                "lr": 1e-5,
            },
        ],
        weight_decay=0.01,
    )
    snapshot = output / "resume.pt"
    if resume and snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError(
                "Resume identity differs; refuse stale data/code/checkpoint"
            )
        model.load_state_dict(state["current"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        generator.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        initial = evaluate(model, evaluation_deadline())
        state = {
            "identity": identity,
            "initial": initial,
            "best": initial,
            "best_state": cpu_state(model),
            "history": [],
            "epoch": 0,
            "pending": False,
            "stale": 0,
        }
        save_state(
            output / "best.pt",
            {"state_dict": state["best_state"], "use_interaction": False},
        )

    def persist():
        state.update(
            current=cpu_state(model),
            optimizer=optimizer.state_dict(),
            random_state=generator.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        )
        save_state(snapshot, state)

    persist()
    while state["epoch"] < 12 and state["stale"] < 3:
        if not state["pending"]:
            loss, exposures = train_epoch(
                model,
                partitions["train"],
                cache,
                optimizer,
                device,
                generator,
                train_deadline(),
            )
            state.update(pending=True, loss=loss, exposures=exposures)
            persist()
        result = evaluate(model, evaluation_deadline())
        improved = protected(result, state["initial"]) and event_rank(
            result
        ) > event_rank(state["best"])
        if improved:
            state.update(best=result, best_state=cpu_state(model), stale=0)
            save_state(
                output / "best.pt",
                {"state_dict": state["best_state"], "use_interaction": False},
            )
        else:
            state["stale"] += 1
        state["epoch"] += 1
        state["pending"] = False
        record = {
            "epoch": state["epoch"],
            "loss": state["loss"],
            "sampled_exposures": state["exposures"],
            "selected": improved,
            **result,
        }
        state["history"].append(record)
        persist()
        write_json(output / "history.json", state["history"])
        print("CLASSROOM_EPOCH " + json.dumps(record), flush=True)
    model.load_state_dict(state["best_state"], strict=True)
    return {
        "initial": state["initial"],
        "selected": state["best"],
        "epochs": state["epoch"],
        "improved": event_rank(state["best"]) > event_rank(state["initial"]),
    }


def run(args):
    manifest, windows, ready = preflight(args.manifest, args.annotations)
    windows, expansion = expand_training(
        windows, manifest["rows"], ClipConfig(**manifest["clip"])
    )
    if args.phase == "check":
        report = {
            "readiness": ready,
            "expansion": expansion,
            "training_windows": sum(row["split"] == "train" for row in windows),
        }
        print(json.dumps(report, indent=2))
        return report
    if not ready["ready_for_temporal_pilot"]:
        raise ValueError("Reviewed labels are not ready; do not allocate GPU")
    started = time.monotonic()
    deadline = started + args.minutes * 60
    train_spent, eval_spent = [0.0], [0.0]
    torch.manual_seed(941)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata, base_sha = load_deployed_base(args.models, False)
    parent = json.loads((args.parent_run / "run.json").read_text(encoding="utf-8"))
    validate_source(parent, ready, base_sha)
    parent_sha = load_checkpoint(
        model, args.parent_run / "rgb_control" / "best.pt", False
    )
    pose_path = args.models / "yolov8n-pose.pt"
    pose_sha = file_sha(pose_path)
    if pose_sha != manifest["pose_sha256"] or metadata["clip"] != manifest["clip"]:
        raise ValueError("Preprocessing or detector provenance mismatch")
    identity = {
        "parent": parent_sha,
        "manifest_file_sha256": file_sha(args.manifest),
        "readiness": ready,
        "expansion": expansion,
        "pose": pose_sha,
        "clip": metadata["clip"],
        "seed": 941,
        "code": {
            name: file_sha(Path(__file__).resolve().parents[1] / name)
            for name in (
                "scripts/fight_classroom_refine.py",
                "scripts/fight_refinement_train.py",
                "fight/demo_training.py",
                "fight/runtime.py",
                "fight/refinement.py",
                "fight/vision.py",
                "fight/pose.py",
                "fight/view_quality.py",
                "fight/temporal_labels.py",
                "fight/reviewed_events.py",
                "scripts/fight_fusion_train.py",
                "scripts/fight_refinement_audit.py",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "train" and (args.output / "selection_frozen.json").exists():
        raise ValueError("Selection already frozen; do not retrain after test exposure")
    protocol_path = args.output / "input_protocol.json"
    if protocol_path.exists():
        if (
            not args.resume
            or json.loads(protocol_path.read_text(encoding="utf-8")) != identity
        ):
            raise ValueError(
                "Existing output requires --resume and identical inputs/code"
            )
    else:
        write_json(protocol_path, identity)
    model = model.to(args.device).eval()
    detector = PoseDetector(
        PoseConfig(**metadata["pose"]), device=args.device, model_path=pose_path
    )
    approved = validate_annotations(
        json.loads(args.annotations.read_text(encoding="utf-8")), manifest["rows"]
    )
    trace_context = {"code": identity["code"], "pose_sha256": pose_sha}
    summary = {
        "phase": args.phase,
        "status": "running",
        "approved_for_demo": False,
        "expansion": expansion,
    }
    try:
        selected = [
            row
            for row in windows
            if row["split"]
            in (
                ("development", "calibration", "test")
                if args.phase == "evaluate"
                else ("train", "development")
            )
        ]
        selected = relocate(selected, args.data)
        if args.phase == "evaluate":
            frozen_path = args.output / "selection_frozen.json"
            if not frozen_path.exists():
                raise ValueError("Finish training selection before any test scoring")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if file_sha(args.output / "best.pt") != frozen["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            load_checkpoint(model, args.output / "best.pt", False)
        cached = cache_windows(
            selected,
            model,
            detector,
            ClipConfig(**metadata["clip"]),
            args.cache,
            parent_sha if args.phase != "evaluate" else frozen["checkpoint_sha256"],
            pose_sha,
            args.device,
            deadline,
        )
        partitions = {
            split: [row for row in cached if row["split"] == split]
            for split in ("train", "development", "calibration", "test")
        }

        def traces(active, rows, limit):
            return trace_sources(
                active,
                rows,
                metadata,
                detector,
                args.device,
                limit,
                args.output / "traces",
                trace_context,
            )

        def development(active, limit):
            scores = score_rows(
                active,
                partitions["development"],
                args.cache,
                args.device,
                deadline=limit,
            )
            diagnostic = window_calibration(partitions["development"], scores)
            continuous = traces(active, partitions["development"], limit)
            gate, choices = select_event_gate(
                continuous, approved, diagnostic["threshold"]
            )
            return {
                "windows": window_development(partitions["development"], scores),
                "events": event_cohorts(
                    continuous,
                    approved,
                    diagnostic["threshold"],
                    gate["confirm_seconds"],
                ),
                "threshold": diagnostic["threshold"],
                "gate": gate,
                "scope": "Development-only selection, not test or calibrated camera accuracy",
            }

        if args.phase == "train":
            train_start = time.monotonic()
            allowance = max(0, (deadline - train_start) * 0.6)
            last = [train_start, "train"]

            def budget(kind):
                now = time.monotonic()
                (train_spent if last[1] == "train" else eval_spent)[0] += now - last[0]
                last[:] = [now, kind]
                return (
                    min(deadline, now + max(0, allowance - train_spent[0]))
                    if kind == "train"
                    else deadline
                )

            report = train(
                model,
                partitions,
                args.cache,
                args.output,
                args.device,
                lambda: budget("train"),
                lambda: budget("evaluate"),
                development,
                identity,
                args.resume,
            )
            summary["training"] = report
            write_json(
                args.output / "selection_frozen.json",
                {
                    "checkpoint_sha256": file_sha(args.output / "best.pt"),
                    "selection": report,
                    "scope": "Frozen before calibration and test; evaluate phase may not retrain or retune on test",
                },
            )
        elif args.phase == "evaluate":
            report_path = args.output / "protocol_frozen.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
                    raise ValueError("Frozen calibration belongs to different weights")
            else:
                calibration = window_calibration(
                    partitions["calibration"],
                    score_rows(
                        model,
                        partitions["calibration"],
                        args.cache,
                        args.device,
                        deadline=deadline,
                    ),
                )
                gate, choices = select_event_gate(
                    traces(model, partitions["development"], deadline),
                    approved,
                    calibration["threshold"],
                )
                report = {
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "threshold": calibration["threshold"],
                    "calibration": calibration,
                    "event_gate": {**metadata.get("event_gate", {}), **gate},
                    "source_classifier_sha256": base_sha,
                    "development_gate_candidates": choices,
                }
                write_json(report_path, report)
            scores = score_rows(
                model, partitions["test"], args.cache, args.device, deadline=deadline
            )
            continuous = traces(model, partitions["test"], deadline)
            report.update(
                test=measured(partitions["test"], scores, report["threshold"]),
                test_by_dataset={
                    name: measured(
                        [row for row in partitions["test"] if row["dataset"] == name],
                        scores[[row["dataset"] == name for row in partitions["test"]]],
                        report["threshold"],
                    )
                    for name in ("airtlab", "rlvs")
                },
                test_by_action=action_window_report(
                    partitions["test"], scores, report["threshold"]
                ),
                test_events_by_dataset=event_cohorts(
                    continuous,
                    approved,
                    report["threshold"],
                    report["event_gate"]["confirm_seconds"],
                ),
                normal_events_by_action=action_event_report(
                    continuous,
                    approved,
                    report["threshold"],
                    report["event_gate"]["confirm_seconds"],
                ),
            )
            report["test"][
                "metric_scope"
            ] = "Reviewed public action windows, not independent camera acceptance"
            for cohort in report["test_by_dataset"].values():
                cohort["metric_scope"] = report["test"]["metric_scope"]
            write_json(args.output / "report.json", report)
            export(model, metadata, report, args.output, pose_path)
            summary["report"] = report
        summary["status"] = "completed"
    except TimeoutError as error:
        summary.update(status="budget_stopped", reason=str(error), resume_required=True)
    except BaseException as error:
        summary.update(status="failed", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(args.output / (args.phase + "_status.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("check", "prepare", "train", "evaluate"), default="check"
    )
    for name in ("models", "parent-run", "data", "cache", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase != "check" and (
        not all((args.models, args.parent_run, args.data, args.cache, args.output))
        or not 1 <= args.minutes <= 60
    ):
        parser.error(
            "Require models/parent-run/data/cache/output and 1–60 minute budget"
        )
    run(args)


if __name__ == "__main__":
    main()
