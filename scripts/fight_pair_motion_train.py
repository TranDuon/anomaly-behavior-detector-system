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
from fight.pair_motion import MotionConfig, PairMotionClassifier, motion_inputs
from scripts.fight_fusion_qa import select_cases
from scripts.fight_fusion_train import (
    development_result,
    progress_allowed,
    selection_key,
)
from scripts.fight_large_train import measured, people_focused_training
from scripts.fight_pilot import write_json


def source_cache_key(row, manifest):
    identity = {
        "video": row["video"],
        "sha256": row["sha256"],
        "start": row["start"],
        "fps": row["fps"],
        "clip": manifest["clip"],
        "pose": manifest["pose"],
        "rgb_weights": manifest["rgb_sha256"],
        "pose_weights": manifest["pose_sha256"],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :24
    ]


def prepare_pool(rows, cache, motion_cache, deadline):
    pool = {}
    for index, row in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("Motion preprocessing budget exhausted")
        with np.load(cache / (row["cache"] + ".npz"), allow_pickle=False) as data:
            target = motion_cache / (row["cache"] + ".npz")
            if not target.exists():
                motion = motion_inputs(data["clips"])
                partial = target.with_suffix(".partial.npz")
                np.savez_compressed(partial, motion=motion)
                partial.replace(target)
            with np.load(target, allow_pickle=False) as computed:
                motion = computed["motion"].copy()
            features = data["features"].astype(np.float32)
            poses = data["poses"].astype(np.float32)
        if (
            motion.shape != (len(features), 2, 15, 32, 32)
            or motion.dtype != np.float32
            or not np.isfinite(motion).all()
        ):
            raise ValueError("Invalid cached motion")
        pool[row["cache"]] = (
            torch.from_numpy(features),
            torch.from_numpy(poses),
            torch.from_numpy(motion),
        )
        if (index + 1) % 64 == 0:
            print(f"MOTION_CACHE {index+1}/{len(rows)}", flush=True)
    return pool


def scores(model, rows, pool, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for row in rows:
            values = [value.to(device) for value in pool[row["cache"]]]
            logits, auxiliary = model.classify_features(*values)
            result.append(float(torch.sigmoid(logits.max()).cpu()))
    return np.asarray(result)


def balanced_order(rows, generator):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["label"]), []).append(row)
    if len(groups) != 4:
        raise ValueError("Expected both binary classes from AIRTLab and RLVS")
    selected = [
        row for group in groups.values() for row in generator.choices(group, k=128)
    ]
    generator.shuffle(selected)
    return selected


def train(model, training, development, pool, output, device, deadline):
    generator = random.Random(751)
    initial = development_result(development, scores(model, development, pool, device))
    best, best_state, history = initial, copy.deepcopy(model.state_dict()), []
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=4e-4,
        weight_decay=0.01,
    )
    stale, reason = 0, "maximum_epochs"
    for epoch in range(16):
        if time.monotonic() >= deadline:
            reason = "budget_stop"
            break
        model.train()
        model.rgb.eval()
        losses = []
        order = balanced_order(training, generator)
        for start in range(0, len(order), 16):
            if time.monotonic() >= deadline:
                reason = "budget_stop"
                break
            batch = order[start : start + 16]
            counts = [len(pool[row["cache"]][0]) for row in batch]
            inputs = [
                torch.cat([pool[row["cache"]][column] for row in batch]).to(device)
                for column in range(3)
            ]
            logits, auxiliary = model.classify_features(*inputs)
            offset, video_logits, actions = 0, [], []
            for count in counts:
                video_logits.append(logits[offset : offset + count].max())
                actions.append(auxiliary[offset : offset + count].amax(dim=0))
                offset += count
            targets = torch.tensor(
                [row["label"] for row in batch], dtype=torch.float32, device=device
            )
            action_targets = torch.tensor(
                [row["action_targets"] for row in batch],
                dtype=torch.float32,
                device=device,
            )
            action_mask = torch.tensor(
                [row["action_mask"] for row in batch],
                dtype=torch.float32,
                device=device,
            )
            primary = nn.functional.binary_cross_entropy_with_logits(
                torch.stack(video_logits), targets
            )
            action_loss = nn.functional.binary_cross_entropy_with_logits(
                torch.stack(actions), action_targets, reduction="none"
            )
            loss = primary + 0.15 * (
                action_loss * action_mask
            ).sum() / action_mask.sum().clamp_min(1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        if reason == "budget_stop":
            break
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "sampled_exposures": len(order),
            **development_result(development, scores(model, development, pool, device)),
        }
        history.append(record)
        if selection_key(record) > selection_key(best):
            best, best_state, stale = record, copy.deepcopy(model.state_dict()), 0
            torch.save(
                {
                    "state_dict": best_state,
                    "use_pair": model.use_pair,
                    "use_motion": model.use_motion,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        write_json(output / "history.json", history)
        print("PAIR_MOTION_EPOCH " + json.dumps(record), flush=True)
        if epoch >= 3 and not progress_allowed(initial, best):
            reason = "no_protected_development_progress"
            break
        if stale >= 4:
            reason = "development_plateau"
            break
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "use_pair": model.use_pair,
            "use_motion": model.use_motion,
        },
        output / "best.pt",
    )
    return {
        "initial": initial,
        "development": best,
        "epochs": len(history),
        "stop_reason": reason,
        "protected_progress": progress_allowed(initial, best),
    }


def export_runtime(model, manifest, report, output, weights):
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
    metadata = {
        "schema_version": 1,
        "architecture": "pair_motion_x3d_s",
        "input_kind": "rgb_pose_motion",
        "clip": manifest["clip"],
        "pose": manifest["pose"],
        "motion": asdict(MotionConfig()),
        "use_pose": model.use_pair,
        "use_motion": model.use_motion,
        "threshold": report["threshold"],
        "event_gate": {"confirm_seconds": 0.4},
        "approved_for_demo": False,
        "real_phone_tested": False,
        "test": report["test"],
        "test_by_dataset": report["test_by_dataset"],
        "development": report["development"],
    }
    write_json(output / "fight_runtime.json", metadata)
    with zipfile.ZipFile(
        output / "sentinel_fight_only_runtime.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for filename in ("fight_model.ts", "fight_runtime.json"):
            archive.write(output / filename, "models/" + filename)
        archive.write(weights, "models/yolov8n-pose.pt")


def run(args):
    if (args.output / "run.json").exists():
        raise ValueError("Existing experiment; do not silently retrain or overwrite")
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    args.output.mkdir(parents=True, exist_ok=True)
    args.motion_cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(751)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["clip"]["frames"] != 16 or manifest["clip"]["fps"] != 12:
        raise ValueError("Unexpected source cache timebase")
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != manifest["rgb_sha256"]:
        raise ValueError("RGB features and source weights do not match")
    if (
        hashlib.sha256(args.pose_weights.read_bytes()).hexdigest()
        != manifest["pose_sha256"]
    ):
        raise ValueError("Pose weights do not match cached pose inputs")
    rows = [
        {**row, "cache": source_cache_key(row, manifest)} for row in manifest["rows"]
    ]
    partitions = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "development", "calibration", "test")
    }
    partitions["train"], focus = people_focused_training(
        partitions["train"], args.cache
    )
    summary = {
        "status": "running",
        "variants": {},
        "seed": 751,
        "training_focus": focus,
        "counts": {name: len(values) for name, values in partitions.items()},
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Frozen RGB features; learned directed pair and optical-motion ablation; repeated public diagnostic, no private phone training/upload",
        "budget_minutes": args.budget_minutes,
    }
    write_json(args.output / "run.json", summary)
    try:
        public = (
            partitions["train"] + partitions["development"] + partitions["calibration"]
        )
        pool = prepare_pool(public, args.cache, args.motion_cache, deadline - 180)
        weights = torch.load(args.baseline, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        states = {}
        for name, enabled in (("rgb_control", False), ("pair_motion", True)):
            torch.manual_seed(751)
            model = PairMotionClassifier(enabled, enabled)
            model.rgb.load_state_dict(weights)
            model = model.to(args.device)
            destination = args.output / name
            destination.mkdir(exist_ok=True)
            trained = train(
                model,
                partitions["train"],
                partitions["development"],
                pool,
                destination,
                args.device,
                min(deadline - 180, time.monotonic() + 240),
            )
            calibration = balanced_calibration(
                partitions["calibration"],
                scores(model, partitions["calibration"], pool, args.device),
            )
            protocol = {
                **trained,
                "threshold": calibration["threshold"],
                "calibration": calibration,
                "event_gate": {"confirm_seconds": 0.4},
                "gate_scope": "Confirmation fixed before test/phone diagnostics; not selected on phone clips",
                "use_pair": enabled,
                "use_motion": enabled,
            }
            write_json(destination / "protocol_frozen.json", protocol)
            summary["variants"][name] = protocol
            states[name] = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            write_json(args.output / "run.json", summary)
        summary["recommended_by_development_only"] = max(
            summary["variants"],
            key=lambda name: selection_key(summary["variants"][name]["development"]),
        )
        summary["pair_progress_over_control"] = progress_allowed(
            summary["variants"]["rgb_control"]["development"],
            summary["variants"]["pair_motion"]["development"],
        )
        write_json(args.output / "selection_frozen.json", summary)
        pool.update(
            prepare_pool(
                partitions["test"], args.cache, args.motion_cache, deadline - 90
            )
        )
        expected = []
        with zipfile.ZipFile(
            args.output / "cpu_validation.zip", "w", zipfile.ZIP_DEFLATED
        ) as qa:
            for name, protocol in summary["variants"].items():
                enabled = protocol["use_pair"]
                model = PairMotionClassifier(enabled, enabled).to(args.device).eval()
                model.load_state_dict(states[name])
                predictions = scores(model, partitions["test"], pool, args.device)
                protocol["test"] = measured(
                    partitions["test"], predictions, protocol["threshold"]
                )
                protocol["test_by_dataset"] = {
                    dataset: measured(
                        [
                            row
                            for row in partitions["test"]
                            if row["dataset"] == dataset
                        ],
                        predictions[
                            [row["dataset"] == dataset for row in partitions["test"]]
                        ],
                        protocol["threshold"],
                    )
                    for dataset in ("airtlab", "rlvs")
                }
                records = [
                    {**row, "score": float(score)}
                    for row, score in zip(partitions["test"], predictions)
                ]
                write_json(args.output / name / "test_predictions.json", records)
                export_runtime(
                    model, manifest, protocol, args.output / name, args.pose_weights
                )
                qa.write(
                    args.output / name / "sentinel_fight_only_runtime.zip",
                    name + "/runtime.zip",
                )
                selected = select_cases(records)
                for row in selected[:1] + [
                    row for row in selected if row["label"] == 1
                ]:
                    qa.write(
                        args.cache / (row["cache"] + ".npz"),
                        name + "/" + row["cache"] + ".npz",
                    )
                    expected.append(
                        {
                            "variant": name,
                            "cache": row["cache"],
                            "score": row["score"],
                            "video": row["video"],
                            "label": row["label"],
                        }
                    )
                write_json(args.output / name / "report.json", protocol)
            qa.writestr("expected.json", json.dumps(expected, indent=2))
        summary["status"] = "completed"
    except Exception as error:
        summary.update(status="failed", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(args.output / "run.json", summary)
        print("PAIR_MOTION_RUN " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Bounded frozen-feature pair/motion pilot; no auto-deployment"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pose-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-minutes", type=float, default=16)
    args = parser.parse_args()
    if not 1 <= args.budget_minutes <= 25:
        parser.error("Budget must be between 1 and 25 minutes")
    run(args)


if __name__ == "__main__":
    main()
