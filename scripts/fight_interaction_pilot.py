"""Bounded temporal/domain adaptation feasibility, never automatic deployment."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from detection.detector import PersonDetector
from fight.interaction_data import pku_sources
from fight.interaction_eval import balanced_calibration, event_metrics, macro_ap
from fight.model import VideoClassifier
from fight.runtime import EventGate, FightRuntime
from fight.vision import ClipConfig, DetectorConfig, augment_handheld, clip_tensor
from scripts.fight_ablation import cached_rows, export_runtime
from scripts.fight_large_train import measured
from scripts.fight_pilot import score_rows, write_json


def public_subset(manifest):
    rows = []
    for split in ("train", "development", "calibration"):
        for dataset in ("airtlab", "rlvs"):
            for label in (0, 1):
                pool = sorted(
                    [
                        row
                        for row in manifest["rows"]
                        if row["split"] == split
                        and row["dataset"] == dataset
                        and row["label"] == label
                    ],
                    key=lambda row: row["sha256"],
                )
                limit = 36 if split == "train" else 20
                rows.extend(pool[:limit])
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Do not train/select/calibrate using the old public test")
    return rows


def balanced_epoch(rows, generator, per_group=24):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["label"]), []).append(row)
    if any(
        {label for source, label in groups if source == dataset} != {0, 1}
        for dataset in {source for source, label in groups}
    ):
        raise ValueError("Both classes required per training source")
    order = []
    for group in groups.values():
        order.extend(generator.choices(group, k=per_group))
    generator.shuffle(order)
    return order


def train_candidate(model, training, development, cache, output, deadline):
    history, best, state = [], -1.0, None
    generator = random.Random(119)
    for frozen, epochs in ((True, 2), (False, 6)):
        if state is not None:
            model.load_state_dict(state)
        model.set_stage("probe" if frozen else "finetune")
        groups = [{"params": model.head.parameters(), "lr": 2e-4}]
        if not frozen:
            groups.append(
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": 2e-5,
                }
            )
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        stale = 0
        for epoch in range(epochs):
            model.train()
            model.backbone.eval()
            losses = []
            order = balanced_epoch(training, generator)
            for batch_start in range(0, len(order), 8):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Interaction training budget exhausted")
                batch = order[batch_start : batch_start + 8]
                optimizer.zero_grad()
                for row in batch:
                    with np.load(cache / f"{row['cache']}.npz") as cached:
                        if frozen:
                            logits = model.head(
                                torch.from_numpy(
                                    cached["features"].astype(np.float32)
                                ).cuda()
                            ).flatten()
                        else:
                            clips = cached["clips"]
                            if generator.random() < 0.5:
                                clips = clips[:, :, :, ::-1].copy()
                            if generator.random() < 0.5:
                                clips = augment_handheld(
                                    clips,
                                    np.random.default_rng(generator.randrange(2**32)),
                                )
                            logits = model(clip_tensor(clips, "cuda"))
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits.max(), torch.tensor(float(row["label"]), device="cuda")
                    )
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            scores = score_rows(model, development, cache, "cuda", frozen)
            selection = macro_ap(development, scores)
            record = {
                "stage": "probe" if frozen else "finetune",
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "development_macro_ap": selection,
                "sampled_training_exposures": len(order),
                "note": "Repeated sampling balances domains, not new independent examples",
            }
            history.append(record)
            write_json(output / "history.json", history)
            print("INTERACTION_EPOCH " + json.dumps(record), flush=True)
            if selection > best:
                best, stale = selection, 0
                state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                torch.save(
                    {"state_dict": state, "architecture": "x3d_s"}, output / "best.pt"
                )
            else:
                stale += 1
            if stale >= 3:
                break
    if state is None:
        raise RuntimeError("No completed checkpoint")
    model.load_state_dict(state)
    return history


class DevicePredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values):
        return self.model(values.to(next(self.model.parameters()).device))


def continuous_diagnostic(model, sources, config, detector, threshold, deadline):
    results = []
    for source in sources:
        runtime = FightRuntime(
            predictor=DevicePredictor(model.eval()), config=config, detector=detector
        )
        runtime.gate = EventGate(threshold, threshold * 0.7)
        capture = cv2.VideoCapture(source["path"])
        events, predictions, active, frame_index = [], [], None, 0
        last_prediction = None
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Continuous diagnostic budget exhausted")
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / source["fps"]
                frame_index += 1
                result = runtime.push(frame, timestamp)
                observed = result.get("prediction_timestamp")
                if observed is not None and observed != last_prediction:
                    predictions.append(
                        {
                            "timestamp": observed,
                            "score": result["score"],
                            "people": result["people"],
                            "state": result["state"],
                        }
                    )
                    last_prediction = observed
                if result.get("event") == "started":
                    active = {"start": timestamp, "end": None}
                    events.append(active)
                elif result.get("event") == "ended" and active is not None:
                    active["end"] = timestamp
                    active = None
        finally:
            capture.release()
        duration = frame_index / source["fps"]
        if active is not None:
            active["end"] = duration
        results.append(
            {
                "video": source["video"],
                "events": events,
                "predictions": predictions,
                "metrics": event_metrics(
                    events, source["intervals"], source["fps"], duration
                ),
            }
        )
        print(
            "CONTINUOUS "
            + json.dumps({"video": source["video"], **results[-1]["metrics"]}),
            flush=True,
        )
    return results


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        raise ValueError(
            "Existing experiment; read saved report, do not automatically retry"
        )
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    baseline_report = json.loads((args.baseline / "report.json").read_text())
    base_weights = args.baseline / "best.pt"
    weight_id = hashlib.sha256(base_weights.read_bytes()).hexdigest()
    initial = torch.load(base_weights, map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    started, deadline = time.monotonic(), time.monotonic() + args.budget_minutes * 60
    detector_config = DetectorConfig()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    source_config = ClipConfig(**manifest["clip"])
    configurations = {
        "aligned_8fps": replace(source_config, version=6),
        "short_12fps": replace(source_config, fps=12, version=7),
    }
    summary = {
        "status": "running",
        "variants": {},
        "baseline_training_weight_sha256": weight_id,
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Tiny PKU source-held feasibility; 0005 train, 0176/0360 diagnostic, actors not independently verified. Already inspected diagnostic sources; not new final test.",
        "pku_calibration_available": False,
        "budget_minutes": args.budget_minutes,
    }
    write_json(output / "comparison.json", summary)
    args.cache.mkdir(parents=True, exist_ok=True)
    model = VideoClassifier(pretrained=False).cuda().eval()
    model.load_state_dict(initial)
    public = public_subset(manifest)
    sources, training_pku, diagnostic_pku = pku_sources(args.pku, source_config)
    if {source["key"] for source in sources} != {"0005-M", "0176-L", "0360-R"}:
        raise ValueError(
            "Pilot source policy changed; create a new reviewed split manifest"
        )
    held_sources = [source for source in sources if source["key"] != "0005-M"]
    try:
        baseline_rows = [row for row in diagnostic_pku if row["key"] != "0005-M"]
        baseline_ready = cached_rows(
            baseline_rows,
            model,
            detector,
            source_config,
            detector_config,
            args.cache,
            "cuda",
            deadline - 120,
            backbone_id=weight_id,
        )
        scores = score_rows(model, baseline_ready, args.cache, "cuda", False)
        summary["baseline_pku_windows"] = measured(
            baseline_ready, scores, baseline_report["threshold"]
        )
        summary["baseline_continuous"] = continuous_diagnostic(
            model,
            held_sources,
            source_config,
            detector,
            baseline_report["threshold"],
            deadline - 120,
        )
        write_json(output / "comparison.json", summary)
        for name, config in configurations.items():
            destination = output / name
            destination.mkdir(exist_ok=True)
            model.load_state_dict(initial)
            model.eval()
            sources, pku_train, pku_diagnostic = pku_sources(args.pku, config)
            rows = (
                public
                + [
                    {**row, "split": "train"}
                    for row in pku_train
                    if row["key"] == "0005-M"
                ]
                + [
                    {**row, "split": "diagnostic"}
                    for row in pku_diagnostic
                    if row["key"] != "0005-M"
                ]
            )
            ready = cached_rows(
                rows,
                model,
                detector,
                config,
                detector_config,
                args.cache,
                "cuda",
                deadline - 180,
                backbone_id=weight_id,
            )
            write_json(
                destination / "manifest.json",
                {
                    "rows": ready,
                    "sources": sources,
                    "clip": config.to_dict(),
                    "scope": summary["scope"],
                    "actors_independent": False,
                    "label_timing_assumption": "1-based inclusive publisher frames converted to 0-based inclusive",
                },
            )
            partitions = {
                split: [row for row in ready if row["split"] == split]
                for split in ("train", "development", "calibration", "diagnostic")
            }
            try:
                history = train_candidate(
                    model,
                    partitions["train"],
                    partitions["development"],
                    args.cache,
                    destination,
                    min(deadline - 150, time.monotonic() + 300),
                )
            except TimeoutError:
                if not (destination / "best.pt").exists():
                    raise
                model.load_state_dict(
                    torch.load(
                        destination / "best.pt", map_location="cuda", weights_only=True
                    )["state_dict"]
                )
                history = json.loads((destination / "history.json").read_text())
            cal_scores = score_rows(
                model, partitions["calibration"], args.cache, "cuda", False
            )
            calibration = balanced_calibration(partitions["calibration"], cal_scores)
            write_json(destination / "calibration_frozen.json", calibration)
            write_json(
                destination / "calibration_predictions.json",
                [
                    {**row, "score": float(score)}
                    for row, score in zip(partitions["calibration"], cal_scores)
                ],
            )
            diagnostic_scores = score_rows(
                model, partitions["diagnostic"], args.cache, "cuda", False
            )
            report = {
                **calibration,
                "pku_diagnostic": measured(
                    partitions["diagnostic"],
                    diagnostic_scores,
                    calibration["threshold"],
                ),
                "scope": summary["scope"],
                "pku_calibration_available": False,
                "approved_for_demo": False,
                "real_phone_tested": False,
                "completed_epochs": len(history),
                "unique_pku_train_sources": 1,
                "unique_pku_train_positive_occurrences": len(
                    {
                        row["occurrence"]
                        for row in partitions["train"]
                        if row["dataset"] == "pku" and row["label"] == 1
                    }
                ),
            }
            report["continuous"] = continuous_diagnostic(
                model,
                held_sources,
                config,
                detector,
                calibration["threshold"],
                deadline - 30,
            )
            write_json(
                destination / "diagnostic_predictions.json",
                [
                    {**row, "score": float(score)}
                    for row, score in zip(partitions["diagnostic"], diagnostic_scores)
                ],
            )
            write_json(destination / "report.json", report)
            export_runtime(model, config, detector_config, report, destination)
            summary["variants"][name] = report
            write_json(output / "comparison.json", summary)
            model.cuda()
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="incomplete", reason=str(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(output / "comparison.json", summary)
        print(
            "INTERACTION_RESULT "
            + json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"baseline_continuous", "variants"}
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pku", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.budget_minutes < 10:
        parser.error("CUDA and >=10 minutes required")
    run(args)


if __name__ == "__main__":
    main()
