"""Tiny AI-curated development feasibility pilot; never a demo acceptance test."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from detection.detector import PersonDetector
from fight.model import VideoClassifier
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import (
    cached_rows,
    export_runtime,
    row_id,
    stable_train,
    threshold_report,
)
from scripts.fight_pilot import metrics, score_rows, write_json
from scripts.fight_regions_ablation import crop_gallery


def reviewed_manifest(source, review, judgments_path):
    source_manifest = json.loads((source / "cached_manifest.json").read_text())
    if source_manifest.get("official_test_used") is not False:
        raise ValueError("Official test is not allowed")
    allowed = {row_id(row): row for row in source_manifest["rows"]}
    mapping = json.loads((review / "private_mapping.json").read_text())
    judgments = json.loads(judgments_path.read_text())
    expected = {row["audit_id"] for row in mapping["rows"]}
    by_id = {row["audit_id"]: row for row in judgments["judgments"]}
    if expected != by_id.keys() or len(by_id) != len(judgments["judgments"]):
        raise ValueError("Freeze a complete unique set of review judgments first")
    rows, uncertain = [], []
    for original in mapping["rows"]:
        if (
            row_id(original) not in allowed
            or original["split"] != allowed[row_id(original)]["split"]
        ):
            raise ValueError("Reviewed clip differs from original source split")
        judgment = by_id[original["audit_id"]]
        if judgment["judgment"] == "uncertain":
            uncertain.append(original["audit_id"])
            continue
        if judgment["judgment"] not in {"fight", "nonfight"} or judgment[
            "confidence"
        ] not in {"high", "medium"}:
            raise ValueError(
                "Unresolved/low-confidence judgments cannot become curated labels"
            )
        rows.append(
            {
                **original,
                "original_dataset_label": original["label"],
                "label": int(judgment["judgment"] == "fight"),
                "review_confidence": judgment["confidence"],
            }
        )
    for key in ("video", "scene_group"):
        if {row[key] for row in rows if row["split"] == "train"} & {
            row[key] for row in rows if row["split"] == "validation"
        }:
            raise ValueError("Curated source/scene leakage")
    for split in ("train", "validation"):
        if any(
            sum(row["split"] == split and row["label"] == label for row in rows) < 4
            for label in (0, 1)
        ):
            raise ValueError("Too few curated clips even for a tiny feasibility run")
    return {
        "rows": rows,
        "excluded_uncertain": uncertain,
        "judgments_sha256": hashlib.sha256(judgments_path.read_bytes()).hexdigest(),
        "rater_type": judgments["rater_type"],
        "human_verified": False,
        "original_annotation_files_modified": False,
        "official_test_used": False,
        "scope": "Tiny deliberately selected AI-curated development cohort; not independent human gold, not population accuracy, not real-phone acceptance",
        "counts": {
            split: {
                "clips": sum(row["split"] == split for row in rows),
                "positive": sum(
                    row["split"] == split and row["label"] == 1 for row in rows
                ),
                "videos": len({row["video"] for row in rows if row["split"] == split}),
                "changed_labels_in_new_manifest": sum(
                    row["split"] == split
                    and row["label"] != row["original_dataset_label"]
                    for row in rows
                ),
            }
            for split in ("train", "validation")
        },
    }


def run(args):
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        previous = json.loads((output / "comparison.json").read_text())
        if previous.get("status") == "completed":
            print(json.dumps(previous), flush=True)
            return
        raise ValueError("Incomplete curated pilot exists; no automatic retry")
    manifest = reviewed_manifest(args.source_run, args.review_dir, args.judgments)
    write_json(output / "manifest.json", manifest)
    summary = {
        "status": "running",
        "variants": {},
        "scope": manifest["scope"],
        "rater_type": manifest["rater_type"],
        "counts": manifest["counts"],
        "approved_for_demo": False,
        "decision": "tiny_curated_feasibility_only",
    }
    validation = [row for row in manifest["rows"] if row["split"] == "validation"]
    prior = json.loads((args.source_run / "predictions.json").read_text())
    lookup = {
        (row["video"], round(row["start_seconds"], 6)): row["score"] for row in prior
    }
    reference_scores = np.array(
        [
            lookup[(row["video"], round(row["start"] / row["fps"], 6))]
            for row in validation
        ]
    )
    summary["expanded_reference_on_curated_validation"] = threshold_report(
        validation, reference_scores
    )
    summary["reference_note"] = (
        "Old model trained on many original-label clips; reference is not matched training budget. Scores existed before curated judgments; tiny selected cohort remains optimistic."
    )
    write_json(output / "comparison.json", summary)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pretrained = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in pretrained.state_dict().items()
    }
    detector_config = DetectorConfig()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    baseline = ClipConfig(region_mode="tubes", trim_black_borders=True, version=3)
    configs = {
        "tubes": baseline,
        "pairs": replace(baseline, region_mode="pairs", version=4),
    }
    ready_by_variant = {}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, config in configs.items():
            destination = output / name
            destination.mkdir(exist_ok=True)
            ready = cached_rows(
                manifest["rows"],
                pretrained,
                detector,
                config,
                detector_config,
                args.cache_dir,
                "cuda",
                deadline - 240,
            )
            ready_by_variant[name] = ready
            write_json(
                destination / "cached_manifest.json",
                {**manifest, "rows": ready, "clip": config.to_dict()},
            )
            crop_gallery(
                [row for row in ready if row["split"] == "train"],
                args.cache_dir,
                config,
                destination,
            )
        del pretrained
        torch.cuda.empty_cache()
        for name, config in configs.items():
            print(f"CURATED_VARIANT {name}", flush=True)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = VideoClassifier("x3d_s", pretrained=False).cuda()
            model.load_state_dict(initial)
            ready = ready_by_variant[name]
            train = [row for row in ready if row["split"] == "train"]
            validation = [row for row in ready if row["split"] == "validation"]
            destination = output / name
            history = stable_train(
                model,
                train,
                validation,
                args.cache_dir,
                "cuda",
                deadline - 50,
                destination,
                epochs=(8, 8),
                camera_augmentation=True,
            )
            scores = score_rows(model, validation, args.cache_dir, "cuda", False)
            report = {
                **threshold_report(validation, scores),
                "scope": manifest["scope"],
                "rater_type": manifest["rater_type"],
                "human_verified": False,
                "official_test_used": False,
                "approved_for_demo": False,
                "decision": "tiny_curated_feasibility_only",
                "completed_epochs": {
                    stage: sum(row["stage"] == stage for row in history)
                    for stage in ("probe", "finetune")
                },
            }
            report["training_fit_not_generalization"] = metrics(
                train,
                score_rows(model, train, args.cache_dir, "cuda", False),
                report["threshold"],
            )
            write_json(destination / "report.json", report)
            write_json(
                destination / "predictions.json",
                [
                    {
                        "video": row["video"],
                        "start_seconds": row["start"] / row["fps"],
                        "label": row["label"],
                        "score": float(score),
                    }
                    for row, score in zip(validation, scores)
                ],
            )
            export_runtime(model, config, detector_config, report, destination)
            summary["variants"][name] = report
            write_json(output / "comparison.json", summary)
            del model
            torch.cuda.empty_cache()
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        raise
    except Exception as error:
        summary["status"] = "incomplete"
        summary["reason"] = str(error)
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(output / "comparison.json", summary)
        print("CURATED_RESULT " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.budget_minutes < 6:
        parser.error("CUDA and >=6 minute budget required")
    run(args)


if __name__ == "__main__":
    main()
