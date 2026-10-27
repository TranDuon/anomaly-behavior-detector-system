"""Temporal-removal diagnostics on the same development clips, without training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fight.vision import clip_tensor
from scripts.fight_pilot import metrics, write_json


def temporal_variants(clips, rng):
    return {
        "static_middle": np.repeat(
            clips[:, clips.shape[1] // 2 : clips.shape[1] // 2 + 1],
            clips.shape[1],
            axis=1,
        ),
        "shuffled": clips[:, rng.permutation(clips.shape[1])].copy(),
    }


def run(output, cache, device="cuda", budget_seconds=180):
    started = time.monotonic()
    manifest = json.loads((output / "cached_manifest.json").read_text())
    rows = [row for row in manifest["rows"] if row["split"] == "validation"]
    predictions = json.loads((output / "predictions.json").read_text())
    if len(rows) != len(predictions) or any(
        row["video"] != prediction["video"]
        or row["label"] != prediction["label"]
        or abs(row["start"] / row["fps"] - prediction["start_seconds"]) > 1e-6
        for row, prediction in zip(rows, predictions)
    ):
        raise ValueError("Diagnostic rows differ from validation predictions")
    report = json.loads((output / "report.json").read_text())
    original = np.array([prediction["score"] for prediction in predictions])
    model = torch.jit.load(
        str(output / "runtime/models/fight_model.ts"), map_location=device
    ).eval()
    rng = np.random.default_rng(42)
    scores = {"static_middle": [], "shuffled": []}
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if time.monotonic() - started > budget_seconds:
                raise TimeoutError(
                    "Temporal audit budget exhausted; no partial result reported"
                )
            with np.load(cache / f"{row['cache']}.npz") as values:
                variants = temporal_variants(values["clips"], rng)
            for name, clips in variants.items():
                scores[name].append(
                    float(torch.sigmoid(model(clip_tensor(clips, device))).max().cpu())
                )
            if (index + 1) % 64 == 0:
                print(f"TEMPORAL_AUDIT {index + 1}/{len(rows)}", flush=True)
    result = {
        "scope": "Paired counterfactual diagnostics on development validation, not a new test or model-selection score.",
        "threshold": report["threshold"],
        "original": metrics(rows, original, report["threshold"]),
        "controls": {},
        "interpretation_note": "Similar static/shuffled scores suggest reliance on appearance; these out-of-distribution interventions alone do not prove a causal failure or phone accuracy.",
        "approved_for_demo": False,
        "seconds": time.monotonic() - started,
    }
    for name, values in scores.items():
        values = np.asarray(values)
        result["controls"][name] = {
            "metrics_at_original_threshold": metrics(rows, values, report["threshold"]),
            "mean_absolute_score_change": float(np.abs(values - original).mean()),
            "prediction_agreement": float(
                (
                    (values >= report["threshold"]) == (original >= report["threshold"])
                ).mean()
            ),
            "score_correlation": (
                float(np.corrcoef(original, values)[0, 1])
                if original.std() > 0 and values.std() > 0
                else None
            ),
        }
    write_json(output / "temporal_controls.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--budget-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("Positive budget required")
    run(args.output_dir, args.cache_dir, args.device, args.budget_seconds)


if __name__ == "__main__":
    main()
