"""Profile CPU classifier threads without changing crops, thresholds or sampling."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fight.pair_motion import MotionConfig, motion_inputs
from fight.vision import clip_tensor
from scripts.fight_pilot import write_json


def profile(models, examples, output, threads=(1, 2, 4), repeats=7):
    if repeats < 3 or not threads or any(value not in range(1, 9) for value in threads):
        raise ValueError("Use 1–8 threads and at least three timed repeats")
    if output.exists():
        raise ValueError("Refuse replacing an existing CPU profile")
    metadata = json.loads((models / "fight_runtime.json").read_text())
    model = torch.jit.load(str(models / "fight_model.ts"), map_location="cpu").eval()
    inputs = []
    for path in examples:
        with np.load(path, allow_pickle=False) as data:
            values = (clip_tensor(data["clips"]),)
            if metadata.get("input_kind") in {"rgb_pose", "rgb_pose_motion"}:
                poses = (
                    torch.zeros_like(torch.from_numpy(data["poses"]))
                    if metadata.get("use_pose") is False
                    else torch.from_numpy(data["poses"])
                )
                values += (poses,)
            if metadata.get("input_kind") == "rgb_pose_motion":
                flow = (
                    motion_inputs(data["clips"], MotionConfig(**metadata["motion"]))
                    if metadata.get("use_motion") is not False
                    else np.zeros((len(data["clips"]), 2, 15, 32, 32), np.float32)
                )
                values += (torch.from_numpy(flow),)
            inputs.append(values)
    if not inputs:
        raise ValueError("At least one deterministic public example required")
    reference, records = [], []
    with torch.inference_mode():
        for count in threads:
            torch.set_num_threads(count)
            timings, differences = [], []
            for index, values in enumerate(inputs):
                for iteration in range(3):
                    scores = torch.sigmoid(model(*values))
                for iteration in range(repeats):
                    started = time.perf_counter()
                    scores = torch.sigmoid(model(*values))
                    timings.append(time.perf_counter() - started)
                if len(reference) < len(inputs):
                    reference.append(scores.clone())
                differences.append(float((scores - reference[index]).abs().max()))
            records.append(
                {
                    "threads": count,
                    "median_seconds": float(np.median(timings)),
                    "p95_seconds": float(np.percentile(timings, 95)),
                    "maximum_score_difference": max(differences),
                    "score_agreement_passed": max(differences) <= 1e-4,
                }
            )
    valid = [row for row in records if row["score_agreement_passed"]]
    selected = min(valid, key=lambda row: row["p95_seconds"]) if valid else None
    report = {
        "scope": "Classifier-only public tensor microbenchmark; excludes source decoding, detection, pose extraction, queues and rendering. Must confirm with paced replay before applying.",
        "records": records,
        "recommended_threads": selected["threads"] if selected else None,
        "classifier_p95_fits_stride": bool(
            selected and selected["p95_seconds"] <= metadata["clip"]["stride_seconds"]
        ),
        "end_to_end_measured": False,
        "approved_for_demo": False,
    }
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--examples", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    profile(args.models, args.examples, args.output, args.threads)


if __name__ == "__main__":
    main()
