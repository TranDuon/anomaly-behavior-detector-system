"""Check exported CPU scores against saved GPU predictions, not camera accuracy."""

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from fight.vision import clip_tensor


def check_bundle(path, references=None):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    results = []
    with zipfile.ZipFile(path) as archive:
        examples = json.loads(archive.read("expected.json"))
        reference_map = {
            (row["variant"], row["cache"]): row for row in references or []
        }
        for variant in sorted({row["variant"] for row in examples}):
            with zipfile.ZipFile(
                io.BytesIO(archive.read(variant + "/runtime.zip"))
            ) as runtime:
                metadata = json.loads(runtime.read("models/fight_runtime.json"))
                model = torch.jit.load(
                    io.BytesIO(runtime.read("models/fight_model.ts")),
                    map_location="cpu",
                ).eval()
            config = metadata["clip"]
            dummy = torch.zeros(1, 3, config["frames"], config["size"], config["size"])
            with torch.inference_mode():
                model(dummy)
                model(dummy)
                for row in examples:
                    if row["variant"] != variant:
                        continue
                    with np.load(
                        io.BytesIO(archive.read(variant + "/" + row["cache"] + ".npz")),
                        allow_pickle=False,
                    ) as cached:
                        inputs = clip_tensor(cached["clips"], "cpu")
                    started = time.perf_counter()
                    score = float(torch.sigmoid(model(inputs).max()))
                    seconds = time.perf_counter() - started
                    reference = reference_map.get((variant, row["cache"]))
                    if references is not None and reference is None:
                        raise ValueError("Missing CPU reference for a smoke example")
                    expected = (
                        reference["colab_cpu_reference"] if reference else row["score"]
                    )
                    error = abs(score - expected)
                    results.append(
                        {
                            **row,
                            "cpu_score": score,
                            "reference_source": (
                                "colab_cpu" if reference else "original_gpu"
                            ),
                            "reference_score": expected,
                            "original_gpu_absolute_error": abs(score - row["score"]),
                            "frozen_threshold_decision_agrees": (
                                score >= metadata["threshold"]
                            )
                            == (row["score"] >= metadata["threshold"]),
                            "absolute_error": error,
                            "cpu_forward_seconds": seconds,
                            "passed": error <= 1e-4,
                        }
                    )
    if not results:
        raise ValueError("No smoke examples supplied")
    return {
        "scope": "Same preprocessed RGB tensors, GPU versus exported CPU classifier; excludes detector, upload/replay queues and physical iPhone",
        "passed": all(row["passed"] for row in results),
        "examples": results,
        "approved_for_demo": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    references = json.loads(args.reference.read_text()) if args.reference else None
    report = check_bundle(args.bundle, references)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("CPU/GPU score mismatch")


if __name__ == "__main__":
    main()
