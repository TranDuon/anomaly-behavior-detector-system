"""Run one real video prediction through each supported local model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fight.mmaction_runtime import MMActionRuntime
from fight.runtime import FightRuntime


def verify(source, output):
    records = []
    for backend in ("legacy", "mmaction_original", "mmaction_interaction_v1", "mmaction_fall_wrestle", "mmaction_transition_redid"):
        runtime = (
            FightRuntime(ROOT / "models", allow_pilot=True, cpu_threads=4, batch_detection=True)
            if backend == "legacy"
            else MMActionRuntime(tuned=backend in {"mmaction_interaction_v1", "mmaction_fall_wrestle", "mmaction_transition_redid"},
                                 fall_wrestle=backend == "mmaction_fall_wrestle",
                                 transition=backend == "mmaction_transition_redid", allow_pilot=True)
        )
        capture = cv2.VideoCapture(str(source))
        try:
            if not runtime.model_ready or not capture.isOpened():
                raise RuntimeError(f"{backend}: model or video unavailable: {runtime.message}")
            runtime.warmup()
            for index in range(runtime.config.frames):
                timestamp = index / runtime.config.fps
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Video is too short for a complete model clip")
                result = runtime.push(frame, timestamp, sampled=True)
            score = result["score"]
            if score is None or not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError(f"{backend}: invalid prediction {score}")
            records.append({
                "backend": backend,
                "state": result["state"],
                "score": score,
                "people": result["people"],
                "input_kind": runtime.input_kind,
                "checkpoint_sha256": runtime.metadata.get("checkpoint_sha256"),
            })
            print(json.dumps(records[-1]), flush=True)
        finally:
            capture.release()
            if hasattr(runtime, "close"):
                runtime.close()
    report = {
        "status": "completed",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "One real video clip per backend; loading/inference smoke check, not accuracy or camera acceptance",
        "models": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/supported_models_smoke.json")
    args = parser.parse_args()
    verify(args.source.resolve(), args.output)
