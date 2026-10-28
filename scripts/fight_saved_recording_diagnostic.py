"""Local-only causal replay of a saved recording; no training or label inference."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fight.runtime import FightRuntime
from scripts.fight_pilot import write_json


def diagnose(models, video, metadata, output):
    if output.exists():
        raise ValueError("Refuse replacing an existing diagnostic")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    runtime = FightRuntime(models_dir=models, allow_pilot=True)
    if not runtime.model_ready:
        raise ValueError(runtime.message)
    runtime.warmup()
    recorded = json.loads(metadata.read_text()) if metadata else {}
    timestamps = recorded.get("frame_source_timestamps", [])
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise ValueError("Saved recording cannot be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Invalid saved recording FPS")
    started, frame_index, predictions = time.monotonic(), 0, []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if timestamps and frame_index >= len(timestamps):
                raise ValueError("Recording has more frames than logged timestamps")
            timestamp = timestamps[frame_index] if timestamps else frame_index / fps
            previous = runtime.result.get("prediction_timestamp")
            result = runtime.push(frame, timestamp, sampled=bool(timestamps))
            if (
                result.get("prediction_timestamp") is not None
                and result["prediction_timestamp"] != previous
            ):
                predictions.append(
                    {
                        key: result.get(key)
                        for key in (
                            "prediction_timestamp",
                            "score",
                            "state",
                            "event",
                            "people",
                            "visible_people",
                            "inference_seconds",
                            "regions",
                            "person_boxes",
                            "view_quality",
                            "message",
                        )
                    }
                )
            frame_index += 1
    finally:
        capture.release()
    if timestamps and len(timestamps) != frame_index:
        raise ValueError("Recording has fewer frames than logged timestamps")
    report = {
        "scope": "Local-only causal saved-video diagnostic; lossy MP4 is not the original live tensor stream. No action labels inferred; no camera opened or video uploaded.",
        "video": str(video),
        "models": str(models),
        "frames": frame_index,
        "source_timestamps_reused": bool(timestamps),
        "seconds": time.monotonic() - started,
        "threshold": runtime.gate.threshold,
        "event_gate": runtime.metadata.get("event_gate"),
        "maximum_score": max((row["score"] for row in predictions), default=None),
        "alarm_starts": [
            row["prediction_timestamp"]
            for row in predictions
            if row["event"] == "started"
        ],
        "prediction_count": len(predictions),
        "predictions": predictions,
        "approved_for_demo": False,
    }
    write_json(output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "predictions"}
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.models, args.video, args.metadata, args.output)


if __name__ == "__main__":
    main()
