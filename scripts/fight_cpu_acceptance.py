"""Real CPU model through uploaded-file and paced camera-replay API paths."""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from scripts.fight_pilot import write_json


def install_candidate(bundle, output):
    models = output / "candidate_models"
    models.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in ("fight_model.ts", "fight_runtime.json"):
            data = archive.read("models/" + name)
            path = models / name
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Different candidate already exists in output")
            path.write_bytes(data)
    return models


def run(args):
    from fastapi.testclient import TestClient
    import api.fight_server as server

    torch.set_num_threads(4)
    server.settings.models_dir = install_candidate(args.bundle, args.output)
    original_factory = server.FightRuntime
    active_trace = []

    def instrumented_factory(*positional, **keywords):
        runtime = original_factory(*positional, **keywords)
        original_push = runtime.push
        destination = active_trace

        def observed_push(*values, **options):
            previous = runtime.result.get("prediction_timestamp")
            result = original_push(*values, **options)
            if (
                result.get("prediction_timestamp") is not None
                and result["prediction_timestamp"] != previous
            ):
                destination.append(
                    {
                        key: result.get(key)
                        for key in (
                            "prediction_timestamp",
                            "score",
                            "state",
                            "people",
                            "event",
                            "inference_seconds",
                        )
                    }
                )
            return result

        runtime.push = observed_push
        return runtime

    server.FightRuntime = instrumented_factory
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    results = []
    deadline = time.monotonic() + args.budget_minutes * 60
    try:
        with TestClient(server.app) as client:
            for source in sources:
                for replay in (False, True):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CPU acceptance budget exhausted")
                    active_trace = []
                    previous_events = {event["id"] for event in server.processor.events}
                    started = time.monotonic()
                    path = Path(source["path"])
                    if not path.is_absolute():
                        path = args.sources.parent / path
                    with path.open("rb") as handle:
                        response = client.post(
                            f"/api/upload?allow_pilot=true&replay={str(replay).lower()}",
                            files={"file": (path.name, handle, "video/mp4")},
                        )
                    response.raise_for_status()
                    ages, lags, stale_polls = [], [], 0
                    while True:
                        status = client.get("/api/status").json()
                        lags.append(status["latency_seconds"])
                        if status["prediction_age_seconds"] is not None:
                            ages.append(status["prediction_age_seconds"])
                        stale_polls += int(
                            status["prediction"].get("score_stale", False)
                        )
                        if not status["running"]:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("CPU acceptance budget exhausted")
                        time.sleep(0.1)
                    if status["phase"] != "completed":
                        raise RuntimeError(status["error"] or status["phase"])
                    durations = [row["inference_seconds"] for row in active_trace]
                    record = {
                        "video": source["video"],
                        "publisher_clip_label": source.get("label"),
                        "mode": "camera_replay" if replay else "uploaded_file",
                        "physical_phone": False,
                        "seconds": time.monotonic() - started,
                        "frames": status["frame_count"],
                        "source_duration_seconds": status["timestamp"],
                        "predictions": active_trace,
                        "prediction_count": len(active_trace),
                        "inference_p50_seconds": (
                            float(np.median(durations)) if durations else None
                        ),
                        "inference_p95_seconds": (
                            float(np.percentile(durations, 95)) if durations else None
                        ),
                        "max_lag_seconds": max(lags, default=0),
                        "max_prediction_age_seconds": max(ages, default=None),
                        "stale_status_polls": stale_polls,
                        "dropped_samples": status["dropped_samples"],
                        "events": [
                            dict(event)
                            for event in server.processor.events
                            if event["id"] not in previous_events
                        ],
                    }
                    results.append(record)
                    write_json(
                        args.output / "cpu_acceptance.json",
                        {
                            "scope": "Real CPU API integration/throughput, not event accuracy or physical iPhone acceptance",
                            "results": results,
                            "completed": False,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in record.items()
                                if key not in {"predictions", "events"}
                            }
                        ),
                        flush=True,
                    )
        comparisons = []
        for index in range(0, len(results), 2):
            offline, replay = results[index : index + 2]
            baseline = {
                round(row["prediction_timestamp"], 6): row["score"]
                for row in offline["predictions"]
            }
            differences = [
                abs(row["score"] - baseline[round(row["prediction_timestamp"], 6)])
                for row in replay["predictions"]
                if round(row["prediction_timestamp"], 6) in baseline
            ]
            comparisons.append(
                {
                    "video": offline["video"],
                    "matching_prediction_timestamps": len(differences),
                    "maximum_matching_score_difference": max(differences, default=None),
                    "offline_events": len(offline["events"]),
                    "replay_events": len(replay["events"]),
                }
            )
        report = {
            "scope": "Real CPU API integration/throughput, not event accuracy or physical iPhone acceptance",
            "completed": True,
            "physical_phone_tested": False,
            "results": results,
            "comparisons": comparisons,
        }
        write_json(args.output / "cpu_acceptance.json", report)
        print("CPU_ACCEPTANCE_COMPLETED " + json.dumps(comparisons), flush=True)
    finally:
        server.processor.stop()
        server.FightRuntime = original_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=10)
    args = parser.parse_args()
    if args.budget_minutes <= 0:
        parser.error("Positive budget required")
    run(args)


if __name__ == "__main__":
    main()
