"""Event scoring over explicit reviewed time, excluding unlabeled gaps."""

import numpy as np


def reviewed_event_metrics(events, intervals, tolerance=1.0):
    positives = [item for item in intervals if item["action"] == "fight"]
    negatives = [
        item for item in intervals if item["action"] not in {"fight", "uncertain"}
    ]
    matched, delays, false_positive, ignored = set(), [], 0, 0
    false_during_normal, duplicate_alerts = 0, 0
    for event in sorted(events, key=lambda item: item["start"]):
        candidates = []
        for index, interval in enumerate(positives):
            start, end = interval["start_seconds"], interval["end_seconds"] + tolerance
            intersection = max(0.0, min(event["end"], end) - max(event["start"], start))
            union = max(event["end"], end) - min(event["start"], start)
            if (
                index not in matched
                and event["start"] >= start - 0.25
                and intersection / max(union, 1e-9) >= 0.1
            ):
                candidates.append(index)
        if candidates:
            index = min(
                candidates,
                key=lambda value: abs(
                    event["start"] - positives[value]["start_seconds"]
                ),
            )
            matched.add(index)
            delays.append(max(0.0, event["start"] - positives[index]["start_seconds"]))
        elif any(
            event["start"] < item["end_seconds"]
            and event["end"] > item["start_seconds"]
            for item in negatives
        ):
            false_positive += 1
            false_during_normal += 1
        elif any(
            event["start"] < item["end_seconds"] + tolerance
            and event["end"] > item["start_seconds"]
            for item in positives
        ):
            false_positive += 1
            duplicate_alerts += 1
        else:
            ignored += 1
    merged_events = []
    for event in sorted(events, key=lambda item: item["start"]):
        if merged_events and event["start"] <= merged_events[-1][1]:
            merged_events[-1][1] = max(merged_events[-1][1], event["end"])
        else:
            merged_events.append([event["start"], event["end"]])
    normal_alert_seconds = sum(
        max(
            0.0,
            min(end, interval["end_seconds"]) - max(start, interval["start_seconds"]),
        )
        for start, end in merged_events
        for interval in negatives
    )
    normal_minutes = (
        sum(item["end_seconds"] - item["start_seconds"] for item in negatives) / 60
    )
    return {
        "ground_truth_events": len(positives),
        "true_positive": len(matched),
        "false_negative": len(positives) - len(matched),
        "false_positive": false_positive,
        "false_alerts_during_normal": false_during_normal,
        "duplicate_alerts": duplicate_alerts,
        "normal_alert_seconds": normal_alert_seconds,
        "ignored_unlabeled_events": ignored,
        "normal_minutes": normal_minutes,
        "start_delays_seconds": delays,
        "scope": "Reviewed intervals only; unlabeled gaps are not normal, and touching an unlabeled gap does not erase a false alert in reviewed normal time",
    }


def summarize_events(records):
    total = {
        key: sum(record[key] for record in records)
        for key in (
            "ground_truth_events",
            "true_positive",
            "false_negative",
            "false_positive",
            "false_alerts_during_normal",
            "duplicate_alerts",
            "normal_alert_seconds",
            "normal_minutes",
            "ignored_unlabeled_events",
        )
    }
    delays = [value for record in records for value in record["start_delays_seconds"]]
    total.update(
        precision=total["true_positive"]
        / max(1, total["true_positive"] + total["false_positive"]),
        recall=total["true_positive"] / max(1, total["ground_truth_events"]),
        false_alerts_per_normal_minute=(
            total["false_alerts_during_normal"] / total["normal_minutes"]
            if total["normal_minutes"]
            else None
        ),
        normal_alert_fraction=(
            total["normal_alert_seconds"] / (60 * total["normal_minutes"])
            if total["normal_minutes"]
            else None
        ),
        p95_start_delay_seconds=float(np.percentile(delays, 95)) if delays else None,
        scope="Small reviewed public-source diagnostic; not independent phone acceptance",
    )
    return total
