"""Read-only comparison of saved interaction pilots, without model selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate_events(records):
    totals = {
        name: sum(record["metrics"][name] for record in records)
        for name in (
            "ground_truth_events",
            "true_positive",
            "false_positive",
            "false_negative",
            "ignored_events",
            "normal_minutes",
        )
    }
    true_positive = totals["true_positive"]
    totals["precision"] = true_positive / max(
        1, true_positive + totals["false_positive"]
    )
    totals["recall"] = true_positive / max(1, totals["ground_truth_events"])
    totals["false_alerts_per_normal_minute"] = (
        totals["false_positive"] / totals["normal_minutes"]
        if totals["normal_minutes"] > 0
        else None
    )
    totals["source_recordings"] = len(records)
    return totals


def summarize(comparison):
    window_scope = "One trailing window per annotated action interval; tiny previously inspected source diagnostic, not public central-clip or camera accuracy"
    conditions = {}
    if "baseline_continuous" in comparison:
        conditions["baseline"] = {
            "windows": {
                **comparison["baseline_pku_windows"],
                "metric_scope": window_scope,
            },
            "events": aggregate_events(comparison["baseline_continuous"]),
        }
    for name, report in comparison["variants"].items():
        conditions[name] = {
            "threshold": report["threshold"],
            "calibration_cohorts": report["cohorts"],
            "calibration_target_met": report["all_calibration_cohorts_meet_target"],
            "windows": {**report["pku_diagnostic"], "metric_scope": window_scope},
            "events": aggregate_events(report["continuous"]),
        }
    return {
        "status": comparison["status"],
        "conditions": conditions,
        "scope": comparison["scope"],
        "approved_for_demo": False,
        "real_phone_tested": False,
        "note": "No automatic ranking/promotion. Calibration is not test; PKU windows/events are tiny previously inspected diagnostics, not camera acceptance.",
    }


def render_timelines(comparison, root):
    conditions = {
        "baseline": (
            comparison.get("baseline_continuous", []),
            None,
        ),
        **{
            name: (report["continuous"], report["threshold"])
            for name, report in comparison["variants"].items()
        },
    }
    sources = {}
    for name in comparison["variants"]:
        manifest = json.loads((root / name / "manifest.json").read_text())
        sources.update({source["video"]: source for source in manifest["sources"]})
    paths = []
    for video, source in sources.items():
        if not any(
            record["video"] == video
            for records, threshold in conditions.values()
            for record in records
        ):
            continue
        figure, axes = plt.subplots(
            len(conditions),
            1,
            figsize=(13, 3 * len(conditions)),
            sharex=True,
            squeeze=False,
        )
        for axis, (name, (records, threshold)) in zip(axes[:, 0], conditions.items()):
            record = next((item for item in records if item["video"] == video), None)
            if record is None:
                continue
            for interval in source["intervals"]:
                if interval["label"] != 0:
                    axis.axvspan(
                        interval["start"] / source["fps"],
                        (interval["end"] + 1) / source["fps"],
                        color="red" if interval["label"] == 1 else "grey",
                        alpha=0.18,
                    )
            axis.plot(
                [item["timestamp"] for item in record["predictions"]],
                [item["score"] for item in record["predictions"]],
                label="model score",
            )
            if threshold is not None:
                axis.axhline(
                    threshold, color="black", linestyle="--", label="threshold"
                )
            for event in record["events"]:
                axis.plot(
                    [event["start"], event["end"]],
                    [1.03, 1.03],
                    color="orange",
                    linewidth=5,
                )
            axis.set(title=name, ylim=(-0.03, 1.08), ylabel="score")
            axis.legend(loc="upper right")
        axes[-1, 0].set_xlabel("source time (seconds)")
        figure.suptitle(video + " | red: target action; grey: unscored; orange: alarm")
        figure.tight_layout()
        path = root / (video.replace("/", "_") + "_timeline.png")
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    comparison = json.loads((args.root / "comparison.json").read_text())
    summary = summarize(comparison)
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    paths = render_timelines(comparison, args.root)
    for name, condition in summary["conditions"].items():
        print(name, json.dumps(condition["events"]))
    print("Timelines:", [str(path) for path in paths])
    print(summary["note"])


if __name__ == "__main__":
    main()
