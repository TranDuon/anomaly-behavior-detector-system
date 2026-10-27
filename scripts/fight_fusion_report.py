"""Render saved fusion diagnostics without changing weights or selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fight.interaction_data import pku_sources
from fight.vision import ClipConfig
from scripts.fight_pilot import write_json


def grouped_negatives(rows, threshold):
    result = {}
    for tag in sorted(
        {tag for row in rows if row["label"] == 0 for tag in row.get("actions", [])}
    ):
        selected = [
            row for row in rows if row["label"] == 0 and tag in row.get("actions", [])
        ]
        groups = {row.get("performance", row["video"]) for row in selected}
        false_groups = {
            row.get("performance", row["video"])
            for row in selected
            if row["score"] >= threshold
        }
        result[tag] = {
            "clips": len(selected),
            "false_positive_clips": sum(row["score"] >= threshold for row in selected),
            "performances": len(groups),
            "performances_with_false_positive": len(false_groups),
            "scope": "Central-window repeated public diagnostic; paired views not independent and publisher tags can overlap",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pku", type=Path)
    args = parser.parse_args()
    run = json.loads((args.root / "run.json").read_text())
    sources = {}
    if args.pku and args.pku.is_dir():
        records, training, diagnostic = pku_sources(args.pku, ClipConfig(fps=12))
        sources = {row["video"]: row for row in records}
        write_json(args.root / "source_intervals.json", sources)
    elif (args.root / "source_intervals.json").exists():
        sources = json.loads((args.root / "source_intervals.json").read_text())
    summary = {
        "status": run["status"],
        "scope": run["scope"],
        "variants": {},
        "approved_for_demo": False,
        "real_phone_tested": False,
    }
    for name, report in run["variants"].items():
        root = args.root / name
        history = json.loads((root / "history.json").read_text())
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = np.arange(1, len(history) + 1)
        axes[0].plot(epochs, [row["loss"] for row in history], label="training loss")
        axes[0].plot(epochs, [row["bce"] for row in history], label="development BCE")
        axes[0].set(xlabel="completed epoch", ylabel="loss")
        axes[1].plot(
            epochs,
            [row["worst_precision_recall"] for row in history],
            label="worst source min(P,R)",
        )
        axes[1].plot(
            epochs, [row["macro_ap"] for row in history], label="development macro AP"
        )
        axes[1].set(
            xlabel="completed epoch", ylabel="development score", ylim=(0, 1.02)
        )
        for axis in axes:
            axis.legend()
        figure.suptitle(name + " | development only, not phone accuracy")
        figure.tight_layout()
        figure.savefig(root / "learning_curves.png", dpi=130)
        plt.close(figure)
        rows = json.loads((root / "test_predictions.json").read_text())
        negatives = grouped_negatives(rows, report["threshold"])
        write_json(root / "grouped_hard_negatives.json", negatives)
        for trace in report.get("pku_continuous", []):
            figure, axis = plt.subplots(figsize=(12, 3.5))
            source = sources.get(trace["video"])
            if source:
                for interval in source["intervals"]:
                    if interval["label"] != 0:
                        axis.axvspan(
                            interval["start"] / source["fps"],
                            (interval["end"] + 1) / source["fps"],
                            color="red" if interval["label"] == 1 else "grey",
                            alpha=0.18,
                        )
            axis.plot(
                [row["timestamp"] for row in trace["predictions"]],
                [row["score"] for row in trace["predictions"]],
                label="fight score",
            )
            axis.axhline(
                report["threshold"],
                linestyle="--",
                color="black",
                label="frozen threshold",
            )
            for event in trace["events"]:
                axis.plot(
                    [event["start"], event["end"]],
                    [1.03, 1.03],
                    color="orange",
                    linewidth=5,
                )
            axis.set(
                xlabel="source seconds",
                ylabel="score",
                ylim=(-0.03, 1.08),
                title=name
                + " / "
                + trace["video"]
                + " | red target, grey unscored, orange alarm",
            )
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                root / (trace["video"].replace("/", "_") + "_timeline.png"), dpi=130
            )
            plt.close(figure)
        summary["variants"][name] = {
            "protected_development_progress": report["protected_development_progress"],
            "completed_epochs": report["completed_epochs"],
            "threshold": report["threshold"],
            "event_gate": report["event_gate"],
            "public_diagnostic": report["test_by_dataset"],
            "hard_negatives": negatives,
            "pku_diagnostic": [
                {"video": row["video"], "metrics": row["metrics"]}
                for row in report.get("pku_continuous", [])
            ],
        }
        print(
            name,
            "epochs",
            report["completed_epochs"],
            "protected progress",
            report["protected_development_progress"],
            "hard negatives",
            negatives,
            flush=True,
        )
    write_json(args.root / "summary.json", summary)
    print("Saved summary/plots; no model changed or promoted.")


if __name__ == "__main__":
    main()
