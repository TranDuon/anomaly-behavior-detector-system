"""Refine the installed MMAction interaction branch with indoor hard negatives.

The original AVA backbone and the previously installed interaction head remain
frozen. This script learns a separate v3 head from their output plus the same
person-pair and motion context used by v2.
"""

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import time
import zipfile

os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import torch

from action_engine import (
    AUX_NAMES,
    AUX_VERSION,
    ActionEngine,
    ROOT,
    SOURCE_SHA,
    SPEC,
    TunedRefinementHead,
    digest,
)
from train_interaction import event_metrics, metrics, write
from train_interaction_v2 import (
    HARD_NEGATIVE_ACTIONS,
    cache_sources,
    development_rank,
    hard_negative_report,
    operating_point,
    validate_indoor_corpus,
)


def load_tuned_features(rows, cache, engine, device, base_sha):
    """Persist a separate score vector produced by the installed fine-tuned head."""
    loaded = []
    for row in rows:
        target = cache / (row["key"] + ".npz")
        with np.load(target, allow_pickle=False) as record:
            cached = {name: record[name] for name in record.files}
        recorded_sha = str(cached.get("tuned_base_sha", ""))
        if recorded_sha != base_sha or "tuned_prior" not in cached:
            cached["tuned_prior"] = engine.tuned_scores(cached["bag"], cached["aux"])
            cached["tuned_base_sha"] = np.asarray(base_sha)
            temporary = target.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **cached)
            temporary.replace(target)
        if len(cached["bag"]) != len(cached["tuned_prior"]):
            raise ValueError("Fine-tuned feature cache has incompatible candidate scores")
        loaded.append(
            {
                **row,
                "bag": torch.from_numpy(cached["bag"]).to(device),
                "aux": torch.from_numpy(cached["aux"]).to(device),
                "prior": torch.from_numpy(cached["tuned_prior"]).to(device),
                "base": float(cached["base"]),
                "people": int(cached["people"]),
            }
        )
    return loaded


@torch.inference_mode()
def scores(head, rows):
    head.eval()
    return [
        float(head(row["bag"], row["aux"], row["prior"]).max().sigmoid().cpu())
        if len(row["bag"])
        else 0.0
        for row in rows
    ]


@torch.inference_mode()
def prior_scores(rows):
    return [float(row["prior"].max()) if len(row["prior"]) else 0.0 for row in rows]


def train_loss(head, row):
    logits = head(row["bag"], row["aux"], row["prior"])
    target = logits.new_tensor(float(row["label"]))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.max(), target)
    if row["label"] == 0:
        loss = loss + 0.35 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)
        )
    return loss


def run(args):
    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_indoor_corpus(corpus)
    base_path = args.weights / "interaction.pt"
    base_metadata_path = args.weights / "interaction.json"
    if not base_path.is_file() or not base_metadata_path.is_file():
        raise ValueError("Install the fine-tuned interaction checkpoint before refinement training")
    base_sha = digest(base_path)
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    if base_metadata.get("checkpoint_sha256") != base_sha:
        raise ValueError("Installed fine-tuned checkpoint hash does not match its metadata")
    protocol = {
        "manifest_sha256": digest(args.manifest),
        "engine_sha256": digest(ROOT / "action_engine.py"),
        "training_code_sha256": digest(Path(__file__)),
        "source_sha256": SOURCE_SHA,
        "spec": SPEC,
        "head_version": 3,
        "aux_version": AUX_VERSION,
        "aux_names": list(AUX_NAMES),
        "base_checkpoint_sha256": base_sha,
        "base_metadata_sha256": digest(base_metadata_path),
        "seed": 42,
        "training_mode": "Frozen AVA plus frozen installed fine-tuned head; train a separate indoor hard-negative refinement head",
        "private_data_used": False,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Run fingerprint changed; use a new output folder")
    write(protocol_path, protocol)
    if (args.output / "report.json").is_file():
        print("Run already completed; retaining frozen selection", flush=True)
        return

    engine = ActionEngine(weights=args.weights, device=args.device, tuned=True)
    public = [source for source in corpus["sources"] if source["split"] != "test"]
    rows = cache_sources(engine, public, args.pack, args.cache, deadline)
    loaded = load_tuned_features(rows, args.cache, engine, args.device, base_sha)
    partitions = {
        split: [row for row in loaded if row["split"] == split]
        for split in ("train", "development", "calibration")
    }
    training = [
        row for row in partitions["train"] if row["label"] is not None and len(row["bag"])
    ]
    if {row["label"] for row in training} != {0, 1}:
        raise ValueError("Training requires both classes with person coverage")
    for split in ("development", "calibration"):
        labels = {row["label"] for row in partitions[split] if row["label"] is not None}
        if labels != {0, 1}:
            raise ValueError(f"{split} split requires both reviewed classes; found {sorted(labels)}")

    head = TunedRefinementHead().to(args.device)
    head.mean.copy_(torch.cat([row["bag"] for row in training]).mean(0))
    head.std.copy_(torch.cat([row["bag"] for row in training]).std(0).clamp_min(0.05))
    all_aux = torch.cat([row["aux"] for row in training])
    head.aux_mean.copy_(all_aux.mean(0))
    head.aux_std.copy_(all_aux.std(0).clamp_min(0.01))
    all_prior = torch.cat([row["prior"] for row in training]).reshape(-1, 1)
    head.prior_mean.copy_(all_prior.mean(0))
    head.prior_std.copy_(all_prior.std(0).clamp_min(0.01))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.00025, weight_decay=0.02)
    counts = Counter((row["dataset"], row["label"], row["group"]) for row in training)
    group_counts = Counter((dataset, label) for dataset, label, _ in counts)
    base_weights = np.asarray(
        [
            HARD_NEGATIVE_ACTIONS.get(row["action"], 1.0)
            / (counts[(row["dataset"], row["label"], row["group"])] * group_counts[(row["dataset"], row["label"])])
            for row in training
        ],
        dtype=np.float64,
    )
    best, best_rank, stale, history = None, -1.0, 0, []
    for epoch in range(args.epochs):
        if time.monotonic() >= deadline - 180:
            break
        hardness = np.ones(len(training))
        if epoch >= 3:
            hardness = np.asarray(
                [1 + 4 * score if row["label"] == 0 else 1 for row, score in zip(training, scores(head, training))]
            )
        order = torch.multinomial(
            torch.as_tensor(base_weights * hardness, dtype=torch.float64),
            max(128, len(training)),
            replacement=True,
        ).tolist()
        head.train()
        loss_sum = 0.0
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            losses = [train_loss(head, training[index]) for index in order[offset : offset + 8]]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(losses)
        rank = development_rank(partitions["development"], scores(head, partitions["development"]))
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(order), "development_worst_cohort_ap": rank})
        print("TRAIN_TUNED_V3", json.dumps(history[-1]), flush=True)
        if rank > best_rank + 1e-4:
            best, best_rank, stale = copy.deepcopy(head.state_dict()), rank, 0
            torch.save({key: value.detach().cpu() for key, value in best.items()}, args.output / "interaction_refined.pt")
        else:
            stale += 1
        write(args.output / "history.json", history)
        if stale >= 15:
            break
    if best is None:
        raise TimeoutError("No refined checkpoint within the budget")
    head.load_state_dict(best)
    calibration_scores = scores(head, partitions["calibration"])
    selected = operating_point(partitions["calibration"], calibration_scores)
    metadata = {
        **protocol,
        "checkpoint_sha256": digest(args.output / "interaction_refined.pt"),
        "threshold": selected["threshold"],
        "confirm_seconds": 0.4,
        "operating_point_feasible": selected["target_feasible"],
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Public-only indoor hard-negative refinement of the installed fine-tuned MMAction branch; private camera clips are local acceptance only",
        "epochs": len(history),
    }
    write(args.output / "interaction_refined.json", metadata)
    test_sources = [source for source in corpus["sources"] if source["split"] == "test"]
    test_rows = cache_sources(engine, test_sources, args.pack, args.cache, deadline)
    testing = load_tuned_features(test_rows, args.cache, engine, args.device, base_sha)
    refined_scores = scores(head, testing)
    base_values = prior_scores(testing)
    report = {
        "status": "completed",
        "scope": metadata["scope"],
        "training": metadata,
        "calibration": selected,
        "development": {
            "fine_tuned_base_worst_cohort_ap": development_rank(partitions["development"], prior_scores(partitions["development"])),
            "refined_worst_cohort_ap": best_rank,
            "hard_negatives": hard_negative_report(partitions["development"], scores(head, partitions["development"]), selected["threshold"]),
        },
        "test": {
            "fine_tuned_base": {
                "windows": metrics(testing, base_values, 0.5),
                "events": event_metrics(testing, base_values, test_sources, 0.5),
            },
            "refined": {
                "windows": metrics(testing, refined_scores, selected["threshold"]),
                "events": event_metrics(testing, refined_scores, test_sources, selected["threshold"]),
                "hard_negatives": hard_negative_report(testing, refined_scores, selected["threshold"]),
            },
        },
        "seconds": time.monotonic() - started,
    }
    write(args.output / "report.json", report)
    with zipfile.ZipFile(args.output / "mmaction_tuned_refinement_v3_pilot.zip", "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in ("interaction_refined.pt", "interaction_refined.json", "report.json", "protocol.json"):
            package.write(args.output / name, name)
    print("FINISHED_TUNED_V3", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pack", "cache", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    arguments = parser.parse_args()
    if not 1 <= arguments.minutes <= 60 or not 1 <= arguments.epochs <= 500:
        parser.error("Bounded budget and epoch count required")
    run(arguments)
