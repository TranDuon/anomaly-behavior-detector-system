"""Continue the installed interaction checkpoint on indoor hard negatives.

AVA and the detector stay frozen. Unlike v3, this updates the existing learned
interaction weights (v1 is widened to v2 without changing its initial logits).
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from action_engine import (
    AUX_NAMES, AUX_VERSION, SOURCE_SHA, PERSON_SHA, SPEC, ROOT,
    ActionEngine, InteractionHead, InteractionHeadV2, digest,
)
from train_interaction import write, metrics, event_metrics
from train_interaction_v2 import (
    HARD_NEGATIVE_ACTIONS, cache_sources, load_features, scores, train_loss,
    validate_indoor_corpus, require_both_classes, development_rank,
    hard_negative_report,
)

CLOSE_ACTIONS = {"stand_close", "approach", "depart", "walk_together"}


def save_state(path, value):
    temporary = path.with_suffix(".partial.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def warm_start(state, version):
    """Keep the v1 function exactly; new context weights start at zero.

    Extra hidden units start with nonzero input weights and zero output weights,
    so they can learn without affecting the inherited prediction at step zero.
    Never recompute visual normalization on the new dataset.
    """
    head = InteractionHeadV2()
    if version == 2:
        head.load_state_dict(state, strict=True)
    elif version == 1:
        old = InteractionHead()
        old.load_state_dict(state, strict=True)
        with torch.no_grad():
            head.mean.copy_(old.mean)
            head.std.copy_(old.std)
            head.layers[0].weight[:128].zero_()
            head.layers[0].weight[:128, :4096].copy_(old.layers[0].weight)
            head.layers[0].bias[:128].copy_(old.layers[0].bias)
            head.layers[3].weight.zero_()
            head.layers[3].weight[:, :128].copy_(old.layers[3].weight)
            head.layers[3].bias.copy_(old.layers[3].bias)
    else:
        raise ValueError("Parent must be an interaction v1/v2 checkpoint")
    return head


def close_report(rows, values, threshold):
    selected = [s for r, s in zip(rows, values)
                if r["label"] == 0 and r["action"] in CLOSE_ACTIONS]
    return {"windows": len(selected),
            "false_positives": int(sum(s >= threshold for s in selected)),
            "false_positive_rate": float(np.mean(np.asarray(selected) >= threshold))
            if selected else None}


def select_threshold(rows, values):
    """Calibration only; constrain close-person FPR as well as overall FPR."""
    unique = np.unique(values)
    thresholds = np.r_[1e-6, (unique[:-1] + unique[1:]) / 2, 1 - 1e-6]
    reports = []
    negatives = np.asarray([s for r, s in zip(rows, values) if r["label"] == 0])
    for threshold in thresholds:
        threshold = float(threshold)
        report = metrics(rows, values, float(threshold))
        close = close_report(rows, values, threshold)
        report.update(threshold=float(threshold), close_people=close,
                      false_positive_rate=float(np.mean(negatives >= threshold)))
        report["target_feasible"] = bool(
            report["precision"] >= .75 and report["recall"] >= .60
            and report["false_positive_rate"] <= .10
            and close["windows"] > 0 and close["false_positive_rate"] <= .10)
        reports.append(report)
    return max(reports, key=lambda r: (
        r["target_feasible"], min(r["precision"], r["recall"]), r["f1"],
        -r["false_positive_rate"], r["threshold"]))


def rank_head(rows, values):
    point = select_threshold(rows, values)
    return (int(point["target_feasible"]), development_rank(rows, values),
            -(point["close_people"]["false_positive_rate"] or 0.0))


def temporal_report(rows, values):
    results = []
    motion_index = AUX_NAMES.index("region_motion_peak")
    for video in sorted({r["video"] for r in rows}):
        pairs = [(r, s) for r, s in zip(rows, values)
                 if r["video"] == video and r["label"] is not None]
        labels = [r["label"] for r, _ in pairs]
        if set(labels) != {0, 1}:
            continue
        motion = [float(r["aux"][:, motion_index].max().cpu())
                  if len(r["aux"]) else 0.0 for r, _ in pairs]
        results.append({"video": video, "windows": len(pairs),
                        "auc": float(roc_auc_score(labels, [s for _, s in pairs])),
                        "motion_auc": float(roc_auc_score(labels, motion))})
    return {"videos": results, "evaluable_videos": len(results),
            "median_within_video_auc": float(np.median([r["auc"] for r in results]))
            if results else None,
            "median_motion_auc": float(np.median([r["motion_auc"] for r in results]))
            if results else None}


def parent_exposure(corpus, metadata):
    """Audit inherited exposure; do not present re-split public data as unseen."""
    matches = [p for p in (ROOT.parent / "benchmarks").glob("*.json")
               if digest(p) == metadata.get("manifest_sha256")]
    if not matches:
        return {"status": "unknown", "note": "Parent training manifest not found; unseen evaluation is unverified"}
    parent = json.loads(matches[0].read_text(encoding="utf-8"))
    exposed = [s for s in parent["sources"] if s["split"] != "test"]
    videos, groups = {s["video"] for s in exposed}, {s["group"] for s in exposed}
    overlap = {split: [s["video"] for s in corpus["sources"]
                       if s["split"] == split and
                       (s["video"] in videos or s["group"] in groups)]
               for split in ("development", "calibration", "test")}
    return {"status": "audited", "parent_manifest": matches[0].name,
            "parent_exposed_videos": overlap,
            "test_is_unseen_from_parent": not overlap["test"],
            "note": "Overlapping sources are regression checks, not independent generalization evidence"}


def evaluate(rows, values, sources, threshold):
    return {"windows": metrics(rows, values, threshold),
            "events": event_metrics(rows, values, sources, threshold),
            "close_people": close_report(rows, values, threshold),
            "hard_negatives": hard_negative_report(rows, values, threshold),
            "temporal": temporal_report(rows, values)}


def run(args):
    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    for path in (args.output, args.cache):
        path.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_indoor_corpus(corpus)
    package_meta = json.loads((args.pack / "PACKAGE.json").read_text(encoding="utf-8"))
    if package_meta["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Drive pack and repository manifest differ")
    base = args.weights / "interaction.pt"
    metadata = json.loads((args.weights / "interaction.json").read_text(encoding="utf-8"))
    if (metadata["checkpoint_sha256"] != digest(base)
            or metadata["source_sha256"] != SOURCE_SHA or metadata["spec"] != SPEC
            or (metadata.get("head_version", 1) == 2 and metadata.get("aux_version") != AUX_VERSION)):
        raise ValueError("Parent checkpoint/source/preprocessing mismatch")
    protocol = {
        "manifest_sha256": digest(args.manifest), "engine_sha256": digest(ROOT / "action_engine.py"),
        "training_code_sha256": digest(Path(__file__)),
        "feature_code_sha256": digest(ROOT / "train_interaction_v2.py"),
        "evaluation_code_sha256": digest(ROOT / "train_interaction.py"),
        "source_sha256": SOURCE_SHA, "person_sha256": PERSON_SHA, "spec": SPEC,
        "head_version": 2, "aux_version": AUX_VERSION, "aux_names": list(AUX_NAMES),
        "parent_checkpoint_sha256": digest(base),
        "parent_metadata_sha256": digest(args.weights / "interaction.json"),
        "seed": 42, "lr": args.lr, "max_epochs": args.epochs, "patience": 20,
        "training_mode": "Warm-start existing interaction weights; frozen AVA/detector; indoor hard-negative mining",
        "private_data_used": False,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Run fingerprint changed; choose a new RUN_TAG")
    write(protocol_path, protocol)
    if (args.output / "mmaction_warmstart_v4_pilot.zip").exists():
        print("Completed bundle already exists", flush=True)
        return
    # Only share caches whose complete extraction fingerprint matches.
    cache_protocol = {k: protocol[k] for k in (
        "manifest_sha256", "engine_sha256", "feature_code_sha256", "source_sha256",
        "person_sha256", "spec", "aux_version")}
    cache_id = hashlib.sha256(json.dumps(cache_protocol, sort_keys=True).encode()).hexdigest()[:24]
    cache = args.cache / cache_id
    cache.mkdir(exist_ok=True)
    write(cache / "protocol.json", cache_protocol)
    write(args.output / "parent_exposure.json", parent_exposure(corpus, metadata))
    engine = ActionEngine(weights=args.weights, device=args.device, tuned=True)
    public = [s for s in corpus["sources"] if s["split"] != "test"]
    rows = load_features(cache_sources(engine, public, args.pack, cache, deadline), cache, args.device)
    partitions = {split: [r for r in rows if r["split"] == split]
                  for split in ("train", "development", "calibration")}
    training = [r for r in partitions["train"] if r["label"] is not None and len(r["bag"])]
    for split, subset in (("train", training), ("development", partitions["development"]),
                          ("calibration", partitions["calibration"])):
        require_both_classes(subset, split)
    write(args.output / "coverage.json", {
        split: {"windows": len(subset), "actions": dict(Counter(r["action"] for r in subset)),
                "positive_windows_without_person": sum(r["label"] == 1 and not len(r["bag"]) for r in subset)}
        for split, subset in partitions.items()})
    if not any(r["label"] == 0 and r["action"] in CLOSE_ACTIONS for r in training):
        raise ValueError("No trainable close-person negatives in this pack")
    head = warm_start(torch.load(base, map_location="cpu", weights_only=True),
                      int(metadata.get("head_version", 1))).to(args.device)
    if metadata.get("head_version", 1) == 1:
        auxiliary = torch.cat([r["aux"] for r in training])
        head.aux_mean.copy_(auxiliary.mean(0))
        head.aux_std.copy_(auxiliary.std(0).clamp_min(.01))
    baseline = {split: scores(head, subset) for split, subset in partitions.items() if split != "train"}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.02)
    counts = Counter((r["dataset"], r["label"], r["group"]) for r in training)
    group_counts = Counter((d, label) for d, label, _ in counts)
    weights = np.asarray([HARD_NEGATIVE_ACTIONS.get(r["action"], 1.0) /
                          (counts[(r["dataset"], r["label"], r["group"])] *
                           group_counts[(r["dataset"], r["label"])]) for r in training])
    state_path, best_path = args.output / "last.pt", args.output / "interaction.pt"
    frozen_path = args.output / "selection_frozen.json"
    best_rank = rank_head(partitions["development"], baseline["development"])
    best_epoch, stale, history = 0, 0, []
    if state_path.exists():
        state = torch.load(state_path, map_location=args.device, weights_only=True)
        head.load_state_dict(state["head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        best_rank, best_epoch, stale = tuple(state["best_rank"]), state["best_epoch"], state["stale"]
        history = state["history"]
        # last.pt includes the best state, making each epoch resume atomic.
        if not frozen_path.exists():
            save_state(best_path, {k: v.cpu() for k, v in state["best"].items()})
        torch.set_rng_state(state["rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    else:
        save_state(best_path, {k: v.detach().cpu() for k, v in head.state_dict().items()})
    if not frozen_path.exists():
        for epoch in range(len(history), args.epochs):
            if stale >= protocol["patience"]:
                break
            if time.monotonic() >= deadline - 180:
                raise TimeoutError("Training paused at epoch boundary; rerun to resume")
            hardness = np.ones(len(training))
            if epoch >= 3:
                hardness = np.asarray([1 + 4 * s if r["label"] == 0 else 1
                                       for r, s in zip(training, scores(head, training))])
            order = torch.multinomial(torch.as_tensor(weights * hardness, dtype=torch.float64),
                                      max(128, len(training)), replacement=True).tolist()
            head.train()
            total = 0.0
            for offset in range(0, len(order), 8):
                optimizer.zero_grad()
                losses = [train_loss(head, training[i]) for i in order[offset:offset + 8]]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
                total += float(loss.detach()) * len(losses)
            rank = rank_head(partitions["development"], scores(head, partitions["development"]))
            if rank > best_rank:
                best_rank, best_epoch, stale = rank, epoch + 1, 0
                save_state(best_path, {k: v.detach().cpu() for k, v in head.state_dict().items()})
            else:
                stale += 1
            history.append({"epoch": epoch + 1, "loss": total / len(order), "development_rank": list(rank)})
            save_state(state_path, {"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                                   "best": torch.load(best_path, map_location="cpu", weights_only=True),
                                   "best_rank": list(best_rank), "best_epoch": best_epoch,
                                   "stale": stale, "history": history, "rng": torch.get_rng_state(),
                                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
            write(args.output / "history.json", history)
            print("WARMSTART_V4", json.dumps(history[-1]), flush=True)
        head.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
        selected = select_threshold(partitions["calibration"], scores(head, partitions["calibration"]))
        write(frozen_path, {"checkpoint_sha256": digest(best_path), "operating_point": selected,
                            "best_epoch": best_epoch, "epochs": len(history)})
    frozen = json.loads(frozen_path.read_text())
    if digest(best_path) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen selection changed")
    head.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
    selected = frozen["operating_point"]
    output_metadata = {**protocol, "checkpoint_sha256": digest(best_path),
                       "threshold": selected["threshold"], "confirm_seconds": .4,
                       "operating_point_feasible": selected["target_feasible"],
                       "approved_for_demo": False, "real_phone_tested": False,
                       "epochs": frozen["epochs"], "best_epoch": frozen["best_epoch"],
                       "scope": "Indoor warm-start pilot; public regression evidence, local camera acceptance pending"}
    write(args.output / "interaction.json", output_metadata)
    test_sources = [s for s in corpus["sources"] if s["split"] == "test"]
    testing = load_features(cache_sources(engine, test_sources, args.pack, cache, deadline), cache, args.device)
    require_both_classes(testing, "test")
    parent_head = warm_start(torch.load(base, map_location="cpu", weights_only=True),
                             int(metadata.get("head_version", 1))).to(args.device)
    parent_values, new_values = scores(parent_head, testing), scores(head, testing)
    parent_calibration = select_threshold(partitions["calibration"], baseline["calibration"])
    report = {"status": "completed", "training": output_metadata, "calibration": selected,
              "parent_calibration": parent_calibration,
              "parent_exposure": parent_exposure(corpus, metadata),
              "test": {
                  "parent_installed_threshold": evaluate(testing, parent_values, test_sources, metadata["threshold"]),
                  "parent_recalibrated": evaluate(testing, parent_values, test_sources, parent_calibration["threshold"]),
                  "warmstart": evaluate(testing, new_values, test_sources, selected["threshold"])},
              "seconds_this_session": time.monotonic() - started}
    head.eval()
    reference = training[0]
    with torch.inference_mode():
        gpu = head(reference["bag"], reference["aux"]).cpu()
        cpu = copy.deepcopy(head).cpu().eval()(reference["bag"].cpu(), reference["aux"].cpu())
    report["head_gpu_cpu_max_logit_difference"] = float((gpu - cpu).abs().max())
    if not torch.allclose(gpu, cpu, atol=1e-4, rtol=1e-4):
        raise ValueError("Head GPU/CPU parity failed")
    write(args.output / "test_predictions.json", [
        {k: v for k, v in r.items() if k not in {"bag", "aux"}} |
        {"parent_score": old, "warmstart_score": new}
        for r, old, new in zip(testing, parent_values, new_values)])
    write(args.output / "report.json", report)
    bundle = args.output / "mmaction_warmstart_v4_pilot.zip"
    with zipfile.ZipFile(bundle.with_suffix(".partial.zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("interaction.pt", "interaction.json", "report.json", "protocol.json",
                     "parent_exposure.json", "coverage.json", "selection_frozen.json"):
            archive.write(args.output / name, name)
    bundle.with_suffix(".partial.zip").replace(bundle)
    write(args.output / "status.json", {"status": "completed", "bundle": str(bundle)})
    print("FINISHED_WARMSTART_V4", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pack", "cache", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=.0001)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 1440 or not 1 <= args.epochs <= 500 or not 0 < args.lr <= .001:
        parser.error("Invalid training budget, epoch count or learning rate")
    try:
        run(args)
    except Exception as error:
        write(args.output / "status.json", {"status": "incomplete", "error": repr(error),
                                            "resume": "Rerun the same revision and RUN_TAG"})
        raise


if __name__ == "__main__":
    main()
