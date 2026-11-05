"""Train a separate neck-alert head while preserving the fall/wrestle/fight parent."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch

from action_engine import InteractionHead, SOURCE_SHA, SPEC, digest
from train_action_transition_v2 import PARENT_SHA, reviewed_sources, sources, write_json
from train_release_neck_v4 import airtlab_sources, features, load_rows, scores


def class_metrics(labels, values, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    selected = values >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, selected, average="binary", zero_division=0)
    return {"windows": int(len(labels)), "positives": int(labels.sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "ap": float(average_precision_score(labels, values))
            if len(set(labels)) == 2 else None}


def combined(parent_values, neck_values, parent_threshold, neck_threshold):
    """The original parent decision is never removed by the supplement."""
    return [old >= parent_threshold or new >= neck_threshold
            for old, new in zip(parent_values, neck_values)]


def rates(rows, decisions):
    groups = {
        "fight": lambda row: row["label"] == 1 and row["dataset"] in {"scfd", "reviewed"},
        "fall": lambda row: row["dataset"] == "gmdcsa24" and row["label"] == 1,
        "choke": lambda row: row["action"] == "choke",
        "hug": lambda row: row["action"] == "hug",
        "ordinary": lambda row: row["label"] == 0 and row["dataset"] in {"scfd", "reviewed"},
    }
    return {name: {"windows": sum(predicate(row) for row in rows),
                   "alert_fraction": (sum(bool(decision) for row, decision in zip(rows, decisions)
                                          if predicate(row)) /
                                      max(1, sum(predicate(row) for row in rows)))}
            for name, predicate in groups.items()}


def choose_threshold(rows, old, neck, parent_threshold):
    if not any(row["action"] == "choke" for row in rows):
        raise ValueError("Development split has no neck positives")
    negative = [value for row, value in zip(rows, neck)
                if row["action"] in {"hug", "sports_wrestle", "ordinary"}]
    if not negative:
        raise ValueError("Development split has no hard negatives")
    candidates = sorted(set([.05, .5, .95] + neck))
    choices = []
    for threshold in candidates:
        decisions = combined(old, neck, parent_threshold, threshold)
        result = rates(rows, decisions)
        # A neck alert must improve choke recall without making hugging worse.
        baseline = rates(rows, [score >= parent_threshold for score in old])
        if result["hug"]["alert_fraction"] > baseline["hug"]["alert_fraction"] + .125:
            continue
        gain = result["choke"]["alert_fraction"] - baseline["choke"]["alert_fraction"]
        choices.append((gain - result["ordinary"]["alert_fraction"] * .25,
                        gain, threshold))
    if not choices:
        raise ValueError("No neck threshold passes the hugging guard")
    _, gain, threshold = max(choices)
    return float(threshold), float(gain)


def train(args, protocol):
    rows = load_rows(args.cache)
    splits = {part: [row for row in rows if row["split"] == part]
              for part in ("train", "development", "test")}
    for part, items in splits.items():
        if not any(row["action"] == "choke" and row["people"] for row in items):
            raise ValueError(part + " lacks detected neck positives")
        if not any(row["action"] == "hug" and row["people"] for row in items):
            raise ValueError(part + " lacks detected hug negatives")
        if not any(row["dataset"] == "scfd" and row["label"] == 1 for row in items):
            raise ValueError(part + " lacks held-out fight examples")
    metadata = json.loads((args.weights / "interaction_fall_wrestle.json").read_text(encoding="utf-8"))
    if (digest(args.weights / "interaction_fall_wrestle.pt") != PARENT_SHA
            or metadata["checkpoint_sha256"] != PARENT_SHA
            or metadata["source_sha256"] != SOURCE_SHA or metadata["spec"] != SPEC):
        raise ValueError("Wrong fall/wrestle parent checkpoint")
    calibration = json.loads((Path(__file__).with_name("config") /
                              "fall_wrestle_calibration.json").read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != PARENT_SHA:
        raise ValueError("Wrong parent calibration")
    parent_threshold = float(calibration["threshold"])
    parent = InteractionHead().to(args.device)
    parent.load_state_dict(torch.load(args.weights / "interaction_fall_wrestle.pt",
                                      map_location="cpu", weights_only=True), strict=True)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    neck_head = copy.deepcopy(parent)
    for parameter in neck_head.parameters():
        parameter.requires_grad_(True)
    # Start the supplement conservatively: parent decisions remain the baseline.
    with torch.no_grad():
        neck_head.layers[-1].bias.fill_(-4.0)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "protocol.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Run protocol changed; choose a new run tag")
    write_json(marker, protocol)
    train_rows = [row for row in splits["train"] if row["people"] and
                  (row["action"] in {"choke", "hug", "ordinary", "sports_wrestle"}
                   or row["dataset"] == "scfd" and row["label"] == 0)]
    counts = Counter(row["action"] == "choke" for row in train_rows)
    if set(counts) != {False, True}:
        raise ValueError("Training needs neck positives and negatives")
    sample_weights = torch.tensor([1 / counts[row["action"] == "choke"]
                                   for row in train_rows], dtype=torch.double)
    sample_weights /= sample_weights.sum()
    optimizer = torch.optim.AdamW(neck_head.parameters(), lr=args.lr, weight_decay=.01)
    state_path = args.output / "train_state.pt"
    history, best_ap, best_state = [], -1.0, None
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        neck_head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_ap, best_state = state["history"], state["best_ap"], state["best"]
        torch.set_rng_state(state["rng"])
        print("RESUME after epoch", len(history), flush=True)
    started = time.monotonic()
    for epoch in range(len(history), args.epochs):
        if time.monotonic() - started > args.minutes * 60 - 120:
            break
        neck_head.train()
        order = torch.multinomial(sample_weights, max(128, len(train_rows)), replacement=True).tolist()
        losses = []
        for offset in range(0, len(order), 8):
            batch = [train_rows[i] for i in order[offset:offset + 8]]
            logits = torch.stack([neck_head(row["bag"].to(args.device)).max()
                                  for row in batch])
            labels = logits.new_tensor([row["action"] == "choke" for row in batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neck_head.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_rows = [row for row in splits["development"] if row["people"] and
                    row["action"] in {"choke", "hug"}]
        dev_scores = scores(neck_head, dev_rows, args.device)
        ap = float(average_precision_score([row["action"] == "choke" for row in dev_rows],
                                           dev_scores))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "dev_neck_ap": ap})
        if ap > best_ap + 1e-5:
            best_ap, best_state = ap, copy.deepcopy(neck_head.state_dict())
            temporary = args.output / "neck_supplement.partial.pt"
            torch.save({key: value.detach().cpu() for key, value in best_state.items()}, temporary)
            temporary.replace(args.output / "neck_supplement.pt")
        temporary = args.output / "train_state.partial.pt"
        torch.save({"head": neck_head.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best_ap": best_ap, "best": best_state,
                    "rng": torch.get_rng_state()}, temporary)
        temporary.replace(state_path)
        write_json(args.output / "history.json", history)
        print("TRAIN", json.dumps(history[-1]), flush=True)
    if best_state is None:
        raise TimeoutError("No checkpoint produced; rerun this cell to resume")
    neck_head.load_state_dict(best_state)
    development = splits["development"]
    old_dev = scores(parent, development, args.device)
    neck_dev = scores(neck_head, development, args.device)
    neck_threshold, dev_gain = choose_threshold(
        development, old_dev, neck_dev, parent_threshold)
    report = {"status": "completed", "parent_checkpoint_sha256": PARENT_SHA,
              "neck_checkpoint_sha256": digest(args.output / "neck_supplement.pt"),
              "parent_threshold": parent_threshold, "neck_threshold": neck_threshold,
              "development_choke_recall_gain": dev_gain,
              "best_epoch": max(history, key=lambda row: row["dev_neck_ap"])["epoch"],
              "approved_for_demo": False, "trained_on_local_machine": False}
    for part in ("development", "test"):
        items = splits[part]
        old = scores(parent, items, args.device)
        new = scores(neck_head, items, args.device)
        baseline = [value >= parent_threshold for value in old]
        enhanced = combined(old, new, parent_threshold, neck_threshold)
        old_rates, new_rates = rates(items, baseline), rates(items, enhanced)
        if new_rates["fight"]["alert_fraction"] + 1e-9 < old_rates["fight"]["alert_fraction"]:
            raise AssertionError("Fight retention failed")
        report[part] = {"parent": old_rates, "supplemented": new_rates,
                        "fight_recall_nonregression": True,
                        "neck": class_metrics([row["action"] == "choke" for row in items
                                               if row["action"] in {"choke", "hug"}],
                                              [score for row, score in zip(items, new)
                                               if row["action"] in {"choke", "hug"}],
                                              neck_threshold)}
    write_json(args.output / "report.json", report)
    write_json(args.output / "neck_supplement.json", {
        "schema_version": 1, "neck_checkpoint_sha256": report["neck_checkpoint_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA, "parent_threshold": parent_threshold,
        "neck_threshold": neck_threshold, "head_version": 1,
        "decision": "parent_alert OR neck_supplement_alert",
        "source_sha256": SOURCE_SHA, "spec": SPEC, "approved_for_demo": False})
    package = args.output / "mmaction_neck_supplement_v5_pilot.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("neck_supplement.pt", "neck_supplement.json", "report.json", "protocol.json"):
            archive.write(args.output / name, name)
    print("FINISHED", json.dumps({"package": str(package), "test": report["test"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "fall-root", "scfd-root", "airtlab-root", "weights", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "train"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--lr", type=float, default=.0001)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 5 <= args.minutes <= 600:
        parser.error("Bounded training budget required")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["publisher_tree_sha"] != "5abac7693229900cf80f722e878fbb119211fc1c":
        raise ValueError("Wrong GMDCSA24 manifest")
    rows = sources(manifest, args.fall_root, args.scfd_root)
    rows.extend(reviewed_sources(args.review_manifest, args.review_root))
    rows.extend(airtlab_sources(args.airtlab_root))
    protocol = {"manifest_sha256": digest(args.manifest),
                "review_manifest_sha256": digest(args.review_manifest),
                "engine_sha256": digest(Path(__file__).with_name("action_engine.py")),
                "trainer_sha256": digest(Path(__file__)),
                "parent_checkpoint_sha256": PARENT_SHA, "source_sha256": SOURCE_SHA,
                "spec": SPEC, "positive_class": "neck_assault", "seed": 42}
    if args.stage == "features":
        features(args, rows, protocol)
    else:
        train(args, protocol)


if __name__ == "__main__":
    main()
