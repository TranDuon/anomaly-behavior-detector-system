"""Conservative MMAction2 head fine-tune for neck assaults and fight release."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch

from action_engine import ActionEngine, InteractionHead, SOURCE_SHA, SPEC, digest
from train_action_transition_v2 import (PARENT_SHA, duration_decision,
                                        frames_for_video, reviewed_sources,
                                        sampled_clip, sources, write_json)


CHOKE_IDS = (26, 33, 37, 38, 80, 81, 83, 87, 89, 98, 106)
HUG_IDS = (1, 5, 6, 12, 25, 28, 29, 32, 33, 36, 37, 38, 44, 47, 49, 58)


def airtlab_sources(root):
    """Two views of one action stay in the same split."""
    rows = []
    for folder, ids, positive, action, development, test in (
        ("violent", CHOKE_IDS, 1, "choke", {38, 83}, {87, 106}),
        ("non-violent", HUG_IDS, 0, "hug", {12, 37}, {38, 49}),
    ):
        for number in ids:
            split = "development" if number in development else "test" if number in test else "train"
            for camera in ("cam1", "cam2"):
                path = root / folder / camera / f"{number}.mp4"
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append({"video": f"airtlab/{folder}/{camera}/{number}.mp4",
                             "path": str(path), "dataset": "airtlab",
                             "group": f"airtlab/{folder}/{number}", "split": split,
                             "label": positive, "action": action, "start": 0.0,
                             "end": None, "publisher_duration": None})
    return rows


def anchors(row, duration):
    start = float(row["start"])
    end = min(duration, float(row["end"]) if row["end"] is not None else duration)
    if not 0 <= start < end:
        raise ValueError("Invalid annotated interval: " + row["video"])
    if row["dataset"] == "reviewed":
        if row["end"] > duration + .1:
            raise ValueError("Reviewed interval exceeds video: " + row["video"])
        action = row["action"]
        step = .5 if action in {"fight", "wrestle", "after_fight"} else 2.0
        if action == "after_fight":
            # The nine-frame window spans roughly 2.13 s. Do not train a
            # nominally calm window while its central frames show the fight.
            first = min(end - .2, start + 1.05)
            return np.arange(first, max(first + .01, end - .2), step).tolist()
        if end - start < .7:
            return [(start + end) / 2]
        return np.arange(start + .2, max(start + .21, end - .2), step).tolist()
    if row["dataset"] in {"scfd", "airtlab"}:
        return [duration / 2]
    if row["label"] == 0:
        return np.linspace(max(.1, duration * .2), duration * .8, 3).tolist()
    return [start + (end - start) * fraction for fraction in (.15, .35, .6)]


def features(args, rows, protocol):
    args.cache.mkdir(parents=True, exist_ok=True)
    marker = args.cache / "protocol.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Feature protocol changed; choose a new run tag")
    write_json(marker, protocol)
    engine = ActionEngine(weights=args.weights, device=args.device)
    index, skipped = [], []
    for number, row in enumerate(rows, 1):
        frames, fps = frames_for_video(row["path"])
        duration = len(frames) / fps
        decision = duration_decision(row, duration)
        if decision == "skip":
            skipped.append({"video": row["video"], "reason": "unsafe_temporal_labels"})
            continue
        for anchor in anchors(row, duration):
            key = hashlib.sha256((row["video"] + f"/{anchor:.6f}/{row['label']}").encode()).hexdigest()[:24]
            target = args.cache / (key + ".npz")
            if not target.exists():
                bag, auxiliary, info = engine.extract_features(sampled_clip(frames, fps, anchor))
                temporary = args.cache / (key + ".partial.npz")
                np.savez_compressed(temporary, bag=bag.astype(np.float32),
                                    auxiliary=auxiliary.astype(np.float32),
                                    people=np.int32(info["people"]))
                temporary.replace(target)
            index.append({"key": key, "video": row["video"], "dataset": row["dataset"],
                          "group": row["group"], "split": row["split"],
                          "label": row["label"], "action": row.get("action"),
                          "anchor": anchor})
        if number % 10 == 0 or number == len(rows):
            print("FEATURES", number, "/", len(rows), flush=True)
    write_json(args.cache / "index.json", index)
    write_json(args.cache / "skipped_sources.json", skipped)


def load_rows(cache):
    rows = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    for row in rows:
        with np.load(cache / (row["key"] + ".npz"), allow_pickle=False) as item:
            row["bag"] = torch.from_numpy(item["bag"].copy())
            row["people"] = int(item["people"])
    return rows


@torch.inference_mode()
def scores(head, rows, device):
    head.eval()
    values = []
    for row in rows:
        if not len(row["bag"]):
            values.append(0.0)
            continue
        bag = row["bag"].to(device)
        logits = head(bag)
        values.append(float(logits.max().sigmoid().cpu()))
    return values


def metrics(rows, values, threshold):
    labels = [row["label"] for row in rows]
    if set(labels) != {0, 1}:
        return {"windows": len(rows), "ap": None, "precision": None, "recall": None, "f1": None}
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, np.asarray(values) >= threshold, average="binary", zero_division=0)
    return {"windows": len(rows), "ap": float(average_precision_score(labels, values)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def action_metrics(rows, values, threshold):
    result = {}
    groups = {
        "fall": lambda row: row["dataset"] == "gmdcsa24" and row["label"] == 1,
        "fight": lambda row: row["label"] == 1 and row["dataset"] in {"scfd", "reviewed"},
        "after_fight": lambda row: row["action"] == "after_fight",
        "choke": lambda row: row["action"] == "choke",
        "hug": lambda row: row["action"] == "hug",
    }
    for name, criterion in groups.items():
        picked = [(row, value) for row, value in zip(rows, values) if criterion(row)]
        result[name] = {"windows": len(picked),
                        "with_person": sum(bool(row["people"]) for row, _ in picked),
                        "alert_fraction": (sum(value >= threshold for _, value in picked) / len(picked)
                                           if picked else None)}
    return result


def guarded_threshold(rows, new_values, parent_values, parent_threshold):
    observed = [(row, new, old) for row, new, old in zip(rows, new_values, parent_values)
                if row["people"]]
    if {row["label"] for row, _, _ in observed} != {0, 1}:
        raise ValueError("Development split needs both classes with detected people")
    parent_actions = action_metrics([r for r, _, _ in observed],
                                    [v for _, _, v in observed], parent_threshold)
    minimum = {name: (parent_actions[name]["alert_fraction"] or 0) - .02
               for name in ("fall", "fight")}
    candidates = np.unique(np.r_[.01, .05, .5, .95, [v for _, v, _ in observed]])
    options = []
    selected_rows = [r for r, _, _ in observed]
    selected_values = [v for _, v, _ in observed]
    for threshold in candidates:
        actions = action_metrics(selected_rows, selected_values, float(threshold))
        if any(actions[name]["windows"] and
               (actions[name]["alert_fraction"] or 0) < minimum[name]
               for name in minimum):
            continue
        overall = metrics(selected_rows, selected_values, float(threshold))
        after = actions["after_fight"]["alert_fraction"] or 0
        hug = actions["hug"]["alert_fraction"] or 0
        choke = actions["choke"]["alert_fraction"] or 0
        options.append((overall["f1"] + .2 * choke - .3 * after - .2 * hug,
                        overall["f1"], float(threshold)))
    if not options:
        raise ValueError("No threshold preserves development fall and fight recall")
    return max(options)[2], {"parent_recall_floor": minimum,
                             "selection": "development F1 plus choke recall minus post-fight and hug false red"}


def train(args, protocol):
    rows = load_rows(args.cache)
    partitions = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "development", "test")}
    train_rows = [row for row in partitions["train"] if row["people"]]
    if any({row["label"] for row in part} != {0, 1}
           for part in (train_rows, partitions["development"], partitions["test"])):
        raise ValueError("Every split needs positive and negative windows")
    metadata = json.loads((args.weights / "interaction_fall_wrestle.json").read_text(encoding="utf-8"))
    if (digest(args.weights / "interaction_fall_wrestle.pt") != PARENT_SHA
            or metadata["checkpoint_sha256"] != PARENT_SHA
            or metadata["source_sha256"] != SOURCE_SHA or metadata["spec"] != SPEC
            or int(metadata.get("head_version", 1)) != 1):
        raise ValueError("Wrong fall/wrestle parent checkpoint")
    calibration = json.loads((Path(__file__).with_name("config") /
                              "fall_wrestle_calibration.json").read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != PARENT_SHA or not 0 < calibration["threshold"] < 1:
        raise ValueError("Wrong parent calibration")
    parent_threshold = calibration["threshold"]
    parent = InteractionHead().to(args.device)
    parent.load_state_dict(torch.load(args.weights / "interaction_fall_wrestle.pt",
                                      map_location="cpu", weights_only=True), strict=True)
    head = copy.deepcopy(parent)
    parent_dev = scores(parent, partitions["development"], args.device)
    parent_test = scores(parent, partitions["test"], args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Training protocol changed; choose a new run tag")
    write_json(path, protocol)
    counts = Counter((r["dataset"], r["action"], r["label"]) for r in train_rows)
    emphasis = {"fight": 3.0, "wrestle": 3.0, "choke": 4.0,
                "hug": 3.0, "after_fight": 1.5}
    weights = torch.tensor([
        emphasis.get(r["action"], 2.0 if r["dataset"] == "gmdcsa24" and r["label"] else 1.0)
        / counts[(r["dataset"], r["action"], r["label"])]
        for r in train_rows], dtype=torch.double)
    weights /= weights.sum()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    best, best_ap, history, stale = None, -1.0, [], 0
    resume = args.output / "train_state.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        best, best_ap, history, stale = (saved["best"], saved["best_ap"],
                                        saved["history"], saved["stale"])
        torch.set_rng_state(saved["torch_rng"])
        print("RESUME after epoch", len(history), flush=True)
    started = time.monotonic()
    for epoch in range(len(history), args.epochs):
        if time.monotonic() - started > args.minutes * 60 - 120:
            break
        head.train()
        order = torch.multinomial(weights, max(256, len(train_rows)), replacement=True).tolist()
        losses = []
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            batch = [train_rows[i] for i in order[offset:offset + 8]]
            logits = torch.stack([head(row["bag"].to(args.device)).max() for row in batch])
            labels = logits.new_tensor([row["label"] for row in batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            # Preserve the parent on broad fight/fall examples, including mild
            # actions which a tiny neck-specific dataset cannot represent.
            with torch.no_grad():
                teacher = torch.stack([parent(row["bag"].to(args.device)).max()
                                       for row in batch]).sigmoid()
            keep = logits.new_tensor([r["dataset"] in {"scfd", "gmdcsa24"}
                                      for r in batch], dtype=torch.bool)
            if keep.any():
                loss = loss + 2.0 * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[keep], teacher[keep])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev = scores(head, partitions["development"], args.device)
        ap = metrics(partitions["development"], dev, .5)["ap"]
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses), "dev_ap": ap})
        write_json(args.output / "history.json", history)
        if ap > best_ap + 1e-4:
            best, best_ap, stale = copy.deepcopy(head.state_dict()), ap, 0
            temporary = args.output / "interaction.partial.pt"
            torch.save({k: v.detach().cpu() for k, v in best.items()}, temporary)
            temporary.replace(args.output / "interaction.pt")
        else:
            stale += 1
        temporary = args.output / "train_state.partial.pt"
        torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                    "best": best, "best_ap": best_ap, "history": history,
                    "stale": stale, "torch_rng": torch.get_rng_state()}, temporary)
        temporary.replace(resume)
        print("TRAIN", json.dumps(history[-1]), flush=True)
        if stale >= 8:
            break
    if best is None:
        raise TimeoutError("No checkpoint produced")
    head.load_state_dict(best)
    new_dev = scores(head, partitions["development"], args.device)
    threshold, guard = guarded_threshold(partitions["development"], new_dev,
                                         parent_dev, parent_threshold)
    new_test = scores(head, partitions["test"], args.device)
    def comparison(part, old, new):
        return {
            "parent": {"overall": metrics(part, old, parent_threshold),
                       "actions": action_metrics(part, old, parent_threshold)},
            "new": {"overall": metrics(part, new, threshold),
                    "actions": action_metrics(part, new, threshold)},
        }
    result = {"status": "completed", "parent_checkpoint_sha256": PARENT_SHA,
              "head_version": 1,
              "training_sources": len({r["video"] for r in train_rows}),
              "skipped_sources": json.loads((args.cache / "skipped_sources.json").read_text()),
              "epochs": len(history), "best_epoch": max(history, key=lambda x: x["dev_ap"])["epoch"],
              "threshold": threshold, "threshold_guard": guard,
              "development": comparison(partitions["development"], parent_dev, new_dev),
              "test": comparison(partitions["test"], parent_test, new_test),
              "approved_for_demo": False, "trained_on_local_machine": False,
              "pat_shoulder_tested": False,
              "release_under_one_second_verified": False}
    write_json(args.output / "report.json", result)
    output_metadata = {**metadata, "checkpoint_sha256": digest(args.output / "interaction.pt"),
                       "parent_checkpoint_sha256": PARENT_SHA, "head_version": 1,
                       "threshold": threshold,
                       "approved_for_demo": False, "real_phone_tested": False,
                       "scope": "Original fight/fall head with choke and calm-transition pilot; runtime acceptance pending"}
    write_json(args.output / "interaction.json", output_metadata)
    with zipfile.ZipFile(args.output / "mmaction_release_neck_v4_pilot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("interaction.pt", "interaction.json", "report.json", "protocol.json"):
            archive.write(args.output / name, name)
    print("FINISHED", json.dumps({"threshold": threshold, "test": result["test"]}), flush=True)


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
    parser.add_argument("--lr", type=float, default=0.0001)
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
                "engine_sha256": digest(Path(__file__).with_name("action_engine.py")),
                "trainer_sha256": digest(Path(__file__)),
                "parent_checkpoint_sha256": PARENT_SHA,
                "review_manifest_sha256": digest(args.review_manifest),
                "source_sha256": SOURCE_SHA, "spec": SPEC,
                "airtlab_choke_ids": list(CHOKE_IDS), "airtlab_hug_ids": list(HUG_IDS),
                "positive_class": "fight_or_wrestle_or_fall_or_choke", "seed": 42}
    if args.stage == "features":
        features(args, rows, protocol)
    else:
        train(args, protocol)


if __name__ == "__main__":
    main()
