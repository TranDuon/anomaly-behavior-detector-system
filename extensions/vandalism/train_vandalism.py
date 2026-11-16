"""V3: extract frozen V-JEPA 2 features and train the vandalism probe.

    python -m extensions.vandalism.train_vandalism --stage features --data /content/vandalism_data --cache /content/vandalism_cache
    python -m extensions.vandalism.train_vandalism --stage train    --data /content/vandalism_data --cache /content/vandalism_cache --output <drive dir>

Windows are what the runtime sees: 8 samples at 3.75 fps, stride 4 samples.
Clip/video labels are weak, so training is multiple-instance: each item is a
bag of windows and the bag score is the mean of its top-k window logits.
Hard-negative bags therefore push down their most vandalism-like windows.
"""

import argparse
import json
import math
import platform
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from extensions.vandalism.model import (FPS, FRAMES, GRID, HF_MODEL, HF_REVISION, SIZE, AttentiveProbe,
                                        Backbone, bag_logits, preprocess)

STRIDE = 4
NAME = "vandalism"


# ------------------------------------------------------------------ windows
def sample_frames(path, start=None, end=None, max_side=480):
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if 0 < fps <= 240 else 25.0
    if start:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 0.5) * 1000)
    frames, times, next_time = [], [], start or 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if start is not None and t + 1e-6 < start:
            continue
        if end is not None and t > end:
            break
        if t + 1e-6 >= next_time:
            height, width = frame.shape[:2]
            scale = max_side / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            times.append(t)
            while next_time <= t + 1e-6:
                next_time += 1 / FPS
    capture.release()
    return frames, times, fps


def item_windows(item, max_windows):
    segment = item.get("segment")
    start = end = None
    if segment:
        start, end = segment
        span = FRAMES / FPS
        if end - start < span:  # short Charades segment: centre a full window on it
            middle = (start + end) / 2
            start, end = max(0.0, middle - span / 2 - 0.1), middle + span / 2 + 0.1
    frames, times, fps = sample_frames(item["path"], start, end)
    if not frames:
        return [], [], fps
    while len(frames) < FRAMES:
        frames.append(frames[-1])
        times.append(times[-1])
    starts = list(range(0, len(frames) - FRAMES + 1, STRIDE))
    if len(starts) > max_windows:  # long CCTV videos: evenly spaced subset
        starts = [starts[round(i)] for i in np.linspace(0, len(starts) - 1, max_windows)]
    clips = [frames[s:s + FRAMES] for s in starts]
    centres = [times[s + FRAMES // 2] for s in starts]
    return clips, centres, fps


def cache_key(item):
    stem = Path(item["path"]).stem
    segment = item.get("segment")
    return f"{item['source']}__{stem}" + (f"__{segment[0]:.1f}_{segment[1]:.1f}" if segment else "")


def stage_features(args, backbone=None, items=None):
    """Pooled tokens of every window -> <cache>/<key>.npz. ``backbone``/``items`` let the
    fine-tune script re-encode dev/test with its fine-tuned encoder tail."""
    items = load_manifest(args.data) if items is None else items
    args.cache.mkdir(parents=True, exist_ok=True)
    backbone = backbone or Backbone(device="cuda" if torch.cuda.is_available() else "cpu")
    started, done = time.perf_counter(), 0
    for number, item in enumerate(items, 1):
        target = args.cache / f"{cache_key(item)}.npz"
        if target.is_file():
            continue
        max_windows = args.max_windows_long if item["source"] == "ashen" else args.max_windows
        clips, centres, fps = item_windows(item, max_windows)
        if not clips:
            print(f"skip (no frames): {item['path']}", flush=True)
            continue
        tokens = []
        for begin in range(0, len(clips), args.batch):
            batch = torch.stack([preprocess(clip) for clip in clips[begin:begin + args.batch]])
            tokens.append(backbone.encode(batch))
        tokens = torch.cat(tokens).numpy()
        labels = np.full(len(centres), item["binary"], np.int8)
        if "intervals_frames" in item:
            frames_at = [round(t * fps) for t in centres]
            labels = np.array([int(any(a <= f <= b for a, b in item["intervals_frames"])) for f in frames_at], np.int8)
        partial = target.with_suffix(".partial.npz")
        np.savez(partial, tokens=tokens.astype(np.float16), times=np.array(centres, np.float32), labels=labels)
        partial.replace(target)
        done += 1
        if done % 50 == 0:
            elapsed = time.perf_counter() - started
            print(f"{number}/{len(items)} items, {done} new, {elapsed / 60:.1f} min", flush=True)
    print(f"features done: {len(items)} items in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ training
def load_manifest(root):
    return [json.loads(line) for line in (Path(root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]


def load_bags(items, cache):
    bags = []
    for item in items:
        path = cache / f"{cache_key(item)}.npz"
        if not path.is_file():
            continue
        with np.load(path) as stored:
            bags.append({"item": item, "tokens": torch.from_numpy(stored["tokens"]),
                         "labels": stored["labels"], "times": stored["times"]})
    return bags


def collate(bags, max_windows):
    width = max(min(len(b["tokens"]), max_windows) for b in bags)
    tokens = torch.zeros(len(bags), width, *bags[0]["tokens"].shape[1:], dtype=torch.float16)
    mask = torch.zeros(len(bags), width, dtype=torch.bool)
    for index, bag in enumerate(bags):
        windows = bag["tokens"]
        if len(windows) > max_windows:  # random subset each epoch for long videos
            windows = windows[torch.randperm(len(windows))[:max_windows]]
        tokens[index, :len(windows)] = windows
        mask[index, :len(windows)] = True
    labels = torch.tensor([b["item"]["binary"] for b in bags], dtype=torch.float32)
    weights = torch.tensor([b["item"].get("weight", 1.0) for b in bags], dtype=torch.float32)
    return tokens, mask, labels, weights


@torch.no_grad()
def window_scores(probe, bags, device, chunk=512):
    probe.eval()
    result = []
    for bag in bags:
        logits = torch.cat([probe(part.to(device)).float().cpu() for part in bag["tokens"].split(chunk)])
        result.append(torch.sigmoid(logits).numpy())
    return result


def average_precision(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(-scores)
    hits = np.cumsum(labels[order])
    return float((hits / np.arange(1, len(labels) + 1) * labels[order]).sum() / labels.sum())


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(labels, scores, min_precision):
    """Highest-recall threshold reaching min_precision; else best F1."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    best, fallback = None, (0.0, 0.5)
    for threshold in np.unique(np.round(scores, 4)):
        predicted = scores >= threshold
        tp = (predicted & (labels == 1)).sum()
        fp = (predicted & (labels == 0)).sum()
        fn = (~predicted & (labels == 1)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > fallback[0]:
            fallback = (f1, float(threshold))
        if precision >= min_precision and (best is None or recall > best[0]):
            best = (recall, float(threshold))
    return (best[1], "min_precision") if best else (fallback[1], "best_f1")


def point(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn, "n": len(labels)}


def evaluate(probe, bags, device, threshold=None, k=3):
    scores = window_scores(probe, bags, device)
    report = {}
    groups = {}
    for bag, values in zip(bags, scores):
        top = float(np.sort(values)[-min(k, len(values)):].mean())
        groups.setdefault(bag["item"]["source"], []).append((bag, values, top))
    all_labels = [b["item"]["binary"] for src in groups.values() for b, _, _ in src]
    all_scores = [top for src in groups.values() for _, _, top in src]
    report["all_bags"] = {"n": len(all_labels), "ap": average_precision(all_labels, all_scores),
                          "roc_auc": roc_auc(all_labels, all_scores)}
    for source, rows in groups.items():
        labels = [b["item"]["binary"] for b, _, _ in rows]
        tops = [top for _, _, top in rows]
        entry = {"n": len(rows), "positives": int(sum(labels)), "ap": average_precision(labels, tops),
                 "roc_auc": roc_auc(labels, tops)}
        if threshold is not None:
            entry["at_threshold"] = point(labels, tops, threshold)
        if source == "ashen":  # frame-annotated test videos: window-level too
            window_labels = np.concatenate([b["labels"] for b, _, _ in rows])
            window_scores_ = np.concatenate([v for _, v, _ in rows])
            entry["window_ap"] = average_precision(window_labels, window_scores_)
            entry["window_roc_auc"] = roc_auc(window_labels, window_scores_)
        if source == "charades" and threshold is not None:
            entry["false_alarm_rate"] = float(np.mean(np.asarray(tops) >= threshold))
        report[source] = entry
    if threshold is not None:
        per_class = {}
        for rows in groups.values():
            for bag, _, top in rows:
                if bag["item"]["binary"] == 0:
                    per_class.setdefault(bag["item"]["label"], []).append(top >= threshold)
        report["false_alarm_by_negative_class"] = {k_: round(float(np.mean(v)), 3) for k_, v in sorted(per_class.items())}
    return report, all_labels, all_scores


def cap_focus_false_alarms(bags, scores, threshold, rule, cap):
    """Raise the threshold until every focus negative class (swing-and-hit look-alikes such as
    chopping wood) alarms on at most ``cap`` of its dev clips. ``scores`` come from evaluate(),
    which orders bags grouped by source. No-op when cap <= 0 or there are no focus bags."""
    if cap <= 0:
        return threshold, rule
    by_source = {}
    for bag in bags:
        by_source.setdefault(bag["item"]["source"], []).append(bag)
    ordered = [bag for group in by_source.values() for bag in group]
    assert len(ordered) == len(scores)
    per_class = {}
    for bag, score in zip(ordered, scores):
        if bag["item"].get("focus") and bag["item"]["binary"] == 0:
            per_class.setdefault(bag["item"]["label"], []).append(score)
    if not per_class:
        return threshold, rule
    needed = threshold
    for label, values in per_class.items():
        values = np.sort(np.asarray(values))[::-1]
        allowed = int(math.floor(cap * len(values)))  # clips of this class allowed above threshold
        if allowed < len(values):
            needed = max(needed, float(values[allowed]) + 1e-6)
    if needed > threshold:
        print(f"threshold {threshold:.4f} -> {needed:.4f} to keep focus false alarms <= {cap:.0%}", flush=True)
        return needed, f"{rule}; raised to cap focus-class false alarms at {cap}"
    return threshold, rule


def stage_train(args):
    seed_everything(args.seed)
    items = load_manifest(args.data)
    splits = {name: load_bags([i for i in items if i["split"] == name], args.cache) for name in ("train", "dev", "test")}
    for name, bags in splits.items():
        print(f"{name}: {len(bags)} bags ({sum(b['item']['binary'] for b in bags)} positive)", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = AttentiveProbe().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(splits["train"]) / args.batch_bags)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.1)
    positives = sum(b["item"]["binary"] for b in splits["train"])
    pos_weight = torch.tensor((len(splits["train"]) - positives) / max(1, positives), device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    best_ap, best_state, patience, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        order = list(range(len(splits["train"])))
        random.shuffle(order)
        losses = []
        for begin in range(0, len(order), args.batch_bags):
            batch = [splits["train"][i] for i in order[begin:begin + args.batch_bags]]
            tokens, mask, labels, weights = collate(batch, args.max_windows_train)
            tokens, mask, labels, weights = tokens.to(device), mask.to(device), labels.to(device), weights.to(device)
            flat = probe(tokens[mask].float())
            window_logits = torch.full(mask.shape, -1e4, device=device)
            window_logits[mask] = flat
            logits = bag_logits(window_logits, mask, k=args.top_k)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        dev_report, _, _ = evaluate(probe, splits["dev"], device, k=args.top_k)
        dev_ap = dev_report["all_bags"]["ap"] or 0.0
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_ap": dev_ap,
                        "dev_auc": dev_report["all_bags"]["roc_auc"]})
        print(f"epoch {epoch:02d} loss {np.mean(losses):.4f} dev AP {dev_ap:.4f} AUC {dev_report['all_bags']['roc_auc']}", flush=True)
        torch.save({"epoch": epoch, "probe": probe.state_dict(), "optimizer": optimizer.state_dict()},
                   args.output / "resume.pt")
        if dev_ap > best_ap + 1e-4:
            best_ap, best_state, patience = dev_ap, {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    probe.load_state_dict(best_state)
    _, dev_labels, dev_scores = evaluate(probe, splits["dev"], device, k=args.top_k)
    threshold, rule = best_threshold(dev_labels, dev_scores, args.min_precision)
    threshold, rule = cap_focus_false_alarms(splits["dev"], dev_scores, threshold, rule, args.max_focus_false_alarm)
    dev_report, _, _ = evaluate(probe, splits["dev"], device, threshold, k=args.top_k)
    test_report, _, _ = evaluate(probe, splits["test"], device, threshold, k=args.top_k)
    export(args, probe, threshold, rule, history, dev_report, test_report, items)


def export(args, probe, threshold, rule, history, dev_report, test_report, items, extra_spec=None, extra_files=()):
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:  # local smoke test without the extension environment
        transformers_version = None

    out = args.output
    state = {k: v.detach().cpu() for k, v in probe.state_dict().items()}
    torch.save(state, out / f"{NAME}.pt")
    spec = {"name": NAME, "display_label": "VANDALISM", "kind": "extension", "backbone": HF_MODEL,
            "backbone_revision": HF_REVISION, "frames": FRAMES, "fps": FPS, "size": SIZE, "resize": "full_frame_no_crop",
            "pooled_grid": list(GRID), "probe": "AttentiveProbe(width=1024, heads=16, queries=4, hidden=512)",
            "bag_pooling": f"top{args.top_k}_mean", "threshold": round(float(threshold), 4), "threshold_rule": rule,
            "min_precision": args.min_precision, "confirm_seconds": 0.8, "release_seconds": 1.0,
            "approved_for_demo": False, "run_tag": args.run_tag, **(extra_spec or {})}
    (out / f"{NAME}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    report = {"spec": spec, "history": history, "dev": dev_report, "test": test_report}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    manifest = Path(args.data) / "manifest.jsonl"
    summary_path = Path(args.data) / "prepare_summary.json"
    protocol = {"run_tag": args.run_tag, "seed": args.seed, "args": {k: str(v) for k, v in vars(args).items()},
                "manifest_sha256": _sha(manifest), "items": len(items),
                "prepare_summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
                "torch": torch.__version__, "transformers": transformers_version, "python": platform.python_version(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "sources": "datasets/vandalism/sources.json (Kinetics-700-2020, ASHEN, Charades)"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    archive = out / f"{NAME}_{args.run_tag}.zip"
    with zipfile.ZipFile(archive.with_suffix(".partial.zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in (f"{NAME}.pt", f"{NAME}.json", "report.json", "protocol.json", *extra_files):
            bundle.write(out / name, name)
    archive.with_suffix(".partial.zip").replace(archive)
    print(json.dumps({"threshold": spec["threshold"], "rule": rule, "dev": dev_report["all_bags"],
                      "test": {k: v for k, v in test_report.items() if k != "false_alarm_by_negative_class"}}, indent=1))
    print("saved:", archive)


def _sha(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["features", "train"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-tag", default=time.strftime("%Y%m%d_%H%M"))
    parser.add_argument("--batch", type=int, default=16, help="windows per backbone batch")
    parser.add_argument("--max-windows", type=int, default=12, help="per short clip/segment")
    parser.add_argument("--max-windows-long", type=int, default=120, help="per long ASHEN video")
    parser.add_argument("--max-windows-train", type=int, default=48, help="windows sampled per bag per step")
    parser.add_argument("--batch-bags", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--max-focus-false-alarm", type=float, default=0.0,
                        help="per focus-class dev false-alarm cap used to raise the threshold (0 = off)")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.stage == "features":
        stage_features(args)
    else:
        if args.output is None:
            parser.error("--output is required for --stage train")
        stage_train(args)


if __name__ == "__main__":
    main()
