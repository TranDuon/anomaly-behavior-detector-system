"""V1: zero-shot vandalism baseline from the frozen core AVA model.

Runs in the core environment (mmaction2_demo/env) and only *uses* the core
ActionEngine (detector + SlowOnly R101 AVA). Nothing in the core is modified.
The core reply keeps only the top-4 AVA labels >= 0.4, so this script reads the
full sigmoid scores itself for the object-interaction labels:
34 hit (an object), 35 kick (an object), 46 push (an object), 58 throw,
plus 64 fight/hit (a person) to see how vandalism overlaps with strike.

Windows match the runtime: 9 samples at 3.75 fps (2.4 s), stride 4 samples.

    run_ava_baseline.cmd     (sets PYTHONPATH and uses mmaction2_demo/env)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "mmaction2_demo"
sys.path.insert(0, str(CORE))

LABELS = {"hit_object": 34, "kick_object": 35, "push_object": 46, "throw": 58, "hit_person": 64}
FPS, WINDOW, STRIDE = 3.75, 9, 4
EVAL_DIR = ROOT / "datasets" / "vandalism" / "raw" / "eval_v1"
REPORT_DIR = ROOT / "artifacts" / "vandalism" / "v1_ava_baseline"


def sample_frames(path, start=None, end=None, max_side=640):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if 0 < fps <= 240 else 25.0
    frames, times, index, next_time = [], [], 0, start or 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = index / fps
        index += 1
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


@torch.inference_mode()
def window_scores(engine, frames):
    from action_engine import rgb_inputs

    inputs, shape = rgb_inputs(frames, engine.device)
    boxes = engine.detect_people(frames[4])
    if not len(boxes):
        return {name: 0.0 for name in LABELS}, 0
    scaled = torch.as_tensor(boxes, device=engine.device).clone()
    scaled[:, [0, 2]] *= shape[1] / frames[0].shape[1]
    scaled[:, [1, 3]] *= shape[0] / frames[0].shape[0]
    rois = torch.cat((torch.zeros(len(boxes), 1, device=engine.device), scaled), dim=1)
    result = engine.model.roi_head._bbox_forward(engine.model.extract_feat(inputs), rois, [dict(img_shape=shape)])
    scores = result["cls_score"].sigmoid().cpu().numpy()
    return {name: float(scores[:, index].max()) for name, index in LABELS.items()}, len(boxes)


def score_item(engine, item):
    segment = item.get("segment")
    path = Path(item["path"])
    path = path if path.is_absolute() else ROOT / path
    frames, times, fps = sample_frames(path, *(segment or (None, None)))
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    if len(frames) < WINDOW:
        frames = frames + [frames[-1]] * (WINDOW - len(frames)) if frames else []
        times = times + [times[-1]] * (WINDOW - len(times)) if times else []
    windows = []
    for begin in range(0, max(1, len(frames) - WINDOW + 1), STRIDE):
        chunk = frames[begin:begin + WINDOW]
        if len(chunk) < WINDOW:
            break
        values, people = window_scores(engine, chunk)
        center_frame = round(times[begin + 4] * fps)
        label = item["binary"]
        if "intervals_frames" in item:
            label = int(any(a <= center_frame <= b for a, b in item["intervals_frames"]))
        windows.append({"t": round(times[begin + 4], 3), "people": people, "label": label, **values})
    return windows


def average_precision(labels, scores):
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(labels)[order]
    if labels.sum() == 0:
        return None
    hits = np.cumsum(labels)
    precision = hits / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def operating_points(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    result = {}
    for threshold in (0.1, 0.2, 0.3, 0.4):
        predicted = scores >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        fn = int((~predicted & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        result[str(threshold)] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    return result


def combined(row, names):
    return max(row[name] for name in names)


SCORES = {
    "hit_object": ("hit_object",), "kick_object": ("kick_object",), "push_object": ("push_object",),
    "throw": ("throw",), "hit_kick_throw": ("hit_object", "kick_object", "throw"),
    "any_object": ("hit_object", "kick_object", "push_object", "throw"), "hit_person": ("hit_person",),
}


def summarize(items, results):
    report = {}
    groups = {"kinetics_clip": [], "charades_clip": [], "ashen_video": []}
    for item, windows in zip(items, results):
        if not windows:
            continue
        clip = {name: max(combined(w, cols) for w in windows) for name, cols in SCORES.items()}
        key = {"kinetics700_2020": "kinetics_clip", "charades": "charades_clip", "ashen": "ashen_video"}[item["source"]]
        groups[key].append((item, clip))
    # Kinetics + Charades share one clip-level pool (Charades clips are all negatives).
    pool = groups["kinetics_clip"] + groups["charades_clip"]
    ashen_windows = [(w["label"], w) for item, windows in zip(items, results) if item["source"] == "ashen" for w in windows]
    for name in SCORES:
        entry = {}
        for title, rows in (("kinetics", groups["kinetics_clip"]), ("kinetics+charades", pool), ("ashen_video", groups["ashen_video"])):
            labels = [item["binary"] for item, _ in rows]
            values = [clip[name] for _, clip in rows]
            entry[title] = {"n": len(rows), "positives": int(sum(labels)), "ap": average_precision(labels, values),
                            "roc_auc": roc_auc(labels, values), "points": operating_points(labels, values)}
        labels = [label for label, _ in ashen_windows]
        values = [combined(w, SCORES[name]) for _, w in ashen_windows]
        entry["ashen_window"] = {"n": len(labels), "positives": int(sum(labels)), "ap": average_precision(labels, values),
                                 "roc_auc": roc_auc(labels, values)}
        report[name] = entry
    false_alarms = {}
    for item, clip in pool:
        if item["binary"] == 0:
            false_alarms.setdefault(item["label"], []).append(clip["hit_kick_throw"])
    report["negative_class_mean_hit_kick_throw"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(false_alarms.items())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EVAL_DIR / "manifest.jsonl")
    parser.add_argument("--output", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    from action_engine import ActionEngine

    torch.set_num_threads(4)
    engine = ActionEngine(device="cuda:0", person_recovery=False)
    engine.warmup()
    args.output.mkdir(parents=True, exist_ok=True)
    windows_path = EVAL_DIR / "ava_windows.jsonl"
    done = {}
    if windows_path.is_file():
        for line in windows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["path"] + str(row.get("segment"))] = row["windows"]
    started = time.perf_counter()
    results = []
    with windows_path.open("a", encoding="utf-8") as stream:
        for number, item in enumerate(items, 1):
            key = item["path"] + str(item.get("segment"))
            if key not in done:
                done[key] = score_item(engine, item)
                stream.write(json.dumps({"path": item["path"], "segment": item.get("segment"), "windows": done[key]}) + "\n")
                stream.flush()
            results.append(done[key])
            if number % 25 == 0:
                print(f"{number}/{len(items)} items  {time.perf_counter() - started:.0f}s", flush=True)
    report = {"scope": "Zero-shot AVA object-interaction scores from the frozen core model; clip score = max over 2.4 s windows.",
              "items": len(items), "seconds": round(time.perf_counter() - started, 1), "metrics": summarize(items, results)}
    (args.output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {t: (v[t]["ap"], v[t]["roc_auc"]) for t in v} for k, v in report["metrics"].items()
                      if isinstance(v, dict) and "kinetics" in v}, indent=1))


if __name__ == "__main__":
    main()
