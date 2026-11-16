"""S3: train / evaluate / export the sharp-object detector (Ultralytics YOLO).

  train     fine-tune a COCO-pretrained YOLO on the merged dataset (build_dataset.py). Weights are
            written to --output/train/weights every epoch, so a run on Drive survives a disconnect;
            calling train again resumes from last.pt.
  evaluate  box mAP (per class and class-agnostic) on val/test and on the Vietnamese subsets, then the
            image-level alert used at runtime: an image alerts when any box >= threshold. The threshold
            is the lowest one whose false-alarm rate on val negatives is <= --max-false-alarm; recall and
            false alarms are then reported on test with that threshold.
  export    sharp_object.pt (fp16 weights) + sharp_object_spec.json + evaluation.json in one zip.
"""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASS_MAP = json.loads((ROOT / "datasets" / "sharp_object" / "class_map.json").read_text(encoding="utf-8"))
CLASSES = CLASS_MAP["classes"]


def weights_dir(output):
    return Path(output) / "train" / "weights"


# ------------------------------------------------------------------ train
def train(args):
    from ultralytics import YOLO

    last = weights_dir(args.output) / "last.pt"
    if last.is_file() and not (Path(args.output) / "train" / "TRAIN_DONE").is_file():
        print(f"resuming from {last}", flush=True)
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as error:  # Ultralytics refuses to resume a run that already finished
            print(f"resume not possible ({error}); keeping the finished run", flush=True)
    elif not last.is_file():
        model = YOLO(args.model)
        model.train(data=str(Path(args.data) / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    patience=args.patience, project=str(args.output), name="train", exist_ok=True,
                    workers=args.workers, seed=3407, deterministic=False, cos_lr=True, close_mosaic=args.close_mosaic,
                    optimizer="AdamW", lr0=args.lr0, cache=args.cache, plots=True, amp=True,
                    fraction=args.fraction, device=args.device)
    (Path(args.output) / "train" / "TRAIN_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


# ------------------------------------------------------------------ evaluate
def box_metrics(model, yaml_path, split, imgsz, single_cls, device):
    result = model.val(data=str(yaml_path), split=split, imgsz=imgsz, single_cls=single_cls, plots=False,
                       verbose=False, device=device, batch=16)
    out = {"map50": float(result.box.map50), "map50_95": float(result.box.map), "precision": float(result.box.mp),
           "recall": float(result.box.mr)}
    if not single_cls:
        per_class = {}
        for index, class_id in enumerate(result.box.ap_class_index):
            per_class[CLASSES[int(class_id)]] = {"ap50": float(result.box.ap50[index]), "ap50_95": float(result.box.ap[index])}
        out["per_class"] = per_class
    return out


def image_scores(model, list_file, imgsz, device):
    """[(path, has_box, max_conf, source)] for every image in a split list."""
    paths = [line.strip() for line in Path(list_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = list(dict.fromkeys(paths))
    rows = []
    for begin in range(0, len(paths), 32):
        chunk = paths[begin:begin + 32]
        for path, result in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=0.01, verbose=False, device=device,
                                                     stream=False)):
            parts = list(Path(path).parts)
            at = len(parts) - 1 - parts[::-1].index("images")
            label = Path(*parts[:at], "labels", *parts[at + 1:]).with_suffix(".txt")
            has_box = label.is_file() and label.read_text(encoding="utf-8").strip() != ""
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros(0)
            rows.append((path, has_box, float(confs.max()) if confs.size else 0.0))
    return rows


def alert_rates(rows, threshold):
    pos = [r for r in rows if r[1]]
    neg = [r for r in rows if not r[1]]
    return {"positives": len(pos), "negatives": len(neg),
            "recall": (sum(r[2] >= threshold for r in pos) / len(pos)) if pos else None,
            "false_alarm": (sum(r[2] >= threshold for r in neg) / len(neg)) if neg else None}


def choose_threshold(rows, max_false_alarm, floor):
    negatives = sorted((r[2] for r in rows if not r[1]), reverse=True)
    if not negatives:
        return max(floor, 0.25), "no negatives in val: default 0.25"
    allowed = int(np.floor(max_false_alarm * len(negatives)))
    # threshold just above the (allowed+1)-th highest negative score
    threshold = negatives[allowed] + 1e-4 if allowed < len(negatives) else floor
    return max(floor, min(threshold, 0.95)), f"false alarm <= {max_false_alarm} on {len(negatives)} val negatives"


def evaluate(args):
    from ultralytics import YOLO

    best = weights_dir(args.output) / "best.pt"
    model = YOLO(str(best))
    data = Path(args.data)
    report = {"weights": str(best), "imgsz": args.imgsz}
    for name, yaml_name in (("all", "data.yaml"), ("vn", "data_vn.yaml")):
        for split in ("val", "test"):
            report[f"box_{name}_{split}"] = box_metrics(model, data / yaml_name, split, args.imgsz, False, args.device)
            report[f"box_{name}_{split}_agnostic"] = box_metrics(model, data / yaml_name, split, args.imgsz, True,
                                                                 args.device)
    val_rows = image_scores(model, data / "val.txt", args.imgsz, args.device)
    test_rows = image_scores(model, data / "test.txt", args.imgsz, args.device)
    threshold, rule = choose_threshold(val_rows, args.max_false_alarm, args.min_threshold)
    report["alert_threshold"] = round(float(threshold), 4)
    report["threshold_rule"] = rule
    vn_val = set(Path(data / "vn_val.txt").read_text(encoding="utf-8").split())
    vn_test = set(Path(data / "vn_test.txt").read_text(encoding="utf-8").split())
    report["alert_val"] = alert_rates(val_rows, threshold)
    report["alert_test"] = alert_rates(test_rows, threshold)
    report["alert_vn_val"] = alert_rates([r for r in val_rows if r[0] in vn_val], threshold)
    report["alert_vn_test"] = alert_rates([r for r in test_rows if r[0] in vn_test], threshold)
    curve = []
    for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        curve.append({"threshold": t, "test": alert_rates(test_rows, t),
                      "vn_test": alert_rates([r for r in test_rows if r[0] in vn_test], t)})
    report["alert_curve"] = curve
    (Path(args.output) / "evaluation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "alert_curve"}, indent=1))


# ------------------------------------------------------------------ export
def export(args):
    import torch

    output = Path(args.output)
    best = weights_dir(output) / "best.pt"
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    package = output / "package"
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir(parents=True)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    for key in ("optimizer", "ema", "updates", "train_results"):
        checkpoint.pop(key, None)
    checkpoint["model"] = checkpoint["model"].half()
    torch.save(checkpoint, package / "sharp_object.pt")
    spec = {"name": "sharp_object", "run_tag": args.run_tag, "framework": "ultralytics", "weights": "sharp_object.pt",
            "classes": CLASSES, "alert": CLASS_MAP["alert"], "imgsz": args.imgsz,
            "alert_threshold": evaluation["alert_threshold"], "threshold_rule": evaluation["threshold_rule"],
            "alert_logic": "alert when a sharp-object box (any class) >= alert_threshold overlaps or is near a person box "
                           "for K of N consecutive frames (runtime side)",
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (package / "sharp_object_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    shutil.copy(output / "evaluation.json", package / "evaluation.json")
    for extra in ("summary.json",):
        if (Path(args.data) / extra).is_file():
            shutil.copy(Path(args.data) / extra, package / "dataset_summary.json")
    archive = output / f"sharp_object_{args.run_tag}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.iterdir()):
            bundle.write(path, path.name)
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["train", "evaluate", "export"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="output of build_dataset.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default=False, help="False / ram / disk")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of train images (smoke tests)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-false-alarm", type=float, default=0.03, help="image-level, on val negatives")
    parser.add_argument("--min-threshold", type=float, default=0.15)
    parser.add_argument("--run-tag", default="sharp_01")
    args = parser.parse_args()
    if isinstance(args.cache, str) and args.cache.lower() in {"false", "0", "no", ""}:
        args.cache = False
    {"train": train, "evaluate": evaluate, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
