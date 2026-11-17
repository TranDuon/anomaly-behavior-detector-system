"""S3: train / evaluate / export the sharp-object detector (Ultralytics YOLO).

  train     stage 1: fine-tune a COCO-pretrained YOLO on the merged dataset (build_dataset.py). Weights
            are written to --output/train/weights every epoch; calling train again resumes from last.pt.
            --single-cls trains one "sharp object" class (fine labels stay in the dataset for review).
  finetune_vn
            stage 2: start from the stage-1 weights (--init), freeze the first --freeze layers and train
            with a low lr on the Vietnamese train images (repeated --vn-repeat times) mixed with a
            --replay share of the public train images, so the model adapts to Vietnamese weapons
            (dao phong lon...) without forgetting the public data. Without Vietnamese train images the
            stage-1 weights are copied as the result.
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
def run_training(args, data_yaml, start, **extra):
    """Train into --output/train, or resume it from last.pt after a disconnect."""
    from ultralytics import YOLO

    last = weights_dir(args.output) / "last.pt"
    done = Path(args.output) / "train" / "TRAIN_DONE"
    if last.is_file() and not done.is_file():
        print(f"resuming from {last}", flush=True)
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as error:  # Ultralytics refuses to resume a run that already finished
            print(f"resume not possible ({error}); keeping the finished run", flush=True)
    elif not last.is_file():
        model = YOLO(str(start))
        model.train(data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    patience=args.patience, project=str(args.output), name="train", exist_ok=True,
                    workers=args.workers, seed=3407, deterministic=False, cos_lr=True, close_mosaic=args.close_mosaic,
                    optimizer="AdamW", lr0=args.lr0, cache=args.cache, plots=True, amp=True,
                    fraction=args.fraction, device=args.device, single_cls=args.single_cls, **extra)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def train(args):
    run_training(args, Path(args.data) / "data.yaml", args.model)


def stage2_lists(data, output, vn_repeat, replay, seed=3407):
    """Stage-2 train list: Vietnamese train images x vn_repeat + a replay share of public train images."""
    import csv
    import random

    with (Path(data) / "manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    vn = sorted(str((Path(data) / r["image"]).resolve()) for r in rows if r["split"] == "train" and r["source"] == "vn")
    public = sorted(str((Path(data) / r["image"]).resolve()) for r in rows
                    if r["split"] == "train" and r["source"] != "vn")
    picked = random.Random(seed).sample(public, int(round(replay * len(public))))
    lines = vn * vn_repeat + picked
    random.Random(seed).shuffle(lines)
    output.mkdir(parents=True, exist_ok=True)
    train_list = output / "stage2_train.txt"
    train_list.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    yaml = output / "data_stage2.yaml"
    yaml.write_text(f"path: {Path(data).resolve().as_posix()}\ntrain: {train_list.resolve().as_posix()}\n"
                    f"val: val.txt\ntest: test.txt\nnames:\n{names}", encoding="utf-8")
    summary = {"vn_train_images": len(vn), "vn_repeat": vn_repeat, "public_replay": len(picked),
               "public_train": len(public), "train_list": len(lines)}
    (output / "stage2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("stage 2 data:", summary, flush=True)
    return yaml, len(vn)


def finetune_vn(args):
    if args.init is None or not Path(args.init).is_file():
        raise SystemExit(f"--init must point to the stage-1 weights (got {args.init})")
    yaml, vn_count = stage2_lists(args.data, Path(args.output), args.vn_repeat, args.replay)
    if vn_count == 0:
        print("no Vietnamese train images: stage-1 weights kept as the result", flush=True)
        target = weights_dir(args.output)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "last.pt"):
            shutil.copy(args.init, target / name)
        (Path(args.output) / "train" / "TRAIN_DONE").write_text("stage 1 copied", encoding="utf-8")
        return
    run_training(args, yaml, args.init, freeze=args.freeze, warmup_epochs=1)


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
    single = len(model.names) == 1  # trained with --single-cls: only class-agnostic metrics exist
    report = {"weights": str(best), "imgsz": args.imgsz, "model_classes": [str(n) for n in model.names.values()]}
    for name, yaml_name in (("all", "data.yaml"), ("vn", "data_vn.yaml")):
        for split in ("val", "test"):
            agnostic = box_metrics(model, data / yaml_name, split, args.imgsz, True, args.device)
            report[f"box_{name}_{split}_agnostic"] = agnostic
            report[f"box_{name}_{split}"] = agnostic if single else box_metrics(model, data / yaml_name, split,
                                                                                args.imgsz, False, args.device)
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
    model_classes = [str(n) for n in dict(checkpoint["model"].names).values()]
    torch.save(checkpoint, package / "sharp_object.pt")
    spec = {"name": "sharp_object", "run_tag": args.run_tag, "framework": "ultralytics", "weights": "sharp_object.pt",
            "classes": model_classes if len(model_classes) == 1 else CLASSES, "label_classes": CLASSES,
            "alert": CLASS_MAP["alert"], "imgsz": args.imgsz, "confirm_seconds": 0.5, "release_seconds": 1.0,
            "alert_threshold": evaluation["alert_threshold"], "threshold_rule": evaluation["threshold_rule"],
            "alert_logic": "alert when a sharp-object box (any class) >= alert_threshold lies inside a person box "
                           "widened for arms/long handles, confirmed over time (extensions/sharp_object/extension.py)",
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
    parser.add_argument("--stage", choices=["train", "finetune_vn", "evaluate", "export"], required=True)
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
    parser.add_argument("--single-cls", action="store_true", help="train one merged 'sharp object' class")
    parser.add_argument("--init", type=Path, default=None, help="finetune_vn: stage-1 best.pt")
    parser.add_argument("--freeze", type=int, default=10, help="finetune_vn: frozen first layers (backbone)")
    parser.add_argument("--vn-repeat", type=int, default=10, help="finetune_vn: Vietnamese image repeats")
    parser.add_argument("--replay", type=float, default=0.25, help="finetune_vn: share of public train images")
    args = parser.parse_args()
    if isinstance(args.cache, str) and args.cache.lower() in {"false", "0", "no", ""}:
        args.cache = False
    {"train": train, "finetune_vn": finetune_vn, "evaluate": evaluate, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
