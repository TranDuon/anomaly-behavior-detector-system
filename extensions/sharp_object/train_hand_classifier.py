"""H2: train / evaluate / export the hand-crop classifier (ResNet) of the sharp-object extension.

  train     fine-tune an ImageNet ResNet on CROPS (build_hand_crops.py) with one logit
            "hand holds a sharp object". Classes are balanced by a weighted sampler; Vietnamese
            crops get --vn-weight, pseudo crops --pseudo-weight. Early stop on val AP of real hand
            crops. last.pt is written every epoch; calling train again resumes from it.
  evaluate  crop-level AP / AUC on val and test (all, Vietnamese), then the image-level alert used at
            runtime: an image alerts when any of its hand crops >= threshold. The threshold is the
            lowest one whose false-alarm rate on val images without any weapon is <= --max-false-alarm
            (same rule as the detector, train_sharp_object.choose_threshold).
  export    sharp_object_<run_tag>.zip: hand_classifier.pt (fp16, tensors only), the pose weights,
            sharp_object_spec.json (method "hand_crop") and evaluation.json.
"""

import argparse
import csv
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from extensions.sharp_object.hand_crop import CROP_DEFAULTS, INPUT_SIZE, MEAN, STD, build_classifier
from extensions.sharp_object.train_sharp_object import CLASS_MAP, alert_rates, choose_threshold


def read_index(crops):
    with (Path(crops) / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = int(row["label"])
        row["image_has_weapon"] = int(row["image_has_weapon"])
    return rows


def average_precision(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class CropDataset:
    def __init__(self, root, rows, train):
        from torchvision import transforms

        self.root, self.rows = Path(root), rows
        normalize = transforms.Normalize(MEAN.tolist(), STD.tolist())
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(20, fill=114)], p=0.5),
                transforms.ColorJitter(0.35, 0.35, 0.3, 0.04),
                transforms.RandomGrayscale(0.05),
                transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.2),
                transforms.ToTensor(), normalize])
        else:  # the runtime resizes the whole crop, no centre crop
            self.transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(),
                                                 normalize])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.root / row["crop"]) as image:
            return self.transform(image.convert("RGB")), float(row["label"])


def loader(root, rows, train, batch, workers, sample_weights=None, samples=None):
    import torch

    dataset = CropDataset(root, rows, train)
    sampler = None
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                                         num_samples=samples, replacement=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, shuffle=False,
                                       num_workers=workers, pin_memory=True, drop_last=train,
                                       persistent_workers=workers > 0)


# collect.py hard negatives: people holding bats / rackets (coco_neg), phones / bottles / cups (coco_hold),
# brooms / vacuums / household objects indoors (charades_neg)
HARD_NEGATIVE_SOURCES = ("coco_neg", "coco_hold", "charades_neg")


def train_weights(rows, vn_weight, pseudo_weight, hard_weight=1.0):
    """Per-crop sampling weights: source/kind weight, then each class gets half of the total mass."""
    base = np.array([(vn_weight if r["source"] == "vn" else hard_weight if r["source"] in HARD_NEGATIVE_SOURCES else 1.0)
                     * (pseudo_weight if r["kind"] == "pseudo" else 1.0) for r in rows])
    labels = np.array([r["label"] for r in rows])
    for value in (0, 1):
        mask = labels == value
        if mask.any():
            base[mask] *= 0.5 / base[mask].sum()
    return base


def predict(model, data_loader, device):
    import torch

    model.eval()
    scores = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for images, _ in data_loader:
            scores.append(torch.sigmoid(model(images.to(device, non_blocking=True)).float().squeeze(1)).cpu())
    return torch.cat(scores).numpy() if scores else np.zeros(0)


def train(args):
    import torch
    from torch import nn

    torch.manual_seed(3407)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "TRAIN_DONE").is_file():
        print("training already finished:", output / "TRAIN_DONE", flush=True)
        return
    rows = read_index(args.crops)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val" and r["kind"] == "hand"]
    print(f"train crops {len(train_rows)} (pos {sum(r['label'] for r in train_rows)}) | "
          f"val hand crops {len(val_rows)} (pos {sum(r['label'] for r in val_rows)})", flush=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = min(len(train_rows), args.max_samples_per_epoch) if args.max_samples_per_epoch else len(train_rows)
    train_loader = loader(args.crops, train_rows, True, args.batch, args.workers,
                          train_weights(train_rows, args.vn_weight, args.pseudo_weight, args.hard_negative_weight), samples)
    val_loader = loader(args.crops, val_rows, False, args.batch * 2, args.workers)
    model = build_classifier(args.arch, pretrained=True).to(device).to(memory_format=torch.channels_last)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
                                   {"params": head, "lr": args.lr * 10}], weight_decay=args.weight_decay)
    steps = max(1, len(train_loader)) * args.epochs
    warmup = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    state = {"epoch": 0, "best_ap": -1.0, "bad_epochs": 0, "history": []}
    last = output / "last.pt"
    if last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        print(f"resuming after epoch {state['epoch']} (best val AP {state['best_ap']:.4f})", flush=True)
    elif args.init:
        initial = torch.load(args.init, map_location="cpu", weights_only=True)
        if initial.get("arch", args.arch) != args.arch:
            raise SystemExit(f"--init is {initial['arch']}, --arch is {args.arch}")
        model.load_state_dict({k: v.float() for k, v in initial["state_dict"].items()})
        print(f"fine-tuning from {args.init}", flush=True)
    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        started = time.time()
        model.train()
        total, seen = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True).float()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images).squeeze(1).float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss) * len(labels)
            seen += len(labels)
        scores = predict(model, val_loader, device)
        labels = [r["label"] for r in val_rows]
        ap, auc = average_precision(labels, scores), roc_auc(labels, scores)
        metric = ap if ap is not None else -total / max(1, seen)
        improved = metric > state["best_ap"]
        if improved:
            state["best_ap"], state["bad_epochs"] = metric, 0
            torch.save({"arch": args.arch, "state_dict": model.state_dict(), "epoch": epoch, "val_ap": ap},
                       output / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch
        state["history"].append({"epoch": epoch, "loss": round(total / max(1, seen), 4), "val_ap": ap, "val_auc": auc,
                                 "minutes": round((time.time() - started) / 60, 2)})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state}, last)
        (output / "history.json").write_text(json.dumps(state["history"], indent=1), encoding="utf-8")
        print(f"epoch {epoch}/{args.epochs} loss {total / max(1, seen):.4f} | val AP "
              f"{'-' if ap is None else f'{ap:.4f}'} AUC {'-' if auc is None else f'{auc:.4f}'}"
              f"{' *' if improved else ''} | {(time.time() - started) / 60:.1f} min", flush=True)
        if state["bad_epochs"] >= args.patience:
            print(f"early stop: no val AP gain for {args.patience} epochs", flush=True)
            break
    (output / "TRAIN_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def image_rows(rows, scores):
    """[(image, has_weapon, max hand score)] for images whose hand crops are clean negatives or include a positive.

    Images with a weapon but no positive hand crop (weapon not held, or the holder's hand not found) are left
    out: they are neither a fair miss nor a fair false alarm for this method."""
    by_image = {}
    for row, score in zip(rows, scores):
        entry = by_image.setdefault(row["image"], {"weapon": row["image_has_weapon"], "pos": 0, "score": 0.0})
        entry["pos"] |= row["label"]
        entry["score"] = max(entry["score"], float(score))
    return [(image, bool(e["pos"]), e["score"]) for image, e in by_image.items() if e["pos"] or not e["weapon"]]


def evaluate(args):
    import torch

    from extensions.sharp_object.hand_crop import load_classifier

    output = Path(args.output)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, arch = load_classifier(output / "best.pt", device)
    rows = read_index(args.crops)
    report = {"arch": arch, "weights": str(output / "best.pt")}
    split_rows, split_scores = {}, {}
    for split in ("val", "test"):
        part = [r for r in rows if r["split"] == split and r["kind"] == "hand"]
        scores = predict(model, loader(args.crops, part, False, args.batch * 2, args.workers), device)
        split_rows[split], split_scores[split] = part, scores
        for name, keep in (("all", lambda r: True), ("vn", lambda r: r["source"] == "vn")):
            mask = [keep(r) for r in part]
            labels = [r["label"] for r, m in zip(part, mask) if m]
            values = [s for s, m in zip(scores, mask) if m]
            report[f"crop_{name}_{split}"] = {"crops": len(labels), "positives": int(sum(labels)),
                                              "ap": average_precision(labels, values), "auc": roc_auc(labels, values)}
    val_images = image_rows(split_rows["val"], split_scores["val"])
    test_images = image_rows(split_rows["test"], split_scores["test"])
    threshold, rule = choose_threshold(val_images, args.max_false_alarm, args.min_threshold)
    report["alert_threshold"] = round(float(threshold), 4)
    report["threshold_rule"] = rule + " (images without any weapon; image alert = max hand-crop score)"
    vn_images = {r["image"] for split in ("val", "test") for r in split_rows[split] if r["source"] == "vn"}
    report["alert_val"] = alert_rates(val_images, threshold)
    report["alert_test"] = alert_rates(test_images, threshold)
    report["alert_vn_val"] = alert_rates([r for r in val_images if r[0] in vn_images], threshold)
    report["alert_vn_test"] = alert_rates([r for r in test_images if r[0] in vn_images], threshold)
    for split, images in (("val", val_images), ("test", test_images)):  # false alarms on held long objects
        hard = {r["image"] for r in split_rows[split] if r["source"] in HARD_NEGATIVE_SOURCES}
        report[f"alert_hard_negative_{split}"] = alert_rates([r for r in images if r[0] in hard], threshold)
    negatives = [s for r, s in zip(split_rows["test"], split_scores["test"]) if not r["label"]]
    report["crop_false_alarm_test"] = float(np.mean(np.asarray(negatives) >= threshold)) if negatives else None
    report["alert_curve"] = [{"threshold": t, "test": alert_rates(test_images, t),
                              "vn_test": alert_rates([r for r in test_images if r[0] in vn_images], t)}
                             for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    crops_summary = Path(args.crops) / "summary.json"
    if crops_summary.is_file():
        report["crops"] = json.loads(crops_summary.read_text(encoding="utf-8"))
    (output / "evaluation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("alert_curve", "crops")}, indent=1))


def export(args):
    import torch
    from ultralytics import YOLO

    output = Path(args.output)
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    package = output / "package"
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir(parents=True)
    torch.save({"arch": best["arch"], "state_dict": {k: v.half() if v.is_floating_point() else v
                                                     for k, v in best["state_dict"].items()}},
               package / "hand_classifier.pt")
    pose_path = Path(YOLO(args.pose).ckpt_path)
    shutil.copy(pose_path, package / pose_path.name)
    crops = evaluation.get("crops", {})
    spec = {"name": "sharp_object", "method": "hand_crop", "run_tag": args.run_tag, "alert": CLASS_MAP["alert"],
            "classifier": "hand_classifier.pt", "arch": best["arch"], "input_size": INPUT_SIZE,
            "pose_weights": pose_path.name, "pose_imgsz": args.runtime_pose_imgsz, "pose_conf": args.runtime_pose_conf,
            "max_people": 8, "crop": crops.get("crop", CROP_DEFAULTS), "confirm_seconds": 0.5, "release_seconds": 1.0,
            "alert_threshold": evaluation["alert_threshold"], "threshold_rule": evaluation["threshold_rule"],
            "alert_logic": "pose -> square crop around each visible hand -> ResNet; step score = max over hands, "
                           "confirmed over time (extensions/sharp_object/extension.py)",
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (package / "sharp_object_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    shutil.copy(output / "evaluation.json", package / "evaluation.json")
    archive = output / f"sharp_object_{args.run_tag}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.iterdir()):
            bundle.write(path, path.name)
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["train", "evaluate", "export"], required=True)
    parser.add_argument("--crops", type=Path, required=True, help="output of build_hand_crops.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="backbone lr; the head uses 10x")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--vn-weight", type=float, default=6.0, help="sampling weight of Vietnamese crops")
    parser.add_argument("--hard-negative-weight", type=float, default=3.0,
                        help="sampling weight of hard-negative crops (people holding bats, rackets, ...)")
    parser.add_argument("--pseudo-weight", type=float, default=0.5, help="sampling weight of pseudo crops")
    parser.add_argument("--max-samples-per-epoch", type=int, default=0, help="0 = number of train crops")
    parser.add_argument("--max-false-alarm", type=float, default=0.03, help="image-level, on val weapon-free images")
    parser.add_argument("--min-threshold", type=float, default=0.3)
    parser.add_argument("--pose", default="yolo11s-pose.pt", help="export: pose weights shipped with the classifier")
    parser.add_argument("--runtime-pose-imgsz", type=int, default=640)
    parser.add_argument("--runtime-pose-conf", type=float, default=0.35)
    parser.add_argument("--init", type=Path, default=None,
                        help="start from this classifier (best.pt / hand_classifier.pt of an earlier run)")
    parser.add_argument("--run-tag", default="hand_01")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
