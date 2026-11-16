"""V6: fine-tune the last V-JEPA 2 encoder blocks together with the vandalism probe.

Runs after ``train_vandalism --stage train`` (the frozen probe, which also picks the
windows to fine-tune on):

    python -m extensions.vandalism.finetune_vandalism --stage cache    --data D --cache C --frozen OUT/frozen --tail-cache T
    python -m extensions.vandalism.finetune_vandalism --stage finetune --data D --frozen OUT/frozen --tail-cache T --output OUT/finetune
    python -m extensions.vandalism.finetune_vandalism --stage evaluate --data D --frozen OUT/frozen --output OUT/finetune --eval-cache E

cache    : for every train/dev bag, the top-K windows of the frozen probe (the most
           vandalism-like ones: what MIL top-k pooling trains on) are decoded again and
           run through the frozen first blocks; the hidden states entering the tail
           (fp16, K x 1024 tokens x 1024) are stored, so each epoch only runs the tail.
finetune : trains the last ``--tail-blocks`` blocks + final LayerNorm + probe, starting
           from the pretrained tail and the frozen probe (epoch 0 = the frozen model), with
           layer-wise lr decay; early stop on dev AP; best weights -> tail_best.pt.
evaluate : re-encodes dev + test with the fine-tuned encoder over all windows (exactly the
           runtime path), picks the threshold on dev like the frozen probe (min precision +
           focus-class false-alarm cap), exports vandalism_<run_tag>.zip with
           vandalism_tail.pt and writes comparison.json against the frozen probe.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from extensions.vandalism.model import (AttentiveProbe, Backbone, bag_logits, encode_head, encode_tail, preprocess,
                                        tail_state_dict)
from extensions.vandalism.train_vandalism import (NAME, average_precision, best_threshold, cache_key,
                                                  cap_focus_false_alarms, evaluate, export, item_windows, load_bags,
                                                  load_manifest, roc_auc, seed_everything, stage_features)

WINDOW_BYTES = 1024 * 1024 * 2  # one window entering the tail: 1024 tokens x 1024 fp16


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def load_frozen_probe(folder, device):
    probe = AttentiveProbe()
    probe.load_state_dict(torch.load(Path(folder) / f"{NAME}.pt", map_location="cpu", weights_only=True), strict=True)
    return probe.to(device).eval()


# ------------------------------------------------------------------ cache
def choose_windows(requested, bags, folder):
    if requested > 0:
        return requested
    free = shutil.disk_usage(folder).free
    for k in (3, 2, 1):
        if bags * k * WINDOW_BYTES * 1.15 < free:
            return k
    raise RuntimeError(f"not enough disk for the tail cache ({free / 1e9:.0f} GB free)")


def stage_cache(args):
    items = [i for i in load_manifest(args.data) if i["split"] in ("train", "dev")]
    device, dtype = device_and_dtype()
    args.tail_cache.mkdir(parents=True, exist_ok=True)
    info_path = args.tail_cache / "cache_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if info["tail_blocks"] != args.tail_blocks:
            raise ValueError(f"{args.tail_cache} holds a {info['tail_blocks']}-block tail cache; use another folder")
        k = info["windows_per_bag"]
    else:
        k = choose_windows(args.windows_per_bag, len(items), args.tail_cache)
        info = {"tail_blocks": args.tail_blocks, "windows_per_bag": k, "selection": "top-k windows of the frozen probe"}
        info_path.write_text(json.dumps(info, indent=1))
    print(f"tail cache: {len(items)} bags x {k} windows, about {len(items) * k * WINDOW_BYTES / 1e9:.0f} GB, "
          f"free {shutil.disk_usage(args.tail_cache).free / 1e9:.0f} GB", flush=True)
    probe = load_frozen_probe(args.frozen, device)
    backbone = Backbone(device=device.type)
    stop = len(backbone.model.encoder.layer) - args.tail_blocks
    started, done, skipped = time.perf_counter(), 0, 0
    for number, item in enumerate(items, 1):
        key = cache_key(item)
        target = args.tail_cache / f"{key}.npy"
        features = args.cache / f"{key}.npz"
        if target.is_file() or not features.is_file():
            continue
        with np.load(features) as stored:
            tokens = torch.from_numpy(stored["tokens"])
        with torch.inference_mode():
            scores = torch.cat([probe(part.to(device)).float().cpu() for part in tokens.split(256)])
        top = sorted(torch.topk(scores, min(k, len(scores))).indices.tolist())
        max_windows = args.max_windows_long if item["source"] == "ashen" else args.max_windows
        clips, _, _ = item_windows(item, max_windows)
        if len(clips) != len(tokens):  # decoding must reproduce the feature windows
            print(f"skip (window mismatch {len(clips)} != {len(tokens)}): {item['path']}", flush=True)
            skipped += 1
            continue
        batch = torch.stack([preprocess(clips[i]) for i in top]).to(device, dtype)
        with torch.inference_mode():
            hidden = encode_head(backbone.model, batch, stop).half().cpu().numpy()
        partial = target.with_suffix(".partial.npy")
        np.save(partial, hidden)
        partial.replace(target)
        done += 1
        if done % 200 == 0:
            print(f"{number}/{len(items)} bags, {done} new, {(time.perf_counter() - started) / 60:.1f} min, "
                  f"free {shutil.disk_usage(args.tail_cache).free / 1e9:.0f} GB", flush=True)
    print(f"tail cache done: {done} new, {skipped} skipped, {(time.perf_counter() - started) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ fine-tune
class TailBags(torch.utils.data.Dataset):
    def __init__(self, items, folder):
        self.rows = [(folder / f"{cache_key(i)}.npy", i) for i in items if (folder / f"{cache_key(i)}.npy").is_file()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, item = self.rows[index]
        return torch.from_numpy(np.load(path)), float(item["binary"]), float(item.get("weight", 1.0))


def collate_tail(rows):
    width = max(len(hidden) for hidden, _, _ in rows)
    hidden = torch.zeros(len(rows), width, *rows[0][0].shape[1:], dtype=torch.float16)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for index, (windows, _, _) in enumerate(rows):
        hidden[index, :len(windows)] = windows
        mask[index, :len(windows)] = True
    labels = torch.tensor([r[1] for r in rows])
    weights = torch.tensor([r[2] for r in rows])
    return hidden, mask, labels, weights


class TailModel(nn.Module):
    def __init__(self, backbone_model, tail_blocks, probe, checkpointing):
        super().__init__()
        self.layers = nn.ModuleList(backbone_model.encoder.layer[-tail_blocks:])
        self.layernorm = backbone_model.encoder.layernorm
        self.probe = probe
        self.checkpointing = checkpointing

    def forward(self, hidden):
        """hidden: (N, 1024, 1024) states entering the tail -> window logits (N,)"""
        for layer in self.layers:
            if self.checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(lambda h, m=layer: m(h, None, None, False)[0], hidden,
                                                           use_reentrant=False)
            else:
                hidden = layer(hidden, None, None, False)[0]
        return self.probe(encode_tail([], self.layernorm, hidden))


def param_groups(model, args):
    groups = []
    count = len(model.layers)
    for depth, layer in enumerate(model.layers):  # depth 0 = first tail block (lowest lr)
        groups.append({"params": list(layer.parameters()), "lr": args.lr_tail * args.layer_decay ** (count - 1 - depth)})
    groups.append({"params": list(model.layernorm.parameters()), "lr": args.lr_tail})
    groups.append({"params": list(model.probe.parameters()), "lr": args.lr_probe})
    for group in groups:
        group["initial_lr"] = group["lr"]
    return groups


@torch.no_grad()
def proxy_dev(model, loader, device, amp, top_k):
    """Dev AP/AUC on the cached windows (fast early-stopping signal)."""
    model.eval()
    labels, scores = [], []
    for hidden, mask, label, _ in loader:
        hidden, mask = hidden.to(device, non_blocking=True), mask.to(device)
        with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
            flat = model(hidden[mask].float())
        window = torch.full(mask.shape, -1e4, device=device)
        window[mask] = flat.float()
        scores += torch.sigmoid(bag_logits(window, mask, k=top_k)).cpu().tolist()
        labels += label.int().tolist()
    return average_precision(labels, scores) or 0.0, roc_auc(labels, scores)


def stage_finetune(args):
    from transformers import AutoModel

    seed_everything(args.seed)
    device, _ = device_and_dtype()
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    items = load_manifest(args.data)
    train = TailBags([i for i in items if i["split"] == "train"], args.tail_cache)
    dev = TailBags([i for i in items if i["split"] == "dev"], args.tail_cache)
    positives = sum(r[1]["binary"] for r in train.rows)
    print(f"tail bags: train {len(train)} ({positives} positive), dev {len(dev)}", flush=True)
    loader_args = {"num_workers": args.workers, "pin_memory": device.type == "cuda", "collate_fn": collate_tail,
                   "persistent_workers": args.workers > 0}
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_bags, shuffle=True, drop_last=True, **loader_args)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.batch_bags, **loader_args)

    backbone_model = AutoModel.from_pretrained(args.hf_model, revision=args.hf_revision, dtype=torch.float32)
    backbone_model.requires_grad_(False)
    probe = AttentiveProbe()
    probe.load_state_dict(torch.load(args.frozen / f"{NAME}.pt", map_location="cpu", weights_only=True), strict=True)
    model = TailModel(backbone_model, args.tail_blocks, probe, args.checkpointing).to(device)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(param_groups(model, args), weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, int(0.05 * steps))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and amp == torch.float16)
    pos_weight = torch.tensor((len(train) - positives) / max(1, positives), device=device)
    args.output.mkdir(parents=True, exist_ok=True)

    def save_best(epoch, dev_ap):
        torch.save({"epoch": epoch, "dev_proxy_ap": dev_ap, "tail_blocks": args.tail_blocks,
                    "tail": tail_state_dict(backbone_model, args.tail_blocks),
                    "probe": {k: v.detach().float().cpu() for k, v in model.probe.state_dict().items()}},
                   args.output / "tail_best.partial.pt")
        (args.output / "tail_best.partial.pt").replace(args.output / "tail_best.pt")

    dev_ap, dev_auc = proxy_dev(model, dev_loader, device, amp, args.top_k)
    print(f"epoch 00 (frozen model) dev proxy AP {dev_ap:.4f} AUC {dev_auc}", flush=True)
    history = [{"epoch": 0, "loss": None, "dev_proxy_ap": dev_ap, "dev_proxy_auc": dev_auc}]
    best_ap, best_epoch, patience = dev_ap, 0, 0
    save_best(0, dev_ap)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, started = [], time.perf_counter()
        for hidden, mask, labels, weights in train_loader:
            hidden, mask = hidden.to(device, non_blocking=True), mask.to(device)
            labels, weights = labels.to(device), weights.to(device)
            with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
                flat = model(hidden[mask].float())
            window = torch.full(mask.shape, -1e4, device=device)
            window[mask] = flat.float()
            logits = bag_logits(window, mask, k=args.top_k)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if args.max_steps and len(losses) >= args.max_steps:
                break
        dev_ap, dev_auc = proxy_dev(model, dev_loader, device, amp, args.top_k)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_proxy_ap": dev_ap, "dev_proxy_auc": dev_auc})
        print(f"epoch {epoch:02d} loss {np.mean(losses):.4f} dev proxy AP {dev_ap:.4f} AUC {dev_auc} "
              f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
        if dev_ap > best_ap + 1e-4:
            best_ap, best_epoch, patience = dev_ap, epoch, 0
            save_best(epoch, dev_ap)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
        (args.output / "finetune_history.json").write_text(json.dumps(history, indent=1))
    (args.output / "finetune_history.json").write_text(json.dumps(history, indent=1))
    print(f"best epoch {best_epoch} (dev proxy AP {best_ap:.4f}); 0 means fine-tuning never beat the frozen model",
          flush=True)


# ------------------------------------------------------------------ evaluate + export
def stage_evaluate(args):
    device, _ = device_and_dtype()
    checkpoint = torch.load(args.output / "tail_best.pt", map_location="cpu", weights_only=True)
    if checkpoint["tail_blocks"] != args.tail_blocks:
        raise ValueError(f"tail_best.pt has {checkpoint['tail_blocks']} tail blocks, not {args.tail_blocks}")
    items = load_manifest(args.data)
    backbone = Backbone(device=device.type, tail_state=checkpoint["tail"])
    encode_args = argparse.Namespace(data=args.data, cache=args.eval_cache, batch=args.batch,
                                     max_windows=args.max_windows, max_windows_long=args.max_windows_long)
    stage_features(encode_args, backbone=backbone, items=[i for i in items if i["split"] in ("dev", "test")])
    del backbone
    torch.cuda.empty_cache() if device.type == "cuda" else None
    probe = AttentiveProbe()
    probe.load_state_dict(checkpoint["probe"], strict=True)
    probe.to(device).eval()
    dev_bags = load_bags([i for i in items if i["split"] == "dev"], args.eval_cache)
    test_bags = load_bags([i for i in items if i["split"] == "test"], args.eval_cache)
    _, dev_labels, dev_scores = evaluate(probe, dev_bags, device, k=args.top_k)
    threshold, rule = best_threshold(dev_labels, dev_scores, args.min_precision)
    threshold, rule = cap_focus_false_alarms(dev_bags, dev_scores, threshold, rule, args.max_focus_false_alarm)
    dev_report, _, _ = evaluate(probe, dev_bags, device, threshold, k=args.top_k)
    test_report, _, _ = evaluate(probe, test_bags, device, threshold, k=args.top_k)
    history_path = args.output / "finetune_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    tail_half = {k: v.half() for k, v in checkpoint["tail"].items()}
    torch.save(tail_half, args.output / f"{NAME}_tail.pt")
    extra_spec = {"backbone_tail_blocks": args.tail_blocks, "backbone_tail_file": f"{NAME}_tail.pt",
                  "finetune": {"best_epoch": checkpoint["epoch"], "dev_proxy_ap": checkpoint["dev_proxy_ap"],
                               "lr_tail": args.lr_tail, "lr_probe": args.lr_probe, "layer_decay": args.layer_decay}}
    export(args, probe, threshold, rule, history, dev_report, test_report, items,
           extra_spec=extra_spec, extra_files=(f"{NAME}_tail.pt",))
    compare(args.frozen / "report.json", args.output / "report.json", args.output / "comparison.json")


def compare(frozen_path, tuned_path, out_path):
    frozen, tuned = json.loads(Path(frozen_path).read_text()), json.loads(Path(tuned_path).read_text())
    rows = {}
    for name, report in (("frozen", frozen), ("finetune", tuned)):
        test = report["test"]
        entry = {"threshold": report["spec"]["threshold"], "test_all_ap": test["all_bags"]["ap"],
                 "test_all_auc": test["all_bags"]["roc_auc"]}
        for source in ("kinetics700_2020", "ashen", "charades"):
            if source in test:
                point = test[source].get("at_threshold", {})
                entry[f"{source}_recall"] = point.get("recall")
                entry[f"{source}_fp"] = point.get("fp")
                entry[f"{source}_auc"] = test[source].get("roc_auc")
        entry["false_alarm_by_class"] = test.get("false_alarm_by_negative_class", {})
        rows[name] = entry
    Path(out_path).write_text(json.dumps(rows, indent=1))
    keys = [k for k in rows["frozen"] if k != "false_alarm_by_class"]
    print(f"\n{'metric':28s} {'frozen':>10s} {'finetune':>10s}")
    for key in keys:
        print(f"{key:28s} {_fmt(rows['frozen'].get(key)):>10s} {_fmt(rows['finetune'].get(key)):>10s}")
    print(f"\n{'false alarms (test)':28s} {'frozen':>10s} {'finetune':>10s}")
    classes = sorted(set(rows["frozen"]["false_alarm_by_class"]) | set(rows["finetune"]["false_alarm_by_class"]))
    for label in classes:
        print(f"{label[:28]:28s} {_fmt(rows['frozen']['false_alarm_by_class'].get(label)):>10s} "
              f"{_fmt(rows['finetune']['false_alarm_by_class'].get(label)):>10s}")
    print("saved:", out_path)


def _fmt(value):
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main():
    from extensions.vandalism.model import HF_MODEL, HF_REVISION

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["cache", "finetune", "evaluate"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="frozen pooled features (train_vandalism --stage features)")
    parser.add_argument("--frozen", type=Path, required=True, help="output folder of the frozen probe")
    parser.add_argument("--tail-cache", type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-tag", default=time.strftime("%Y%m%d_%H%M"))
    parser.add_argument("--tail-blocks", type=int, default=4, help="last encoder blocks to fine-tune (of 24)")
    parser.add_argument("--windows-per-bag", type=int, default=0, help="0 = 3, or fewer if the disk is short")
    parser.add_argument("--max-windows", type=int, default=12, help="must match the features stage")
    parser.add_argument("--max-windows-long", type=int, default=120, help="must match the features stage")
    parser.add_argument("--batch", type=int, default=32, help="windows per backbone batch when re-encoding")
    parser.add_argument("--batch-bags", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    parser.add_argument("--lr-tail", type=float, default=2e-5)
    parser.add_argument("--lr-probe", type=float, default=1e-4)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--max-focus-false-alarm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--hf-model", default=HF_MODEL)
    parser.add_argument("--hf-revision", default=HF_REVISION)
    args = parser.parse_args()
    needed = {"cache": ("cache", "tail_cache"), "finetune": ("tail_cache", "output"), "evaluate": ("output", "eval_cache")}
    missing = [name for name in needed[args.stage] if getattr(args, name) is None]
    if missing:
        parser.error(f"--stage {args.stage} needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
    {"cache": stage_cache, "finetune": stage_finetune, "evaluate": stage_evaluate}[args.stage](args)


if __name__ == "__main__":
    random.seed(0)
    main()
