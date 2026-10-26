"""Diagnose the visual fight stream on cached S3D features: is there a fight signal at all?

visual_fight_v1 reached frame AUC 0.872 on UBI-Fights but a median in-video
AUC of 0.495, below a motion-energy baseline (0.562), while its training
ranking loss fell to 0.002: it memorised the training scenes. Two causes are
possible, and they call for different next steps:
  (a) the features hold a fight signal the head memorised past -> retrain
      with a head that cannot memorise (and more scenes, e.g. RLVS);
  (b) whole-frame S3D at 224 px does not see small CCTV fights -> extract
      features on crops around groups of people.

Methods, all on the same cached features (nothing is re-extracted):
  motion        motion-energy baseline
  zero_shot     probability mass of S3D's own Kinetics fight classes; no
                training, so nothing to memorise (S3D's classifier is a
                linear map of the cached mean feature, so this is exact)
  linear*       a linear probe
  temporal*     a small causal temporal head (64 units, 2 blocks, dropout 0.5)
  *_scene       features minus the running mean of the earlier clips of the
                same video (a fixed camera's scene removed; causal, so it
                also works live)
  *_videomean   features minus the mean of the whole video (not causal;
                upper bound of scene removal, diagnostic only)
Trained probes use UBI-Fights training videos (same split as training) and
keep the epoch with the best median in-video AUC on validation.

Reported per method: validation and test median in-video AUC, test frame AUC
(whole test set and the "_short" set), Surveillance Camera Fight clip AUC.
The verdict compares the best validation method's test in-video AUC with the
motion baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import load_items
from anomaly.visual_fight import ClipFeatures, FightTemporalHead, predict_video
from scripts.evaluation_metrics import aggregate_frame_scores
from scripts.train_fight import split_videos, write_json
from scripts.train_visual_fight import VideoData, fight_loss, frame_level, make_batch


FIGHT_CLASSES = (
    "punching person (boxing)", "wrestling", "slapping", "headbutting",
    "drop kicking", "side kick", "high kick",
)
MEAN_DIM = 1024                       # first half of a cached feature: mean over (T, H, W)


def kinetics_fight_score(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, fight: list[int]) -> np.ndarray:
    """Probability mass of the Kinetics fight classes for each clip."""
    logits = features[:, :MEAN_DIM].astype(np.float32) @ weight.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, fight].sum(axis=1).astype(np.float64)


def load_kinetics_head() -> tuple[np.ndarray, np.ndarray, list[int]]:
    from torchvision.models.video import S3D_Weights

    weights = S3D_Weights.KINETICS400_V1
    state = weights.get_state_dict(progress=False)
    weight = state["classifier.1.weight"].reshape(400, MEAN_DIM).numpy()
    bias = state["classifier.1.bias"].numpy()
    categories = weights.meta["categories"]
    return weight, bias, [categories.index(name) for name in FIGHT_CLASSES]


def remove_scene(features: np.ndarray, mode: str) -> np.ndarray:
    """Subtract the scene: running mean of earlier clips ('scene') or the whole-video mean ('videomean')."""
    values = features.astype(np.float32)
    if mode == "videomean":
        return (values - values.mean(axis=0, keepdims=True)).astype(np.float16)
    cumulative = np.cumsum(values, axis=0)
    earlier = np.empty_like(values)
    earlier[0] = values[0]                                     # first clip: nothing earlier yet
    earlier[1:] = cumulative[:-1] / np.arange(1, len(values))[:, None]
    return (values - earlier).astype(np.float16)


def transformed(videos: list[VideoData], mode: str | None) -> list[VideoData]:
    if mode is None:
        return videos
    result = []
    for video in videos:
        copy_ = copy.copy(video)
        copy_.clips = copy.copy(video.clips)
        copy_.clips.features = remove_scene(video.clips.features, mode)
        result.append(copy_)
    return result


def train_probe(train, val, head_kwargs, args, device, rng) -> tuple[FightTemporalHead, list[dict]]:
    """Train on train videos; keep the epoch with the best median in-video AUC on validation."""
    fight = [index for index, video in enumerate(train) if video.item.has_positive]
    normal = [index for index, video in enumerate(train) if not video.item.has_positive]
    model = FightTemporalHead(**head_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_score, history = None, -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(args.steps_per_epoch):
            features, labels = make_batch(train, fight, normal, args.batch_size, args.sequence_length, 0.6, rng)
            loss, _ = fight_loss(model(features.to(device)), labels.to(device), 1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation = frame_level(val, [predict_video(model, video.clips.features, device) for video in val])
        score = float(np.nan_to_num(validation["video_auc_median"], nan=0.0))
        history.append({"epoch": epoch, "val_auc": validation["auc"], "val_video_auc_median": score})
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def test_metrics(test_ubi, test_scf, score_fn, short_seconds: float) -> dict:
    """Frame AUC (whole and short test set), median in-video AUC, SCF clip AUC."""
    labels_all, scores_all, labels_short, scores_short, inside = [], [], [], [], []
    for video in test_ubi:
        scores = score_fn(video)
        frame_scores, _ = aggregate_frame_scores(scores, video.clips.last_frames + 1, video.clips.frame_count)
        labels = video.frame_labels
        labels_all.append(labels)
        scores_all.append(frame_scores)
        kept = len(labels) if labels.any() else min(len(labels), int(round(short_seconds * video.clips.fps)))
        labels_short.append(labels[:kept])
        scores_short.append(frame_scores[:kept])
        if 0 < labels.sum() < len(labels):
            inside.append(roc_auc_score(labels, frame_scores))
    result = {
        "test_frame_auc": float(roc_auc_score(np.concatenate(labels_all), np.concatenate(scores_all))),
        "test_short_frame_auc": float(roc_auc_score(np.concatenate(labels_short), np.concatenate(scores_short))),
        "test_video_auc_median": float(np.median(inside)),
        "test_videos_below_half": int(np.sum(np.asarray(inside) < 0.5)),
    }
    if test_scf is not None:
        clip_labels = [int(video.item.has_positive) for video in test_scf]
        clip_scores = [float(np.max(score_fn(video))) for video in test_scf]
        result["scf_clip_auc"] = float(roc_auc_score(clip_labels, clip_scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the visual fight stream on cached features")
    parser.add_argument("--dataset", action="append", required=True, help="ubi:root (train+test) and scf:root (test)")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude", default=None, help="exclude.json from dedup_videos.py")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Same split as train_visual_fight.py")
    parser.add_argument("--short-normal-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root).resolve()
    excluded = set()
    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text(encoding="utf-8"))["exclude_from_training"])

    items = [item for spec in args.dataset for item in load_items(spec)]
    cached = [item for item in items if (feature_root / item.dataset / f"{item.key}.npz").exists()]
    weight, bias, fight_index = load_kinetics_head()

    def load(group):
        videos = []
        for item in group:
            path = feature_root / item.dataset / f"{item.key}.npz"
            video = VideoData(item, ClipFeatures.load(path), 1.0)
            video.crop_features = video.crop_count = None
            with np.load(path) as data:
                if "crop_features" in data:                 # person-group crops (extract_visual.py --crops)
                    video.crop_features, video.crop_count = data["crop_features"], data["crop_count"]
            if len(video.clips):
                videos.append(video)
        return videos

    def zero_shot(video) -> np.ndarray:
        if video.crop_features is None:
            return kinetics_fight_score(video.clips.features, weight, bias, fight_index)
        clips, slots, dim = video.crop_features.shape
        scores = kinetics_fight_score(video.crop_features.reshape(-1, dim), weight, bias, fight_index)
        scores = scores.reshape(clips, slots)
        scores[np.arange(slots)[None, :] >= video.crop_count[:, None]] = 0.0
        return scores.max(axis=1)                            # the most fight-like crop of each clip

    usable = [item for item in cached if item.dataset == "ubi" and item.split == "train" and item.uid not in excluded]
    train_items, val_items = split_videos(usable, args.val_ratio, args.seed)
    train, val = load(train_items), load(val_items)
    test_ubi = load([item for item in cached if item.dataset == "ubi" and item.split == "test"])
    scf_items = [item for item in cached if item.dataset == "scf"]
    test_scf = load(scf_items) if scf_items else None
    print(
        f"[diagnose-visual] train {len(train)} / val {len(val)} / test {len(test_ubi)} UBI videos"
        f"{f', {len(test_scf)} SCF clips' if test_scf else ''}; device={device}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    results, histories = {}, {}

    def record(name, val_videos, score_fn, scf_ok=True):
        validation = frame_level(val_videos, [score_fn(video) for video in val_videos])
        row = {"val_frame_auc": validation["auc"], "val_video_auc_median": validation["video_auc_median"]}
        row.update(test_metrics(test_ubi_view[name_mode(name)], test_scf if scf_ok else None, score_fn,
                                args.short_normal_seconds))
        results[name] = row
        print(
            f"  {name:20s} val in-video={row['val_video_auc_median']:.3f} | test in-video="
            f"{row['test_video_auc_median']:.3f} frame={row['test_frame_auc']:.3f} "
            f"short={row['test_short_frame_auc']:.3f} SCF={row.get('scf_clip_auc', float('nan')):.3f}",
            flush=True,
        )

    def name_mode(name):
        return "scene" if name.endswith("_scene") else ("videomean" if name.endswith("_videomean") else None)

    test_ubi_view = {mode: transformed(test_ubi, mode) for mode in (None, "scene", "videomean")}

    record("motion", val, lambda video: video.clips.motion.astype(np.float64))
    record("zero_shot", val, zero_shot)

    heads = {
        "linear": {"hidden": 64, "layers": 0, "dropout": 0.0, "input_dropout": 0.2},
        "temporal": {"hidden": 64, "layers": 2, "dropout": 0.5, "input_dropout": 0.5},
    }
    for mode in (None, "scene", "videomean"):
        train_view, val_view = transformed(train, mode), transformed(val, mode)
        for head_name, head_kwargs in heads.items():
            name = head_name + (f"_{mode}" if mode else "")
            model, history = train_probe(train_view, val_view, head_kwargs, args, device, rng)
            histories[name] = history
            # SCF clips are 1-3 s: no earlier clips to estimate a scene from.
            record(name, val_view, lambda video, m=model: predict_video(m, video.clips.features, device),
                   scf_ok=mode is None)

    motion = results["motion"]["test_video_auc_median"]
    best = max((name for name in results if name != "motion"), key=lambda name: results[name]["val_video_auc_median"])
    best_test = results[best]["test_video_auc_median"]
    if best_test >= motion + 0.05:
        verdict = (f"SIGNAL: '{best}' reaches test in-video AUC {best_test:.3f} vs motion {motion:.3f}; "
                   "the features see fights -> retrain with this setup (+ more scenes, e.g. RLVS)")
    else:
        verdict = (f"NO SIGNAL BEYOND MOTION: best on validation '{best}' gives test in-video AUC {best_test:.3f} "
                   f"vs motion {motion:.3f}; whole-frame S3D does not see the fight -> extract features on "
                   "person-group crops")
    write_json(output_dir / "visual_diagnostics.json", {
        "results": results, "best_on_validation": best, "verdict": verdict,
        "fight_classes": list(FIGHT_CLASSES), "histories": histories, "settings": vars(args),
    })
    print(f"[diagnose-visual] {verdict}", flush=True)


if __name__ == "__main__":
    main()
