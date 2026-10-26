"""Extract frozen S3D clip features (fight stream A) from the fight datasets.

One .npz per video in <feature-root>/<dataset>/<key>.npz (features float16
(N, 2048), motion energy, first/last decoded frame of each clip, frame count,
source fps), so a Colab disconnect only loses the video being processed.
Whole videos are always extracted. Long videos are streamed: only the frames
of the clips in flight are held in memory.

Decoding is the bottleneck, not the GPU; --shard k/n runs the k-th of n
interleaved slices of the video list, and n such processes share one GPU.
Each shard writes its own manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import load_items
from anomaly.visual_fight import EXTRACTION_CONFIG, ClipFeatureExtractor, extract_clips, sample_video


def write_manifest(path: Path, manifest: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract S3D clip features for the visual fight stream")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]; repeatable")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16, help="Clips per backbone call")
    parser.add_argument(
        "--half", choices=("auto", "on", "off"), default="auto",
        help="FP16 autocast; auto = on for CUDA devices",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, help="k/n: process every n-th video starting at k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[visual] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)
    half = args.half == "on" or (args.half == "auto" and device.type == "cuda")

    feature_root = Path(args.feature_root).resolve()
    items = [item for spec in args.dataset for item in load_items(spec)]
    if args.split != "all":
        items = [item for item in items if item.split == args.split]
    if args.limit is not None:
        items = items[: args.limit]
    manifest_name = "manifest.json"
    if args.shard:
        index_text, _, count_text = args.shard.partition("/")
        shard_index, shard_count = int(index_text), int(count_text)
        if not 0 <= shard_index < shard_count:
            parser.error("--shard must be k/n with 0 <= k < n")
        items = items[shard_index::shard_count]
        manifest_name = f"manifest_shard{shard_index}of{shard_count}.json"
    if not items:
        raise RuntimeError("No videos found for the given datasets")

    extractor = ClipFeatureExtractor().to(device)
    print(f"[visual] {len(items)} video(s) -> {feature_root} (device={device}, half={half})", flush=True)

    manifests: dict[str, dict] = {}
    for index, item in enumerate(items, start=1):
        dataset_dir = feature_root / item.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / manifest_name
        if item.dataset not in manifests:
            manifest = {"config": EXTRACTION_CONFIG, "videos": {}}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("config") != EXTRACTION_CONFIG and not args.force:
                    raise RuntimeError(
                        f"Feature cache {dataset_dir} was built with {manifest.get('config')}; "
                        "use a new --feature-root or --force."
                    )
                manifest["config"] = EXTRACTION_CONFIG
            manifests[item.dataset] = manifest
        manifest = manifests[item.dataset]

        output_path = dataset_dir / f"{item.key}.npz"
        if output_path.exists() and not args.force:
            print(f"[{index}/{len(items)}] Skip {item.uid}: cached", flush=True)
            continue
        started = time.perf_counter()
        payload = extract_clips(sample_video(item.path), extractor, device, args.batch_size, half)
        temporary_path = output_path.with_name(output_path.stem + ".partial.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(output_path)
        seconds = time.perf_counter() - started
        clips = int(len(payload["features"]))
        manifest["videos"][item.key] = {
            "source": str(item.path),
            "split": item.split,
            "frames": int(payload["frame_count"]),
            "clips": clips,
            "seconds": round(seconds, 2),
        }
        write_manifest(manifest_path, manifest)
        video_seconds = int(payload["frame_count"]) / max(float(payload["fps"]), 1e-6)
        print(
            f"[{index}/{len(items)}] {item.uid}: clips={clips} "
            f"({video_seconds / max(seconds, 1e-6):.1f}x real time)",
            flush=True,
        )

    for dataset, manifest in manifests.items():
        write_manifest(feature_root / dataset / manifest_name, manifest)
    print("[visual] Complete.", flush=True)


if __name__ == "__main__":
    main()
