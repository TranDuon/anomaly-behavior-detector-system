"""Download only the official demo's two pretrained checkpoints."""

import hashlib
import json
from pathlib import Path
from download_file import download

ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "action_slowonly_r101.pth": "https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth",
    "person_faster_rcnn_r50.pth": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
}


def main():
    destination = ROOT / "weights"
    destination.mkdir(exist_ok=True)
    records = {}
    for name, url in WEIGHTS.items():
        target = destination / name
        if not target.exists():
            temporary = target.with_suffix(".partial")
            print("Downloading:", name, flush=True)
            download(url, temporary)
            if temporary.stat().st_size < 1024 * 1024:
                raise ValueError("Incomplete checkpoint: " + name)
            temporary.replace(target)
        digest = (
            hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(target.read_bytes()).hexdigest()
        )
        records[name] = {"url": url, "sha256": digest, "bytes": target.stat().st_size}
        print(name, target.stat().st_size, digest, flush=True)
    (destination / "provenance.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
