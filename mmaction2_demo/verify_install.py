"""Verify isolated CUDA operators and official checkpoint formats without a camera."""

import hashlib
import json
from pathlib import Path
import subprocess

import torch
import mmcv
import mmaction
import mmdet
from mmcv.ops import roi_align
from mmaction.utils import register_all_modules


def main():
    root = Path(__file__).resolve().parent
    assert mmaction.__version__ == "1.2.0"
    revision = subprocess.check_output(
        ["git", "-C", str(root / "repo"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "4d6c93474730cad2f25e51109adcf96824efc7a3"
    assert (
        torch.cuda.is_available()
    ), "CUDA must work before declaring the GPU demo ready"
    register_all_modules()
    image = torch.ones(1, 1, 8, 8, device="cuda")
    boxes = torch.tensor([[0, 1, 1, 6, 6]], dtype=torch.float32, device="cuda")
    assert roi_align(image, boxes, (2, 2), 1.0).shape == (1, 1, 2, 2)
    records = {}
    for name in ("action_slowonly_r101.pth", "person_faster_rcnn_r50.pth"):
        path = root / "weights" / name
        checkpoint = torch.load(path, map_location="cpu")
        assert "state_dict" in checkpoint, "Unexpected official checkpoint format"
        if name.startswith("action"):
            head = checkpoint["state_dict"]["roi_head.bbox_head.fc_cls.weight"]
            assert head.shape[0] == 81
        records[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "mmaction": mmaction.__version__,
        "mmcv": mmcv.__version__,
        "mmdet": mmdet.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "revision": revision,
        "checkpoint_sha256": records,
        "scope": "Dependency and checkpoint format check; not action accuracy or physical camera acceptance",
    }
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "install_verified.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
