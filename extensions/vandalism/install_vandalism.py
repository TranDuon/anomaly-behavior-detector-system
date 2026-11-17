"""V4: verify and install a vandalism checkpoint zip produced by the Colab notebook.

    .venv\\Scripts\\python.exe -m extensions.vandalism.install_vandalism <path to vandalism_<RUN_TAG>.zip>

Checks: zip members, spec fields matching extensions/vandalism/model.py, the probe
state_dict loading strictly into AttentiveProbe (weights_only), finite weights.
Installs atomically into weights/extensions/vandalism/ and writes install.json
(SHA-256 of every file). Existing core weights are never touched.
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch

from extensions.vandalism.model import FRAMES, FPS, GRID, HF_MODEL, HF_REVISION, SIZE, AttentiveProbe

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "weights" / "extensions" / "vandalism"
MEMBERS = ("vandalism.pt", "vandalism.json", "report.json", "protocol.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify(archive):
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = [m for m in MEMBERS if m not in names]
        if missing:
            raise ValueError(f"zip is missing {missing}")
        blobs = {m: bundle.read(m) for m in MEMBERS}
        spec = json.loads(blobs["vandalism.json"])
        tail = spec.get("backbone_tail_file")
        if tail:
            if tail not in names:
                raise ValueError(f"spec names {tail} but the zip does not hold it")
            blobs[tail] = bundle.read(tail)
    expected = {"name": "vandalism", "backbone": HF_MODEL, "backbone_revision": HF_REVISION, "frames": FRAMES,
                "fps": FPS, "size": SIZE, "pooled_grid": list(GRID)}
    wrong = {k: (spec.get(k), v) for k, v in expected.items() if spec.get(k) != v}
    if wrong:
        raise ValueError(f"spec does not match the runtime model: {wrong}")
    if not 0 < float(spec["threshold"]) < 1:
        raise ValueError("threshold must be in (0, 1)")
    state = torch.load(io.BytesIO(blobs["vandalism.pt"]), map_location="cpu", weights_only=True)
    probe = AttentiveProbe()
    probe.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("non-finite weights")
    if tail:
        tail_state = torch.load(io.BytesIO(blobs[tail]), map_location="cpu", weights_only=True)
        check_tail(tail_state, spec)
    return blobs, spec


def check_tail(state, spec):
    """Fine-tuned encoder tail: complete last blocks that fit V-JEPA 2 ViT-L (checked on the meta device)."""
    from transformers import AutoConfig, AutoModel

    from extensions.vandalism.model import load_tail

    config = AutoConfig.from_pretrained(spec["backbone"], revision=spec["backbone_revision"])
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    blocks = {int(k.split(".")[2]) for k in state if k.startswith("encoder.layer.")}
    if len(blocks) != int(spec["backbone_tail_blocks"]):
        raise ValueError(f"tail holds {len(blocks)} blocks, spec says {spec['backbone_tail_blocks']}")
    load_tail(model, {k: v.to("meta") for k, v in state.items()})
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("non-finite tail weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--force", action="store_true", help="replace an installed checkpoint")
    args = parser.parse_args()
    blobs, spec = verify(args.archive)
    if (TARGET / "vandalism.pt").exists() and not args.force:
        sys.exit(f"{TARGET} already has a checkpoint; use --force to replace it")
    staging = TARGET.with_name("vandalism.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, data in blobs.items():
        (staging / name).write_bytes(data)
    record = {"installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_zip": args.archive.name,
              "source_zip_sha256": sha256(args.archive.read_bytes()), "run_tag": spec.get("run_tag"),
              "files": {name: sha256(data) for name, data in blobs.items()}}
    (staging / "install.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if TARGET.exists():
        shutil.rmtree(TARGET.with_name("vandalism.previous"), ignore_errors=True)
        TARGET.rename(TARGET.with_name("vandalism.previous"))
    staging.rename(TARGET)
    report = json.loads(blobs["report.json"])
    print(f"installed {spec['run_tag']} -> {TARGET}")
    print("threshold", spec["threshold"], "| test", json.dumps({k: v.get("roc_auc") for k, v in report["test"].items()
                                                                  if isinstance(v, dict) and "roc_auc" in v}))


if __name__ == "__main__":
    main()
