"""S5: verify and install a sharp-object detector for the runtime extension.

    extensions\\env\\Scripts\\python.exe -m extensions.sharp_object.install_sharp_object <sharp_object_<RUN_TAG>.zip>
    extensions\\env\\Scripts\\python.exe -m extensions.sharp_object.install_sharp_object --weights best.pt --threshold 0.35

The zip comes from the Colab notebook (train_sharp_object.py --stage export). --weights installs
an intermediate Ultralytics checkpoint (e.g. train/weights/best.pt from Drive) with a hand-set
alert threshold, for trying the runtime before the run finishes.

Checks: zip members, spec fields, the weights load in Ultralytics and predict on a blank image.
Installs atomically into weights/extensions/sharp_object/ (previous install kept in
sharp_object.previous) and writes install.json with SHA-256 of every file.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "weights" / "extensions" / "sharp_object"
MEMBERS = ("sharp_object.pt", "sharp_object_spec.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(folder):
    import numpy as np
    from ultralytics import YOLO

    spec = json.loads((folder / "sharp_object_spec.json").read_text(encoding="utf-8"))
    missing = [k for k in ("name", "weights", "alert_threshold", "imgsz") if k not in spec]
    if missing or spec["name"] != "sharp_object":
        raise ValueError(f"bad spec (missing {missing}, name {spec.get('name')})")
    if not 0 < float(spec["alert_threshold"]) < 1:
        raise ValueError("alert_threshold must be in (0, 1)")
    model = YOLO(str(folder / spec["weights"]))
    if model.task != "detect":
        raise ValueError(f"expected a detection model, got {model.task}")
    model.predict(np.zeros((int(spec["imgsz"]), int(spec["imgsz"]), 3), np.uint8), imgsz=int(spec["imgsz"]),
                  verbose=False, device="cpu")
    return spec, model.names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--weights", type=Path, help="intermediate Ultralytics .pt instead of a zip")
    parser.add_argument("--threshold", type=float, default=0.35, help="alert threshold for --weights")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if bool(args.archive) == bool(args.weights):
        sys.exit("give either a zip or --weights")
    staging = TARGET.with_name("sharp_object.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if args.archive:
        with zipfile.ZipFile(args.archive) as bundle:
            names = set(bundle.namelist())
            missing = [m for m in MEMBERS if m not in names]
            if missing:
                raise ValueError(f"zip is missing {missing}")
            for name in names:
                if "/" in name or "\\" in name or name.startswith("."):
                    raise ValueError(f"unexpected path in zip: {name}")
                (staging / name).write_bytes(bundle.read(name))
        source = {"source_zip": args.archive.name, "source_sha256": sha256(args.archive)}
    else:
        shutil.copy(args.weights, staging / "sharp_object.pt")
        spec = {"name": "sharp_object", "run_tag": f"manual:{args.weights.name}", "framework": "ultralytics",
                "weights": "sharp_object.pt", "alert": "sharp_object", "imgsz": args.imgsz,
                "alert_threshold": args.threshold, "threshold_rule": "set by hand (intermediate checkpoint)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (staging / "sharp_object_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        source = {"source_weights": str(args.weights), "source_sha256": sha256(args.weights)}
    spec, names = check(staging)
    record = {"installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_tag": spec.get("run_tag"), **source,
              "classes": names, "files": {p.name: sha256(p) for p in sorted(staging.iterdir())}}
    (staging / "install.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if TARGET.exists():
        shutil.rmtree(TARGET.with_name("sharp_object.previous"), ignore_errors=True)
        TARGET.rename(TARGET.with_name("sharp_object.previous"))
    staging.rename(TARGET)
    print(f"installed {spec.get('run_tag')} -> {TARGET} | threshold {spec['alert_threshold']} | classes {names}")


if __name__ == "__main__":
    main()
