"""S5: verify and install a sharp-object detector for the runtime extension.

    extensions\\env\\Scripts\\python.exe -m extensions.sharp_object.install_sharp_object <sharp_object_<RUN_TAG>.zip>
    extensions\\env\\Scripts\\python.exe -m extensions.sharp_object.install_sharp_object --weights best.pt --threshold 0.35

The zip comes from a Colab notebook: the YOLO detector (train_sharp_object.py --stage export) or the
hand-crop classifier (train_hand_classifier.py --stage export, spec "method": "hand_crop"). --weights installs
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
# Weights members depend on the spec method, checked in check(). The handgun extension uses the
# same installer (extensions/handgun/install_handgun.py) with name="handgun".


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hand_crop(folder, spec, name="sharp_object"):
    """Hand-crop install: classifier + pose weights load, and one blank step runs end to end."""
    import numpy as np
    from ultralytics import YOLO

    from extensions.sharp_object.hand_crop import load_classifier, to_tensor

    missing = [k for k in ("classifier", "arch", "pose_weights", "alert_threshold") if k not in spec]
    if missing or spec["name"] != name:
        raise ValueError(f"bad spec (missing {missing}, name {spec.get('name')})")
    if not 0 < float(spec["alert_threshold"]) < 1:
        raise ValueError("alert_threshold must be in (0, 1)")
    for member in (spec["classifier"], spec["pose_weights"]):
        if not (folder / member).is_file():
            raise ValueError(f"missing {member}")
    model, arch = load_classifier(folder / spec["classifier"])
    size = int(spec.get("input_size", 224))
    model(to_tensor([np.zeros((size, size, 3), np.uint8)]))
    pose = YOLO(str(folder / spec["pose_weights"]))
    if pose.task != "pose":
        raise ValueError(f"expected a pose model, got {pose.task}")
    pose.predict(np.zeros((640, 640, 3), np.uint8), verbose=False, device="cpu")
    return spec, {"method": "hand_crop", "arch": arch}


def check(folder, name="sharp_object", hand_crop_only=False):
    import numpy as np
    from ultralytics import YOLO

    spec = json.loads((folder / f"{name}_spec.json").read_text(encoding="utf-8"))
    if spec.get("method") == "hand_crop":
        return check_hand_crop(folder, spec, name)
    if hand_crop_only:
        raise ValueError(f"{name} needs a hand_crop spec, got method {spec.get('method')}")
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


def install(name="sharp_object", archive=None, weights=None, threshold=0.35, imgsz=960, hand_crop_only=False):
    target = TARGET.with_name(name)
    staging = target.with_name(f"{name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if archive:
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            missing = [m for m in (f"{name}_spec.json",) if m not in names]
            if missing:
                raise ValueError(f"zip is missing {missing}")
            for name in names:
                if "/" in name or "\\" in name or name.startswith("."):
                    raise ValueError(f"unexpected path in zip: {name}")
                (staging / name).write_bytes(bundle.read(name))
        source = {"source_zip": archive.name, "source_sha256": sha256(archive)}
    else:
        shutil.copy(weights, staging / f"{name}.pt")
        spec = {"name": name, "run_tag": f"manual:{weights.name}", "framework": "ultralytics",
                "weights": f"{name}.pt", "alert": name, "imgsz": imgsz,
                "alert_threshold": threshold, "threshold_rule": "set by hand (intermediate checkpoint)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (staging / f"{name}_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        source = {"source_weights": str(weights), "source_sha256": sha256(weights)}
    spec, names = check(staging, name, hand_crop_only)
    record = {"installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_tag": spec.get("run_tag"), **source,
              "classes": names, "files": {p.name: sha256(p) for p in sorted(staging.iterdir())}}
    (staging / "install.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target.with_name(f"{name}.previous"), ignore_errors=True)
        target.rename(target.with_name(f"{name}.previous"))
    staging.rename(target)
    print(f"installed {spec.get('run_tag')} -> {target} | threshold {spec['alert_threshold']} | classes {names}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--weights", type=Path, help="intermediate Ultralytics .pt instead of a zip")
    parser.add_argument("--threshold", type=float, default=0.35, help="alert threshold for --weights")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if bool(args.archive) == bool(args.weights):
        sys.exit("give either a zip or --weights")
    install("sharp_object", archive=args.archive, weights=args.weights, threshold=args.threshold, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
