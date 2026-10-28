"""Launch the unmodified official MMAction2 person-action demo in isolation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CONFIG = "configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py"


def command(source, output, show=True, device="cuda:0"):
    args = [
        sys.executable,
        str(ROOT / "run_official.py"),
        "--config",
        str(ROOT / "repo" / CONFIG),
        "--checkpoint",
        str(ROOT / "weights/action_slowonly_r101.pth"),
        "--det-config",
        str(ROOT / "repo/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py"),
        "--det-checkpoint",
        str(ROOT / "weights/person_faster_rcnn_r50.pth"),
        "--label-map",
        str(ROOT / "repo/tools/data/ava/label_map.txt"),
        "--input-video",
        str(source),
        "--device",
        device,
        "--out-filename",
        str(output),
        "--action-score-thr",
        "0.4",
        "--predict-stepsize",
        "8",
        "--output-fps",
        "15",
        "--display-width",
        "640",
        "--cfg-options",
        "test_dataloader.dataset.proposal_file=None",
    ]
    if show:
        args.append("--show")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.camera:
        source = (
            args.source
            or input("Camera index (DroidCam often 1 or 2; laptop often 0): ").strip()
        )
        if not source.isdecimal():
            parser.error("Enter the camera index shown by Windows/DroidCam")
    else:
        source = (args.source or input("Video path: ")).strip().strip('"')
        source = str(Path(source).resolve())
        if not Path(source).is_file():
            parser.error("Video file not found")
    if not (ROOT / "results/install_verified.json").is_file():
        raise RuntimeError("Installation not verified; run setup.cmd first")
    for name in ("action_slowonly_r101.pth", "person_faster_rcnn_r50.pth"):
        if not (ROOT / "weights" / name).is_file():
            raise FileNotFoundError("Missing checkpoint; run setup.ps1: " + name)
    output = ROOT / "results" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".mp4")
    output.parent.mkdir(exist_ok=True)
    print("Official MMAction2 v1.2.0 / SlowOnly R101 AVA demo", flush=True)
    print(
        "Action labels, not an accepted violence alarm system. Stop with Ctrl+C in this console.",
        flush=True,
    )
    print("Output:", output, flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "repo")
    environment["TORCH_HOME"] = str(ROOT / "weights" / "torch-cache")
    environment["CUDA_MODULE_LOADING"] = "LAZY"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    with (output.with_suffix(".log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command(source, output, not args.no_preview, args.device),
            cwd=ROOT / "repo",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        except KeyboardInterrupt:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=20)
            return
        except BaseException:
            process.terminate()
            process.wait(timeout=20)
            raise
    if result:
        raise RuntimeError(
            f"Official demo exited {result}; inspect {output.with_suffix('.log')}"
        )
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("No playable result was written; inspect the log")
    print("Saved:", output)


if __name__ == "__main__":
    main()
