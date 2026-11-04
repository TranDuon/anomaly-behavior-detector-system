"""Install the Colab fall/wrestle pilot beside interaction v1."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from action_engine import InteractionHead, ROOT, SOURCE_SHA, SPEC, digest


PARENT_SHA = "8c5808db106aacd0f43b72a0e76178b2dec039e0a1766d996bb8374b7e4dddb7"


def install(archive, root=ROOT):
    root = Path(root)
    calibration = json.loads((root / "config/fall_wrestle_calibration.json").read_text())
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "results") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(archive) as package:
            if set(package.namelist()) != {"interaction.pt", "interaction.json", "report.json", "protocol.json"}:
                raise ValueError("Unexpected checkpoint package contents")
            for name in package.namelist():
                if package.getinfo(name).file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large checkpoint file")
                (staging / name).write_bytes(package.read(name))
        metadata = json.loads((staging / "interaction.json").read_text(encoding="utf-8"))
        report = json.loads((staging / "report.json").read_text(encoding="utf-8"))
        if (metadata.get("source_sha256") != SOURCE_SHA or metadata.get("spec") != SPEC
                or metadata.get("checkpoint_sha256") != digest(staging / "interaction.pt")
                or metadata.get("checkpoint_sha256") != calibration["checkpoint_sha256"]
                or metadata.get("parent_checkpoint_sha256") != PARENT_SHA
                or metadata.get("head_version") != 1
                or report.get("status") != "completed"
                or report.get("parent_checkpoint_sha256") != PARENT_SHA
                or report.get("threshold") != metadata.get("threshold")
                or not 0 < calibration["threshold"] < 1):
            raise ValueError("Incomplete, incompatible, or uncalibrated checkpoint")
        head = InteractionHead()
        head.load_state_dict(torch.load(staging / "interaction.pt", map_location="cpu",
                                        weights_only=True), strict=True)
        for source, target in (
            ("interaction.pt", "weights/interaction_fall_wrestle.pt"),
            ("interaction.json", "weights/interaction_fall_wrestle.json"),
            ("report.json", "results/interaction_fall_wrestle_report.json"),
            ("protocol.json", "results/interaction_fall_wrestle_protocol.json"),
        ):
            destination = root / target
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(staging / source, temporary)
            temporary.replace(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    metadata = install(args.archive)
    print("Fall/wrestle pilot installed:", metadata["checkpoint_sha256"])


if __name__ == "__main__":
    main()
