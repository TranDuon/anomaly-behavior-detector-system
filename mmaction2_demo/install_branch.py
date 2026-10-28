"""Install a locally downloaded pilot branch after checking its public source and schema."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from action_engine import InteractionHead, ROOT, SOURCE_SHA, SPEC, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(args.archive) as package:
            for name in ("interaction.pt", "interaction.json", "report.json"):
                item = package.getinfo(name)
                if item.file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpected large pilot branch")
                (staging / name).write_bytes(package.read(name))
        metadata = json.loads((staging / "interaction.json").read_text())
        report = json.loads((staging / "report.json").read_text())
        if (
            metadata["source_sha256"] != SOURCE_SHA
            or metadata["spec"] != SPEC
            or metadata["checkpoint_sha256"] != digest(staging / "interaction.pt")
            or report.get("status") != "completed"
        ):
            raise ValueError("Incomplete or incompatible training bundle")
        if (
            metadata.get("private_data_used") is not False
            or not 0 < metadata["threshold"] < 1
        ):
            raise ValueError("Unexpected data policy or threshold")
        head = InteractionHead()
        head.load_state_dict(
            torch.load(
                staging / "interaction.pt", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        backup = (
            ROOT
            / "results"
            / ("branch_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        for name in ("interaction.pt", "interaction.json"):
            target = ROOT / "weights" / name
            if target.exists():
                backup.mkdir(exist_ok=True)
                shutil.copyfile(target, backup / name)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(staging / name, temporary)
            temporary.replace(target)
    print(
        "Pilot branch installed. Original AVA weights and default web model unchanged."
    )
    print(
        "On the web: choose MMAction2 + fine-tuned branch and allow pilot; stop the previous session first."
    )


if __name__ == "__main__":
    main()
