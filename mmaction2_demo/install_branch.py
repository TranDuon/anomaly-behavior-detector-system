"""Install interaction v1 after checking model and branch fingerprints."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from action_engine import (
    InteractionHead,
    ROOT,
    SOURCE_SHA,
    SPEC,
    digest,
)


def install(archive, root=ROOT):
    root = Path(root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "results") as directory:
        staging = Path(directory)
        names = ("interaction.pt", "interaction.json", "report.json")
        with zipfile.ZipFile(archive) as package:
            for name in names:
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
        if metadata.get("private_data_used") is not False or not 0 < metadata["threshold"] < 1:
            raise ValueError("Unexpected data policy or threshold")
        if int(metadata.get("head_version", 1)) != 1:
            raise ValueError("Only mmaction_interaction_v1 is supported")
        if report.get("training", {}).get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise ValueError("Report does not describe this checkpoint")
        head = InteractionHead()
        head.load_state_dict(
            torch.load(staging / "interaction.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        backup = root / "results" / ("branch_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        for name in ("interaction.pt", "interaction.json"):
            target = root / "weights" / name
            if target.exists():
                backup.mkdir(exist_ok=True)
                shutil.copyfile(target, backup / name)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(staging / name, temporary)
            temporary.replace(target)
        shutil.copyfile(staging / "report.json", root / "results/interaction_v1_report.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    install(args.archive)
    print("mmaction_interaction_v1 installed and verified.")
    print("On the web: choose MMAction2 interaction v1 and allow pilot.")


if __name__ == "__main__":
    main()
