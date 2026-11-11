"""Validate and install the running package beside the verified parent."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from action_engine import ROOT, SOURCE_SHA, SPEC
from fixed_running_engine import PARENT_SHA, load_running_checkpoint


def install(archive, root=ROOT):
    root = Path(root)
    names = {"interaction_fixed_running.pt", "interaction_fixed_running.json", "report.json", "protocol.json"}
    with tempfile.TemporaryDirectory(dir=root / "results") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(archive) as package:
            if len(package.namelist()) != len(names) or set(package.namelist()) != names:
                raise ValueError("Unexpected running package contents")
            for name in names:
                if package.getinfo(name).file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large running package member")
                (staging / name).write_bytes(package.read(name))
        shutil.copyfile(root / "weights/interaction_fall_wrestle.pt", staging / "interaction_fall_wrestle.pt")
        metadata, _ = load_running_checkpoint(staging)
        report = json.loads((staging / "report.json").read_text())
        protocol = json.loads((staging / "protocol.json").read_text())
        if (report.get("status") != "completed" or report.get("epochs", 0) < 1
                or protocol.get("source_sha256") != SOURCE_SHA or protocol.get("spec") != SPEC
                or protocol.get("parent_checkpoint_sha256") != PARENT_SHA
                or protocol.get("dataset_revision") != metadata["dataset_revision"]
                or report["development"]["threshold"] != metadata["running_threshold"]
                or report["test"]["threshold"] != metadata["running_threshold"]):
            raise ValueError("Incomplete or incompatible running report")
        for name in names:
            target = root / ("weights" if name.startswith("interaction_") else "results") / (
                name if name.startswith("interaction_") else "fixed_running_" + name)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(staging / name, temporary)
            temporary.replace(target)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    print("Running pilot installed:", install(arguments.archive)["checkpoint_sha256"])
