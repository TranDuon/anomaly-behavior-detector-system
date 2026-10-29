"""Install an interaction pilot after checking model and branch fingerprints."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from action_engine import (
    AUX_VERSION,
    InteractionHead,
    InteractionHeadV2,
    ROOT,
    SOURCE_SHA,
    SPEC,
    TunedRefinementHead,
    digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(args.archive) as package:
            refined = {
                "interaction_refined.pt",
                "interaction_refined.json",
            }.issubset(package.namelist())
            names = (
                ("interaction_refined.pt", "interaction_refined.json", "report.json")
                if refined
                else ("interaction.pt", "interaction.json", "report.json")
            )
            for name in names:
                item = package.getinfo(name)
                if item.file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpected large pilot branch")
                (staging / name).write_bytes(package.read(name))
        checkpoint_name = "interaction_refined.pt" if refined else "interaction.pt"
        metadata_name = "interaction_refined.json" if refined else "interaction.json"
        metadata = json.loads((staging / metadata_name).read_text())
        report = json.loads((staging / "report.json").read_text())
        if (
            metadata["source_sha256"] != SOURCE_SHA
            or metadata["spec"] != SPEC
            or metadata["checkpoint_sha256"] != digest(staging / checkpoint_name)
            or report.get("status") != "completed"
        ):
            raise ValueError("Incomplete or incompatible training bundle")
        if metadata.get("private_data_used") is not False or not 0 < metadata["threshold"] < 1:
            raise ValueError("Unexpected data policy or threshold")
        head_version = int(metadata.get("head_version", 1))
        if refined:
            installed_base = ROOT / "weights/interaction.pt"
            if (
                head_version != 3
                or metadata.get("aux_version") != AUX_VERSION
                or metadata.get("operating_point_feasible") is not True
                or not installed_base.is_file()
                or metadata.get("base_checkpoint_sha256") != digest(installed_base)
            ):
                raise ValueError("Refinement does not match the installed fine-tuned branch")
            head = TunedRefinementHead()
        else:
            if head_version not in {1, 2} or (
                head_version == 2 and metadata.get("aux_version") != AUX_VERSION
            ):
                raise ValueError("Unsupported interaction head version")
            if head_version == 2 and metadata.get("operating_point_feasible") is not True:
                raise ValueError("Fusion v2 calibration gate did not pass")
            head = InteractionHeadV2() if head_version == 2 else InteractionHead()
        head.load_state_dict(
            torch.load(staging / checkpoint_name, map_location="cpu", weights_only=True),
            strict=True,
        )
        backup = ROOT / "results" / ("branch_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        names = (
            ("interaction_refined.pt", "interaction_refined.json")
            if refined
            else ("interaction.pt", "interaction.json")
        )
        for name in names:
            target = ROOT / "weights" / name
            if target.exists():
                backup.mkdir(exist_ok=True)
                shutil.copyfile(target, backup / name)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(staging / name, temporary)
            temporary.replace(target)
    print(
        "Refined fine-tuned branch installed. Original AVA weights and the prior branch remain available."
        if refined
        else "Pilot branch installed. Original AVA weights and default web model unchanged."
    )
    print(
        "On the web: choose MMAction2 fine-tuned + hiệu chỉnh indoor and allow pilot; stop the previous session first."
        if refined
        else "On the web: choose MMAction2 + nhánh fine-tuned and allow pilot; stop the previous session first."
    )


if __name__ == "__main__":
    main()
