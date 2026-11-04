"""Install the Re-DID action-transition pilot as a separate MMAction head."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from action_engine import InteractionHead, ROOT, SOURCE_SHA, SPEC, digest


PARENT_SHA = "ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344"
PACKAGE_SHA = "e6a883467779afeeaa9db48c3cd27c23f79511e46a885e9575405fb88394e656"


def install(archive, root=ROOT):
    root = Path(root)
    archive = Path(archive)
    if digest(archive) != PACKAGE_SHA:
        raise ValueError("Unexpected action-transition pilot ZIP")
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "results") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(archive) as package:
            expected = {"interaction.pt", "interaction.json", "report.json", "protocol.json"}
            if set(package.namelist()) != expected:
                raise ValueError("Unexpected checkpoint package contents")
            for name in expected:
                if package.getinfo(name).file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large checkpoint file")
                (staging / name).write_bytes(package.read(name))
        metadata = json.loads((staging / "interaction.json").read_text(encoding="utf-8"))
        report = json.loads((staging / "report.json").read_text(encoding="utf-8"))
        protocol = json.loads((staging / "protocol.json").read_text(encoding="utf-8"))
        reviewed = report.get("test_reviewed_by_action", {}).get("new", {})
        if (metadata.get("source_sha256") != SOURCE_SHA
                or metadata.get("spec") != SPEC
                or metadata.get("checkpoint_sha256") != digest(staging / "interaction.pt")
                or metadata.get("parent_checkpoint_sha256") != PARENT_SHA
                or metadata.get("head_version") != 1
                or report.get("status") != "completed"
                or report.get("parent_checkpoint_sha256") != PARENT_SHA
                or report.get("threshold") != metadata.get("threshold")
                or protocol.get("parent_checkpoint_sha256") != PARENT_SHA
                or protocol.get("review_manifest_sha256") is None
                or not {"fight", "after_fight", "sports_wrestle"} <= set(reviewed)
                or not 0 < metadata.get("threshold", 0) < 1):
            raise ValueError("Incomplete or incompatible action-transition checkpoint")
        head = InteractionHead()
        head.load_state_dict(torch.load(staging / "interaction.pt", map_location="cpu",
                                        weights_only=True), strict=True)
        for source, target in (
            ("interaction.pt", "weights/interaction_transition.pt"),
            ("interaction.json", "weights/interaction_transition.json"),
            ("report.json", "results/interaction_transition_report.json"),
            ("protocol.json", "results/interaction_transition_protocol.json"),
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
    print("Action-transition pilot installed:", metadata["checkpoint_sha256"])


if __name__ == "__main__":
    main()
