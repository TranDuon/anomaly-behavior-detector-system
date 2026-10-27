"""Package deterministic public examples for isolated CPU upload/replay checks."""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def select_cases(rows):
    selected, seen = [], set()
    for tag in ("hug", "highfive", "handgestures"):
        pool = sorted(
            [
                row
                for row in rows
                if row["label"] == 0 and tag in row.get("actions", [])
            ],
            key=lambda row: row["sha256"],
        )
        for row in pool:
            group = row.get("performance", row["video"])
            if group not in seen:
                selected.append(row)
                seen.add(group)
                break
    for dataset in ("airtlab", "rlvs"):
        pool = sorted(
            [row for row in rows if row["label"] == 1 and row["dataset"] == dataset],
            key=lambda row: row["sha256"],
        )
        if pool:
            selected.append(pool[0])
    return selected


def package(root, cache, pku=None):
    root, cache = Path(root), Path(cache)
    run = json.loads((root / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("Package only a completed experiment")
    names = sorted(run["variants"])
    if set(names) != {"rgb_aux", "rgb_pose_aux"}:
        raise ValueError("Unexpected variants")
    sources, expected = [], []
    destination = root / "cpu_validation.zip"
    partial = destination.with_suffix(".partial.zip")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
        for name in names:
            rows = json.loads((root / name / "test_predictions.json").read_text())
            cases = select_cases(rows)
            archive.write(
                root / name / "sentinel_fight_only_runtime.zip", name + "/runtime.zip"
            )
            smoke = cases[:1] + [row for row in cases if row["label"] == 1]
            for row in smoke:
                archive.write(
                    cache / (row["cache"] + ".npz"), name + "/" + row["cache"] + ".npz"
                )
                expected.append(
                    {
                        "variant": name,
                        "cache": row["cache"],
                        "score": row["score"],
                        "video": row["video"],
                        "label": row["label"],
                    }
                )
            if not sources:
                for row in cases:
                    path = Path(row["path"])
                    relative = (
                        "videos/"
                        + hashlib.sha256(row["video"].encode()).hexdigest()[:12]
                        + path.suffix
                    )
                    archive.write(path, relative)
                    sources.append(
                        {
                            "path": relative,
                            "video": row["video"],
                            "label": row["label"],
                            "actions": row.get("actions", []),
                            "selection": "Fixed action tags and SHA ordering, not prediction scores",
                        }
                    )
        if pku and (Path(pku) / "rgb/0360-R.avi").is_file():
            archive.write(Path(pku) / "rgb/0360-R.avi", "videos/pku_0360-R.avi")
            sources.append(
                {
                    "path": "videos/pku_0360-R.avi",
                    "video": "pku/0360-R",
                    "label": None,
                    "selection": "Predeclared mixed-action continuous source; integration only",
                }
            )
        archive.writestr("expected.json", json.dumps(expected, indent=2))
        archive.writestr("sources.json", json.dumps(sources, indent=2))
    partial.replace(destination)
    print(
        "CPU validation package:",
        destination,
        destination.stat().st_size,
        "bytes;",
        len(sources),
        "sources;",
        len(expected),
        "tensor examples",
    )
    return destination


def unpack(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Refuse overwriting an existing QA directory")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or any(":" in part for part in path.parts)
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe QA archive member")
        if sum(member.file_size for member in archive.infolist()) > 1024**3:
            raise ValueError("QA archive exceeds 1 GiB limit")
        if archive.testzip() is not None:
            raise ValueError("Corrupt QA archive")
        archive.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pku", type=Path)
    args = parser.parse_args()
    if args.mode == "pack":
        package(args.source, args.destination, args.pku)
    else:
        unpack(args.source, args.destination)


if __name__ == "__main__":
    main()
