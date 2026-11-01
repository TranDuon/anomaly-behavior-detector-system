"""Prepare public temporal review or audit/export explicitly reviewed labels. No training."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fight.assault_extension import audit, binary_corpus, build_manifest, pending_annotations


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare(parent, pool, archives, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; existing reviews must not be overwritten")
    manifest = build_manifest(read(parent), read(pool))
    annotations = pending_annotations(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir(exist_ok=True)
    available, missing = 0, []
    with ExitStack() as stack:
        packages = [stack.enter_context(zipfile.ZipFile(path)) for path in archives]
        indexes = [set(package.namelist()) for package in packages]
        for source in manifest["sources"]:
            name = source["media"]
            if not re.fullmatch(r"media/[a-f0-9]{24}\.mp4", name):
                raise ValueError("Unexpected public preview path")
            found = False
            for package, names in zip(packages, indexes):
                if name not in names:
                    continue
                item = package.getinfo(name)
                if item.file_size > 128 * 1024 * 1024:
                    raise ValueError("Preview too large for the short-clip review pack")
                data = package.read(name)
                if hashlib.sha256(data).hexdigest() != source["preview_sha256"]:
                    continue
                (output / name).write_bytes(data)
                found = True
                break
            if found:
                available += 1
            else:
                missing.append(source["video"])
    # No timestamp or subtype is inherited from weak clip tags or old AI annotations.
    write(output / "manifest.json", manifest)
    write(output / "annotations.pending.json", annotations)
    write(output / "review.json", {"manifest": manifest, "annotations": annotations, "missing_media": missing})
    report, _ = audit(manifest, annotations)
    report.update({"available_previews": available, "missing_media": missing,
                   "parent_sources": sum(row["from_parent_v1"] for row in manifest["sources"])})
    write(output / "readiness.json", report)
    for name in ("assault_review.html", "assault_review.js"):
        shutil.copyfile(ROOT / "frontend" / name, output / ("index.html" if name.endswith(".html") else name))
    print(json.dumps(report, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--parent", type=Path, default=ROOT / "benchmarks/mmaction_interaction_public_v1.json")
    build.add_argument("--pool", type=Path, default=ROOT / "benchmarks/mmaction_interaction_indoor_v2.json")
    build.add_argument("--archive", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--pack", type=Path, required=True)
    check.add_argument("--annotations", type=Path, required=True)
    check.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent, args.pool, args.archive, args.output)
    else:
        manifest, labels = read(args.pack / "manifest.json"), read(args.annotations)
        report, _ = audit(manifest, labels)
        # Never accept reviewed labels for a preview that was unavailable or changed.
        for source in manifest["sources"]:
            if any(item["video"] == source["video"] and item["review_status"] == "reviewed" for item in labels["sources"]):
                preview = args.pack / source["media"]
                if not preview.is_file() or hashlib.sha256(preview.read_bytes()).hexdigest() != source["preview_sha256"]:
                    raise ValueError("Reviewed preview is missing or changed: " + source["video"])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.export:
            corpus = binary_corpus(manifest, labels)
            if args.export.exists():
                raise ValueError("Refuse overwriting an existing exported corpus")
            write(args.export, corpus)
        if not report["ready_for_training"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
