"""Reuse the official Oxford VIA3 app instead of implementing a temporal timeline."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from fight.via_bridge import import_via, via_project
from scripts.fight_pilot import write_json

VIA_URL = "https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html"
VIA_SHA256 = "79c53de67a8b38b2e9fbbbd82357af45071a3a330b16576162f764a22b804926"


def install_app(output):
    destination = output / "via_video_annotator.html"
    if destination.exists():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(VIA_URL, timeout=60) as response:
            data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != VIA_SHA256:
        raise ValueError(
            "VIA upstream bytes changed; review the official version before updating the locked checksum"
        )
    if not destination.exists():
        partial = destination.with_suffix(".partial.html")
        partial.write_bytes(data)
        partial.replace(destination)
    write_json(
        output / "via_provenance.json",
        {
            "source": VIA_URL,
            "version": "3.0.13",
            "data_format_version": "3.1.1",
            "sha256": VIA_SHA256,
            "license": "Upstream BSD-2-Clause notices retained verbatim inside the unmodified app",
        },
    )
    return destination


def prepare(output, with_app=False):
    pack = json.loads((output / "review.json").read_text(encoding="utf-8"))
    project = via_project(pack)
    if not project["file"]:
        raise ValueError("Prepare review media before generating a VIA project")
    path = output / "via_project.json"
    if not path.exists():
        write_json(path, project)
    if with_app:
        install_app(output)
    launcher = Path(__file__).resolve().parents[1] / "frontend" / "via_review.html"
    (output / "via_review.html").write_bytes(launcher.read_bytes())
    print(
        f"VIA ready: {path}. Serve the pack on localhost, open via_review.html and click the pending-project button. Annotate temporal segments and save using VIA. No automatic approvals."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--pack", type=Path, required=True)
    build.add_argument("--with-app", action="store_true")
    convert = commands.add_parser("import")
    convert.add_argument("--pack", type=Path, required=True)
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--project", type=Path, required=True)
    convert.add_argument("--reviewer", required=True)
    convert.add_argument("--method", choices=("human", "ai_assisted"), required=True)
    convert.add_argument("--approve-reviewed", action="store_true")
    convert.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.pack, args.with_app)
        return
    if args.output.exists():
        raise ValueError("Refuse overwriting existing reviewed labels")
    pack = json.loads((args.pack / "review.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    project = json.loads(args.project.read_text(encoding="utf-8"))
    document = import_via(
        project,
        pack,
        manifest["rows"],
        args.reviewer,
        args.method,
        args.approve_reviewed,
    )
    write_json(args.output, document)
    print(
        f"Imported {sum(bool(source['intervals']) for source in document['sources'])} sources; approved={sum(source['review_status']=='approved' for source in document['sources'])}. Preflight still required; no training started."
    )


if __name__ == "__main__":
    main()
