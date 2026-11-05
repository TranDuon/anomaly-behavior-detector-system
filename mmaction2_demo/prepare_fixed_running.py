"""Download only fixed ground-camera UCF-ARG walk/jog/run clips from SPHAR.

The upstream SPHAR action folder also mixes aerial, rooftop, CASIA and VIRAT
clips.  Select exact UCF-ARG ground-camera filenames and verify Git blob hashes
so a silent upstream change cannot alter the training set.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


DATASET_REVISION = "6f2be47ef810865c1a15595d976e15169a0df4a1"
TREE_URL = (
    "https://api.github.com/repos/AlexanderMelde/SPHAR-Dataset/git/trees/"
    + DATASET_REVISION + "?recursive=1"
)
PATH_PATTERN = re.compile(
    r"^videos/(running|walking)/"
    r"ucarg_person(\d{2})_(0[1-4])_ground_(running|jogging|walking)\.mp4$"
)
LABELS = {"running": 1, "jogging": 0, "walking": 0}
EXCLUDED = {
    # The tracked file is not decodable (ffprobe reports duration=N/A).
    "videos/running/ucarg_person01_02_ground_jogging.mp4": "invalid_video",
}


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def select_entries(tree):
    if tree.get("truncated"):
        raise ValueError("GitHub returned a truncated dataset tree")
    selected = []
    for item in tree["tree"]:
        match = PATH_PATTERN.fullmatch(item["path"])
        if not match or item["type"] != "blob":
            continue
        if item["path"] in EXCLUDED:
            continue
        folder, actor, take, action = match.groups()
        if folder != ("running" if action in {"running", "jogging"} else "walking"):
            raise ValueError("Action folder mismatch: " + item["path"])
        selected.append((item, int(actor), int(take), action))
    counts = {action: sum(row[3] == action for row in selected) for action in LABELS}
    if counts != {"running": 48, "jogging": 47, "walking": 48}:
        raise ValueError(f"Incomplete UCF-ARG ground set: {counts}")
    keys = {(actor, take, action) for _, actor, take, action in selected}
    if len(keys) != len(selected):
        raise ValueError("Duplicate actor/take/action in UCF-ARG set")
    return sorted(selected, key=lambda row: row[0]["path"])


def split_for_actor(actor):
    if 1 <= actor <= 8:
        return "train"
    if 9 <= actor <= 10:
        return "development"
    if 11 <= actor <= 12:
        return "test"
    raise ValueError(f"Unexpected UCF-ARG actor {actor}")


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(TREE_URL, headers={"User-Agent": "fixed-running-colab"})
    with urllib.request.urlopen(request, timeout=60) as response:
        selected = select_entries(json.load(response))
    rows = []
    for index, (item, actor, take, action) in enumerate(selected, 1):
        relative = Path(item["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and git_blob_sha(target.read_bytes()) == item["sha"]:
            pass
        else:
            url = (
                "https://raw.githubusercontent.com/AlexanderMelde/SPHAR-Dataset/"
                + DATASET_REVISION + "/" + item["path"]
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        payload = response.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            if len(payload) != item["size"] or git_blob_sha(payload) != item["sha"]:
                raise ValueError("Dataset file hash mismatch: " + item["path"])
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(target)
        rows.append({
            "path": str(target), "source_path": item["path"],
            "actor": actor, "take": take, "action": action,
            "label": LABELS[action], "split": split_for_actor(actor),
            "git_blob_sha1": item["sha"],
        })
        if index % 24 == 0:
            print(f"Dataset verified: {index}/{len(selected)}", flush=True)
    manifest = {
        "dataset": "SPHAR UCF-ARG ground only",
        "source_revision": DATASET_REVISION,
        "source_url": "https://github.com/AlexanderMelde/SPHAR-Dataset",
        "classes": ["walking", "jogging", "running"],
        "split_policy": "actors 01-08 train, 09-10 development, 11-12 test",
        "excluded": EXCLUDED,
        "rows": rows,
    }
    output = destination / "fixed_running_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest:", output, flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.destination)
