"""Measure detection and clear delays against reviewed video intervals."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evaluate(trace_path, review_path):
    entries = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines()]
    header = entries[0]
    if header.get("kind") != "session":
        raise ValueError("Missing trace session header")
    source = Path(header["source"])
    checksum = digest(source)
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    matches = [item for item in review["sources"] if item["sha256"] == checksum]
    if len(matches) != 1:
        raise ValueError("Trace source must match exactly one reviewed video")
    predictions = [entry for entry in entries if entry.get("kind") == "prediction"]
    results = []
    for interval in matches[0]["intervals"]:
        start, end = interval["start_seconds"], interval["end_seconds"]
        inside = [row for row in predictions if start <= row["timestamp"] < end]
        target = "normal" if interval["action"] == "after_fight" else "fighting"
        if interval["action"] in {"ordinary", "sports_wrestle"}:
            results.append({"action": interval["action"], "start_seconds": start,
                            "red_fraction": sum(row["state"] == "fighting" for row in inside)
                            / len(inside) if inside else None,
                            "observations": len(inside)})
            continue
        reached = next((row for row in inside if row["state"] == target), None)
        results.append({"action": interval["action"], "start_seconds": start,
                        "end_seconds": end, "delay_seconds":
                        round(reached["timestamp"] - start, 3) if reached else None,
                        "observations": len(inside)})
    return {"session_id": header["session_id"], "video_sha256": checksum,
            "backend": header["backend"], "intervals": results,
            "predictions": len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.trace, args.review), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
