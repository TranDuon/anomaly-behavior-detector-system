"""Record or compare golden behaviour traces for the four core branches.

Phase-0 safety net for the extension work: every change must leave the
strike / grapple / fast_running / fall output of the default backend
unchanged on a fixed set of reference videos.

    python scripts/compare_behavior_traces.py record     # once, before any change
    python scripts/compare_behavior_traces.py compare    # after every change
    python scripts/compare_behavior_traces.py compare --backend mmaction_separate_actions_ext

Videos are processed offline (every decoded frame, timestamps = index / fps),
the same causal path ``FightProcessor.run`` uses for uploaded files, so runs
are repeatable. Only the core labels are compared; keys an extension adds are
ignored, which is exactly the "extensions may add, never change" contract.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DIR = ROOT / "tests" / "golden"
MANIFEST = GOLDEN_DIR / "behavior_reference.json"
TRACE_DIR = GOLDEN_DIR / "traces"
VIDEO_DIR = GOLDEN_DIR / "videos"
CORE_LABELS = ("strike", "grapple", "fast_running", "fall")
DEFAULT_BACKEND = "mmaction_separate_actions"
SCORE_TOLERANCE = 1e-3
BOX_TOLERANCE = 2.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _round(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _boxes(values):
    return [[round(float(c), 1) for c in box] for box in values or []]


def snapshot(result):
    """Core-only view of one prediction; extension keys are dropped."""
    regions = result.get("action_regions") or {}
    return {
        "timestamp": _round(result.get("prediction_timestamp"), 6),
        "state": result.get("state"),
        "event": result.get("event"),
        "people": result.get("people"),
        "active_behaviors": sorted(b for b in result.get("active_behaviors", []) if b in CORE_LABELS),
        "action_states": {k: v for k, v in sorted((result.get("action_states") or {}).items()) if k in CORE_LABELS},
        "action_scores": {k: _round(v) for k, v in sorted((result.get("action_scores") or {}).items()) if k in CORE_LABELS},
        "action_regions": {k: _boxes(regions.get(k)) for k in CORE_LABELS if regions.get(k)},
        "person_boxes": _boxes(result.get("person_boxes")),
    }


def make_runtime(backend, extensions=None, extension_python=None):
    from fight.mmaction_runtime import MMActionRuntime

    if backend == DEFAULT_BACKEND:
        runtime = MMActionRuntime(allow_pilot=True, separate_actions=True)
    elif backend == "mmaction_separate_actions_ext":
        names = None if extensions is None else [n for n in extensions.split(",") if n]
        runtime = MMActionRuntime(allow_pilot=True, separate_actions=True, extensions=True,
                                  extension_names=names, extension_python=extension_python)
    else:
        raise ValueError(f"Unsupported backend {backend}")
    if not runtime.model_ready:
        raise RuntimeError(runtime.message)
    runtime.warmup()
    if backend != DEFAULT_BACKEND:
        status = runtime.extension_status()
        print(f"extensions: requested={status['requested']} active={status['active']} error={status.get('error')}")
    return runtime


def run_video(runtime, path):
    import cv2

    runtime.reset()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if 0 < fps <= 240 else 25.0
    rows, index, last = [], 0, None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = runtime.push(frame, index / fps, predict=True, sampled=False)
            index += 1
            stamp = result.get("prediction_timestamp")
            if stamp is not None and stamp != last:
                last = stamp
                rows.append(snapshot(result))
    finally:
        capture.release()
    return {"frames": index, "fps": fps, "predictions": rows}


def load_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["videos"]:
        local = VIDEO_DIR / f"{item['sha256'][:16]}{Path(item['source']).suffix}"
        item["path"] = local if local.is_file() else ROOT / item["source"]
        if not Path(item["path"]).is_file():
            raise FileNotFoundError(f"Reference video missing: {item['source']} (sha {item['sha256'][:12]})")
    return manifest


def record(args):
    manifest = load_manifest()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    runtime = make_runtime(DEFAULT_BACKEND)
    try:
        for item in manifest["videos"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"SHA mismatch for {item['source']}")
            backup = VIDEO_DIR / f"{item['sha256'][:16]}{Path(item['source']).suffix}"
            if not backup.is_file():
                shutil.copy2(item["path"], backup)
            target = TRACE_DIR / f"{item['sha256'][:16]}.json"
            if target.exists() and not args.force:
                raise FileExistsError(f"{target} exists; golden traces are recorded once (use --force deliberately)")
            started = time.perf_counter()
            trace = run_video(runtime, item["path"])
            trace.update(source=item["source"], sha256=item["sha256"], backend=DEFAULT_BACKEND,
                         recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            target.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"recorded {item['label']:<14} {len(trace['predictions']):>4} predictions "
                  f"{time.perf_counter() - started:6.1f}s  {item['source']}")
    finally:
        runtime.close()


def diff_rows(golden, current):
    problems = []
    if len(golden) != len(current):
        problems.append(f"prediction count {len(golden)} -> {len(current)}")
    for index, (want, got) in enumerate(zip(golden, current)):
        where = f"t={want['timestamp']}"
        if want["timestamp"] is None or got["timestamp"] is None or abs(want["timestamp"] - got["timestamp"]) > 1e-4:
            problems.append(f"#{index} timestamp {want['timestamp']} -> {got['timestamp']}")
            continue
        for key in ("state", "event", "people", "active_behaviors", "action_states"):
            if want[key] != got[key]:
                problems.append(f"{where} {key}: {want[key]} -> {got[key]}")
        for name in set(want["action_scores"]) | set(got["action_scores"]):
            a, b = want["action_scores"].get(name), got["action_scores"].get(name)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > SCORE_TOLERANCE):
                problems.append(f"{where} score[{name}]: {a} -> {b}")
        pairs = [("person_boxes", want["person_boxes"], got["person_boxes"])]
        for name in set(want["action_regions"]) | set(got["action_regions"]):
            pairs.append((f"regions[{name}]", want["action_regions"].get(name, []), got["action_regions"].get(name, [])))
        for key, a, b in pairs:
            if len(a) != len(b) or any(abs(x - y) > BOX_TOLERANCE for p, q in zip(a, b) for x, y in zip(p, q)):
                problems.append(f"{where} {key}: {a} -> {b}")
    return problems


def compare(args):
    manifest = load_manifest()
    runtime = make_runtime(args.backend, args.extensions, args.extension_python)
    failed = 0
    try:
        for item in manifest["videos"]:
            golden = json.loads((TRACE_DIR / f"{item['sha256'][:16]}.json").read_text(encoding="utf-8"))
            trace = run_video(runtime, item["path"])
            problems = diff_rows(golden["predictions"], trace["predictions"])
            status = "OK  " if not problems else "FAIL"
            print(f"{status} {item['label']:<14} {len(trace['predictions']):>4} predictions  {item['source']}")
            for line in problems[: args.max_lines]:
                print("      " + line)
            if len(problems) > args.max_lines:
                print(f"      ... {len(problems) - args.max_lines} more")
            failed += bool(problems)
    finally:
        runtime.close()
    total = len(manifest["videos"])
    print(f"\n{total - failed}/{total} reference videos match the golden traces ({args.backend}).")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record golden traces with the current default backend")
    rec.add_argument("--force", action="store_true")
    cmp_ = sub.add_parser("compare", help="re-run and compare against the golden traces")
    cmp_.add_argument("--backend", default=DEFAULT_BACKEND)
    cmp_.add_argument("--max-lines", type=int, default=8)
    cmp_.add_argument("--extensions", default=None,
                      help="comma separated extension names for the _ext backend (default: installed ones)")
    cmp_.add_argument("--extension-python", default=None,
                      help="interpreter for the extension worker (default: extensions/env)")
    args = parser.parse_args()
    if args.command == "record":
        record(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
