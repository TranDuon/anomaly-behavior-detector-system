"""Extension worker: a separate process for behaviours added after the core four.

Runs in ``extensions/env`` (new torch) and talks JSON lines over stdin/stdout,
like ``mmaction2_demo/worker.py``:

    -> {"op": "infer", "transport": "shared_memory", "mapping": ..., "shape": [9,H,W,3],
        "timestamp": 12.3, "people": {...}}
    <- {"timestamp": 12.3, "extensions": {name: {"score", "boxes", "details"} | {"error"}},
        "timings_ms": {name: ms, "read": ms, "total": ms}}
    -> {"op": "reset"} / {"op": "close"}

One failing extension reports an error for itself only; the others keep running.
"""

import argparse
import contextlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "mmaction2_demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extensions import base  # noqa: E402
from frame_transport import SharedFrameReader  # noqa: E402

# Modules that register real extensions. Missing optional dependencies only
# disable the extension that needs them.
KNOWN_MODULES = ("extensions.vandalism", "extensions.sharp_object")


def reply(value):
    sys.__stdout__.write(json.dumps(value, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def load_extensions(names):
    errors = {}
    for module in KNOWN_MODULES:
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(module)
    loaded = {}
    for name in names:
        try:
            extension = base.create(name)
            extension.load()
            loaded[name] = extension
        except Exception as error:  # one broken extension must not stop the others
            traceback.print_exc(file=sys.stderr)
            errors[name] = f"{type(error).__name__}: {error}"
    return loaded, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extensions", default="selftest", help="comma separated names")
    args = parser.parse_args()
    names = [name.strip() for name in args.extensions.split(",") if name.strip()]
    with contextlib.redirect_stdout(sys.stderr):
        extensions, load_errors = load_extensions(names)
    meta = {"device": "cpu", "torch": None}
    with contextlib.suppress(Exception):
        import torch

        meta = {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
    reply({"ready": True, "extensions": {name: ext.spec() for name, ext in extensions.items()},
           "load_errors": load_errors, "metadata": meta})
    reader = SharedFrameReader()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request.get("op", "infer")
                if op == "close":
                    break
                if op == "reset":
                    for extension in extensions.values():
                        if hasattr(extension, "reset"):
                            extension.reset()
                    reply({"reset": True})
                    continue
                started = time.perf_counter()
                frames = reader.read(request)
                read_done = time.perf_counter()
                results, timings = {}, {"read": (read_done - started) * 1000}
                for name, extension in extensions.items():
                    step = time.perf_counter()
                    try:
                        with contextlib.redirect_stdout(sys.stderr):
                            output = extension.observe(frames, request["timestamp"], request.get("people"))
                        results[name] = {"score": float(output.get("score", 0.0)),
                                         "boxes": [list(map(float, box)) for box in output.get("boxes", [])],
                                         "details": output.get("details", {})}
                    except Exception as error:
                        traceback.print_exc(file=sys.stderr)
                        results[name] = {"error": f"{type(error).__name__}: {error}"}
                    timings[name] = (time.perf_counter() - step) * 1000
                timings["total"] = (time.perf_counter() - started) * 1000
                reply({"timestamp": request["timestamp"], "extensions": results, "timings_ms": timings})
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply({"error": f"{type(error).__name__}: {error}"})
    finally:
        reader.close()


if __name__ == "__main__":
    main()
