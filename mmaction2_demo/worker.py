"""Local file-backed JSON IPC; no network listener and no camera access."""

import contextlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from action_engine import ActionEngine, ROOT


def reply(value):
    sys.__stdout__.write(json.dumps(value, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def main():
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        engine = ActionEngine(tuned="--tuned" in sys.argv)
    reply({"ready": True, "metadata": engine.metadata})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            path = Path(request["frames"]).resolve()
            if not path.is_relative_to((ROOT / "results/ipc").resolve()):
                raise ValueError("Invalid local frame path")
            with np.load(path, allow_pickle=False) as source:
                frames = source["frames"]
            with contextlib.redirect_stdout(sys.stderr):
                result = engine.infer(frames)
            reply({"result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply({"error": str(error)})


if __name__ == "__main__":
    main()
