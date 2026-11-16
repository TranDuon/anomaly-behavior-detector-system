"""Smoke test: start the extension worker with this interpreter and run one step."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fight.extension_client import ExtensionClient  # noqa: E402


def main():
    client = ExtensionClient(["selftest"], python=sys.executable)
    try:
        if not client.start():
            raise SystemExit(f"worker failed: {client.error}")
        frames = np.zeros((9, 72, 96, 3), dtype=np.uint8)
        frames[-1] += 64
        client.submit(frames, 1.0, {"person_boxes": [[1, 2, 30, 40]]})
        deadline = time.monotonic() + 20
        while client.latest(1.0) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        reply = client.latest(1.0)
        if reply is None:
            raise SystemExit(f"no reply: {client.status()}")
        print("extension worker OK:", reply["extensions"], client.metadata)
    finally:
        client.close()


if __name__ == "__main__":
    main()
