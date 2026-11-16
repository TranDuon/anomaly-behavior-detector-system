"""Asynchronous, fail-safe link from the runtime to the extension worker.

The core never waits for an extension: ``submit`` returns immediately (and
drops the step if the worker is still busy), ``latest`` returns the newest
reply if it is fresh enough. A crashed, slow or misbehaving worker only
disables extensions; nothing here raises into the core pipeline.
"""

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from mmaction2_demo.frame_transport import SharedFrameBuffer

ROOT = Path(__file__).resolve().parents[1]
EXT_ROOT = ROOT / "extensions"
DEFAULT_PYTHON = EXT_ROOT / "env" / "Scripts" / "python.exe"
STALE_SECONDS = 1.5


class ExtensionClient:
    def __init__(self, names, python=None, stale_seconds=STALE_SECONDS, ready_timeout=120):
        self.names = list(names)
        self.python = Path(python) if python else DEFAULT_PYTHON
        self.stale_seconds = stale_seconds
        self.ready_timeout = ready_timeout
        self.process = None
        self.log = None
        self.buffer = SharedFrameBuffer()
        self.lock = threading.Lock()
        self.busy = False
        self.busy_since = None
        self.reply = None
        self.specs = {}
        self.load_errors = {}
        self.metadata = {}
        self.error = None
        self.submitted = self.dropped_busy = self.received = 0
        self._ready = queue.Queue(maxsize=1)

    # ---------------------------------------------------------------- lifecycle
    @property
    def enabled(self):
        return self.process is not None and self.process.poll() is None and self.error is None

    def start(self):
        """Start the worker. Returns False (never raises) if it cannot run."""
        if self.process is not None:
            return self.enabled
        if not self.python.is_file():
            self.error = f"Extension environment missing: {self.python}"
            return False
        try:
            (ROOT / "logs").mkdir(exist_ok=True)
            self.log = (ROOT / "logs" / f"extension_worker_{uuid.uuid4().hex[:8]}.log").open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
            self.process = subprocess.Popen(
                [str(self.python), str(EXT_ROOT / "worker.py"), "--extensions", ",".join(self.names)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf-8", env=environment, cwd=ROOT,
            )
        except OSError as error:
            self.error = f"Extension worker failed to start: {error}"
            self.process = None
            return False
        threading.Thread(target=self._receive, daemon=True).start()
        try:
            ready = self._ready.get(timeout=self.ready_timeout)
        except queue.Empty:
            self._fail("Extension worker did not become ready")
            return False
        if not ready.get("ready"):
            self._fail(f"Extension worker refused to start: {ready}")
            return False
        self.specs = ready.get("extensions", {})
        self.load_errors = ready.get("load_errors", {})
        self.metadata = ready.get("metadata", {})
        return True

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"op":"close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.log is not None:
            self.log.close()
            self.log = None
        self.buffer.close()

    def _fail(self, message):
        self.error = message
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def _receive(self):
        process = self.process
        try:
            for line in process.stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if "ready" in value:
                    try:
                        self._ready.put_nowait(value)
                    except queue.Full:
                        pass
                    continue
                with self.lock:
                    self.busy = False
                    self.busy_since = None
                    if "extensions" in value and _finite_reply(value):
                        self.reply = value
                        self.received += 1
                    elif "error" in value:
                        self.reply = None
        except (OSError, ValueError):
            pass
        with self.lock:
            self.busy = False
        if self.error is None:
            self.error = "Extension worker stopped"
        try:
            self._ready.put_nowait({"ready": False})
        except queue.Full:
            pass

    # ---------------------------------------------------------------- per step
    def submit(self, frames, timestamp, people=None):
        """Send one step if the worker is idle. Never blocks, never raises."""
        if not self.enabled:
            return False
        with self.lock:
            if self.busy:
                # A reply missing for far too long means the worker hung.
                if self.busy_since is not None and time.monotonic() - self.busy_since > 10:
                    self._fail("Extension worker stopped answering")
                self.dropped_busy += 1
                return False
            self.busy = True
            self.busy_since = time.monotonic()
        try:
            request = self.buffer.write(list(frames))
            request.update(op="infer", timestamp=float(timestamp), people=people or {})
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            self.submitted += 1
            return True
        except (OSError, ValueError) as error:
            with self.lock:
                self.busy = False
            self._fail(f"Extension worker unreachable: {error}")
            return False

    def latest(self, now):
        """Newest reply no older than ``stale_seconds`` source time, else None."""
        with self.lock:
            reply = self.reply
        if reply is None or not self.enabled:
            return None
        age = now - reply["timestamp"]
        if age < -1e-6 or age > self.stale_seconds:
            return None
        return reply

    def reset(self):
        with self.lock:
            self.reply = None

    def status(self):
        return {"enabled": self.enabled, "error": self.error, "extensions": list(self.specs),
                "load_errors": self.load_errors, "metadata": self.metadata,
                "submitted": self.submitted, "received": self.received, "dropped_busy": self.dropped_busy}


def _finite_reply(value):
    try:
        if not math.isfinite(float(value["timestamp"])):
            return False
        for result in value["extensions"].values():
            if "error" not in result and not (0.0 <= float(result["score"]) <= 1.0):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def current_python():
    """Interpreter for tests: run the worker with the server's own Python."""
    return Path(sys.executable)
