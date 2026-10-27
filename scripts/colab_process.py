"""Bounded subprocess logging with cleanup on notebook interruption."""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def run_logged(command, log_path, timeout, revision=None):
    if timeout <= 0:
        raise ValueError("Positive subprocess timeout required")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=os.name == "posix",
    )
    expired = threading.Event()

    def stop(hard=False):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
            elif process.poll() is None:
                process.kill() if hard else process.terminate()
        except ProcessLookupError:
            pass

    def timeout_stop():
        expired.set()
        stop()
        fallback = threading.Timer(5, lambda: stop(True))
        fallback.daemon = True
        fallback.start()

    watchdog = threading.Timer(timeout, timeout_stop)
    watchdog.daemon = True
    watchdog.start()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nREVISION {revision or 'unspecified'} | {time.ctime()}\n")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        return_code = process.wait()
        if return_code != 0 or expired.is_set():
            raise RuntimeError(
                f"Subprocess stopped ({return_code}, timeout={expired.is_set()}); inspect {log_path}. No automatic retry."
            )
    except BaseException:
        stop()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop(True)
            process.wait(timeout=5)
        raise
    finally:
        watchdog.cancel()
        process.stdout.close()
