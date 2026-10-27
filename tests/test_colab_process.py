import sys

import pytest

from scripts.colab_process import run_logged


def test_logged_process_writes_output(tmp_path):
    log = tmp_path / "process.txt"
    run_logged(
        [sys.executable, "-c", "print('finished')"], log, timeout=5, revision="test"
    )
    assert "finished" in log.read_text() and "REVISION test" in log.read_text()


def test_logged_process_timeout_stops_child(tmp_path):
    with pytest.raises(RuntimeError, match="timeout=True"):
        run_logged(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "timeout.txt",
            timeout=0.2,
        )
