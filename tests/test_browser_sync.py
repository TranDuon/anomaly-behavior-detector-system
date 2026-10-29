import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fight.runtime import FightRuntime
from fight.vision import ClipConfig


def test_frontend_and_api_share_identity_and_disallow_cached_state():
    import api.fight_server as server

    with TestClient(server.app) as client:
        page = client.get("/")
        status = client.get("/api/status")
        identity = status.json()["server"]
        assert identity["process_id"] == os.getpid()
        assert identity["project_dir"] == str(server.BASE_DIR)
        assert (
            re.search(r'data-ui-revision="([a-f0-9]+)"', page.text).group(1)
            == identity["ui_revision"]
        )
        assert (
            re.search(r'data-server-instance="([a-f0-9]+)"', page.text).group(1)
            == identity["instance_id"]
        )
        assert "__FIGHTWATCH_UI_VERSION__" not in page.text
        for response in (
            page,
            status,
            client.get("/api/events"),
            client.get("/api/report"),
        ):
            assert "no-store" in response.headers["cache-control"]
            assert response.headers["x-fightwatch-instance"] == identity["instance_id"]


def test_ui_file_change_changes_the_revision_used_by_reload_guard(
    tmp_path, monkeypatch
):
    import api.fight_server as server

    page = tmp_path / "fight.html"
    page.write_text(
        '<html data-ui-revision="__FIGHTWATCH_UI_VERSION__">one</html>',
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "UI_PATH", page)
    before = server.server_identity()["ui_revision"]
    page.write_text(
        '<html data-ui-revision="__FIGHTWATCH_UI_VERSION__">two</html>',
        encoding="utf-8",
    )
    after = server.server_identity()["ui_revision"]
    assert before != after
    assert after in server.frontend().body.decode()


def test_start_keeps_events_in_separate_sessions(monkeypatch):
    import api.fight_server as server

    runtime = SimpleNamespace(model_ready=True, warmup=lambda: None)
    monkeypatch.setattr(server, "FightRuntime", lambda *args, **kwargs: runtime)
    processor = server.FightProcessor()
    monkeypatch.setattr(processor, "run", lambda source: None)
    processor.events = [{"id": "old"}]
    processor.start("first")
    processor.thread.join()
    first_id = processor.session_id
    assert processor.events == []
    processor.events.append({"id": "first-event"})
    processor.start("second")
    processor.thread.join()
    assert processor.session_id != first_id
    assert processor.events == []


def test_rejected_start_from_another_tab_does_not_stop_or_replace_active_model(
    monkeypatch,
):
    import api.fight_server as server

    original = SimpleNamespace(model_ready=True, message="loaded")
    monkeypatch.setattr(server, "FightRuntime", lambda *args, **kwargs: original)
    processor = server.FightProcessor()
    processor.phase, processor.source = "processing", 0
    processor.thread = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(
        processor,
        "stop",
        lambda: pytest.fail("A rejected request must not stop the camera"),
    )
    monkeypatch.setattr(
        server,
        "FightRuntime",
        lambda *args, **kwargs: SimpleNamespace(
            model_ready=False, message="Enable pilot"
        ),
    )
    with pytest.raises(ValueError, match="Enable pilot"):
        processor.start(1, allow_pilot=False)
    assert processor.runtime is original
    assert processor.phase == "processing" and processor.source == 0
    assert processor.is_running()


def test_blocked_pilot_is_exposed_separately_from_missing_model(tmp_path, monkeypatch):
    import api.fight_server as server

    (tmp_path / "fight_model.ts").write_bytes(b"Pilot gate runs before loading")
    (tmp_path / "fight_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": ClipConfig().to_dict(),
                "threshold": 0.8,
                "approved_for_demo": False,
            }
        ),
        encoding="utf-8",
    )
    runtime = FightRuntime(tmp_path, allow_pilot=False)
    assert runtime.pilot_required and not runtime.model_ready
    monkeypatch.setattr(server.processor, "runtime", runtime)
    model = server.processor.status()["model"]
    assert model["pilot_required"]
    assert "tích Cho phép checkpoint pilot" in model["message"]
    assert not model["approved_for_demo"]


def test_browser_state_helpers_without_network_or_camera():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed; backend/browser identity checks still run")
    path = Path(__file__).with_name("frontend_state.test.cjs")
    result = subprocess.run(
        [node, "--test", str(path)],
        cwd=path.parent.parent,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
