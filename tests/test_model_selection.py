"""Public model selection and installation must not activate retired branches."""

import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import zipfile

import pytest
import torch


def test_web_exposes_only_the_three_supported_backends():
    from fastapi.testclient import TestClient
    import api.fight_server as server

    with TestClient(server.app) as client:
        page = client.get("/")
    selector = re.search(r'<select id="model-backend">(.*?)</select>', page.text)
    assert selector is not None
    assert re.findall(r'<option value="([^"]+)"', selector.group(1)) == [
        "legacy", "mmaction_original", "mmaction_interaction_v1"
    ]
    assert "FightWatch X3D-S + pose" in selector.group(1)


@pytest.mark.parametrize(
    "backend,tuned",
    [("legacy", None), ("mmaction_original", False), ("mmaction_interaction_v1", True)],
)
def test_backend_routes_without_replacing_active_session(monkeypatch, backend, tuned):
    import api.fight_server as server
    import fight.mmaction_runtime as mmaction

    processor = server.FightProcessor()
    previous = processor.runtime
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_ready=False, message="not installed")

    monkeypatch.setattr(server, "FightRuntime", unavailable)
    monkeypatch.setattr(mmaction, "MMActionRuntime", unavailable)
    monkeypatch.setattr(processor, "stop", lambda: pytest.fail("must keep session"))
    with pytest.raises(ValueError, match="not installed"):
        processor.start("unused.mp4", backend=backend)
    assert processor.runtime is previous
    assert len(calls) == 1
    if tuned is not None:
        assert calls[0] == {"tuned": tuned, "allow_pilot": False}


@pytest.mark.parametrize("backend", ["mmaction_tuned", "mmaction_tuned_refined"])
def test_retired_backends_are_rejected_before_stopping_session(monkeypatch, backend):
    import api.fight_server as server

    processor = server.FightProcessor()
    monkeypatch.setattr(processor, "stop", lambda: pytest.fail("must keep session"))
    with pytest.raises(ValueError, match="Model"):
        processor.start("unused.mp4", backend=backend)


def installer(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "mmaction2_demo"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("v1_installer", directory / "install_branch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(module, tmp_path, *, version=1, corrupt=False):
    checkpoint = tmp_path / "candidate.pt"
    torch.save(module.InteractionHead().state_dict(), checkpoint)
    metadata = {
        "source_sha256": module.SOURCE_SHA,
        "spec": module.SPEC,
        "checkpoint_sha256": module.digest(checkpoint),
        "head_version": version,
        "private_data_used": False,
        "threshold": 0.33,
    }
    report = {"status": "completed", "training": metadata}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("interaction.pt", b"corrupt" if corrupt else checkpoint.read_bytes())
        package.writestr("interaction.json", json.dumps(metadata))
        package.writestr("report.json", json.dumps(report))
    return archive, metadata


def test_verified_v1_bundle_installs_report_and_preserves_original_weights(tmp_path, monkeypatch):
    module = installer(monkeypatch)
    archive, metadata = make_bundle(module, tmp_path)
    root = tmp_path / "installation"
    (root / "weights").mkdir(parents=True)
    original = root / "weights/action_slowonly_r101.pth"
    original.write_bytes(b"unchanged original")
    module.install(archive, root=root)
    assert module.digest(root / "weights/interaction.pt") == metadata["checkpoint_sha256"]
    assert json.loads((root / "results/interaction_v1_report.json").read_text())["status"] == "completed"
    assert original.read_bytes() == b"unchanged original"


@pytest.mark.parametrize("version,corrupt", [(2, False), (1, True)])
def test_bad_bundle_cannot_replace_installed_model(tmp_path, monkeypatch, version, corrupt):
    module = installer(monkeypatch)
    archive, _ = make_bundle(module, tmp_path, version=version, corrupt=corrupt)
    root = tmp_path / "installation"
    (root / "weights").mkdir(parents=True)
    previous = root / "weights/interaction.pt"
    previous.write_bytes(b"keep installed checkpoint")
    with pytest.raises(ValueError):
        module.install(archive, root=root)
    assert previous.read_bytes() == b"keep installed checkpoint"
