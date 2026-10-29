import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from action_engine import AUX_NAMES, InteractionHead, InteractionHeadV2, SOURCE_SHA, SPEC, digest
import train_warmstart_v4 as trainer
from train_interaction import write


@pytest.mark.parametrize("version", [1, 2])
def test_warm_start_preserves_parent_logits_and_learns_context(version):
    torch.manual_seed(7)
    old = (InteractionHead() if version == 1 else InteractionHeadV2()).eval()
    old.mean.copy_(torch.randn(4096))
    old.std.copy_(torch.rand(4096) + .2)
    original = copy.deepcopy(old.state_dict())
    new = trainer.warm_start(original, version).eval()
    bag, aux = torch.randn(5, 4096), torch.randn(5, len(AUX_NAMES))
    expected = old(bag) if version == 1 else old(bag, aux)
    torch.testing.assert_close(new(bag, aux), expected, atol=1e-6, rtol=1e-5)
    new(bag, aux).sum().backward()
    assert new.layers[0].weight.grad[:, 4096:].abs().sum() > 0
    for key, value in old.state_dict().items():
        assert torch.equal(value, original[key])


def test_close_person_constraint_cannot_hide_inside_many_easy_negatives():
    rows = [{"label": 0, "action": "normal", "people": 2, "group": str(i)} for i in range(30)]
    rows += [{"label": 0, "action": "stand_close", "people": 2, "group": "close"}]
    rows += [{"label": 1, "action": "fight", "people": 2, "group": str(i)} for i in range(10)]
    point = trainer.select_threshold(rows, [.01] * 30 + [.95] + [.8] * 10)
    assert not point["target_feasible"]
    rows[30]["action"] = "uncertain"
    assert not trainer.select_threshold(rows, [.01] * 31 + [.8] * 10)["target_feasible"]


def test_tiny_training_bundle_and_frozen_resume(tmp_path, monkeypatch):
    """Exercise optimizer, checkpoint, calibration, test export, and test-only retry."""
    torch.set_num_threads(1)
    paths = {name: tmp_path / name for name in ("pack", "weights", "cache", "output")}
    for path in paths.values():
        path.mkdir()
    parent = InteractionHead()
    torch.save(parent.state_dict(), paths["weights"] / "interaction.pt")
    write(paths["weights"] / "interaction.json", {
        "source_sha256": SOURCE_SHA, "spec": SPEC, "threshold": .5,
        "checkpoint_sha256": digest(paths["weights"] / "interaction.pt")})
    sources, cached_rows = [], []
    for split in ("train", "development", "calibration", "test"):
        for label in (0, 1):
            video = f"{split}/{label}"
            sources.append({"video": video, "split": split, "group": video,
                            "dataset": "sbu", "environment": "indoor", "duration": 4,
                            "intervals": [{"start_seconds": 0, "end_seconds": 4,
                                           "action": "fight" if label else "stand_close"}]})
            cached_rows.append({"video": video, "split": split, "group": video,
                                "dataset": "sbu", "duration": 4, "timestamp": 2,
                                "label": label, "people": 2, "action": "fight" if label else "stand_close",
                                "bag": torch.randn(2, 4096), "aux": torch.randn(2, len(AUX_NAMES))})
    manifest = tmp_path / "manifest.json"
    write(manifest, {"schema_version": 2, "scope_profile": "indoor_room_fight_v2",
                     "readiness": "ready", "sources": sources})
    write(paths["pack"] / "PACKAGE.json", {"manifest_sha256": digest(manifest)})
    monkeypatch.setattr(trainer, "ActionEngine", lambda **kw: object())
    monkeypatch.setattr(trainer, "cache_sources", lambda engine, sources, *args:
                        [r for r in cached_rows if r["video"] in {s["video"] for s in sources}])
    monkeypatch.setattr(trainer, "load_features", lambda rows, *args: rows)
    args = SimpleNamespace(**paths, manifest=manifest, device="cpu", epochs=2, minutes=5, lr=.0001)
    original_write = trainer.write
    interrupted = False

    def interrupt_after_epoch(path, value):
        nonlocal interrupted
        if Path(path).name == "history.json" and not interrupted:
            interrupted = True
            raise RuntimeError("Simulated runtime interruption")
        original_write(path, value)

    monkeypatch.setattr(trainer, "write", interrupt_after_epoch)
    with pytest.raises(RuntimeError, match="Simulated"):
        trainer.run(args)
    saved = torch.load(paths["output"] / "last.pt", weights_only=True)
    assert len(saved["history"]) == 1
    first_epoch_loss = saved["history"][0]["loss"]
    monkeypatch.setattr(trainer, "write", original_write)
    trainer.run(args)
    bundle = paths["output"] / "mmaction_warmstart_v4_pilot.zip"
    assert bundle.is_file()
    report = json.loads((paths["output"] / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["training"]["head_version"] == 2
    assert report["training"]["approved_for_demo"] is False
    assert report["test"]["warmstart"]["close_people"]["windows"] == 1
    history = json.loads((paths["output"] / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[0]["loss"] == first_epoch_loss
    checkpoint = digest(paths["output"] / "interaction.pt")
    bundle.unlink()
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("Optimizer ran after selection freeze"))
    trainer.run(args)
    assert bundle.is_file()
    assert digest(paths["output"] / "interaction.pt") == checkpoint
    args.lr = .0002
    with pytest.raises(ValueError, match="fingerprint"):
        trainer.run(args)


def test_temporal_metric_reports_missing_mixed_videos_explicitly():
    rows = [{"video": "one", "label": 1, "aux": torch.zeros(2, len(AUX_NAMES))}]
    report = trainer.temporal_report(rows, [.9])
    assert report["evaluable_videos"] == 0
    assert report["median_within_video_auc"] is None
