import pytest
import torch

from scripts.fight_refinement_audit import (
    load_checkpoint,
    non_test_rows,
    validate_source,
)


def test_audit_excludes_training_and_test_before_any_scoring():
    rows = [
        {"split": split} for split in ("train", "development", "calibration", "test")
    ]
    assert non_test_rows(rows) == rows[1:3]
    with pytest.raises(ValueError, match="Both non-test"):
        non_test_rows([{"split": "test"}])


def test_audit_requires_the_exact_frozen_annotations_and_source_manifest():
    ready = {
        "annotations_sha256": "labels",
        "source_manifest_sha256": "manifest",
        "ready_for_temporal_pilot": True,
    }
    source = {"source_classifier_sha256": "model", "readiness": dict(ready)}
    validate_source(source, ready, "model")
    with pytest.raises(ValueError, match="classifier"):
        validate_source(source, ready, "another")
    for key in ("annotations_sha256", "source_manifest_sha256"):
        with pytest.raises(ValueError, match=key):
            validate_source(source, {**ready, key: "changed"}, "model")


def test_audit_checkpoint_loads_strictly_and_preserves_mode(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.pt"
    torch.save({"state_dict": model.state_dict(), "use_interaction": False}, path)
    assert len(load_checkpoint(torch.nn.Linear(2, 1), path, False)) == 64
    with pytest.raises(ValueError, match="mode"):
        load_checkpoint(model, path, True)
    with pytest.raises(RuntimeError):
        load_checkpoint(torch.nn.Linear(3, 1), path, False)


def test_audit_refuses_non_finite_checkpoint_weights(tmp_path):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": model.state_dict(), "use_interaction": True}, path)
    with pytest.raises(ValueError, match="non-finite"):
        load_checkpoint(model, path, True)
