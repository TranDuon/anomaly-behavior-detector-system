import random

import pytest
import torch
from torch import nn

from scripts import fight_classroom_refine as repair


def report(precision=0.8, false_alerts=1, ap=1.0):
    return {
        "events": {
            "public": {
                "precision": precision,
                "recall": 1.0,
                "false_alerts_per_normal_minute": false_alerts,
                "normal_alert_fraction": 0.1,
            }
        },
        "windows": {"macro_ap": ap, "cohorts": {"public": {"ap": ap}}},
    }


def test_selection_uses_continuous_events_not_just_confidence():
    initial = report()
    assert repair.event_rank(report(false_alerts=0)) > repair.event_rank(initial)
    assert repair.event_rank(report(precision=0.7, false_alerts=0)) < repair.event_rank(
        initial
    )
    assert not repair.protected(report(ap=0.9), initial)


def test_balanced_order_rejects_holdouts_and_balances_performance_groups():
    rows = [{"video": "one", "dataset": "public", "label": 0, "split": "train"}]
    with pytest.raises(ValueError, match="training sources only"):
        repair.balanced_order([{**rows[0], "split": "test"}], random.Random(1))
    selected = repair.balanced_order(rows, random.Random(1), exposures=8)
    assert len(selected) == 8


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)

    def set_stage(self, stage):
        assert stage == "finetune"


def test_resume_finishes_pending_evaluation_without_repeating_optimizer_epoch(
    tmp_path, monkeypatch
):
    calls = []

    def epoch(*args):
        calls.append("epoch")
        if len(calls) > 1:
            raise TimeoutError("stop after verifying resume")
        return 0.2, 8

    evaluations = []

    def evaluate(model, deadline):
        evaluations.append("eval")
        if len(evaluations) == 2:
            raise TimeoutError("interrupted event scoring")
        return report(precision=0.9 if len(evaluations) > 2 else 0.8)

    monkeypatch.setattr(repair, "train_epoch", epoch)
    model = TinyModel()
    arguments = (
        model,
        {"train": []},
        tmp_path,
        tmp_path,
        "cpu",
        lambda: float("inf"),
        lambda: float("inf"),
        evaluate,
        {"source": "sha"},
    )
    with pytest.raises(TimeoutError, match="interrupted event"):
        repair.train(*arguments, resume=False)
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert state["pending"] and state["epoch"] == 0
    with pytest.raises(TimeoutError, match="verifying resume"):
        repair.train(*arguments, resume=True)
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert not state["pending"] and state["epoch"] == 1
    assert state["best"]["events"]["public"]["precision"] == 0.9
    assert len(evaluations) == 3
    with pytest.raises(ValueError, match="identity differs"):
        repair.train(*arguments[:-1], {"source": "changed"}, resume=True)


def test_close_view_cannot_use_stale_pose_coordinates(tmp_path):
    model = TinyModel()
    model.use_interaction = True
    with pytest.raises(ValueError, match="RGB-only"):
        repair.train_epoch(
            model, [], tmp_path, None, "cpu", random.Random(1), float("inf")
        )
