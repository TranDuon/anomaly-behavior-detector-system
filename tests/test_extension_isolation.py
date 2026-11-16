"""Extensions may add labels but can never change, delay or break the core output."""

import copy
import sys
import time

import numpy as np
import pytest

from fight import behaviors
from fight.extension_client import ExtensionClient
from fight.mmaction_runtime import MMActionRuntime
from fight.runtime import EventGate

CORE_KEYS = ("state", "event", "people", "person_boxes", "regions", "score", "actions", "posture")


def core_output(timestamp, active=("strike",)):
    return {
        "state": "fighting" if active else "normal", "event": None, "people": 2, "score": .9,
        "prediction_timestamp": timestamp, "person_boxes": [[10, 10, 50, 90], [60, 10, 100, 90]],
        "regions": [[10, 10, 50, 90]], "actions": [], "posture": {"boxes": []},
        "active_behaviors": list(active),
        "action_states": {"strike": "fighting" if "strike" in active else "normal", "grapple": "normal",
                          "fall": "normal", "fast_running": "normal"},
        "action_scores": {"strike": .9, "grapple": .1, "fall": 0., "fast_running": 0.},
        "action_regions": {"strike": [[10, 10, 50, 90]], "fall": [], "grapple": [], "fast_running": []},
        "timings": {"pipeline_seconds": .2},
    }


def core_view(output):
    view = {key: copy.deepcopy(output.get(key)) for key in CORE_KEYS}
    for key in ("action_states", "action_scores", "action_regions"):
        view[key] = {k: v for k, v in output[key].items() if k in behaviors.CORE_NAMES}
    view["active_behaviors"] = [b for b in output["active_behaviors"] if b in behaviors.CORE_NAMES]
    return view


class FakeClient:
    def __init__(self, replies=None, enabled=True, explode=False):
        self.replies = replies or {}
        self._enabled = enabled
        self.explode = explode
        self.submitted = []
        self.error = None

    @property
    def enabled(self):
        return self._enabled

    def submit(self, frames, timestamp, people=None):
        self.submitted.append(timestamp)
        return True

    def latest(self, now):
        if self.explode:
            raise RuntimeError("boom")
        reply = self.replies.get(round(now, 3))
        if reply is None:
            return None
        return reply if 0 <= now - reply["timestamp"] <= 1.5 else None

    def reset(self):
        pass

    def status(self):
        return {"enabled": self._enabled}

    def close(self):
        pass


def make_runtime(client, alarm=True):
    runtime = MMActionRuntime(allow_pilot=True, separate_actions=True, extensions=True,
                              extension_names=["fake_ext"])
    runtime.extension_client = client
    runtime.extension_specs = {"fake_ext": {"threshold": .6, "confirm_seconds": .5,
                                            "release_seconds": .25, "alarm_enabled": alarm}}
    runtime.extension_gates = {"fake_ext": EventGate(.6, .42, confirm_seconds=.5, release_seconds=.25)}
    runtime.samples.extend(np.zeros((9, 8, 8, 3), np.uint8))
    runtime.sample_times.extend([i / 3.75 for i in range(9)])
    return runtime


def test_base_backend_is_untouched_by_extension_code():
    runtime = MMActionRuntime(allow_pilot=True, separate_actions=True)
    assert runtime.backend_name == "mmaction_separate_actions"
    assert not runtime.extensions_enabled and runtime.extension_status() is None
    assert runtime.behavior_names == behaviors.CORE_NAMES
    with pytest.raises(ValueError):
        MMActionRuntime(allow_pilot=True, extensions=True)


@pytest.mark.parametrize("reply", [
    None,                                                                  # worker silent / slow
    {"timestamp": 3.0, "extensions": {"fake_ext": {"error": "crashed"}}},  # extension crashed
    {"timestamp": 3.0, "extensions": {"other": {"score": 1., "boxes": []}}},  # unknown name
    {"timestamp": 3.0, "extensions": {"fake_ext": {"score": .99, "boxes": [[0, 0, 5, 5]]}}},  # real alarm
])
def test_core_fields_identical_whatever_the_extension_returns(reply):
    runtime = make_runtime(FakeClient({3.0: reply} if reply else {}))
    output = core_output(3.0)
    before = core_view(output)
    runtime.apply_extensions(output, 3.0)
    assert core_view(output) == before
    assert output["active_behaviors"][: len(before["active_behaviors"])] == before["active_behaviors"]
    assert "fake_ext" in output["action_states"] and "extensions" in output


def test_extension_gate_confirms_then_appends_after_core_labels():
    stamps = [round(3 + i * .2667, 3) for i in range(6)]
    replies = {t: {"timestamp": t, "extensions": {"fake_ext": {"score": .95, "boxes": [[1, 2, 3, 4]]}}} for t in stamps}
    runtime = make_runtime(FakeClient(replies))
    states = []
    for t in stamps:
        output = core_output(t)
        runtime.apply_extensions(output, t)
        states.append(output["action_states"]["fake_ext"])
    assert states[0] == "verifying" and states[-1] == "fighting"
    assert output["active_behaviors"] == ["strike", "fake_ext"]
    assert output["action_regions"]["fake_ext"] == [[1, 2, 3, 4]]
    assert output["state"] == "fighting"  # overall state still comes from the core


def test_alarm_disabled_extension_never_enters_active_behaviors():
    stamps = [round(3 + i * .2667, 3) for i in range(6)]
    replies = {t: {"timestamp": t, "extensions": {"fake_ext": {"score": .95, "boxes": []}}} for t in stamps}
    runtime = make_runtime(FakeClient(replies), alarm=False)
    for t in stamps:
        output = core_output(t, active=())
        runtime.apply_extensions(output, t)
    assert output["action_states"]["fake_ext"] == "fighting"
    assert output["active_behaviors"] == [] and output["state"] == "normal"


def test_stale_extension_result_is_dropped():
    runtime = make_runtime(FakeClient({5.0: {"timestamp": 3.0, "extensions": {"fake_ext": {"score": .99, "boxes": []}}}}))
    output = core_output(5.0)
    runtime.apply_extensions(output, 5.0)
    assert output["action_states"]["fake_ext"] == "normal"


def test_cadence_submits_every_second_step_and_slows_down_with_slow_core():
    client = FakeClient()
    runtime = make_runtime(client)
    for i in range(6):
        runtime.apply_extensions(core_output(3 + i * .2667), 3 + i * .2667)
    assert len(client.submitted) == 3
    slow = core_output(10.)
    slow["timings"]["pipeline_seconds"] = .6
    runtime.apply_extensions(slow, 10.)
    assert runtime.extension_cadence == 4


def test_extension_failure_inside_push_never_reaches_core(monkeypatch):
    runtime = make_runtime(FakeClient(explode=True))
    runtime.model_ready = True
    expected = core_output(3.0)
    monkeypatch.setattr(runtime, "push_separate", lambda *a, **k: copy.deepcopy(expected))
    output = runtime.push(np.zeros((8, 8, 3), np.uint8), 3.0)
    assert core_view(output) == core_view(expected)
    assert "Extension merge failed" in runtime.extension_client.error


def test_missing_extension_environment_only_disables_extensions(tmp_path):
    client = ExtensionClient(["selftest"], python=tmp_path / "missing" / "python.exe")
    assert client.start() is False and "missing" in client.error
    assert client.submit(np.zeros((9, 8, 8, 3), np.uint8), 1.0) is False
    assert client.latest(1.0) is None
    client.close()


def test_real_worker_round_trip_and_crash_are_contained():
    client = ExtensionClient(["selftest", "does_not_exist"], python=sys.executable)
    try:
        assert client.start(), client.error
        assert "does_not_exist" in client.load_errors and "selftest" in client.specs
        frames = np.zeros((9, 24, 32, 3), np.uint8)
        frames[-1] += 32
        assert client.submit(frames, 2.0, {"person_boxes": [[1, 1, 9, 9]]})
        deadline = time.monotonic() + 20
        while client.latest(2.0) is None and time.monotonic() < deadline:
            time.sleep(.05)
        reply = client.latest(2.0)
        assert reply["extensions"]["selftest"]["score"] == pytest.approx(.5)
        client.process.kill()
        client.process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while client.enabled and time.monotonic() < deadline:
            time.sleep(.05)
        assert not client.enabled
        assert client.submit(frames, 3.0) is False and client.latest(3.0) is None
    finally:
        client.close()
