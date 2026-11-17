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
    runtime.samples.extend(np.stack([np.full((8, 8, 3), 25 * i, np.uint8) for i in range(9)]))  # moving scene
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
    slow["inference_seconds"] = .3
    runtime.apply_extensions(slow, 10.)
    assert runtime.extension_cadence == 3


def test_vandalism_gate_uses_raw_score():
    # probe_02_ft scores are ~0 or ~1: the gate takes the raw score (smooth_window 1), so an
    # isolated spike is not averaged away (it still needs the confirmation time to alarm).
    stamps = [round(3 + i * .2667, 3) for i in range(6)]
    pattern = [.95, .1, .1, .95, .1, .1]
    replies = {t: {"timestamp": t, "extensions": {"vandalism": {"score": s, "boxes": []}}} for t, s in zip(stamps, pattern)}
    runtime = make_runtime(FakeClient(replies))
    runtime.extension_specs = {"vandalism": runtime.extension_specs["fake_ext"]}
    runtime.extension_gates = {"vandalism": runtime.extension_gates["fake_ext"]}
    gated = []
    for t in stamps:
        output = core_output(t, active=())
        runtime.apply_extensions(output, t)
        gated.append(output["extensions"]["behaviors"]["vandalism"]["gated_score"])
    assert max(gated) > .9
    assert "vandalism" not in output["active_behaviors"]  # single spikes do not confirm


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


# --------------------------------------------------------------- vandalism branch uses strike as context
from fight.mmaction_runtime import has_target_person  # noqa: E402


def lone_strike_output(timestamp, people=([10, 10, 50, 90],)):
    output = core_output(timestamp)
    output["person_boxes"] = [list(p) for p in people]
    output["people"] = len(people)
    return output


def vandalism_runtime(score):
    stamps = [round(3 + i * .2667, 3) for i in range(6)]
    replies = {t: {"timestamp": t, "extensions": {"vandalism": {"score": score, "boxes": []}}} for t in stamps}
    runtime = make_runtime(FakeClient(replies))
    runtime.extension_specs = {"vandalism": runtime.extension_specs["fake_ext"]}   # threshold .6
    runtime.extension_gates = {"vandalism": runtime.extension_gates["fake_ext"]}
    return runtime, stamps


def test_target_person_rule():
    striker = [100, 0, 250, 240]
    assert has_target_person(striker, [striker, [260, 10, 400, 240]])          # victim next to striker
    assert not has_target_person(striker, [striker])                             # alone
    assert not has_target_person(striker, [striker, [0, 140, 115, 238]])         # small partial box
    assert not has_target_person(striker, [striker, [600, 0, 750, 240]])         # person far away


@pytest.mark.parametrize("make_output,score,expect_vandalism,context", [
    (lone_strike_output, .5, True, "lone_strike"),        # .5 >= .7 x .6: lone swing supports vandalism
    (lone_strike_output, .3, False, "lone_strike"),
    (core_output, .8, False, "strike_with_victim"),       # fight with a victim: needs >= .9
    (core_output, .95, True, "strike_with_victim"),
])
def test_strike_context_only_changes_vandalism(make_output, score, expect_vandalism, context):
    runtime, stamps = vandalism_runtime(score)
    for t in stamps:
        output = make_output(t)
        before = core_view(output)
        runtime.apply_extensions(output, t)
        assert core_view(output) == before                 # strike branch untouched, every step
    assert output["extensions"]["behaviors"]["vandalism"]["context"] == context
    assert ("vandalism" in output["active_behaviors"]) is expect_vandalism
    assert output["active_behaviors"][0] == "strike"


# --------------------------------------------------------------- group-A strike context (information only)
from fight.strike_context import StrikeContext  # noqa: E402


def strike_out(people, strike=(100, 0, 200, 240), fall=()):
    return {"action_states": {"strike": "fighting"}, "person_boxes": [list(p) for p in people],
            "action_regions": {"strike": [list(strike)], "fall": [list(f) for f in fall]}}


def test_group_a_alone_and_far_are_vandalism():
    ctx = StrikeContext()
    assert ctx.observe(strike_out([(100, 0, 200, 240)]), 1.0)["verdict"] == "vandalism"            # A1
    ctx.reset()
    assert ctx.observe(strike_out([(100, 0, 200, 240), (600, 0, 700, 240)]), 1.0)["verdict"] == "vandalism"  # A3


def test_group_a_victim_that_reacts_is_fight_and_static_is_vandalism():
    striker = (100, 0, 200, 240)
    ctx = StrikeContext()
    for i, x in enumerate((210, 230, 260, 300)):                         # victim pushed away -> A5
        result = ctx.observe(strike_out([striker, (x, 0, x + 100, 240)]), 1 + i * .27)
    assert result["verdict"] == "fight"
    ctx = StrikeContext()
    for i in range(4):                                                     # bystander never moves -> A4
        result = ctx.observe(strike_out([striker, (210, 0, 310, 240)]), 1 + i * .27)
    assert result["verdict"] == "vandalism"


def test_strike_context_never_changes_core_output():
    runtime = make_runtime(FakeClient())
    output = core_output(3.0)
    before = core_view(output)
    runtime.apply_extensions(output, 3.0)
    assert core_view(output) == before and output["strike_context"]["verdict"] in {"fight", "uncertain", "vandalism"}


def test_a_plus_d1_needs_vandalism_support():
    # lone striker, vandalism silent -> stays a fight (e.g. two people sharing one box)
    runtime, stamps = vandalism_runtime(.0)
    output = lone_strike_output(stamps[0])
    runtime.apply_extensions(output, stamps[0])
    assert output["strike_context"]["group_a_verdict"] == "vandalism"
    assert output["strike_context"]["verdict"] == "fight"
    # lone striker and the vandalism branch fires -> suspected vandalism
    runtime, stamps = vandalism_runtime(.95)
    for t in stamps:
        output = lone_strike_output(t)
        runtime.apply_extensions(output, t)
    assert output["strike_context"]["verdict"] == "vandalism"
    assert output["active_behaviors"][0] == "strike"          # strike itself still reported


def test_presented_labels_replace_strike_but_detection_keeps_it():
    runtime, stamps = vandalism_runtime(.95)
    for t in stamps:
        output = lone_strike_output(t)
        runtime.apply_extensions(output, t)
    assert "strike" in output["active_behaviors"]                     # detection unchanged
    assert "strike" not in output["presented_behaviors"] and "vandalism" in output["presented_behaviors"]
    assert output["presented_regions"]["strike"] == []
    assert output["presented_regions"]["vandalism"]                  # striker box now shown as vandalism


def test_presented_labels_keep_real_fight():
    runtime, stamps = vandalism_runtime(.0)
    for t in stamps:
        output = core_output(t)                                       # striker with a victim next to them
        runtime.apply_extensions(output, t)
    assert output["presented_behaviors"][0] == "strike"


def test_annotate_and_record_use_presented_labels(tmp_path):
    import numpy as np
    from api.fight_server import annotate
    frame = np.zeros((120, 160, 3), np.uint8)
    result = {"state": "fighting", "behavior_scope": "three_actions", "active_behaviors": ["strike"],
              "presented_behaviors": ["vandalism"], "presented_regions": {"vandalism": [[10, 10, 50, 90]]},
              "person_boxes": [[10, 10, 50, 90]], "regions": [[10, 10, 50, 90]], "actions": []}
    assert annotate(frame, result).shape == frame.shape


def test_m1_still_people_do_not_count_as_vandalism():
    runtime, stamps = vandalism_runtime(.95)
    runtime.samples.clear()
    runtime.samples.extend(np.zeros((9, 8, 8, 3), np.uint8))           # nobody moves
    for t in stamps:
        output = core_output(t, active=())
        runtime.apply_extensions(output, t)
    assert output["extensions"]["behaviors"]["vandalism"]["motion"] == 0.
    assert "vandalism" not in output["active_behaviors"]


def test_m2_quick_release_after_four_low_scores():
    stamps = [round(3 + i * .2667, 3) for i in range(14)]
    pattern = [.95] * 6 + [.1, .5, .1, .5, .1, .5, .1, .5]             # low with spikes, never 1 s continuous
    replies = {t: {"timestamp": t, "extensions": {"vandalism": {"score": s, "boxes": []}}} for t, s in zip(stamps, pattern)}
    runtime = make_runtime(FakeClient(replies))
    runtime.extension_specs = {"vandalism": runtime.extension_specs["fake_ext"]}
    runtime.extension_gates = {"vandalism": runtime.extension_gates["fake_ext"]}
    seen = []
    for t in stamps:
        output = core_output(t, active=())
        runtime.apply_extensions(output, t)
        seen.append(output["action_states"]["vandalism"])
    assert "fighting" in seen[:7]
    assert seen[-1] != "fighting"                                       # released despite the spikes
