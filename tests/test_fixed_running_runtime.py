from fight.mmaction_runtime import MMActionRuntime
from fight.runtime import EventGate


def runtime():
    result = MMActionRuntime(fixed_running=True, allow_pilot=True)
    result.gate = EventGate(0.83, 0.58, confirm_seconds=0.4, release_seconds=0.25)
    result.running_gate = EventGate(0.16, 0.11, confirm_seconds=0.4, release_seconds=0.25)
    return result


def output(fight=0.01, running=0.01, people=1):
    return {"score": fight, "running_score": running, "people": people,
            "regions": [[1, 2, 3, 4]], "running_regions": [[5, 6, 7, 8]]}


def test_running_does_not_get_a_fight_label():
    model = runtime()
    assert model.update_decision(output(running=0.8), 0) == ("verifying", None)
    prediction = output(running=0.8)
    assert model.update_decision(prediction, 0.54) == ("fighting", "started")
    assert prediction["active_behaviors"] == ["fast_running"]
    assert prediction["regions"] == [[5, 6, 7, 8]]
    assert prediction["fight_state"] == "normal"


def test_independent_gates_keep_existing_fight_and_release_running():
    model = runtime()
    model.update_decision(output(fight=0.9, running=0.8), 0)
    prediction = output(fight=0.9, running=0.8)
    assert model.update_decision(prediction, 0.54) == ("fighting", "started")
    assert prediction["active_behaviors"] == ["fight_wrestle_fall", "fast_running"]
    model.update_decision(output(fight=0.9), 1.08)
    prediction = output(fight=0.9)
    assert model.update_decision(prediction, 1.62) == ("fighting", None)
    assert prediction["active_behaviors"] == ["fight_wrestle_fall"]


def test_lost_person_ends_running_and_reset_clears_both_gates():
    model = runtime()
    model.update_decision(output(running=0.8), 0)
    model.update_decision(output(running=0.8), 0.54)
    assert model.update_decision(output(people=0), 1.08) == ("insufficient", "ended")
    assert not model.running_gate.active
    model.update_decision(output(running=0.8), 1.62)
    model.reset()
    assert model.running_gate.pending_since is None
    assert model.result["active_behaviors"] == []


def test_rejected_running_evidence_clears_only_running_alarm():
    model = runtime()
    model.update_decision(output(running=0.8), 0)
    model.update_decision(output(running=0.8), 1.08)
    rejected = output(running=0.0)
    rejected['running_guard_version'] = 2
    assert model.update_decision(rejected, 1.62) == ('normal', 'ended')
    assert not model.running_gate.active


def test_one_running_spike_does_not_start_an_alarm():
    model = runtime()
    model.running_gate = EventGate(0.16, 0.11, confirm_seconds=0.8, release_seconds=0.25)
    assert model.update_decision(output(running=0.8), 0) == ("verifying", None)
    assert model.update_decision(output(), 0.54) == ("normal", None)
    assert not model.alarm_active
