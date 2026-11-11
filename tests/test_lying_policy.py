from pathlib import Path

import pytest

from mmaction2_demo.lying_policy import LYING_THRESHOLD, compatible_engine_hashes, lying_evidence


def test_already_lying_does_not_require_a_fall_transition():
    person = {'box': [0, 0, 100, 30], 'ava_lying_score': .85, 'ava_fall_score': .01}
    assert lying_evidence([person]) == (.85, [person['box']])


def test_sitting_or_bending_does_not_trigger_from_box_or_old_fall_score():
    people = [{'box': [0, 0, 100, 30], 'ava_lying_score': .1, 'ava_fall_score': .95},
              {'box': [120, 0, 150, 100], 'ava_lying_score': .04}]
    assert lying_evidence(people) == (.1, [])
    assert lying_evidence([]) == (0., [])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True, None, '0.9'])
def test_invalid_scores_cannot_trigger(value):
    assert lying_evidence([{'box': [0, 0, 100, 30], 'ava_lying_score': value}]) == (0., [])


def test_regions_only_include_the_lying_person():
    standing = {'box': [0, 0, 10, 30], 'ava_lying_score': .1}
    lying = {'box': [30, 20, 80, 40], 'ava_lying_score': LYING_THRESHOLD}
    assert lying_evidence([standing, lying])[1] == [lying['box']]


def test_fingerprint_compatibility_is_limited_to_the_additive_output():
    root = Path(__file__).resolve().parents[1] / 'mmaction2_demo'
    hashes = compatible_engine_hashes(root/'action_engine.py')
    assert '236bb65ef07297e043efa0fdba168d83d9f9600fda86763acaba81d89de0e39e' in hashes


def test_lying_alert_confirms_and_clears_independently():
    from fight.mmaction_runtime import MMActionRuntime
    from fight.runtime import EventGate
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.action_gates = {name: EventGate(.5, .35, confirm_seconds=.4, release_seconds=.25)
                            for name in ('strike', 'grapple', 'fall', 'fast_running')}
    def output(lying):
        return {'people': 2, 'score': lying, 'regions': [],
                'action_scores': {'strike': .01, 'grapple': .01, 'fall': lying, 'fast_running': 0.},
                'action_regions': {'strike': [], 'grapple': [], 'fall': [[0, 0, 100, 30]], 'fast_running': []}}
    first = output(.9)
    assert runtime.update_decision(first, 0.)[0] == 'verifying'
    second = output(.9)
    assert runtime.update_decision(second, .54)[0] == 'fighting'
    assert second['active_behaviors'] == ['fall']
    assert runtime.update_decision(output(0.), 1.08)[0] == 'normal'


def posture_runtime():
    from fight.mmaction_runtime import MMActionRuntime
    from fight.runtime import EventGate
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.action_gates = {name: EventGate(.4, .28, confirm_seconds=.4, release_seconds=.25)
                            for name in ('strike', 'grapple', 'fall', 'fast_running')}
    return runtime


def posture_output(score, box):
    return {'people': 1, 'score': score, 'regions': [],
            'actions': [{'box': box, 'ava_lying_score': score}],
            'action_scores': {'strike': .01, 'grapple': .01, 'fall': score, 'fast_running': 0.},
            'action_regions': {'strike': [], 'grapple': [],
                               'fall': [box] if score >= .4 else [], 'fast_running': []}}


def test_recorded_bed_clip_high_then_weak_same_person_confirms_without_lowering_on_threshold():
    runtime = posture_runtime()
    first = posture_output(.7047739625, [25.1169, 267.0089, 383.514, 434.6419])
    assert runtime.update_decision(first, 2.141027715)[0] == 'verifying'
    second = posture_output(.3359889984, [20.2967, 313.8716, 400.697, 460.4281])
    assert runtime.update_decision(second, 2.676284644)[0] == 'fighting'
    assert second['action_scores']['fall'] == .3359889984
    assert second['action_thresholds']['fall'] == .4
    assert second['fall_confirmation']['continuation']
    assert second['regions'] == [second['actions'][0]['box']]


@pytest.mark.parametrize('score,box', [(.2, [0, 0, 100, 30]),
                                       (.33, [300, 0, 400, 30]),
                                       (.33, [0, 0, 30, 100])])
def test_weak_unsupported_or_different_person_cannot_finish_confirmation(score, box):
    runtime = posture_runtime()
    runtime.update_decision(posture_output(.7, [0, 0, 100, 30]), 0.)
    result = posture_output(score, box)
    assert runtime.update_decision(result, .54)[0] == 'normal'


def test_persistent_weak_horizontal_detection_cannot_initiate_alert():
    runtime = posture_runtime()
    for timestamp in (0., .54, 1.08):
        assert runtime.update_decision(posture_output(.33, [0, 0, 100, 30]), timestamp)[0] == 'normal'


def test_regular_gates_still_require_two_strong_observations():
    from fight.runtime import EventGate
    gate = EventGate(.4, .28, confirm_seconds=.4)
    gate.update(.7, 0.)
    assert gate.update(.33, .54) == ('normal', None)


def test_active_lying_alert_clears_when_support_moves_to_another_person():
    runtime = posture_runtime()
    runtime.update_decision(posture_output(.7, [0, 0, 100, 30]), 0.)
    assert runtime.update_decision(posture_output(.7, [0, 0, 100, 30]), .54)[0] == 'fighting'
    assert runtime.update_decision(posture_output(.33, [300, 0, 400, 30]), 1.08) == ('normal', 'ended')
    runtime.reset()
    assert runtime.lying_support_boxes == []


def test_lying_confirmation_cannot_bridge_a_stale_observation_gap():
    runtime = posture_runtime()
    runtime.update_decision(posture_output(.7, [0, 0, 100, 30]), 0.)
    assert runtime.update_decision(posture_output(.33, [0, 0, 100, 30]), 2.)[0] == 'normal'


def test_lost_people_clear_lying_confirmation_before_reappearance():
    runtime = posture_runtime()
    runtime.update_decision(posture_output(.7, [0, 0, 100, 30]), 0.)
    missing = posture_output(0., [0, 0, 100, 30])
    missing['people'] = 0
    assert runtime.update_decision(missing, .54)[0] == 'insufficient'
    assert runtime.lying_support_boxes == []
    assert runtime.update_decision(posture_output(.33, [0, 0, 100, 30]), 1.08)[0] == 'normal'


def test_bed_clip_weak_then_strong_can_confirm_before_end_of_video():
    runtime = posture_runtime()
    first = posture_output(.32705909, [82.8766, 195.631, 369.6471, 274.8347])
    assert runtime.update_decision(first, 3.752805755)[0] == 'normal'
    last = posture_output(.45921937, [79.814, 197.0629, 357.2913, 275.5017])
    assert runtime.update_decision(last, 4.288920863)[0] == 'fighting'
    assert last['fall_confirmation']['previous_support_used']


@pytest.mark.parametrize('timestamp,box', [(2., [0, 0, 100, 30]), (.54, [300, 0, 400, 30]),
                                         (.54, [0, 0, 30, 100])])
def test_retrospective_support_rejects_stale_different_or_upright_person(timestamp, box):
    runtime = posture_runtime()
    runtime.update_decision(posture_output(.33, [0, 0, 100, 30]), 0.)
    output = posture_output(.7, box)
    assert runtime.update_decision(output, timestamp)[0] == 'verifying'
    assert not output['fall_confirmation']['previous_support_used']
