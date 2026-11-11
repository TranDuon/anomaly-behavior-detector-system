from collections import deque
from types import SimpleNamespace

import numpy as np

from fight.mmaction_runtime import MMActionRuntime
from fight.runtime import EventGate


LABELS = ('strike', 'grapple', 'fall', 'fast_running')


def model():
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.action_gates = {name: EventGate(.5, .35, confirm_seconds=.4, release_seconds=.25) for name in LABELS}
    return runtime


def prediction(**values):
    scores = {name: values.get(name, .01) for name in LABELS}
    return {'people': 2, 'score': max(scores.values()), 'regions': [], 'action_scores': scores,
            'action_regions': {name: [[index * 10, 0, index * 10 + 9, 20]] for index, name in enumerate(LABELS)}}


def test_four_labels_are_independent_and_can_overlap():
    runtime = model()
    runtime.separate_actions = False
    runtime.update_decision(prediction(grapple=.9), 0.)
    output = prediction(grapple=.9, fall=.9)
    runtime.update_decision(output, .54)
    assert output['active_behaviors'] == ['grapple']
    output = prediction(grapple=.9, fall=.9)
    runtime.update_decision(output, 1.08)
    assert output['active_behaviors'] == ['grapple', 'fall']
    assert output['action_states']['strike'] == 'normal'
    runtime.reset()
    assert all(not gate.active for gate in runtime.action_gates.values())


def test_weak_grapple_pilot_does_not_relabel_striking_as_headlock():
    runtime = model()
    runtime.update_decision(prediction(grapple=.99, strike=.9), 0.)
    output = prediction(grapple=.99, strike=.9)
    runtime.update_decision(output, .54)
    assert output['active_behaviors'] == ['strike']
    assert output['action_scores']['grapple'] == .99
    assert output['action_evidence']['grapple'] == 'diagnostic_only_unvalidated_grapple_pilot'


def test_pose_merge_never_feeds_modified_boxes_back_into_rgb(monkeypatch):
    runtime = model()
    runtime.model_ready = True
    box = [0, 0, 100, 100]
    seen = []
    def rgb(frame, timestamp, predict, *, sampled):
        seen.append(runtime.result.get('posture'))
        return {'person_boxes': [box], 'actions': [], 'box_timestamp': timestamp,
                'action_scores': {'strike': .8}, 'action_states': {'strike': 'fighting'},
                'active_behaviors': ['strike'], 'action_regions': {'strike': [box]}}
    runtime.posture_detector = SimpleNamespace(observe=lambda *args, **kwargs:
                                               {'boxes': [], 'confirmed': [], 'candidates': []})
    monkeypatch.setattr(runtime, 'push_actions', rgb)
    frame = np.random.default_rng(8).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    try:
        for timestamp in (0., .27):
            result = runtime.push(frame, timestamp, sampled=True)
            assert result['people'] == 1 and result['active_behaviors'] == ['strike']
        assert seen == [None, None]
        assert 'posture' not in runtime.rgb_result
    finally:
        runtime.frame_buffer.close()


def test_event_journal_keeps_grapple_and_fall_as_separate_events(monkeypatch):
    from api.fight_server import FightProcessor
    processor = FightProcessor.__new__(FightProcessor)
    processor.before = deque(maxlen=12)
    processor.timestamp = 0.
    processor.runtime = SimpleNamespace(backend_name='mmaction_separate_actions')
    processor.label_recordings = {}
    processor.active_event, processor.event_frames = None, []
    calls = []

    def record_single(frame, result):
        calls.append((tuple(result['active_behaviors']), result['event'], result['score']))
        if result['event'] == 'started':
            processor.active_event = {'label': result['active_behaviors'][0]}
        elif result['event'] == 'ended':
            processor.active_event = None

    monkeypatch.setattr(processor, 'record_single', record_single)
    frame = np.zeros((10, 10, 3), np.uint8)
    output = prediction(grapple=.8, fall=.7)
    output['active_behaviors'] = ['grapple', 'fall']
    processor.record(frame, output)
    assert set(processor.label_recordings) == {'grapple', 'fall'}
    assert (('grapple',), 'started', .8) in calls
    assert (('fall',), 'started', .7) in calls
    output['active_behaviors'] = ['fall']
    processor.record(frame, output)
    assert set(processor.label_recordings) == {'fall'}


def test_running_label_can_be_rendered_with_the_other_three():
    from api.fight_server import annotate
    image = annotate(np.zeros((100, 200, 3), np.uint8),
                     {'state': 'fighting', 'behavior_scope': 'three_actions',
                      'active_behaviors': list(LABELS), 'regions': [], 'actions': []})
    assert image.shape == (100, 200, 3)
