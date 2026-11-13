import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fight.box_tracking import DisplayBoxTracker
from fight.display_alerts import RecentRunningAlert
from fight.mmaction_runtime import MMActionRuntime


def test_preview_cannot_read_in_progress_or_rewound_result(monkeypatch):
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.model_ready = True
    runtime.result = {'prediction_timestamp': 1., 'state': 'normal'}
    runtime.rgb_result = {'prediction_timestamp': .5, 'state': 'normal'}
    entered, release = threading.Event(), threading.Event()
    errors = []
    box = [5, 5, 60, 90]
    def infer(*args, **kwargs):
        runtime.result = {'prediction_timestamp': .5, 'state': 'insufficient'}
        entered.set()
        assert release.wait(5)
        return {'prediction_timestamp': 2., 'box_timestamp': 2., 'state': 'normal',
                'person_boxes': [box], 'actions': [], 'action_regions': {}, 'timings': {}}
    runtime.posture_detector = SimpleNamespace(observe=lambda *args, **kwargs:
                                               {'boxes': [], 'confirmed': [], 'candidates': []})
    monkeypatch.setattr(runtime, 'push_actions', infer)
    def run():
        try:
            runtime.push(np.random.default_rng(5).integers(0, 255, (100, 100, 3), dtype=np.uint8), 2., sampled=True)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    try:
        thread.start()
        assert entered.wait(5)
        assert runtime.result == {'prediction_timestamp': 1., 'state': 'normal'}
        release.set()
        thread.join(5)
        assert not thread.is_alive() and not errors
        assert runtime.result['prediction_timestamp'] == 2.
    finally:
        release.set()
        thread.join(5)
        runtime.frame_buffer.close()


def test_failed_inference_does_not_publish_partial_result(monkeypatch):
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.model_ready = True
    runtime.result = {'prediction_timestamp': 1., 'state': 'normal'}
    def fail(*args, **kwargs):
        runtime.result = {'prediction_timestamp': .5, 'state': 'insufficient'}
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(runtime, 'push_actions', fail)
    try:
        with pytest.raises(RuntimeError):
            runtime.push(np.zeros((100, 100, 3), np.uint8), 2., sampled=True)
        assert runtime.result == {'prediction_timestamp': 1., 'state': 'normal'}
    finally:
        runtime.frame_buffer.close()


def running_result():
    box = [20, 60, 60, 120]
    return {'state': 'fighting', 'behavior_scope': 'three_actions',
            'active_behaviors': ['fast_running'], 'display_policy': 'running_confirmed_geometry_v1',
            'prediction_timestamp': 1., 'rgb_prediction_timestamp': 1., 'box_timestamp': 1.,
            'person_boxes': [box], 'regions': [box], 'actions': []}


def test_lost_texture_does_not_erase_confirmed_running_or_invent_a_box():
    frame = np.zeros((150, 180, 3), np.uint8)
    result = DisplayBoxTracker().align(frame, 1., running_result())
    assert result['state'] == 'fighting'
    assert result['geometry_status'] == 'running_track_unavailable'
    assert result['person_boxes'] == result['regions'] == []
    from api.fight_server import annotate
    image = annotate(frame, result)
    assert image[90, 20].tolist() == [0, 0, 0]
    assert np.any(np.all(image[:36] == [50, 65, 235], axis=2))


def test_recent_banner_expires_without_extending_model_alarm_or_other_labels():
    recent = RecentRunningAlert()
    recent.apply(running_result(), 1.2)
    normal = {**running_result(), 'state': 'normal', 'active_behaviors': [], 'regions': []}
    display = recent.apply(normal, 1.8)
    assert display['state'] == 'normal' and display['active_behaviors'] == []
    assert display['recent_running_alert'] == {'confirmed_at': 1.}
    assert 'recent_running_alert' not in recent.apply(normal, 2.3)
    strike = {**normal, 'state': 'fighting', 'active_behaviors': ['strike']}
    assert 'recent_running_alert' not in recent.apply(strike, 1.9)


def test_preview_can_show_a_confirmed_event_even_if_it_missed_the_active_snapshot():
    normal = {**running_result(), 'state': 'normal', 'active_behaviors': [], 'regions': [],
              'last_confirmed_running_at': 1.}
    display = RecentRunningAlert().apply(normal, 1.6)
    assert display['recent_running_alert'] == {'confirmed_at': 1.}
    assert display['state'] == 'normal' and display['regions'] == []
