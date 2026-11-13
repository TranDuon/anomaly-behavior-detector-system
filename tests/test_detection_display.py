import numpy as np

from fight.box_tracking import DisplayBoxTracker


def observed(timestamp, anchor, boxes):
    return {'prediction_timestamp': timestamp, 'state': 'normal', 'active_behaviors': [],
            'display_geometry': {'box_timestamp': anchor, 'person_boxes': boxes,
                                 'actions': [{'box': box, 'labels': []} for box in boxes],
                                 'regions': [], 'appearances': [[1., 0.] for box in boxes]}}


def test_detector_kalman_tracks_motion_even_on_textureless_background():
    tracker = DisplayBoxTracker()
    frame = np.zeros((180, 400, 3), np.uint8)
    centers = []
    for index in range(41):
        timestamp = index / 30
        anchor = (index // 8) * 8 / 30
        left = 20 + 120 * anchor
        result = tracker.align(frame, timestamp, observed(timestamp, anchor, [[left, 40, left + 30, 110]]))
        assert result['box_tracking'] == 'detector_kalman'
        if .1 < timestamp < 8 / 30:
            assert result['person_boxes'] == []
            continue
        assert len(result['person_boxes']) == 1
        box = result['person_boxes'][0]
        centers.append((box[0] + box[2]) / 2)
        assert result['box_timestamp'] == anchor
        if timestamp >= .55:
            assert abs(centers[-1] - (35 + 120 * timestamp)) < 8
    assert all(after >= before - 2 for before, after in zip(centers, centers[1:]))


def test_pose_clock_cannot_reseed_an_old_detector_observation():
    tracker = DisplayBoxTracker()
    frame = np.zeros((180, 300, 3), np.uint8)
    tracker.align(frame, 0., observed(0., 0., [[20, 40, 50, 110]]))
    tracker.align(frame, .2, observed(.2, 0., [[20, 40, 50, 110]]))
    assert tracker.detector_tracker.prediction_key == 0.
    assert len(tracker.detector_tracker.filters) == 1


def test_expired_geometry_is_hidden_and_requests_detection():
    tracker = DisplayBoxTracker()
    frame = np.zeros((180, 300, 3), np.uint8)
    source = observed(0., 0., [[20, 40, 50, 110]])
    tracker.align(frame, 0., source)
    result = tracker.align(frame, .8, source)
    assert result['person_boxes'] == []
    assert result['tracking_needs_detection']


def test_one_person_update_does_not_add_a_second_trailing_box():
    tracker = DisplayBoxTracker()
    frame = np.zeros((180, 300, 3), np.uint8)
    tracker.align(frame, 0., observed(0., 0., [[20, 40, 50, 110]]))
    result = tracker.align(frame, .3, observed(.3, .267, [[52, 40, 82, 110]]))
    assert len(result['person_boxes']) == 1
    assert len(tracker.detector_tracker.filters) == 1


def test_extra_display_geometry_does_not_mutate_model_result():
    tracker = DisplayBoxTracker()
    frame = np.zeros((180, 300, 3), np.uint8)
    source = observed(0., 0., [[20, 40, 50, 110]])
    source['person_boxes'] = [[0, 0, 10, 10]]
    result = tracker.align(frame, 0., source)
    assert source['person_boxes'] == [[0, 0, 10, 10]]
    assert result['person_boxes'] != source['person_boxes']
    assert source['state'] == 'normal'


def test_display_detection_does_not_advance_pose_or_action_state():
    import io
    import json
    from types import SimpleNamespace
    from fight.mmaction_runtime import MMActionRuntime
    runtime = MMActionRuntime(separate_actions=True, allow_pilot=True)
    runtime.process = SimpleNamespace(stdin=io.StringIO())
    runtime.replies.put({'result': {'person_boxes': [[10, 10, 30, 60]]}})
    runtime.last_prediction = .25
    runtime.latest_posture_time = .25
    try:
        result = runtime.detect_display(np.zeros((80, 100, 3), np.uint8), .5)
        assert json.loads(runtime.process.stdin.getvalue())['op'] == 'display_detect'
        assert result['timestamp'] == .5
        assert runtime.last_prediction == runtime.latest_posture_time == .25
        assert not runtime.samples
    finally:
        runtime.frame_buffer.close()


def test_small_runner_display_is_disabled_for_closeups_and_other_action_branches():
    from fight.display_geometry import needs_small_person_display
    empty_pose = {'boxes': []}
    small = {'person_boxes': [[100, 100, 140, 180]]}
    assert needs_small_person_display(small, empty_pose, (540, 960, 3), {'fast_running': 'fighting'})
    assert not needs_small_person_display({'person_boxes': [[100, 20, 420, 530]]}, empty_pose, (540, 960, 3), {})
    for name in ('strike', 'grapple', 'fall'):
        for state in ('fighting', 'verifying'):
            assert not needs_small_person_display(small, empty_pose, (540, 960, 3), {name: state})
