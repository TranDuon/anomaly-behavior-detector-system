import cv2
import numpy as np

from fight.box_tracking import DisplayBoxTracker


def frames():
    texture = np.random.default_rng(7).integers(20, 255, (60, 30, 3), dtype=np.uint8)
    for index in range(40):
        frame = np.zeros((120, 240, 3), np.uint8)
        left = 20 + index * 3
        frame[30:90, left:left + 30] = texture
        yield frame


def prediction(timestamp=1.0):
    return {"state": "fighting", "prediction_timestamp": timestamp, "box_timestamp": 0.0,
            "person_boxes": [[20, 30, 50, 90]], "regions": [[20, 30, 50, 90]],
            "actions": [{"box": [20, 30, 50, 90], "labels": []}]}


def test_delayed_box_is_replayed_to_current_person_and_keeps_color():
    tracker = DisplayBoxTracker()
    original = prediction()
    for index, frame in enumerate(frames()):
        result = tracker.align(frame, index / 30, original if index >= 30 else {})
        if index >= 30:
            assert len(result["person_boxes"]) == 1
            assert abs(result["person_boxes"][0][0] - (20 + index * 3)) < 2
            assert result["regions"] == result["person_boxes"]
            assert result["actions"][0]["box"] == result["person_boxes"][0]
    assert original["person_boxes"] == [[20, 30, 50, 90]]


def test_no_reference_or_expired_detection_does_not_draw_ghost_box():
    tracker = DisplayBoxTracker()
    frame = next(frames())
    assert tracker.align(frame, 1.0, prediction())["person_boxes"] == []
    assert tracker.align(frame, 4.0, prediction())["person_boxes"] == []


def test_occlusion_drops_box_instead_of_hallucinating_a_person():
    tracker = DisplayBoxTracker()
    frame = next(frames())
    assert tracker.align(frame, 0, prediction(0))["person_boxes"]
    result = tracker.align(np.zeros_like(frame), 1 / 30, prediction(0))
    assert result["person_boxes"] == []
    assert result["state"] == "insufficient"


def test_coordinate_scaling_and_pair_highlights_only_member_people():
    tracker = DisplayBoxTracker(width=240)
    frame = cv2.resize(next(frames()), (480, 240))
    result = prediction(0)
    result["person_boxes"] = [[40, 60, 100, 180]]
    result["regions"] = [[40, 60, 100, 180]]
    result["actions"][0]["box"] = [40, 60, 100, 180]
    tracked = tracker.align(frame, 0, result)
    assert np.allclose(tracked["person_boxes"], result["person_boxes"])
    assert tracked["regions"] == tracked["person_boxes"]


def test_two_people_move_independently_and_bystander_stays_unhighlighted():
    tracker = DisplayBoxTracker()
    texture = np.random.default_rng(9).integers(20, 255, (60, 30, 3), dtype=np.uint8)
    result = prediction(0)
    result["person_boxes"].append([170, 30, 200, 90])
    for index in range(20):
        frame = np.zeros((120, 240, 3), np.uint8)
        frame[30:90, 20 + index * 2:50 + index * 2] = texture
        frame[30:90, 170 - index:200 - index] = texture
        display = tracker.align(frame, index / 30, result)
        assert len(display["person_boxes"]) == 2
        assert abs(display["person_boxes"][0][0] - (20 + index * 2)) < 2
        assert abs(display["person_boxes"][1][0] - (170 - index)) < 2
        assert display["regions"] == [display["person_boxes"][0]]


def test_sampled_rgb_boxes_and_running_regions_move_to_pose_timestamp():
    tracker = DisplayBoxTracker(maximum_gap=.4)
    result = prediction(0)
    result['action_regions'] = {'fast_running': result['regions'], 'strike': []}
    for index, frame in enumerate(frames()):
        if index % 8:
            continue
        aligned = tracker.align(frame, index / 30, result)
        assert len(aligned['person_boxes']) == 1
        assert abs(aligned['person_boxes'][0][0] - (20 + index * 3)) < 2
        assert aligned['action_regions']['fast_running'] == aligned['person_boxes']
        assert aligned['action_regions']['strike'] == []
