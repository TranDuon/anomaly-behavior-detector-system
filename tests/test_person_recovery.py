import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mmaction2_demo'))
from person_recovery import select_people


def test_primary_detection_is_sensitive_without_rotation():
    calls = []

    def detect(frame):
        calls.append(frame.shape)
        return np.array([[10, 10, 30, 70], [11, 10, 31, 70]]), np.array([0.65, 0.7])

    result = select_people(np.zeros((80, 120, 3), np.uint8), detect)
    assert result.tolist() == [[11, 10, 31, 70]]
    assert len(calls) == 1


def test_horizontal_candidate_requires_rotated_confirmation():
    calls = []

    def detect(frame):
        calls.append(frame.shape)
        if len(calls) == 1:
            return np.array([[20, 30, 100, 50]]), np.array([0.15])
        if len(calls) == 2:
            return np.array([[30, 20, 50, 100]]), np.array([0.99])
        return np.array([[20, 20, 40, 100]]), np.array([0.1])

    result = select_people(np.zeros((80, 120, 3), np.uint8), detect)
    np.testing.assert_allclose(result, [[20, 30, 100, 50]])


def test_weak_boxes_are_not_accepted_without_confirmation():
    def detect(frame):
        return np.array([[10, 20, 80, 40]]), np.array([0.15])

    assert select_people(np.zeros((80, 120, 3), np.uint8), detect).shape == (0, 4)


def test_counterclockwise_recovery_maps_back_and_does_not_duplicate():
    calls = []

    def detect(frame):
        calls.append(1)
        if len(calls) == 1:
            return np.array([[20, 30, 100, 50]]), np.array([0.15])
        if len(calls) == 2:
            return np.empty((0, 4)), np.array([])
        return np.array([[30, 20, 50, 100], [30, 20, 50, 100]]), np.array([0.99, 0.9])

    result = select_people(np.zeros((80, 120, 3), np.uint8), detect)
    np.testing.assert_allclose(result, [[20, 30, 100, 50]])


def test_unrelated_rotated_detection_is_rejected():
    calls = []

    def detect(frame):
        calls.append(1)
        if len(calls) == 1:
            return np.array([[10, 10, 50, 20]]), np.array([0.15])
        return np.array([[0, 90, 10, 119]]), np.array([0.99])

    assert not len(select_people(np.zeros((80, 120, 3), np.uint8), detect))


def test_invalid_and_empty_boxes_are_rejected():
    frame = np.zeros((80, 120, 3), np.uint8)
    assert select_people(frame, lambda _: ([], [])).shape == (0, 4)
    assert not len(select_people(frame, lambda _: ([[0, 0, 0, 0], [0, 0, float('nan'), 80]], [.99, .99])))


def test_overlapping_distinct_people_are_not_suppressed_as_duplicates():
    result = select_people(np.zeros((200, 200, 3), np.uint8),
                           lambda _: ([[20, 20, 100, 180], [40, 20, 120, 180]], [.99, .95]))
    assert len(result) == 2


def test_crop_confirmation_recovers_one_occluded_candidate():
    calls = []

    def detect(frame):
        calls.append(1)
        if len(calls) == 1:
            return np.array([[10, 10, 30, 60], [100, 30, 150, 180]]), np.array([.99, .99])
        return np.array([[15, 22, 65, 172], [35, 30, 85, 130]]), np.array([.99, .95])

    result = select_people(np.zeros((200, 240, 3), np.uint8), detect)
    assert len(result) == 3
    assert len(calls) == 2


def test_same_lying_person_with_different_box_sizes_is_not_counted_twice():
    result = select_people(np.zeros((200, 300, 3), np.uint8),
                           lambda _: ([[20, 70, 250, 130], [25, 65, 250, 150]], [.99, .95]))
    assert len(result) == 1
