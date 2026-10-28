from types import SimpleNamespace

import numpy as np

from fight.pose import PoseConfig, PoseDetector


def detector():
    instance = PoseDetector.__new__(PoseDetector)
    instance.config = PoseConfig()
    instance.device = "cpu"
    instance.cache = {}
    instance.calls = []

    def predict(frames, **kwargs):
        instance.calls.append(frames)
        return [
            SimpleNamespace(keypoints=None)
            for frame in (frames if isinstance(frames, list) else [frames])
        ]

    instance.model = SimpleNamespace(predict=predict)
    return instance


def test_prefetch_disabled_by_default_preserves_training_extraction():
    instance = detector()
    instance.prefetch([np.zeros((16, 16, 3), np.uint8) for index in range(3)])
    assert not instance.calls and not instance.cache


def test_batch_prefetch_reuses_results_and_deduplicates_same_frame():
    instance = detector()
    instance.batch_detection = True
    frames = [np.zeros((16, 16, 3), np.uint8) for index in range(3)]
    instance.prefetch(frames + [frames[0]])
    assert len(instance.calls) == 1 and len(instance.calls[0]) == 3
    assert all(instance.poses(frame) == [] for frame in frames)
    assert len(instance.calls) == 1
    instance.retain(frames[1:])
    assert len(instance.cache) == 2


def test_mixed_shapes_do_not_change_letterbox_preprocessing_by_batching():
    instance = detector()
    instance.batch_detection = True
    instance.prefetch([np.zeros((16, width, 3), np.uint8) for width in (16, 32)])
    assert not instance.calls
