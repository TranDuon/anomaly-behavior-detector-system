import threading

import cv2
import numpy as np
import pytest

from fight.browser_video import finalize_browser_video
from fight.live_queue import fresh_live_samples
from mmaction2_demo.detection_cache import ExactDetectionCache


def test_detector_cache_checks_pixels_not_shared_memory_address():
    calls = []
    def detect(frame):
        calls.append(int(frame[0, 0, 0]))
        return np.array([[frame[0, 0, 0], 0, 10, 20]], dtype=float)
    cache = ExactDetectionCache(detect, capacity=2)
    frame = np.zeros((20, 20, 3), np.uint8)
    first = cache(frame)
    assert np.array_equal(cache(frame.copy()), first)
    first[:] = -1
    assert cache(frame)[0, 0] == 0
    frame[:] = 2
    assert cache(frame)[0, 0] == 2
    frame[:] = 3
    cache(frame)
    frame[:] = 0
    cache(frame)
    assert calls == [0, 2, 3, 0]
    assert len(cache.entries) == 2


def test_live_queue_rejects_old_samples_but_preserves_causal_order():
    samples = [('old', 0.), ('recent', 1.4), ('latest', 1.7)]
    fresh, dropped = fresh_live_samples(samples, 2.)
    assert fresh == samples[1:] and dropped == 1
    assert fresh_live_samples(samples, 4.) == ([], 3)


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15., (64, 48))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((48, 64, 3), index * 30, np.uint8))
    writer.release()


def test_offline_result_is_h264_with_original_fps_and_frame_count(tmp_path):
    path = tmp_path / 'result.mp4'
    make_video(path)
    finalize_browser_video(path, threading.Event())
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(15.)
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 6
        assert int(capture.get(cv2.CAP_PROP_FOURCC)) in {
            cv2.VideoWriter_fourcc(*codec) for codec in ('avc1', 'h264', 'H264')}
    finally:
        capture.release()
    assert list(tmp_path.glob('*.h264.mp4')) == []


def test_stopping_encoder_keeps_original_generated_video(tmp_path):
    path = tmp_path / 'result.mp4'
    make_video(path)
    original = path.read_bytes()
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        finalize_browser_video(path, stop)
    assert path.read_bytes() == original
    assert list(tmp_path.glob('*.h264.mp4')) == []
