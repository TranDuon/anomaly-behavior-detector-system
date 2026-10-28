import cv2
import numpy as np
import pytest

from fight.vision import ClipConfig
from scripts.fight_pilot import read_frames


class FakeCapture:
    def __init__(self, count):
        self.count = count
        self.position = 0
        self.read_count = 0
        self.released = False

    def set(self, property_id, value):
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def grab(self):
        success = self.position < self.count
        self.position += 1
        return success

    def read(self):
        self.read_count += 1
        if self.position >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


@pytest.mark.parametrize("source_fps", [6, 11, 12, 30])
def test_reader_preserves_repeated_and_skipped_frame_indices(monkeypatch, source_fps):
    config = ClipConfig(fps=12)
    count = 55
    start = count - round((config.frames - 1) * source_fps / config.fps) - 1
    positions = np.rint(start + np.arange(config.frames) * source_fps / config.fps)
    capture = FakeCapture(count)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    frames = read_frames(
        {"path": "public.mp4", "start": start, "fps": source_fps}, config
    )
    assert [int(frame[0, 0, 0]) for frame in frames] == positions.tolist()
    assert capture.read_count == len(set(positions))
    assert capture.released
    if source_fps < config.fps:
        assert frames[0] is not frames[1]


def test_reader_reports_true_decode_failure_without_fabricating_frames(monkeypatch):
    capture = FakeCapture(10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    with pytest.raises(ValueError, match="frame 10: public.mp4"):
        read_frames({"path": "public.mp4", "start": 0, "fps": 12}, ClipConfig(fps=12))
    assert capture.released
