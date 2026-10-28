"""Run upstream detection unchanged, with bounded queues for a 16 GB laptop."""

from pathlib import Path
import faulthandler
import queue
import runpy
import time

def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(45, repeat=False)
    print('[STARTUP] Importing OpenCV and PyTorch...', flush=True)
    import cv2
    import torch

    root = Path(__file__).resolve().parent
    print('[STARTUP] Loading official MMAction2 modules...', flush=True)
    upstream = runpy.run_path(str(root / "repo/demo/webcam_demo_spatiotemporal_det.py"))
    original_helper = upstream["ClipHelper"]
    original_capture = cv2.VideoCapture

    def capture(source):
        if isinstance(source, int):
            print(f'[STARTUP] Opening camera {source} with DirectShow...', flush=True)
            result = original_capture(source, cv2.CAP_DSHOW)
            if not result.isOpened():
                raise RuntimeError(f'Cannot open camera {source}; close other camera apps or select another index')
            result.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            result.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            result.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print('[STARTUP] Camera opened.', flush=True)
            return result
        return original_capture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])

    class PendingQueue(queue.Queue):
        def __init__(self, owner):
            super().__init__(maxsize=1)
            self.owner = owner

        def put(self, item, block=True, timeout=None):
            while not self.owner.stopped:
                try:
                    super().put(item, timeout=0.2)
                    return
                except queue.Full:
                    pass

    class BoundedClipHelper(original_helper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_queue = PendingQueue(self)
            faulthandler.cancel_dump_traceback_later()
            print('[STARTUP] Ready; collecting the first action clip...', flush=True)

        def display(self, task):
            while not self.stopped:
                with self.display_lock:
                    available = len(self.display_queue) < 1
                if available:
                    return super().display(task)
                time.sleep(0.02)

    print('[STARTUP] Setting CPU thread count...', flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cv2.VideoCapture = capture
    upstream["main"].__globals__["ClipHelper"] = BoundedClipHelper
    print('[STARTUP] Loading detector and action checkpoints...', flush=True)
    try:
        upstream["main"](upstream["parse_args"]())
    finally:
        cv2.VideoCapture = original_capture


if __name__ == "__main__":
    main()
