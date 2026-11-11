"""Local file-backed JSON IPC; no network listener and no camera access."""

import contextlib
import argparse
import json
from pathlib import Path
import sys
import traceback
import time

import numpy as np
import torch

from action_engine import ActionEngine, ROOT
from frame_transport import SharedFrameReader


def reply(value):
    sys.__stdout__.write(json.dumps(value, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned", action="store_true", help="Use interaction v1")
    parser.add_argument("--fall-wrestle", action="store_true", help="Use fall/wrestle pilot head")
    parser.add_argument("--transition", action="store_true", help="Use Re-DID transition pilot head")
    parser.add_argument("--fixed-running", action="store_true", help="Use separate running and preserved fall/wrestle heads")
    parser.add_argument('--three-actions', action='store_true', help='Use independent strike, grapple and fall heads')
    parser.add_argument('--separate-actions', action='store_true', help='Independent original strike/fall, grapple pilot and preserved running')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        if args.separate_actions:
            from separate_actions_engine import SeparateActionsEngine
            engine = SeparateActionsEngine(person_recovery=True)
        elif args.three_actions:
            from three_actions_engine import ThreeActionsEngine
            engine = ThreeActionsEngine(person_recovery=True)
        elif args.fixed_running:
            from fixed_running_engine import FixedRunningEngine
            engine = FixedRunningEngine(person_recovery=True)
        else:
            engine = ActionEngine(tuned=args.tuned or args.fall_wrestle or args.transition,
                                  fall_wrestle=args.fall_wrestle, transition=args.transition,
                                  person_recovery=True)
        engine.metadata['person_detection'] = {'profile': 'sensitive_overlap_v2',
                                               'threshold': 0.6, 'recovery_threshold': 0.8,
                                               'nms_iou': 0.7, 'crop_confirmation': 0.85}
        engine.warmup()
        from detection_cache import ExactDetectionCache
        engine.detect_people = ExactDetectionCache(engine.detect_people)
        posture = None
        if args.separate_actions:
            sys.path.insert(0, str(ROOT.parent))
            from fight.lying_posture import LyingPosture
            posture = LyingPosture(ROOT.parent / 'models/yolov8n-pose.pt', device='cuda:0')
            posture.detector.warmup(640)
            engine.metadata['posture_device'] = 'cuda:0'
        engine.metadata['detector_cache'] = 'exact_frame_bytes_bounded_24'

    reply({"ready": True, "metadata": engine.metadata})
    reader = SharedFrameReader()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            if request.get('op') == 'reset':
                if posture is not None:
                    posture.reset()
                    posture.detector.cache.clear()
                reply({'reset': True})
                continue
            started = time.perf_counter()
            if request.get("transport") == "shared_memory":
                frames = reader.read(request)
            else:
                path = Path(request["frames"]).resolve()
                if not path.is_relative_to((ROOT / "results/ipc").resolve()):
                    raise ValueError("Invalid local frame path")
                with np.load(path, allow_pickle=False) as source:
                    frames = source["frames"]
            loaded = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if request.get('op') == 'pose':
                    if posture is None:
                        raise ValueError('Pose is not enabled for this worker')
                    result = {}
                    inferred = time.perf_counter()
                elif request.get('op') == 'detect':
                    boxes = engine.detect_people(frames[-1]).tolist()
                    result = {'person_boxes': boxes, 'people': len(boxes),
                              'actions': [{'box': box, 'labels': [], 'fight_score': 0.} for box in boxes]}
                    inferred = time.perf_counter()
                else:
                    result = engine.infer(frames)
                    inferred = time.perf_counter()
                    result = engine.current_view(frames, result)
                viewed = time.perf_counter()
                observed = None
                if posture is not None:
                    posture.detector.cache.clear()
                    observed = posture.observe(frames[-1], request['timestamp'],
                                               recovery_boxes=request.get('recovery_boxes', result.get('person_boxes', [])))
            finished = time.perf_counter()
            reply({"result": result, 'posture': observed, "timings": {
                "worker_read_seconds": loaded - started,
                "action_seconds": inferred - loaded,
                "current_detector_seconds": viewed - inferred,
                'pose_seconds': finished - viewed,
                'detector_cache_hits': engine.detect_people.hits,
            }})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply({"error": str(error)})
    reader.close()


if __name__ == "__main__":
    main()
