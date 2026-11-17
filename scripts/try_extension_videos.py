"""Run the _ext backend on videos and summarise what each extension did.

    python scripts/try_extension_videos.py VIDEO [VIDEO ...] [--label NAME ...]

For every video: max extension score, seconds spent in the confirmed state,
first alarm time, and the core behaviours seen, so false alarms on core-only
videos (strike / running / fall) are visible next to true detections.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2

    from fight.mmaction_runtime import MMActionRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--paced", action="store_true", help="feed frames in real time (like a camera)")
    args = parser.parse_args()
    runtime = MMActionRuntime(allow_pilot=True, separate_actions=True, extensions=True)
    runtime.warmup()
    status = runtime.extension_status()
    print("extensions:", status["active"], "error:", status.get("error"))
    names = list(runtime.extension_specs)
    rows = []
    try:
        for path in args.videos:
            runtime.reset()
            capture = cv2.VideoCapture(str(path))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            index, last, started = 0, None, time.monotonic()
            summary = {name: {"max": 0.0, "fighting_s": 0.0, "first_alarm": None, "steps": 0} for name in names}
            core = set()
            verdicts = {}
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                t = index / fps
                index += 1
                if args.paced:
                    delay = started + t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                result = runtime.push(frame, t, predict=True, sampled=False)
                stamp = result.get("prediction_timestamp")
                if stamp is None or stamp == last:
                    continue
                step = (stamp - last) if last is not None else 0.0
                last = stamp
                core |= {b for b in result.get("active_behaviors", []) if b not in names}
                context = result.get("strike_context")
                if context and "strike" in result.get("active_behaviors", []):
                    verdicts[context["verdict"]] = verdicts.get(context["verdict"], 0) + 1
                for name in names:
                    state = result.get("action_states", {}).get(name)
                    score = result.get("action_scores", {}).get(name, 0.0)
                    entry = summary[name]
                    entry["max"] = max(entry["max"], score)
                    entry["steps"] += 1
                    if state == "fighting":
                        entry["fighting_s"] += step
                        if entry["first_alarm"] is None:
                            entry["first_alarm"] = round(stamp, 2)
            capture.release()
            duration = index / fps
            for name, entry in summary.items():
                print(f"{path.name[:60]:60s} {duration:6.1f}s  {name}: max {entry['max']:.2f}  "
                      f"alarm {entry['fighting_s']:5.1f}s  first {entry['first_alarm']}  core {sorted(core)}  strike-context {verdicts}", flush=True)
            rows.append({"video": str(path), "duration": duration, "core": sorted(core), "extensions": summary, "strike_context": verdicts})
    finally:
        print("worker:", json.dumps(runtime.extension_status(), default=str))
        runtime.close()
    out = ROOT / "logs" / f"try_extension_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
