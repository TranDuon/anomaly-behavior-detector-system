"""Interface every extension behaviour implements.

An extension runs inside the extension worker process (``extensions/worker.py``)
with its own Python environment (``extensions/env``). It never imports from
the core worker and never changes core outputs; the runtime only *adds* its
result next to the core labels.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT.parent / "weights" / "extensions"


class Extension:
    """Base class. Subclasses set ``name`` and implement ``load`` + ``observe``."""

    name = ""
    display_label = ""
    confirm_seconds = 0.8
    release_seconds = 1.0
    threshold = 0.5
    alarm_enabled = True

    def load(self, weights_dir=WEIGHTS):
        """Load checkpoints. Raise to disable only this extension."""

    def observe(self, frames, timestamp, people=None):
        """Score one step.

        frames:    (9, H, W, 3) uint8 BGR, the same 3.75 fps samples the core used.
        timestamp: source time (s) of the last frame.
        people:    {"person_boxes": [[x1, y1, x2, y2], ...], "posture_boxes": [...]}
                   copied from the core reply, so no second person detector runs.

        Returns {"score": float in [0, 1], "boxes": [[x1, y1, x2, y2], ...],
                 "details": {...}} — boxes are what the overlay highlights.
        """
        raise NotImplementedError

    def spec(self):
        return {"name": self.name, "display_label": self.display_label or self.name.upper(),
                "confirm_seconds": self.confirm_seconds, "release_seconds": self.release_seconds,
                "threshold": self.threshold, "alarm_enabled": self.alarm_enabled}


_REGISTRY = {}


def register(cls):
    if not cls.name:
        raise ValueError("Extension needs a name")
    _REGISTRY[cls.name] = cls
    return cls


def available():
    return dict(_REGISTRY)


def create(name):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown extension '{name}'")
    return _REGISTRY[name]()


@register
class SelfTestExtension(Extension):
    """Dependency-free extension used by tests and to check the IPC path.

    Score = mean frame brightness change between first and last sample, so it
    reacts to motion without any model. Never use it as a real alarm.
    """

    name = "selftest"
    display_label = "SELFTEST"
    alarm_enabled = False

    def observe(self, frames, timestamp, people=None):
        import numpy as np

        first = frames[0].astype(np.float32)
        last = frames[-1].astype(np.float32)
        score = float(min(1.0, np.abs(last - first).mean() / 64.0))
        boxes = list((people or {}).get("person_boxes", []))
        return {"score": score, "boxes": boxes, "details": {"kind": "mean_abs_difference"}}
