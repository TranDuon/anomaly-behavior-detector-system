"""S5: runtime "person holding a sharp object" extension (extensions/worker.py, extensions/env).

Approach 1 (detector + association):
  1. The YOLO detector trained by train_sharp_object.py finds sharp objects (any class) in the
     latest core sample.
  2. Each object is linked to the person box (from the core reply, no second person detector)
     that contains most of it, after widening the person box for arms and long handles.
  3. The step score is the highest confidence among *held* objects; an object nobody holds
     (on a table, on the floor away from people) scores 0 and is only reported in details.
  4. Time confirmation (K of N steps) is the runtime EventGate: threshold / confirm_seconds /
     release_seconds come from sharp_object_spec.json.

Boxes returned to the runtime are the holders' person boxes (overlay highlight); the weapon
boxes are in details["weapons"], all in source-frame pixels.
"""

import json

import numpy as np

from extensions.base import WEIGHTS, Extension, register

# Person box widening used to decide "held": arms stretch sideways and a raised long knife
# (dao phong lon) can stick out above the head.
SIDE_MARGIN = 0.35   # x person width, each side
TOP_MARGIN = 0.30    # x person height, above the head
BOTTOM_MARGIN = 0.0  # objects below the feet are on the floor, not held
MIN_INSIDE = 0.30    # share of the object box that must fall inside the widened person box
MAX_OBJECT_RATIO = 1.5  # object box area <= this x person box area (rejects huge false boxes)
MAX_REPORTED = 20     # weapon boxes kept in details (highest confidence first)
FLOOR_BAND = 0.12    # object entirely in the lowest 12 % of the person box and wider than tall
#                      -> lying on the floor next to the feet, not held


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _intersection(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def holder_of(obj, people):
    """Index of the person holding ``obj`` (x1, y1, x2, y2) or None."""
    best, best_inside = None, 0.
    obj_area = _area(obj)
    if obj_area <= 0:
        return None
    for index, person in enumerate(people):
        left, top, right, bottom = person
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0 or obj_area > MAX_OBJECT_RATIO * width * height:
            continue
        widened = (left - SIDE_MARGIN * width, top - TOP_MARGIN * height,
                   right + SIDE_MARGIN * width, bottom + BOTTOM_MARGIN * height)
        inside = _intersection(obj, widened) / obj_area
        on_floor = obj[1] >= bottom - FLOOR_BAND * height and (obj[2] - obj[0]) > (obj[3] - obj[1])
        if inside >= MIN_INSIDE and not on_floor and inside > best_inside:
            best, best_inside = index, inside
    return best


@register
class SharpObjectExtension(Extension):
    name = "sharp_object"
    display_label = "SHARP OBJECT"
    confirm_seconds = 0.5
    release_seconds = 1.0
    threshold = 0.35

    def load(self, weights_dir=WEIGHTS):
        import torch
        from ultralytics import YOLO

        folder = weights_dir / "sharp_object"
        spec = json.loads((folder / "sharp_object_spec.json").read_text(encoding="utf-8"))
        self.threshold = float(spec["alert_threshold"])
        self.confirm_seconds = float(spec.get("confirm_seconds", self.confirm_seconds))
        self.release_seconds = float(spec.get("release_seconds", self.release_seconds))
        self.imgsz = int(spec.get("imgsz", 960))
        # Keep low-confidence boxes for the details; the runtime gate applies the alert threshold.
        self.min_conf = min(0.1, self.threshold)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.model = YOLO(str(folder / spec.get("weights", "sharp_object.pt")))
        self.names = self.model.names
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz, verbose=False,
                           device=self.device, half=self.half)  # warm-up

    def observe(self, frames, timestamp, people=None):
        people = people or {}
        frame = frames[-1]
        height, width = frame.shape[:2]
        source_w, source_h = people.get("frame_size") or (width, height)
        sx, sy = source_w / width, source_h / height
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.min_conf, verbose=False, device=self.device,
                                    half=self.half)[0]
        persons = [list(map(float, box)) for box in people.get("person_boxes", [])]
        weapons, held = [], set()
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                box = [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]
                holder = holder_of(box, persons)
                if holder is not None:
                    held.add(holder)
                label = "sharp" if len(self.names) == 1 else str(self.names.get(int(cls), cls))
                weapons.append({"box": [round(v, 1) for v in box], "conf": round(float(conf), 4),
                                "class": label, "holder": holder})
        weapons = sorted(weapons, key=lambda w: -w["conf"])[:MAX_REPORTED]
        held_confs = [w["conf"] for w in weapons if w["holder"] is not None]
        return {"score": float(max(held_confs, default=0.)), "boxes": [persons[i] for i in sorted(held)],
                "details": {"weapons": weapons, "held": len(held_confs), "unheld": len(weapons) - len(held_confs)}}
