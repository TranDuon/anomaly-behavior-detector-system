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

Approach 2 (spec "method": "hand_crop", train_hand_classifier.py):
  1. A pose model (shipped in the install folder) finds people and their elbow/wrist keypoints
     in the latest sample.
  2. A square crop around each visible hand (extensions/sharp_object/hand_crop.py, same
     geometry as training) goes through a ResNet: "this hand holds a sharp object".
  3. The step score is the highest hand score; boxes are the person boxes of hands at or above
     the alert threshold, details["hands"] lists every hand crop with its score.
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
LOCATOR_BASE = "yolov8s-worldv2.pt"   # hand_crop: full weapon box for the overlay (YOLO-World)
LOCATOR_FILE = "weapon_world.pt"      # vocabulary baked in, saved in the install folder
LOCATOR_CLASSES = ["knife", "machete", "sword", "spear", "axe", "sickle", "cleaver", "long blade on a pole"]
LOCATOR_IMGSZ = 640
LOCATOR_CONF = 0.01                   # very low: display only, and only on hands the classifier flagged
LOCATOR_REACH = 4.0                   # search square = 4x the hand crop side, centred on the hand
FLOOR_BAND = 0.12   # object entirely in the lowest 12 % of the person box and wider than tall
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
    method = "detector"

    def load(self, weights_dir=WEIGHTS):
        import torch
        from ultralytics import YOLO

        folder = weights_dir / "sharp_object"
        spec = json.loads((folder / "sharp_object_spec.json").read_text(encoding="utf-8"))
        self.threshold = float(spec["alert_threshold"])
        self.confirm_seconds = float(spec.get("confirm_seconds", self.confirm_seconds))
        self.release_seconds = float(spec.get("release_seconds", self.release_seconds))
        self.method = spec.get("method", "detector")
        if self.method == "hand_crop":
            self._load_hand_crop(folder, spec)
            return
        self.imgsz = int(spec.get("imgsz", 960))
        # Keep low-confidence boxes for the details; the runtime gate applies the alert threshold.
        self.min_conf = min(0.1, self.threshold)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.model = YOLO(str(folder / spec.get("weights", "sharp_object.pt")))
        self.names = self.model.names
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz, verbose=False,
                           device=self.device, half=self.half)  # warm-up

    def _load_hand_crop(self, folder, spec):
        import torch
        from ultralytics import YOLO

        from extensions.sharp_object.hand_crop import crop_params, load_classifier

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.crop = crop_params(spec)
        self.input_size = int(spec.get("input_size", 224))
        self.pose_imgsz = int(spec.get("pose_imgsz", 640))
        self.pose_conf = float(spec.get("pose_conf", 0.35))
        self.max_people = int(spec.get("max_people", 8))
        self.pose = YOLO(str(folder / spec["pose_weights"]))
        self.classifier, _ = load_classifier(folder / spec["classifier"], self.device)
        if self.half:
            self.classifier.half()
        self.locator = self._load_locator(folder)
        blank = np.zeros((self.pose_imgsz, self.pose_imgsz, 3), np.uint8)
        self.pose.predict(blank, imgsz=self.pose_imgsz, verbose=False, device=self.device, half=self.half)  # warm-up
        self._classify([np.zeros((self.input_size, self.input_size, 3), np.uint8)])

    def _load_locator(self, folder):
        """Open-vocabulary YOLO-World that draws the full weapon box around an alerted hand.

        Display only: the alert decision stays with the hand classifier. The vocabulary is baked
        into weapon_world.pt on first use (needs CLIP once); later loads skip CLIP.
        """
        from ultralytics import YOLOWorld

        baked = folder / LOCATOR_FILE
        try:
            if baked.is_file():
                model = YOLOWorld(str(baked))
            else:
                model = YOLOWorld(LOCATOR_BASE)
                model.set_classes(LOCATOR_CLASSES)
                model.save(str(baked))
            model.predict(np.zeros((LOCATOR_IMGSZ, LOCATOR_IMGSZ, 3), np.uint8), imgsz=LOCATOR_IMGSZ,
                          verbose=False, device=self.device)  # warm-up
            return model
        except Exception as error:  # the extension still works with hand boxes only
            print(f"sharp_object: weapon locator disabled ({type(error).__name__}: {error})", flush=True)
            return None

    def _locate(self, frame, hand_box):
        """Weapon box touching ``hand_box`` (frame pixels) or the hand box itself."""
        height, width = frame.shape[:2]
        side = max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
        cx, cy = (hand_box[0] + hand_box[2]) / 2, (hand_box[1] + hand_box[3]) / 2
        reach = side * LOCATOR_REACH / 2
        x1, y1 = int(max(0, cx - reach)), int(max(0, cy - reach))
        x2, y2 = int(min(width, cx + reach)), int(min(height, cy + reach))
        if self.locator is None or x2 - x1 < 16 or y2 - y1 < 16:
            return list(hand_box), None
        result = self.locator.predict(frame[y1:y2, x1:x2], imgsz=LOCATOR_IMGSZ, conf=LOCATOR_CONF,
                                      verbose=False, device=self.device)[0]
        best, best_conf = None, 0.
        if result.boxes is not None and len(result.boxes):
            for (bx1, by1, bx2, by2), conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                box = [float(bx1 + x1), float(by1 + y1), float(bx2 + x1), float(by2 + y1)]
                if _intersection(box, hand_box) > 0:  # union of every candidate touching the hand
                    best = box if best is None else [min(best[0], box[0]), min(best[1], box[1]),
                                                     max(best[2], box[2]), max(best[3], box[3])]
                    best_conf = max(best_conf, float(conf))
        if best is None:
            return list(hand_box), None
        return [min(best[0], hand_box[0]), min(best[1], hand_box[1]),
                max(best[2], hand_box[2]), max(best[3], hand_box[3])], best_conf

    def _classify(self, crops):
        import torch

        from extensions.sharp_object.hand_crop import to_tensor

        batch = to_tensor(crops).to(self.device)
        with torch.inference_mode():
            logits = self.classifier(batch.half() if self.half else batch)
        return torch.sigmoid(logits.float().squeeze(1)).cpu().numpy()

    def _observe_hand_crop(self, frames, people):
        from extensions.sharp_object.hand_crop import cut, hand_crops, people_from_pose

        frame = frames[-1]
        height, width = frame.shape[:2]
        source_w, source_h = people.get("frame_size") or (width, height)
        sx, sy = source_w / width, source_h / height
        result = self.pose.predict(frame, imgsz=self.pose_imgsz, conf=self.pose_conf, verbose=False,
                                   device=self.device, half=self.half)[0]
        persons = people_from_pose(result, self.pose_conf, self.max_people)
        crops, hands, frame_boxes = [], [], []
        for index, (box, keypoints) in enumerate(persons):
            for hand in hand_crops(box, keypoints, self.crop):
                crops.append(cut(frame, hand["box"], self.input_size))
                frame_boxes.append([float(v) for v in hand["box"]])
                hands.append({"person": index, "arm": hand["arm"],
                              "box": [round(float(v) * s, 1) for v, s in zip(hand["box"], (sx, sy, sx, sy))]})
        scores = self._classify(crops) if crops else []
        for hand, score in zip(hands, scores):
            hand["score"] = round(float(score), 4)
        # Holders are reported with the core person boxes (exact match needed for the red overlay);
        # a pose person with no matching core box keeps its own box.
        core = [list(map(float, box)) for box in people.get("person_boxes", [])]
        person_boxes, weapons = [], []
        for index in sorted({h["person"] for h in hands if h["score"] >= self.threshold}):
            pose_box = [float(v) * s for v, s in zip(persons[index][0], (sx, sy, sx, sy))]
            ious = [_intersection(pose_box, c) / max(1e-6, _area(pose_box) + _area(c) - _intersection(pose_box, c))
                    for c in core]
            best = int(np.argmax(ious)) if ious else -1
            box = core[best] if best >= 0 and ious[best] >= 0.3 else pose_box
            if box not in person_boxes:
                person_boxes.append(box)
            for hand, frame_box in zip(hands, frame_boxes):
                if hand["person"] == index and hand["score"] >= self.threshold:
                    located, conf = self._locate(frame, frame_box)
                    weapons.append({"box": [round(v * s, 1) for v, s in zip(located, (sx, sy, sx, sy))],
                                    "conf": hand["score"], "class": "sharp", "holder": len(person_boxes) - 1,
                                    "located": conf is not None})
        hands = sorted(hands, key=lambda h: -h["score"])[:MAX_REPORTED]
        return {"score": float(max(scores, default=0.)), "boxes": person_boxes,
                "details": {"method": "hand_crop", "people": len(persons), "hands": hands, "weapons": weapons}}

    def observe(self, frames, timestamp, people=None):
        people = people or {}
        if self.method == "hand_crop":
            return self._observe_hand_crop(frames, people)
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
