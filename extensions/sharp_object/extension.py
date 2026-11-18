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
LOCATOR_MAX_SIDE = 3.0                # weapon box sides capped at 3x the hand crop side (drops huge merges)
# A: everyday-object veto. A COCO detector looks around each alerted hand; a confidently detected
# harmless object held in that hand (most of its box inside the hand crop) cancels the hand's alert.
# Measured on a live bottle/phone session: 15/23 false-alarm frames removed, 5/229 spear frames lost.
VETO_BASE = "yolo11s.pt"
VETO_FILE = "object_veto.pt"          # copied into the install folder on first use
VETO_CLASSES = {"bottle", "cup", "cell phone", "remote", "book", "wine glass", "toothbrush", "hair drier",
                "mouse", "vase"}       # never knife / scissors / fork
VETO_CONF = 0.35
VETO_INSIDE = 0.6                     # share of the object box inside the hand crop
VETO_IMGSZ = 320
VETO_REACH = 2.0                      # search square = 2x the hand crop side
# B (18/11/2026): long household-tool veto. A hand gripping a mop / broom handle looks like a hand
# gripping a spear or dao phong lon in the hand crop (the head of the tool is outside the crop),
# and COCO has no mop/broom class. CLIP (ViT-B/32, zero-shot) looks at the whole alerted person:
# if "holding a mop / broom / rake" outweighs the weapon prompts, that person's hands are cancelled.
# Measured offline: mop video 22/22 alerted people cancelled; self-filmed weapon video 0/75;
# 28 alerted people in Vietnamese news weapon photos: 3 cancelled (1 holds a machete and a broom).
TOOL_PROMPTS = ["a person holding a mop", "a person holding a broom", "a person mopping the floor",
                "a person sweeping the floor", "a person holding a rake"]
WEAPON_PROMPTS = ["a person holding a knife", "a person holding a machete", "a person holding a spear",
                  "a person holding a sword", "a person holding a long blade on a pole", "a person holding an axe"]
TOOL_VETO_PROB = 0.8   # share of the softmax on the tool prompts
CLIP_LOCAL = WEIGHTS.parent / "clip" / "ViT-B-32.pt"   # used when present, else the CLIP cache
FLOOR_BAND = 0.12   # object entirely in the lowest 12 % of the person box and wider than tall
#                      -> lying on the floor next to the feet, not held

# One pose pass per worker step for every hand-crop extension (sharp_object, handgun): the worker
# hands the same ``frames`` array to each extension, so a second extension with the same pose
# weights and settings reuses the first one's people instead of running pose again.
_POSE_CACHE = {}
# Loaded models shared by every hand-crop extension in the worker (pose, COCO veto): the same file
# (by SHA-256) is loaded once. Results are unchanged; each extension keeps its own settings.
_MODEL_CACHE = {}


def _file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_yolo(path):
    """One Ultralytics model per distinct weights file in this process."""
    from ultralytics import YOLO

    key = _file_sha256(path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key], key


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _intersection(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


WEAPON_MERGE_OVERLAP = 0.6  # share of the smaller box covered by the other -> same weapon


def _merge_weapon_boxes(weapons):
    """One weapon held in both hands is located once per alerted hand, giving two near-identical
    boxes. Keep one box per weapon (the union of overlapping boxes of the same holder).
    Display only: the alert score and holders are unchanged."""
    merged = []
    for weapon in sorted(weapons, key=lambda w: -w["conf"]):
        for kept in merged:
            a, b = kept["box"], weapon["box"]
            if kept["holder"] == weapon["holder"] and                     _intersection(a, b) >= WEAPON_MERGE_OVERLAP * max(1e-6, min(_area(a), _area(b))):
                kept["box"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                kept["located"] = kept["located"] or weapon["located"]
                break
        else:
            merged.append(dict(weapon, box=list(weapon["box"])))
    return merged


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
    # Per-weapon settings; subclasses (extensions/handgun) override these, never the module constants.
    locator_classes = LOCATOR_CLASSES
    veto_classes = VETO_CLASSES
    use_tool_veto = True       # B: mop / broom CLIP veto (long tools only)
    weapon_class = "sharp"     # "class" of hand-crop weapon boxes in details["weapons"]

    def load(self, weights_dir=WEIGHTS):
        import torch
        from ultralytics import YOLO

        folder = weights_dir / self.name
        spec = json.loads((folder / f"{self.name}_spec.json").read_text(encoding="utf-8"))
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
        self.model = YOLO(str(folder / spec.get("weights", f"{self.name}.pt")))
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
        self.pose, pose_sha = shared_yolo(folder / spec["pose_weights"])
        self.pose_key = (pose_sha, self.pose_imgsz, self.pose_conf, self.max_people)
        self.classifier, _ = load_classifier(folder / spec["classifier"], self.device)
        if self.half:
            self.classifier.half()
        self.locator = self._load_locator(folder)
        self.veto = self._load_veto(folder)
        self.tool_veto = self._load_tool_veto() if self.use_tool_veto else None
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
                model.set_classes(list(self.locator_classes))
                model.save(str(baked))
            model.predict(np.zeros((LOCATOR_IMGSZ, LOCATOR_IMGSZ, 3), np.uint8), imgsz=LOCATOR_IMGSZ,
                          verbose=False, device=self.device)  # warm-up
            return model
        except Exception as error:  # the extension still works with hand boxes only
            print(f"{self.name}: weapon locator disabled ({type(error).__name__}: {error})", flush=True)
            return None

    def _load_veto(self, folder):
        import shutil

        from ultralytics import YOLO

        local = folder / VETO_FILE
        try:
            if not local.is_file():
                model = YOLO(VETO_BASE)  # downloads the official COCO weights once
                shutil.copy(model.ckpt_path, local)
            model, _ = shared_yolo(local)
            model.predict(np.zeros((VETO_IMGSZ, VETO_IMGSZ, 3), np.uint8), imgsz=VETO_IMGSZ, verbose=False,
                          device=self.device)  # warm-up
            return model
        except Exception as error:  # the extension still works without the veto
            print(f"{self.name}: object veto disabled ({type(error).__name__}: {error})", flush=True)
            return None

    def _load_tool_veto(self):
        try:
            import clip
            import torch

            model, preprocess = clip.load(str(CLIP_LOCAL) if CLIP_LOCAL.is_file() else "ViT-B/32", device=self.device)
            with torch.no_grad():
                text = model.encode_text(clip.tokenize(TOOL_PROMPTS + WEAPON_PROMPTS).to(self.device))
                text = text / text.norm(dim=-1, keepdim=True)
            return model, preprocess, text
        except Exception as error:  # the extension still works without this veto
            print(f"{self.name}: tool veto disabled ({type(error).__name__}: {error})", flush=True)
            return None

    def _tool_vetoed(self, frame, person_box):
        """Probability (0-1) that the person in ``person_box`` (frame pixels) holds a household long
        tool rather than a weapon, or None when the veto is unavailable."""
        if self.tool_veto is None:
            return None
        import cv2
        import torch
        from PIL import Image

        model, preprocess, text = self.tool_veto
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        pad_x, pad_y = .3 * (x2 - x1), .1 * (y2 - y1)  # arms and long handles stick out
        x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        x2, y2 = int(min(width, x2 + pad_x)), int(min(height, y2 + pad_y))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        image = preprocess(Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(image.to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
            probs = (100 * features @ text.T).softmax(-1)[0].float().cpu().numpy()
        return float(probs[:len(TOOL_PROMPTS)].sum())

    def _vetoed(self, frame, hand_box):
        """Name of a harmless COCO object held in ``hand_box`` (frame pixels), else None."""
        if self.veto is None:
            return None
        height, width = frame.shape[:2]
        side = max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
        cx, cy = (hand_box[0] + hand_box[2]) / 2, (hand_box[1] + hand_box[3]) / 2
        reach = side * VETO_REACH / 2
        x1, y1 = int(max(0, cx - reach)), int(max(0, cy - reach))
        x2, y2 = int(min(width, cx + reach)), int(min(height, cy + reach))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        result = self.veto.predict(frame[y1:y2, x1:x2], imgsz=VETO_IMGSZ, conf=VETO_CONF, verbose=False,
                                   device=self.device)[0]
        if result.boxes is None or not len(result.boxes):
            return None
        for (bx1, by1, bx2, by2), cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            name = self.veto.names[int(cls)]
            box = [float(bx1 + x1), float(by1 + y1), float(bx2 + x1), float(by2 + y1)]
            if name in self.veto_classes and _intersection(box, hand_box) / max(1e-6, _area(box)) >= VETO_INSIDE:
                return name
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
        box = [min(best[0], hand_box[0]), min(best[1], hand_box[1]),
               max(best[2], hand_box[2]), max(best[3], hand_box[3])]
        limit = side * LOCATOR_MAX_SIDE
        for low, high in ((0, 2), (1, 3)):  # oversized side: keep the window that holds the hand
            if box[high] - box[low] > limit:  # and overlaps the detected weapon the most
                start = min(max(best[low], hand_box[high] - limit), hand_box[low])
                box[low], box[high] = start, start + limit
        return box, best_conf

    def _classify(self, crops):
        import torch

        from extensions.sharp_object.hand_crop import to_tensor

        batch = to_tensor(crops).to(self.device)
        with torch.inference_mode():
            logits = self.classifier(batch.half() if self.half else batch)
        return torch.sigmoid(logits.float().squeeze(1)).cpu().numpy()

    def _pose_people(self, frames):
        """[(box, keypoints)] on the newest sample, shared with other hand-crop extensions this step."""
        from extensions.sharp_object.hand_crop import people_from_pose

        key = getattr(self, "pose_key", None)
        cached = _POSE_CACHE.get("step")
        if key is not None and cached is not None and cached["frames"] is frames and cached["key"] == key:
            return cached["persons"]
        result = self.pose.predict(frames[-1], imgsz=self.pose_imgsz, conf=self.pose_conf, verbose=False,
                                   device=self.device, half=self.half)[0]
        persons = people_from_pose(result, self.pose_conf, self.max_people)
        _POSE_CACHE["step"] = {"frames": frames, "key": key, "persons": persons}
        return persons

    def _observe_hand_crop(self, frames, people):
        from extensions.sharp_object.hand_crop import cut, hand_crops

        frame = frames[-1]
        height, width = frame.shape[:2]
        source_w, source_h = people.get("frame_size") or (width, height)
        sx, sy = source_w / width, source_h / height
        persons = self._pose_people(frames)
        crops, hands, frame_boxes = [], [], []
        for index, (box, keypoints) in enumerate(persons):
            for hand in hand_crops(box, keypoints, self.crop):
                crops.append(cut(frame, hand["box"], self.input_size))
                frame_boxes.append([float(v) for v in hand["box"]])
                hands.append({"person": index, "arm": hand["arm"],
                              "box": [round(float(v) * s, 1) for v, s in zip(hand["box"], (sx, sy, sx, sy))]})
        scores = self._classify(crops) if crops else []
        for hand, score, frame_box in zip(hands, scores, frame_boxes):
            hand["score"] = round(float(score), 4)
            if score >= self.threshold:
                held = self._vetoed(frame, frame_box)
                if held is not None:  # A: harmless object in this hand -> no alert from it
                    hand.update(raw_score=hand["score"], vetoed=held, score=0.0)
        for index in sorted({h["person"] for h in hands if h["score"] >= self.threshold}):
            tool = self._tool_vetoed(frame, [float(v) for v in persons[index][0]])
            if tool is not None and tool >= TOOL_VETO_PROB:  # B: mop / broom, not a weapon
                for hand in hands:
                    if hand["person"] == index and hand["score"] >= self.threshold:
                        hand.update(raw_score=hand["score"], vetoed="long_tool", tool_prob=round(tool, 3), score=0.0)
        scores = [hand["score"] for hand in hands]
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
                                    "conf": hand["score"], "class": self.weapon_class, "holder": len(person_boxes) - 1,
                                    "located": conf is not None})
        weapons = _merge_weapon_boxes(weapons)
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
