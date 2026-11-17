"""Hand-crop method shared by dataset building, training and the runtime extension.

Pipeline: a pose model finds people and their elbow/wrist keypoints -> a square crop is cut
around each visible hand (sized by the person, pushed a little past the wrist along the
forearm so a held blade/handle falls inside) -> a ResNet classifies each crop as
"holding a sharp object" or not. Keeping the geometry here guarantees that training crops
and runtime crops are cut the same way.
"""

import numpy as np

# COCO-17 keypoint indices (Ultralytics pose models): (elbow, wrist) per arm.
ARMS = {"left": (7, 9), "right": (8, 10)}
CROP_DEFAULTS = {
    "crop_scale": 0.45,    # crop side = crop_scale x max(person height, person width)
    "forward": 0.5,        # crop centre = wrist + forward x (wrist - elbow): towards the hand / held object
    "min_side": 48,        # pixels
    "keypoint_conf": 0.3,  # wrist (and elbow, for the push) must be at least this confident
}
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_VALUE = 114


def crop_params(spec=None):
    params = dict(CROP_DEFAULTS)
    params.update((spec or {}).get("crop", {}))
    return params


def hand_crops(person_box, keypoints, params=None):
    """Square crop boxes around the visible hands of one person.

    person_box: (x1, y1, x2, y2); keypoints: (17, 3) array of x, y, conf.
    Returns [{"arm", "box": [x1, y1, x2, y2], "wrist": [x, y]}] in the same pixel frame.
    """
    params = params or CROP_DEFAULTS
    x1, y1, x2, y2 = (float(v) for v in person_box)
    side = max(float(params["min_side"]), params["crop_scale"] * max(y2 - y1, x2 - x1))
    threshold = params["keypoint_conf"]
    crops = []
    for arm, (elbow, wrist) in ARMS.items():
        wx, wy, wc = (float(v) for v in keypoints[wrist])
        if wc < threshold:
            continue
        cx, cy = wx, wy
        ex, ey, ec = (float(v) for v in keypoints[elbow])
        if ec >= threshold:
            cx += params["forward"] * (wx - ex)
            cy += params["forward"] * (wy - ey)
        half = side / 2
        crops.append({"arm": arm, "box": [cx - half, cy - half, cx + half, cy + half], "wrist": [wx, wy]})
    return crops


def cut(frame, box, size):
    """Crop ``box`` from a BGR frame (padding outside the frame) and resize to size x size."""
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        return np.full((size, size, 3), PAD_VALUE, np.uint8)
    pad = (max(0, -y1), max(0, y2 - height), max(0, -x1), max(0, x2 - width))
    patch = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    interpolation = cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interpolation)


def to_tensor(crops_bgr):
    """List of size x size BGR uint8 crops -> normalised float tensor (N, 3, size, size)."""
    import torch

    stack = np.stack(crops_bgr)[:, :, :, ::-1].astype(np.float32) / 255.0
    stack = (stack - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(stack.transpose(0, 3, 1, 2)))


def people_from_pose(result, min_conf=0.4, max_people=8):
    """[(box xyxy, keypoints (17, 3))] from one Ultralytics pose result, largest people first."""
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    if keypoints.shape[-1] == 2:  # no visibility channel: treat every point as visible
        keypoints = np.concatenate([keypoints, np.ones(keypoints.shape[:-1] + (1,), keypoints.dtype)], -1)
    keep = [i for i in np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if confs[i] >= min_conf]
    return [(boxes[i], keypoints[i]) for i in keep[:max_people]]


def build_classifier(arch="resnet50", pretrained=True):
    """torchvision ResNet with a single-logit head ("holding a sharp object")."""
    import torchvision
    from torch import nn

    builders = {"resnet18": ("resnet18", "IMAGENET1K_V1"), "resnet34": ("resnet34", "IMAGENET1K_V1"),
                "resnet50": ("resnet50", "IMAGENET1K_V2")}
    if arch not in builders:
        raise ValueError(f"unknown arch {arch}; choose from {sorted(builders)}")
    name, weights = builders[arch]
    model = getattr(torchvision.models, name)(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_classifier(path, device="cpu"):
    """Load {"arch", "state_dict"} saved by train_hand_classifier export (tensors only, weights_only)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_classifier(payload["arch"], pretrained=False)
    model.load_state_dict({k: v.float() for k, v in payload["state_dict"].items()})
    return model.to(device).eval(), payload["arch"]
