"""Shared RGB AVA inference and a trainable interaction branch; original weights stay frozen."""

import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "163785949c5178484012a6593d12d6fcf4d36be404fa42b5899b314e91f4703b"
PERSON_SHA = "a5d8aa15c75b961b9c73f28a6e92af370835639a2939ff890852b052a5256a5b"
SPEC = {
    "sample_fps": 3.75,
    "buffer_frames": 9,
    "model_frames": 8,
    "keyframe_index": 4,
    "short_side": 256,
    "decision_stride_seconds": 16 / 30,
    "color": "RGB",
    "version": 1,
}
AUX_VERSION = 2
AUX_NAMES = (
    "base_score",
    "people_fraction",
    "is_pair",
    "motion_mean",
    "motion_peak",
    "motion_recent",
    "motion_change",
    "pair_center_distance",
    "pair_iou",
    "region_motion_mean",
    "region_motion_peak",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rgb_inputs(frames, device="cpu"):
    images = np.asarray(frames)
    if (
        images.ndim != 4
        or len(images) != 9
        or images.shape[-1] != 3
        or images.dtype != np.uint8
    ):
        raise ValueError("Nine uniform BGR uint8 source samples required")
    height, width = images.shape[1:3]
    scale = 256 / min(height, width)
    size = (round(width * scale), round(height * scale))
    rgb = np.stack(
        [
            cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)[..., ::-1]
            for frame in images[:8]
        ]
    )
    inputs = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    mean = inputs.new_tensor([123.675, 116.28, 103.53])
    std = inputs.new_tensor([58.395, 57.12, 57.375])
    return ((inputs - mean) / std).permute(3, 0, 1, 2).unsqueeze(0), (size[1], size[0])


def interaction_candidates(count):
    candidates = [(index,) for index in range(count)]
    candidates += list(itertools.combinations(range(count), 2))
    return candidates


def interaction_bag(features, boxes):
    candidates = interaction_candidates(len(features))
    values, regions = [], []
    for indices in candidates:
        first = features[indices[0]]
        second = features[indices[-1]]
        values.append(torch.cat(((first + second) / 2, (first - second).abs())))
        selected = np.asarray([boxes[index] for index in indices])
        regions.append(
            [
                float(selected[:, 0].min()),
                float(selected[:, 1].min()),
                float(selected[:, 2].max()),
                float(selected[:, 3].max()),
            ]
        )
    return (torch.stack(values) if values else features.new_empty((0, 4096))), regions


def fusion_auxiliary(frames, people, base_score, candidates, boxes=None):
    small = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        for frame in frames
    ]
    motion_maps = np.asarray(
        [
            np.abs(current.astype(np.float32) - previous.astype(np.float32)) / 255
            for previous, current in zip(small, small[1:])
        ],
        dtype=np.float32,
    )
    motion = motion_maps.mean(axis=(1, 2)) if len(motion_maps) else np.empty(0)
    mean = float(motion.mean()) if len(motion) else 0.0
    peak = float(motion.max()) if len(motion) else 0.0
    recent = float(motion[-2:].mean()) if len(motion) else 0.0
    early = float(motion[:2].mean()) if len(motion) else 0.0
    boxes = np.asarray(boxes, dtype=np.float32) if boxes is not None else None
    source_height, source_width = frames[0].shape[:2]
    values = []
    for indices in candidates:
        pair_distance = pair_iou = 0.0
        region_mean, region_peak = mean, peak
        if boxes is not None and len(boxes) > max(indices):
            selected = boxes[list(indices)]
            left, top = selected[:, :2].min(axis=0)
            right, bottom = selected[:, 2:].max(axis=0)
            x1 = max(0, min(63, int(np.floor(left * 64 / source_width))))
            y1 = max(0, min(35, int(np.floor(top * 36 / source_height))))
            x2 = max(x1 + 1, min(64, int(np.ceil(right * 64 / source_width))))
            y2 = max(y1 + 1, min(36, int(np.ceil(bottom * 36 / source_height))))
            region_motion = motion_maps[:, y1:y2, x1:x2]
            if region_motion.size:
                per_frame = region_motion.mean(axis=(1, 2))
                region_mean = float(per_frame.mean())
                region_peak = float(per_frame.max())
            if len(indices) == 2:
                first, second = selected
                centers = (selected[:, :2] + selected[:, 2:]) / 2
                heights = (selected[:, 3] - selected[:, 1]).clip(min=1)
                pair_distance = min(
                    1.0, float(np.linalg.norm(centers[0] - centers[1]) / heights.mean() / 3)
                )
                intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
                    0, min(first[3], second[3]) - max(first[1], second[1])
                )
                first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
                second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
                pair_iou = float(intersection / max(first_area + second_area - intersection, 1))
        values.append(
            (
                float(base_score),
                min(1.0, people / 6),
                float(len(indices) == 2),
                mean,
                peak,
                recent,
                recent - early,
                pair_distance,
                pair_iou,
                region_mean,
                region_peak,
            )
        )
    return np.asarray(values, dtype=np.float32).reshape(-1, len(AUX_NAMES))


class InteractionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4096))
        self.register_buffer("std", torch.ones(4096))
        self.layers = nn.Sequential(
            nn.Linear(4096, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, features):
        return self.layers(
            ((features - self.mean) / self.std.clamp_min(0.05)).clamp(-12, 12)
        ).flatten()


class InteractionHeadV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4096))
        self.register_buffer("std", torch.ones(4096))
        self.register_buffer("aux_mean", torch.zeros(len(AUX_NAMES)))
        self.register_buffer("aux_std", torch.ones(len(AUX_NAMES)))
        self.layers = nn.Sequential(
            nn.Linear(4096 + len(AUX_NAMES), 192),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(192, 1),
        )

    def forward(self, features, auxiliary):
        visual = ((features - self.mean) / self.std.clamp_min(0.05)).clamp(-12, 12)
        context = (
            (auxiliary - self.aux_mean) / self.aux_std.clamp_min(0.01)
        ).clamp(-12, 12)
        return self.layers(torch.cat((visual, context), dim=1)).flatten()


class InteractionDistanceHead(nn.Module):
    """V1-compatible head with candidate geometry and motion as extra inputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4096))
        self.register_buffer("std", torch.ones(4096))
        self.register_buffer("aux_mean", torch.zeros(len(AUX_NAMES)))
        self.register_buffer("aux_std", torch.ones(len(AUX_NAMES)))
        self.layers = nn.Sequential(
            nn.Linear(4096 + len(AUX_NAMES), 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    @torch.no_grad()
    def warm_start(self, parent):
        self.mean.copy_(parent.mean)
        self.std.copy_(parent.std)
        self.layers[0].weight[:, :4096].copy_(parent.layers[0].weight)
        self.layers[0].weight[:, 4096:].zero_()
        self.layers[0].bias.copy_(parent.layers[0].bias)
        self.layers[3].load_state_dict(parent.layers[3].state_dict())
        return self

    def forward(self, features, auxiliary):
        visual = ((features - self.mean) / self.std.clamp_min(0.05)).clamp(-12, 12)
        geometry = ((auxiliary - self.aux_mean) / self.aux_std.clamp_min(0.01)).clamp(-12, 12)
        return self.layers(torch.cat((visual, geometry), dim=1)).flatten()


class TunedRefinementHead(nn.Module):
    """Calibrate the installed v1/v2 interaction head against indoor hard negatives."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4096))
        self.register_buffer("std", torch.ones(4096))
        self.register_buffer("aux_mean", torch.zeros(len(AUX_NAMES)))
        self.register_buffer("aux_std", torch.ones(len(AUX_NAMES)))
        self.register_buffer("prior_mean", torch.zeros(1))
        self.register_buffer("prior_std", torch.ones(1))
        self.layers = nn.Sequential(
            nn.Linear(4096 + len(AUX_NAMES) + 1, 192),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(192, 1),
        )

    def forward(self, features, auxiliary, prior):
        visual = ((features - self.mean) / self.std.clamp_min(0.05)).clamp(-12, 12)
        context = ((auxiliary - self.aux_mean) / self.aux_std.clamp_min(0.01)).clamp(
            -12, 12
        )
        prior = ((prior.reshape(-1, 1) - self.prior_mean) / self.prior_std.clamp_min(0.01)).clamp(
            -12, 12
        )
        return self.layers(torch.cat((visual, context, prior), dim=1)).flatten()


def match_display_people(reference, current_frame, original, current):
    def histogram(frame, box):
        height, width = frame.shape[:2]
        left, top, right, bottom = np.rint(box).astype(int)
        crop = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
        if not crop.size:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        values = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(values, values).flatten()

    old_histograms = [histogram(reference, box) for box in original]
    new_histograms = [histogram(current_frame, box) for box in current]
    proposals = []
    for old_index, old in enumerate(original):
        old = np.asarray(old)
        candidates = []
        for new_index, new in enumerate(current):
            new = np.asarray(new)
            if old_histograms[old_index] is None or new_histograms[new_index] is None:
                continue
            distance = np.linalg.norm((old[:2] + old[2:] - new[:2] - new[2:]) / 2)
            scale = max(old[3] - old[1], new[3] - new[1], 1)
            appearance = cv2.compareHist(old_histograms[old_index], new_histograms[new_index],
                                         cv2.HISTCMP_BHATTACHARYYA)
            if distance / scale > 3 or appearance > 0.65:
                continue
            candidates.append((appearance + 0.12 * distance / scale, old_index, new_index))
        candidates.sort()
        if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 0.1):
            proposals.append(candidates[0])
    assignments, used = {}, set()
    for _, old_index, new_index in sorted(proposals):
        if new_index not in used:
            assignments[old_index] = new_index
            used.add(new_index)
    return assignments


def remap_display_regions(original, current, assignments, regions):
    mapped = []
    for region in regions:
        for count in (1, 2):
            matched = False
            for indices in itertools.combinations(range(len(original)), count):
                boxes = np.asarray([original[index] for index in indices])
                bounds = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
                if np.allclose(bounds, region, atol=2):
                    if all(index in assignments for index in indices):
                        mapped.extend(current[assignments[index]] for index in indices)
                    matched = True
                    break
            if matched:
                break
    return mapped


class ActionEngine:
    def __init__(self, weights=None, device="cuda:0", tuned=False, refined=False, calibrated=False,
                 fall_wrestle=False, transition=False, person_threshold=0.9, person_recovery=False):
        from mmengine import Config
        from mmaction.utils import register_all_modules
        from mmdet.apis import init_detector

        register_all_modules()
        self.device = device
        if not 0 < person_threshold <= 1:
            raise ValueError("person_threshold must be in (0, 1]")
        self.person_threshold = person_threshold
        self.person_recovery = person_recovery
        self.weights = Path(weights or ROOT / "weights")
        for name, expected in (
            ("action_slowonly_r101.pth", SOURCE_SHA),
            ("person_faster_rcnn_r50.pth", PERSON_SHA),
        ):
            if digest(self.weights / name) != expected:
                raise ValueError("Unexpected official checkpoint: " + name)
        config = Config.fromfile(
            str(
                ROOT
                / "repo/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py"
            )
        )
        config.test_dataloader.dataset.proposal_file = None
        self.model = init_detector(
            config, str(self.weights / "action_slowonly_r101.pth"), device=device
        ).eval()
        detector_config = Config.fromfile(str(ROOT / "repo/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py"))
        if person_recovery:
            detector_config.model.test_cfg.rcnn.nms.iou_threshold = 0.7
        self.detector = init_detector(
            detector_config,
            str(self.weights / "person_faster_rcnn_r50.pth"),
            device=device,
        ).eval()
        for model in (self.model, self.detector):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        lines = (ROOT / "repo/tools/data/ava/label_map.txt").read_text().splitlines()
        self.labels = {
            int(line.split(": ", 1)[0]): line.split(": ", 1)[1] for line in lines
        }
        self.head = None
        self.head_version = 0
        self.metadata = {
            "threshold": 0.4,
            "confirm_seconds": 0.4,
            "approved_for_demo": False,
            "source_sha256": SOURCE_SHA,
            "spec": SPEC,
            "scope": "Original AVA action scores; not a calibrated violence classifier",
        }
        calibration_path = ROOT / "config/original_calibration.json"
        if calibrated and not tuned:
            calibration = json.loads(calibration_path.read_text())
            if (
                calibration["source_sha256"] != SOURCE_SHA
                or calibration["engine_sha256"] != hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                or calibration["spec"] != SPEC
                or not 0 < calibration["threshold"] < 1
                or calibration["confirm_seconds"] not in (0.4, 0.8, 1.2)
            ):
                raise ValueError("Incompatible original-model calibration")
            self.metadata.update(calibration)
        if refined and not tuned:
            raise ValueError("A refinement head requires the fine-tuned branch")

        if (fall_wrestle or transition) and not tuned:
            raise ValueError("Pilot checkpoint requires the interaction head")
        if fall_wrestle and transition:
            raise ValueError("Select one pilot checkpoint")
        if tuned:
            stem = "interaction_transition" if transition else "interaction_fall_wrestle" if fall_wrestle else "interaction"
            checkpoint_name = stem + ".pt"
            metadata_name = stem + ".json"
            metadata = json.loads((self.weights / metadata_name).read_text())
            if (
                metadata["source_sha256"] != SOURCE_SHA
                or metadata["spec"] != SPEC
                or digest(self.weights / checkpoint_name)
                != metadata["checkpoint_sha256"]
            ):
                raise ValueError(
                    "Fine-tuned branch/source/preprocessing fingerprint mismatch"
                )
            self.head_version = int(metadata.get("head_version", 1))
            if self.head_version != 1:
                raise ValueError("Only mmaction_interaction_v1 is supported")
            if fall_wrestle:
                calibration = json.loads((ROOT / "config/fall_wrestle_calibration.json").read_text())
                if (calibration["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                        or not 0 < calibration["threshold"] < 1):
                    raise ValueError("Fall/wrestle calibration does not match checkpoint")
                metadata = {**metadata, "threshold": calibration["threshold"],
                            "calibration": calibration}
            self.head = InteractionHead().to(device)
            self.head.load_state_dict(
                torch.load(
                    self.weights / checkpoint_name,
                    map_location=device,
                    weights_only=True,
                ),
                strict=True,
            )
            self.head.eval()
            self.metadata = metadata

    @torch.inference_mode()
    def warmup(self):
        from mmdet.apis import inference_detector

        frames = np.zeros((9, 360, 640, 3), dtype=np.uint8)
        inference_detector(self.detector, frames[4])
        inputs, shape = rgb_inputs(frames, self.device)
        rois = inputs.new_tensor([[0, 0, 0, shape[1] - 1, shape[0] - 1]])
        self.model.roi_head._bbox_forward(self.model.extract_feat(inputs), rois,
                                          [dict(img_shape=shape)])
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def tuned_scores(self, bag, auxiliary):
        if self.head is None:
            raise ValueError("Fine-tuned interaction head is required")
        features = torch.from_numpy(bag).to(self.device)
        context = torch.from_numpy(auxiliary).to(self.device)
        logits = (
            self.head(features, context)
            if self.head_version == 2
            else self.head(features)
        )
        return logits.sigmoid().cpu().numpy()

    @torch.inference_mode()
    def detect_people(self, frame):
        from mmdet.apis import inference_detector

        if self.person_recovery:
            from person_recovery import select_people

            def detect(image):
                predictions = inference_detector(self.detector, image).pred_instances
                selected = predictions[predictions.labels == 0]
                return selected.bboxes.detach().cpu().numpy(), selected.scores.detach().cpu().numpy()

            return select_people(frame, detect)

        predictions = inference_detector(self.detector, frame).pred_instances
        keep = (predictions.labels == 0) & (predictions.scores >= self.person_threshold)
        selected = predictions[keep]
        order = selected.scores.argsort(descending=True)[:6]
        return selected.bboxes[order].detach().cpu().numpy()

    @torch.inference_mode()
    def current_view(self, frames, output):
        original = output["person_boxes"]
        current = self.detect_people(frames[-1]).tolist()
        assignments = match_display_people(frames[4], frames[-1], original, current)
        actions = [{"box": box, "labels": [], "fight_score": 0.0} for box in current]
        for old_index, new_index in assignments.items():
            if old_index < len(output.get("actions", [])):
                actions[new_index] = {**output["actions"][old_index], "box": current[new_index],
                                      "action_box": original[old_index]}
        return {**output, "person_boxes": current, "actions": actions,
                "regions": remap_display_regions(original, current, assignments, output.get("regions", [])),
                "running_regions": remap_display_regions(original, current, assignments, output.get("running_regions", [])),
                "action_person_boxes": original, "box_frame_index": -1,
                "attribution_uncertain": len(assignments) < min(len(original), len(current))}

    @torch.inference_mode()
    def extract_features(self, frames):
        inputs, shape = rgb_inputs(frames, self.device)
        boxes = self.detect_people(frames[4])
        if not len(boxes):
            return (
                np.empty((0, 4096), np.float32),
                np.empty((0, len(AUX_NAMES)), np.float32),
                {
                "people": 0,
                "person_boxes": [],
                "actions": [],
                "base_score": 0.0,
                "candidate_regions": [],
                },
            )
        scaled = torch.as_tensor(boxes, device=self.device).clone()
        scaled[:, [0, 2]] *= shape[1] / frames[0].shape[1]
        scaled[:, [1, 3]] *= shape[0] / frames[0].shape[0]
        rois = torch.cat(
            (torch.zeros(len(boxes), 1, device=self.device), scaled), dim=1
        )
        result = self.model.roi_head._bbox_forward(
            self.model.extract_feat(inputs), rois, [dict(img_shape=shape)]
        )
        scores = result["cls_score"].sigmoid().cpu().numpy()
        features = result["bbox_feats"].mean(dim=(2, 3, 4))
        candidates = interaction_candidates(len(features))
        bag, regions = interaction_bag(features, boxes)
        actions = []
        for box, values in zip(boxes, scores):
            labels = [
                {"id": index, "name": name, "score": float(values[index])}
                for index, name in self.labels.items()
                if values[index] >= 0.4
            ]
            actions.append(
                {
                    "box": box.tolist(),
                    "labels": sorted(labels, key=lambda item: -item["score"])[:4],
                    "fight_score": float(values[64]),
                    "ava_running_score": float(values[10]),
                    "ava_grab_score": float(values[66]),
                    "ava_fall_score": float(values[5]),
                    "ava_lying_score": float(values[8]),
                    "ava_hug_score": float(values[70]),
                }
            )
        base_score = float(scores[:, 64].max())
        auxiliary = fusion_auxiliary(
            frames, len(boxes), base_score, candidates, boxes=boxes
        )
        return bag.cpu().numpy(), auxiliary, {
            "people": len(boxes),
            "person_boxes": boxes.tolist(),
            "actions": actions,
            "base_score": base_score,
            "candidate_regions": regions,
        }

    def extract(self, frames):
        bag, _, output = self.extract_features(frames)
        return bag, output

    @torch.inference_mode()
    def infer(self, frames):
        bag, auxiliary, output = self.extract_features(frames)
        score = output["base_score"]
        regions = output["person_boxes"]
        if self.head is not None and len(bag):
            features = torch.from_numpy(bag).to(self.device)
            prior = (
                (
                    self.head(
                        features, torch.from_numpy(auxiliary).to(self.device)
                    )
                    if self.head_version == 2
                    else self.head(features)
                )
                .sigmoid()
                .cpu()
                .numpy()
            )
            scores = prior
            index = int(scores.argmax())
            score, regions = float(scores[index]), [output["candidate_regions"][index]]
            if self.head_version == 2:
                output["fusion_context"] = {
                    name: float(value)
                    for name, value in zip(AUX_NAMES, auxiliary[index])
                }
        return {**output, "score": score, "regions": regions}
