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


class ActionEngine:
    def __init__(self, weights=None, device="cuda:0", tuned=False, refined=False):
        from mmengine import Config
        from mmaction.utils import register_all_modules
        from mmdet.apis import init_detector

        register_all_modules()
        self.device = device
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
        self.detector = init_detector(
            str(ROOT / "repo/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py"),
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
        self.refinement = None
        self.head_version = 0
        self.metadata = {
            "threshold": 0.4,
            "confirm_seconds": 0.4,
            "approved_for_demo": False,
            "source_sha256": SOURCE_SHA,
            "spec": SPEC,
            "scope": "Original AVA action scores; not a calibrated violence classifier",
        }
        if refined and not tuned:
            raise ValueError("A refinement head requires the fine-tuned branch")
        if tuned:
            metadata = json.loads((self.weights / "interaction.json").read_text())
            if (
                metadata["source_sha256"] != SOURCE_SHA
                or metadata["spec"] != SPEC
                or digest(self.weights / "interaction.pt")
                != metadata["checkpoint_sha256"]
            ):
                raise ValueError(
                    "Fine-tuned branch/source/preprocessing fingerprint mismatch"
                )
            self.head_version = int(metadata.get("head_version", 1))
            if self.head_version not in {1, 2} or (
                self.head_version == 2
                and metadata.get("aux_version") != AUX_VERSION
            ):
                raise ValueError("Unsupported interaction head version")
            self.head = (
                InteractionHeadV2() if self.head_version == 2 else InteractionHead()
            ).to(device)
            self.head.load_state_dict(
                torch.load(
                    self.weights / "interaction.pt",
                    map_location=device,
                    weights_only=True,
                ),
                strict=True,
            )
            self.head.eval()
            self.metadata = metadata
        if refined:
            refinement_path = self.weights / "interaction_refined.pt"
            refinement_metadata_path = self.weights / "interaction_refined.json"
            refinement_metadata = json.loads(refinement_metadata_path.read_text())
            if (
                refinement_metadata.get("source_sha256") != SOURCE_SHA
                or refinement_metadata.get("spec") != SPEC
                or refinement_metadata.get("aux_version") != AUX_VERSION
                or refinement_metadata.get("base_checkpoint_sha256")
                != digest(self.weights / "interaction.pt")
                or refinement_metadata.get("checkpoint_sha256") != digest(refinement_path)
            ):
                raise ValueError("Fine-tuned refinement/source fingerprint mismatch")
            self.refinement = TunedRefinementHead().to(device)
            self.refinement.load_state_dict(
                torch.load(refinement_path, map_location=device, weights_only=True),
                strict=True,
            )
            self.refinement.eval()
            self.metadata = refinement_metadata

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
    def extract_features(self, frames):
        from mmdet.apis import inference_detector

        inputs, shape = rgb_inputs(frames, self.device)
        predictions = inference_detector(self.detector, frames[4]).pred_instances
        keep = (predictions.labels == 0) & (predictions.scores >= 0.9)
        selected = predictions[keep]
        order = selected.scores.argsort(descending=True)[:6]
        boxes = selected.bboxes[order].detach().cpu().numpy()
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
            if self.refinement is not None:
                scores = (
                    self.refinement(
                        features,
                        torch.from_numpy(auxiliary).to(self.device),
                        torch.from_numpy(prior).to(self.device),
                    )
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
            index = int(scores.argmax())
            score, regions = float(scores[index]), [output["candidate_regions"][index]]
            if self.head_version == 2:
                output["fusion_context"] = {
                    name: float(value)
                    for name, value in zip(AUX_NAMES, auxiliary[index])
                }
            if self.refinement is not None:
                output["legacy_tuned_score"] = float(prior[index])
        return {**output, "score": score, "regions": regions}
