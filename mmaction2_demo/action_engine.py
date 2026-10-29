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


def interaction_bag(features, boxes):
    candidates = [(index,) for index in range(len(features))]
    candidates += list(itertools.combinations(range(len(features)), 2))
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


class ActionEngine:
    def __init__(self, weights=None, device="cuda:0", tuned=False, calibrated=False):
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
            self.head = InteractionHead().to(device)
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

    @torch.inference_mode()
    def extract(self, frames):
        from mmdet.apis import inference_detector

        inputs, shape = rgb_inputs(frames, self.device)
        predictions = inference_detector(self.detector, frames[4]).pred_instances
        keep = (predictions.labels == 0) & (predictions.scores >= 0.9)
        selected = predictions[keep]
        order = selected.scores.argsort(descending=True)[:6]
        boxes = selected.bboxes[order].detach().cpu().numpy()
        if not len(boxes):
            return np.empty((0, 4096), np.float32), {
                "people": 0,
                "person_boxes": [],
                "actions": [],
                "base_score": 0.0,
                "candidate_regions": [],
            }
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
        return bag.cpu().numpy(), {
            "people": len(boxes),
            "person_boxes": boxes.tolist(),
            "actions": actions,
            "base_score": float(scores[:, 64].max()),
            "candidate_regions": regions,
        }

    @torch.inference_mode()
    def infer(self, frames):
        bag, output = self.extract(frames)
        score = output["base_score"]
        regions = output["person_boxes"]
        if self.head is not None and len(bag):
            scores = (
                self.head(torch.from_numpy(bag).to(self.device)).sigmoid().cpu().numpy()
            )
            index = int(scores.argmax())
            score, regions = float(scores[index]), [output["candidate_regions"][index]]
        return {**output, "score": score, "regions": regions}
