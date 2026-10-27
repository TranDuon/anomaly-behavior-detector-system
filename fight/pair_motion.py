from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from fight.fusion import ACTION_NAMES
from fight.model import VideoClassifier


@dataclass(frozen=True)
class MotionConfig:
    version: int = 1
    flow_size: int = 64
    output_size: int = 32
    displacement_scale: float = 8.0

    def __post_init__(self):
        if (
            self.version,
            self.flow_size,
            self.output_size,
            self.displacement_scale,
        ) != (1, 64, 32, 8.0):
            raise ValueError("Unsupported motion preprocessing")


def motion_inputs(clips, config=None):
    config = config or MotionConfig()
    clips = np.asarray(clips)
    if (
        clips.ndim != 5
        or clips.shape[1] != 16
        or clips.shape[-1] != 3
        or clips.dtype != np.uint8
    ):
        raise ValueError("Motion requires 16 uint8 RGB frames per region")
    output = np.zeros(
        (len(clips), 2, 15, config.output_size, config.output_size), np.float32
    )
    for region_index, region in enumerate(clips):
        greys = [
            cv2.cvtColor(
                cv2.resize(frame, (config.flow_size, config.flow_size)),
                cv2.COLOR_RGB2GRAY,
            )
            for frame in region
        ]
        for frame_index, (previous, current) in enumerate(zip(greys, greys[1:])):
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, None, 0.5, 2, 15, 2, 5, 1.1, 0
            )
            flow -= np.median(flow, axis=(0, 1), keepdims=True)
            flow = cv2.resize(flow, (config.output_size, config.output_size))
            output[region_index, :, frame_index] = np.clip(
                flow / config.displacement_scale, -1, 1
            ).transpose(2, 0, 1)
    return output


def pair_descriptors(poses, deltas):
    poses = torch.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    points = poses[..., :2].clamp(-5, 5)
    confidence = poses[..., 2].clamp(0, 1)
    torso_indices = [5, 6, 11, 12]
    torso_mask = (confidence[..., torso_indices] >= 0.3).to(points.dtype)
    torso = (points[..., torso_indices, :] * torso_mask.unsqueeze(-1)).sum(
        dim=-2
    ) / torso_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    torso_valid = (torso_mask.sum(dim=-1) >= 2).to(points.dtype)
    opposite = torso.flip(2)
    valid = (confidence[..., [9, 10]] >= 0.3).to(points.dtype) * torso_valid.flip(
        2
    ).unsqueeze(-1)
    offsets = (points[..., [9, 10], :] - opposite.unsqueeze(-2)) * valid.unsqueeze(-1)
    distance = offsets.square().sum(dim=-1).clamp_min(1e-8).sqrt()
    velocity = (offsets[:, 1:] - offsets[:, :-1]) / deltas
    velocity = velocity.clamp(-20, 20) * (valid[:, 1:] * valid[:, :-1]).unsqueeze(-1)
    velocity = torch.cat((torch.zeros_like(offsets[:, :1]), velocity), dim=1)
    closing = (-(offsets * velocity).sum(dim=-1) / distance.clamp_min(1e-4)).clamp(
        -20, 20
    )
    speed = velocity.square().sum(dim=-1).clamp_min(1e-8).sqrt().clamp_max(20)
    directed = torch.cat(
        (
            offsets,
            distance.unsqueeze(-1) * valid.unsqueeze(-1),
            valid.unsqueeze(-1),
            velocity,
            closing.unsqueeze(-1),
            speed.unsqueeze(-1) * valid.unsqueeze(-1),
        ),
        dim=-1,
    ).flatten(-2)
    pair_valid = torso_valid * torso_valid.flip(2)
    torso_offset = (torso - opposite) * pair_valid.unsqueeze(-1)
    torso_distance = torso_offset.square().sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    ).sqrt() * pair_valid.unsqueeze(-1)
    quality = confidence.mean(dim=-1, keepdim=True)
    descriptors = torch.cat(
        (directed, torso_offset, torso_distance, pair_valid.unsqueeze(-1), quality),
        dim=-1,
    )
    return descriptors.permute(0, 2, 3, 1)


class PairMotionClassifier(nn.Module):
    def __init__(self, use_pair=True, use_motion=True):
        super().__init__()
        self.rgb = VideoClassifier(pretrained=False)
        self.rgb.requires_grad_(False)
        self.use_pair = use_pair
        self.use_motion = use_motion
        self.register_buffer(
            "pose_deltas",
            torch.tensor([2, 2, 2, 2, 2, 2, 3], dtype=torch.float32).reshape(
                1, 7, 1, 1, 1
            )
            / 12,
        )
        self.rgb_projection = nn.Sequential(nn.Linear(2048, 64), nn.ReLU())
        self.pair_encoder = nn.Sequential(
            nn.Conv1d(21, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(2, 8, (3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1, stride=(1, 2, 2)),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(161, 64), nn.ReLU(), nn.Dropout(0.2))
        self.delta = nn.Linear(64, 1)
        self.actions = nn.Linear(64, len(ACTION_NAMES))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def classify_features(self, features, poses, motion):
        descriptors = pair_descriptors(poses, self.pose_deltas)
        pair_quality = descriptors[:, :, 19].mean(dim=-1)
        encoded_pair = self.pair_encoder(descriptors.reshape(-1, 21, 8))
        encoded_pair = torch.cat(
            (encoded_pair.mean(dim=-1), encoded_pair.amax(dim=-1)), dim=-1
        ).reshape(-1, 2, 64)
        encoded_pair = (encoded_pair * pair_quality.unsqueeze(-1)).amax(dim=1)
        encoded_motion = self.motion_encoder(motion)
        encoded_motion = torch.cat(
            (encoded_motion.mean(dim=(2, 3, 4)), encoded_motion.amax(dim=(2, 3, 4))),
            dim=-1,
        )
        quality = pair_quality.mean(dim=1, keepdim=True)
        if not self.use_pair:
            encoded_pair = encoded_pair * 0
            quality = quality * 0
        if not self.use_motion:
            encoded_motion = encoded_motion * 0
        combined = self.fusion(
            torch.cat(
                (self.rgb_projection(features), encoded_pair, encoded_motion, quality),
                dim=-1,
            )
        )
        return self.rgb.head(features).flatten() + self.delta(
            combined
        ).flatten(), self.actions(combined)

    def forward(self, clips, poses, motion):
        return self.classify_features(self.rgb.features(clips), poses, motion)[0]
