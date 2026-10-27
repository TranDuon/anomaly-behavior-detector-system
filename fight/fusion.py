"""Small RGB/pose residual fusion with weak multi-label action supervision."""

import torch
from torch import nn

from fight.model import VideoClassifier

ACTION_NAMES = (
    "fight",
    "hug",
    "handshake_greet",
    "highfive_clap",
    "gesture",
    "solo_motion",
)


class InteractionFusion(nn.Module):
    def __init__(self, use_pose=True):
        super().__init__()
        self.rgb = VideoClassifier(pretrained=False)
        self.use_pose = use_pose
        self.rgb_projection = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(102, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fusion = nn.Sequential(nn.Linear(193, 128), nn.ReLU(), nn.Dropout(0.2))
        self.delta = nn.Linear(128, 1)
        self.actions = nn.Linear(128, len(ACTION_NAMES))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def classify_features(self, features, poses):
        quality = poses[..., 2].mean(dim=(1, 2, 3)).unsqueeze(1)
        sequence = poses.flatten(2).transpose(1, 2)
        encoded = self.pose_encoder(sequence).flatten(1)
        if not self.use_pose:
            encoded = encoded * 0
            quality = quality * 0
        combined = self.fusion(
            torch.cat(
                (self.rgb_projection(features), encoded * quality, quality), dim=1
            )
        )
        logits = self.rgb.head(features).flatten() + self.delta(combined).flatten()
        return logits, self.actions(combined)

    def forward(self, clips, poses):
        logits, actions = self.classify_features(self.rgb.features(clips), poses)
        return logits

    def set_stage(self, stage):
        self.rgb.set_stage(stage)
        for parameter in self.rgb.head.parameters():
            parameter.requires_grad = False
