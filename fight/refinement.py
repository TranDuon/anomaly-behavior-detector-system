"""Residual adaptation that preserves the complete deployed RGB-aux classifier."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from fight.fusion import ACTION_NAMES, InteractionFusion
from fight.pair_motion import pair_descriptors


class RefinedFightClassifier(nn.Module):
    def __init__(self, use_interaction=True):
        super().__init__()
        self.base = InteractionFusion(use_pose=False)
        self.use_interaction = use_interaction
        self.register_buffer(
            "pose_deltas",
            torch.tensor([2, 2, 2, 2, 2, 2, 3], dtype=torch.float32).reshape(
                1, 7, 1, 1, 1
            )
            / 12,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 128), nn.ReLU()
        )
        self.pair_encoder = nn.Sequential(
            nn.Conv1d(21, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, stride=(1, 2, 2), padding=1),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(225, 128), nn.ReLU(), nn.Dropout(0.1))
        self.delta = nn.Linear(128, 1)
        self.actions = nn.Linear(128, len(ACTION_NAMES))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.set_stage("probe")

    def set_stage(self, stage):
        if stage not in {"probe", "finetune"}:
            raise ValueError("Expected probe or finetune")
        self.requires_grad_(True)
        self.base.requires_grad_(False)
        if stage == "finetune":
            for block in list(self.base.rgb.backbone.blocks)[-2:]:
                block.requires_grad_(True)
        if not self.use_interaction:
            self.pair_encoder.requires_grad_(False)
            self.motion_encoder.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def classify_features(self, features, poses, motion):
        base_logits, unused = self.base.classify_features(
            features, torch.zeros_like(poses)
        )
        pair = features.new_zeros((features.shape[0], 64))
        flow = features.new_zeros((features.shape[0], 32))
        quality = features.new_zeros((features.shape[0], 1))
        if self.use_interaction:
            descriptors = pair_descriptors(poses, self.pose_deltas)
            visible = descriptors[:, :, 19].mean(dim=-1)
            encoded = self.pair_encoder(descriptors.reshape(-1, 21, 8))
            encoded = torch.cat(
                (encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1
            ).reshape(-1, 2, 64)
            pair = (encoded * visible.unsqueeze(-1)).amax(dim=1)
            quality = visible.mean(dim=1, keepdim=True)
            encoded_flow = self.motion_encoder(motion)
            flow = torch.cat(
                (encoded_flow.mean(dim=(2, 3, 4)), encoded_flow.amax(dim=(2, 3, 4))),
                dim=-1,
            )
        combined = self.fusion(
            torch.cat((self.projection(features), pair, flow, quality), dim=-1)
        )
        return base_logits + self.delta(combined).flatten(), self.actions(combined)

    def forward(self, clips, poses, motion):
        return self.classify_features(self.base.rgb.features(clips), poses, motion)[0]


def load_deployed_base(models, use_interaction=True):
    models = Path(models)
    metadata = json.loads((models / "fight_runtime.json").read_text(encoding="utf-8"))
    if (
        metadata.get("architecture") != "rgb_pose_x3d_s"
        or metadata.get("use_pose") is not False
    ):
        raise ValueError(
            "Expected the complete RGB-aux source classifier, not legacy RGB or pose-aux weights"
        )
    clip = metadata["clip"]
    if (clip["frames"], clip["fps"], clip["size"]) != (16, 12, 182):
        raise ValueError("Unexpected deployed source timebase/size")
    source = models / "fight_model.ts"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "7cd20e23e1590689a1d08bf63a9aa967c831a71dd985fed1a09fae5bd32b7d09":
        raise ValueError(
            "Unrecognized source checkpoint; update the protocol explicitly, not silently"
        )
    scripted = torch.jit.load(str(source), map_location="cpu").eval()
    model = RefinedFightClassifier(use_interaction)
    model.base.load_state_dict(scripted.state_dict(), strict=True)
    return model.eval(), metadata, digest
