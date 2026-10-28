"""Thin CPU adapter for the official PyTorchVideo AVA person-action detector."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as torch_functional

AVA_ACTIONS = {
    9: "martial_art",
    58: "throw",
    64: "fight_hit_person",
    66: "grab_person",
    67: "hand_clap",
    68: "hand_shake",
    69: "hand_wave",
    70: "hug_person",
    71: "kick_person",
    73: "lift_person",
    76: "push_person",
}
AVA_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ava/SLOW_4x16_R50_DETECTION.pyth"
AVA_CHECKPOINTS = {
    "slow_r50": AVA_CHECKPOINT_URL,
    "slowfast_r50": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ava/SLOWFAST_8x8_R50_DETECTION.pyth",
}


def ava_inputs(frames, boxes, short_side=256, num_frames=4):
    images = np.asarray(frames)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Expected nonempty uint8 BGR video frames")
    if not len(images) or not 0 < min(images.shape[1:3]) or short_side != 256:
        raise ValueError(
            "Official AVA input requires positive dimensions and short side 256"
        )
    if num_frames not in {4, 32}:
        raise ValueError("Official AVA Slow/SlowFast requires 4/32 frames")
    proposals = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if not np.isfinite(proposals).all():
        raise ValueError("Finite AVA person proposals required")
    height, width = images.shape[1:3]
    proposals[:, [0, 2]] = np.clip(proposals[:, [0, 2]], 0, width - 1)
    proposals[:, [1, 3]] = np.clip(proposals[:, [1, 3]], 0, height - 1)
    if np.any(
        (proposals[:, 2] <= proposals[:, 0]) | (proposals[:, 3] <= proposals[:, 1])
    ):
        raise ValueError("Nonempty AVA person proposals required")
    indices = torch.linspace(0, len(images) - 1, num_frames).long().numpy()
    rgb = (
        torch.from_numpy(images[indices, :, :, ::-1].copy()).permute(3, 0, 1, 2).float()
        / 255
    )
    scale = short_side / min(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    rgb = torch_functional.interpolate(
        rgb, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    proposals *= np.asarray([scale, scale, scale, scale], dtype=np.float32)
    proposals[:, [0, 2]] = np.clip(proposals[:, [0, 2]], 0, new_width - 1)
    proposals[:, [1, 3]] = np.clip(proposals[:, [1, 3]], 0, new_height - 1)
    batch_boxes = torch.cat(
        (torch.zeros(len(proposals), 1), torch.from_numpy(proposals)), dim=1
    )
    return ((rgb - 0.45) / 0.225).unsqueeze(0), batch_boxes


def action_scores(predictions):
    values = torch.as_tensor(predictions).detach().cpu()
    if values.ndim != 2 or values.shape[1] != 80:
        raise ValueError(
            "AVA checkpoint must return 80 sigmoid action scores per person"
        )
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("AVA detector did not return finite sigmoid scores")
    return [
        {name: float(row[action_id - 1]) for action_id, name in AVA_ACTIONS.items()}
        for row in values
    ]


class AvaPersonActions:
    def __init__(self, checkpoint, variant="slow_r50", device="cpu"):
        from pytorchvideo.models.hub import slow_r50_detection, slowfast_r50_detection

        if variant not in AVA_CHECKPOINTS:
            raise ValueError("Unsupported official AVA architecture")
        self.variant, self.device = variant, device
        checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        factory = (
            slow_r50_detection if variant == "slow_r50" else slowfast_r50_detection
        )
        self.model = factory(pretrained=False).eval()
        self.model.load_state_dict(weights["model_state"], strict=True)
        self.model.to(device)

    def tensors(self, frames, boxes):
        inputs, proposals = ava_inputs(
            frames, boxes, num_frames=4 if self.variant == "slow_r50" else 32
        )
        inputs, proposals = inputs.to(self.device), proposals.to(self.device)
        if self.variant == "slowfast_r50":
            indices = torch.linspace(0, 31, 8, device=inputs.device).long()
            inputs = [inputs.index_select(2, indices), inputs]
        return inputs, proposals

    def predict(self, frames, boxes):
        if not len(boxes):
            return []
        inputs, proposals = self.tensors(frames, boxes)
        with torch.inference_mode():
            return action_scores(self.model(inputs, proposals))
