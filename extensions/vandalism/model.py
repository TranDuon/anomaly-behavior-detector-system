"""Vandalism model shared by Colab training and the runtime extension.

Input matches what the runtime gives an extension: the last 8 of the 9 core
samples (3.75 fps, ~2.1 s). Frames are resized to 256x256 without cropping, so
the whole room stays in view (the stock V-JEPA 2 processor centre-crops away
half of a 16:9 frame).

Backbone: V-JEPA 2 ViT-L (frozen). Its 4x16x16 = 1024 output tokens are
average-pooled to a 4x4x4 grid (64 tokens x 1024), cached, and scored by a
small attentive probe. probe_01 trains only the probe; the fine-tuned variant
also retrains the last few encoder blocks (+ final LayerNorm), shipped as a
separate "tail" state dict that Backbone(tail_state=...) loads over the
pretrained weights. Same compute at runtime, only the weights differ.
"""

import math

import numpy as np
import torch
from torch import nn

HF_MODEL = "facebook/vjepa2-vitl-fpc64-256"
HF_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
FRAMES = 8
FPS = 3.75
SIZE = 256
GRID = (4, 4, 4)  # pooled (time, height, width)
TOKENS = GRID[0] * GRID[1] * GRID[2]
WIDTH = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(frames_bgr):
    """List of FRAMES BGR uint8 frames -> float tensor (FRAMES, 3, SIZE, SIZE)."""
    import cv2

    if len(frames_bgr) != FRAMES:
        raise ValueError(f"expected {FRAMES} frames, got {len(frames_bgr)}")
    stack = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1] for f in frames_bgr])
    stack = (stack.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(stack.transpose(0, 3, 1, 2)))


def pool_tokens(hidden):
    """(B, 1024 tokens, D) in t-h-w order -> (B, 64, D) on a 4x4x4 grid."""
    batch, count, width = hidden.shape
    t = FRAMES // 2
    side = int(math.isqrt(count // t))
    if t * side * side != count:
        raise ValueError(f"unexpected token count {count}")
    grid = hidden.reshape(batch, t, side, side, width).permute(0, 4, 1, 2, 3)
    pooled = nn.functional.adaptive_avg_pool3d(grid, GRID)
    return pooled.flatten(2).transpose(1, 2).contiguous()


def tail_keys(model, tail_blocks):
    """State-dict key prefixes of the last ``tail_blocks`` encoder blocks + final LayerNorm."""
    count = len(model.encoder.layer)
    if not 0 < tail_blocks <= count:
        raise ValueError(f"tail_blocks must be in 1..{count}")
    return tuple(f"encoder.layer.{i}." for i in range(count - tail_blocks, count)) + ("encoder.layernorm.",)


def tail_state_dict(model, tail_blocks):
    prefixes = tail_keys(model, tail_blocks)
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items() if k.startswith(prefixes)}


def load_tail(model, state):
    """Overwrite the encoder tail with fine-tuned weights; every key must exist with the same shape."""
    own = model.state_dict()
    wrong = [k for k, v in state.items() if k not in own or own[k].shape != v.shape]
    if wrong or not state:
        raise ValueError(f"tail state does not fit the backbone: {wrong[:5]}")
    blocks = {int(k.split(".")[2]) for k in state if k.startswith("encoder.layer.")}
    expected = set(tail_keys(model, len(blocks)))
    if {k for k in own if k.startswith(tuple(expected))} != set(state):
        raise ValueError("tail state must hold complete last blocks + encoder.layernorm")
    model.load_state_dict(state, strict=False)


def encode_head(model, clips, stop):
    """Embeddings + encoder blocks [0, stop) -> hidden states (B, 1024 tokens, D)."""
    hidden = model.encoder.embeddings(clips)
    for layer in model.encoder.layer[:stop]:
        hidden = layer(hidden, None, None, False)[0]
    return hidden


def encode_tail(layers, layernorm, hidden):
    """Remaining blocks + final LayerNorm, then pooling -> (B, 64, D)."""
    for layer in layers:
        hidden = layer(hidden, None, None, False)[0]
    return pool_tokens(layernorm(hidden))


class Backbone:
    """V-JEPA 2 encoder (frozen at inference) producing pooled tokens.

    ``tail_state``: optional fine-tuned last blocks (see load_tail)."""

    def __init__(self, device="cuda", dtype=torch.float16, model_id=HF_MODEL, revision=HF_REVISION, tail_state=None):
        from transformers import AutoModel

        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, revision=revision, dtype=self.dtype)
        if tail_state is not None:
            load_tail(self.model, tail_state)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, clips):
        """clips: (B, FRAMES, 3, SIZE, SIZE) float -> (B, 64, 1024) float16 on CPU."""
        clips = clips.to(self.device, self.dtype, non_blocking=True)
        try:
            output = self.model(pixel_values_videos=clips, skip_predictor=True)
        except TypeError:  # older/newer signatures
            output = self.model(pixel_values_videos=clips)
        return pool_tokens(output.last_hidden_state.float()).half().cpu()


class AttentiveProbe(nn.Module):
    """Learned queries attend over the 64 pooled tokens -> one logit per window."""

    def __init__(self, width=WIDTH, heads=16, queries=4, hidden=512, dropout=0.1):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, TOKENS, width))
        self.queries = nn.Parameter(torch.randn(1, queries, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(width * queries), nn.Dropout(dropout),
                                  nn.Linear(width * queries, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, tokens):
        """tokens: (N, 64, width) -> logits (N,)"""
        x = self.norm(tokens.float() + self.position)
        attended, _ = self.attention(self.queries.expand(len(x), -1, -1), x, x, need_weights=False)
        return self.head(attended.flatten(1)).squeeze(-1)


def bag_logits(window_logits, mask, k=3):
    """Multiple-instance pooling: mean of the top-k window logits of each bag.

    window_logits, mask: (B, W). Bags with fewer than k windows use all of them.
    """
    filled = window_logits.masked_fill(~mask, float("-inf"))
    counts = mask.sum(1).clamp(min=1)
    top = filled.topk(min(k, filled.shape[1]), dim=1).values
    keep = torch.arange(top.shape[1], device=top.device)[None] < counts[:, None].clamp(max=k)
    return (top.masked_fill(~keep, 0).sum(1)) / keep.sum(1)
