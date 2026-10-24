"""
anomaly/stgcn.py — Compact ST-GCN on the two-person skeleton graph.

Input  : (N, C, T, V) pair tensors from anomaly/skeleton.py
Output : (N, num_classes) logits per pair

PairBagClassifier scores a window as a bag of pairs (multiple-instance
learning): the window logit pools its pair logits (max, or a smooth max that
does not grow with the number of pairs), so a window labelled "fight" only
needs one fighting pair, and the label never has to say which.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from anomaly.skeleton import NUM_CHANNELS, NUM_NODES, two_person_adjacency


class GraphConv(nn.Module):
    """Spatial graph convolution over K adjacency subsets with learnable edge weights."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.subsets = adjacency.shape[0]
        self.register_buffer("adjacency", adjacency)
        self.edge_importance = nn.Parameter(torch.ones_like(adjacency))
        self.conv = nn.Conv2d(in_channels, out_channels * self.subsets, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        y = self.conv(x).view(n, self.subsets, -1, t, v)
        weights = self.adjacency * self.edge_importance
        return torch.einsum("nkctv,kvw->nctw", y, weights)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_classes: int = 1,
        channels: Sequence[int] = (64, 64, 128, 128, 256),
        strides: Sequence[int] = (1, 1, 2, 1, 2),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "channels": list(channels),
            "strides": list(strides),
            "dropout": dropout,
        }
        adjacency = torch.from_numpy(two_person_adjacency())
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)
        blocks = []
        previous = in_channels
        for index, (width, stride) in enumerate(zip(channels, strides)):
            blocks.append(
                STGCNBlock(
                    previous,
                    width,
                    adjacency,
                    stride=stride,
                    dropout=0.0 if index == 0 else dropout,
                )
            )
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(previous, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        x = self.blocks(x)
        return self.head(x.mean(dim=(2, 3)))


class PairBagClassifier(nn.Module):
    """Window-level classifier pooling the pair logits of each window.

    pool_r None: max over pairs (models trained before pool_r existed).
    pool_r = r: mean-normalised log-sum-exp, (1/r) * log(mean_k exp(r * l_k)).
    The max of more pairs is higher by chance alone, so with max pooling a
    crowded normal scene scored like a fight (video score rose with the number
    of people); the mean-normalised smooth max stays put when calm pairs are
    added and still follows the strongest pair as r grows.
    """

    def __init__(self, pool_r: float | None = None, **stgcn_kwargs):
        super().__init__()
        self.pool_r = float(pool_r) if pool_r else None
        self.encoder = STGCN(**stgcn_kwargs)

    @property
    def config(self) -> dict:
        config = dict(self.encoder.config)
        if self.pool_r is not None:
            config["pool_r"] = self.pool_r
        return config

    def pool(self, pair_logits: torch.Tensor) -> torch.Tensor:
        """(B, K, num_classes) pair logits, -inf at padding -> (B, num_classes)."""
        if self.pool_r is None:
            return pair_logits.max(dim=1).values
        count = torch.isfinite(pair_logits).sum(dim=1).clamp(min=1).to(pair_logits.dtype)
        return (torch.logsumexp(self.pool_r * pair_logits, dim=1) - torch.log(count)) / self.pool_r

    def pair_logits(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """bags (B, K, C, T, V), mask (B, K) bool -> pair logits (B, K, num_classes).

        Only real pairs go through the encoder. Feeding the all-zero padding
        slots as well put them into every BatchNorm batch statistic, so the
        normalisation depended on how many slots each batch happened to pad
        and differed between training and evaluation.
        """
        b, k = bags.shape[:2]
        flat_mask = mask.reshape(-1)
        logits = torch.full(
            (b * k, self.config["num_classes"]), float("-inf"),
            dtype=torch.float32, device=bags.device,
        )
        if flat_mask.any():
            logits[flat_mask] = self.encoder(bags.flatten(0, 1)[flat_mask]).float()
        return logits.view(b, k, -1)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Window logits (B, num_classes) pooled over each window's pairs."""
        return self.pool(self.pair_logits(bags, mask))

    def save(self, path: Path, metadata: dict | None = None) -> None:
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "PairBagClassifier":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


def predict_windows(
    model: PairBagClassifier,
    pose,
    starts: np.ndarray,
    window: dict,
    device: torch.device,
    batch_size: int = 64,
    return_pairs: bool = False,
):
    """Fight probability per window start; windows without people score 0.

    With return_pairs=True also returns, per window, the track ids of the
    highest-scoring pair (None when the window had no candidate) — the people
    to highlight on the dashboard.
    """
    from anomaly.skeleton import window_pairs

    probabilities = np.zeros(len(starts), dtype=np.float64)
    top_pairs: list = [None] * len(starts)
    pending_bags, pending_ids, pending_index = [], [], []

    def flush() -> None:
        bags, mask = collate_bags(pending_bags)
        with torch.no_grad():
            pair_logits = model.pair_logits(bags.to(device), mask.to(device))
        best = pair_logits[..., 0].argmax(dim=1).cpu().numpy()
        probabilities[pending_index] = torch.sigmoid(model.pool(pair_logits)[:, 0]).cpu().numpy()
        for slot, index in enumerate(pending_index):
            top_pairs[index] = pending_ids[slot][int(best[slot])]
        pending_bags.clear()
        pending_ids.clear()
        pending_index.clear()

    model.eval()
    for index, start in enumerate(starts):
        bag, ids = window_pairs(
            pose,
            int(start),
            window["length"],
            window["min_presence"],
            window["max_pairs"],
            window["pair_radius"],
        )
        if len(bag):
            pending_bags.append(bag)
            pending_ids.append(ids)
            pending_index.append(index)
        if len(pending_bags) >= batch_size:
            flush()
    if pending_bags:
        flush()
    return (probabilities, top_pairs) if return_pairs else probabilities


def collate_bags(bags: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (K_i, C, T, V) arrays (K_i >= 1) into (B, K_max, ...) + mask."""
    k_max = max(len(bag) for bag in bags)
    shape = bags[0].shape[1:]
    padded = np.zeros((len(bags), k_max, *shape), dtype=np.float32)
    mask = np.zeros((len(bags), k_max), dtype=bool)
    for index, bag in enumerate(bags):
        padded[index, : len(bag)] = bag
        mask[index, : len(bag)] = True
    return torch.from_numpy(padded), torch.from_numpy(mask)
