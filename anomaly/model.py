"""
anomaly/model.py — LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
  Encoder: LSTM (num_layers) → latent vector (last hidden state)
  Decoder: LSTM (num_layers) → reconstructs input sequence

Training signal: MSE reconstruction loss on NORMAL behaviour sequences.
Inference:       reconstruction error > threshold → anomaly.

Input shape:  (batch, seq_len, feature_dim)
Output shape: (batch, seq_len, feature_dim)  — reconstructed sequence
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_size)
        Returns:
            outputs : (batch, seq_len, hidden_size)
            (h_n, c_n): final hidden / cell states
        """
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        z    : (batch, hidden_size) — latent vector (last encoder hidden state)
        Returns: (batch, seq_len, output_size)
        """
        # Repeat latent vector across time axis
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)   # (batch, seq_len, hidden)
        out, _ = self.lstm(z_rep)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.

    Attributes:
        input_size  : feature dimension per timestep
        hidden_size : LSTM hidden units
        num_layers  : stacked LSTM layers
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        Returns reconstructed sequence of same shape.
        """
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        # Use the top-most layer's hidden state as the latent vector
        z = h_n[-1]                       # (batch, hidden_size)
        x_hat = self.decoder(z, seq_len)  # (batch, seq_len, input_size)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector (no decoding)."""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE: (batch,)."""
        return ((x - x_hat) ** 2).mean(dim=(1, 2))

    # ── Persistence helpers ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "LSTMAutoencoder":
        ckpt = torch.load(path, map_location=device)
        cfg  = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
