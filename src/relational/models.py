"""Tiny MLPs sized for CPU."""

from __future__ import annotations

import torch
from torch import nn


class PDFRegressor(nn.Module):
    """The 2024-04-30 PDF's MNIST regressor, downsized for CPU.

    Original: ``Linear(784, 4096) → ReLU → Linear(4096, 3) → Linear(3, 1)``.
    We use 256 hidden — preserves the qualitative behaviour and runs in
    seconds per epoch on a laptop.

    Split into ``encoder`` (everything up to the bottleneck) and ``head``
    (final projection) so that exp1 can inject noise either at the input
    or at the bottleneck embedding.
    """

    def __init__(self, hidden: int = 256, bottleneck: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.head = nn.Linear(bottleneck, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def forward_with_emb_noise(
        self, x: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Run the forward with ``noise`` added at the bottleneck embedding."""
        return self.head(self.encoder(x) + noise)


class TinyMLP(nn.Module):
    """Small MLP for the 1D synthetic task."""

    def __init__(self, in_dim: int = 1, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
