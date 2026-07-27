"""Shared-backbone, four-head neural surrogate for dispersion curves."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class ResidualBlock(nn.Module):
    """A normalized residual multilayer-perceptron block."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value + self.layers(value))


class FourHeadForwardModel(nn.Module):
    """Map 20 Vs values to four independent 120-frequency modal curves."""

    def __init__(
        self,
        input_size: int = 20,
        width: int = 256,
        blocks: int = 4,
        frequencies: int = 120,
    ) -> None:
        super().__init__()
        if input_size <= 0 or width <= 0 or blocks < 0 or frequencies <= 0:
            raise ValueError("network dimensions must be positive")
        self.input_size = input_size
        self.frequencies = frequencies
        self.input = nn.Sequential(nn.Linear(input_size, width), nn.GELU())
        self.backbone = nn.Sequential(
            *(ResidualBlock(width) for _ in range(blocks))
        )
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, frequencies),
            )
            for _ in range(4)
        )

    def forward(self, vs: Tensor) -> Tensor:
        if vs.ndim != 2 or vs.shape[1] != self.input_size:
            raise ValueError(
                f"vs must have shape (batch, {self.input_size})"
            )
        features = self.backbone(self.input(vs))
        return torch.stack([head(features) for head in self.heads], dim=1)


def masked_smooth_l1(
    prediction: Tensor, target: Tensor, valid_mask: Tensor
) -> Tensor:
    """Average Smooth-L1 independently over each nonempty modal head."""
    if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
        raise ValueError("prediction, target, and valid_mask must have the same shape")
    if prediction.ndim != 3 or prediction.shape[1] != 4:
        raise ValueError("dispersion tensors must have shape (batch, 4, frequency)")
    mask = valid_mask.to(dtype=torch.bool)
    cell_loss = functional.smooth_l1_loss(
        prediction, target, reduction="none"
    )
    mode_losses = [
        cell_loss[:, mode][mask[:, mode]].mean()
        for mode in range(4)
        if torch.any(mask[:, mode])
    ]
    if not mode_losses:
        return prediction.sum() * 0.0
    return torch.stack(mode_losses).mean()
