"""Shared-backbone, four-head neural surrogate for dispersion curves."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

# Feature count appended to the raw Vs input when ``profile_features`` is on:
# 20 deficit + 20 excess + 6 scalars (see ``profile_feature_expansion``).
PROFILE_FEATURE_COUNT = 46


def profile_feature_expansion(vs: Tensor) -> Tensor:
    """Derive anomaly-localizing features from a Vs profile.

    The synthetic model kinds differ by thin anomalous zones: low-velocity
    layers dip below the running maximum, high-velocity layers rise above
    the running minimum of everything beneath them. These piecewise-linear
    profiles (plus a few global coordinates) let the network gate its
    capacity per structure kind instead of inferring the anomaly locations
    from raw layer values. The operations are differentiable, so Jacobians
    taken through the model (inversion sensitivities, kernel training)
    remain consistent.
    """
    if vs.ndim != 2:
        raise ValueError("vs must have shape (batch, layers)")
    deficit = torch.relu(torch.cummax(vs, dim=1).values - vs)
    # Minimum of the strictly-below layers: inclusive running minimum
    # shifted left, with +inf at the last layer (nothing beneath it).
    inclusive_min = torch.flip(
        torch.cummin(torch.flip(vs, dims=[1]), dim=1).values, dims=[1]
    )
    below_min = torch.cat(
        [
            inclusive_min[:, 1:],
            torch.full(
                (vs.shape[0], 1), torch.inf, dtype=vs.dtype, device=vs.device
            ),
        ],
        dim=1,
    )
    excess = torch.relu(vs - below_min)
    max_deficit = deficit.amax(dim=1, keepdim=True)
    max_excess = excess.amax(dim=1, keepdim=True)
    depth_deficit = torch.argmax(deficit, dim=1, keepdim=True).to(vs.dtype)
    depth_excess = torch.argmax(excess, dim=1, keepdim=True).to(vs.dtype)
    layers = torch.tensor(
        max(vs.shape[1] - 1, 1), dtype=vs.dtype, device=vs.device
    )
    return torch.cat(
        [
            deficit,
            excess,
            max_deficit,
            depth_deficit / layers,
            max_excess,
            depth_excess / layers,
            vs.mean(dim=1, keepdim=True),
            vs.amin(dim=1, keepdim=True),
        ],
        dim=1,
    )


class HigherOrderLayerNorm(nn.LayerNorm):
    """LayerNorm with an autograd graph for its mean and variance.

    The native LayerNorm JVP can have correct values but incorrect parameter
    gradients when backpropagating a derivative loss. Expressing the same
    normalization with elementary operations keeps the mixed derivatives
    intact. Inheriting LayerNorm preserves legacy weight/bias state dict keys.
    """

    def forward(self, value: Tensor) -> Tensor:
        dimensions = tuple(range(-len(self.normalized_shape), 0))
        centered = value - value.mean(dim=dimensions, keepdim=True)
        variance = centered.square().mean(dim=dimensions, keepdim=True)
        output = centered * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class ResidualBlock(nn.Module):
    """A normalized residual multilayer-perceptron block."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.norm = HigherOrderLayerNorm(width)

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
        profile_features: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0 or width <= 0 or blocks < 0 or frequencies <= 0:
            raise ValueError("network dimensions must be positive")
        self.input_size = input_size
        self.frequencies = frequencies
        self.profile_features = profile_features
        expanded = input_size + (PROFILE_FEATURE_COUNT if profile_features else 0)
        self.input = nn.Sequential(nn.Linear(expanded, width), nn.GELU())
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
        if self.profile_features:
            vs = torch.cat([vs, profile_feature_expansion(vs)], dim=1)
        features = self.backbone(self.input(vs))
        return torch.stack([head(features) for head in self.heads], dim=1)


def model_from_checkpoint(payload: Mapping[str, object]) -> FourHeadForwardModel:
    """Build the network described by a checkpoint and load its weights."""
    architecture = payload.get("architecture") or {}
    model = FourHeadForwardModel(**dict(architecture))
    model.load_state_dict(payload["model"])
    return model


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
