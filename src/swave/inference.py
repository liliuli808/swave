"""Stable NumPy inference interface for trained forward surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .network import FourHeadForwardModel
from .splits import validate_checkpoint_split_policy

VS_MIN = 0.3
VS_MAX = 2.6


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


@dataclass
class ForwardPredictor:
    """Loaded model plus the normalization required for physical predictions."""

    model: FourHeadForwardModel
    input_mean: NDArray[np.float32]
    input_std: NDArray[np.float32]
    target_mean: NDArray[np.float32]
    target_std: NDArray[np.float32]
    device: torch.device

    @classmethod
    def load(
        cls, checkpoint: Path | str, device: str = "auto"
    ) -> ForwardPredictor:
        selected_device = resolve_device(device)
        payload = torch.load(
            Path(checkpoint),
            map_location=selected_device,
            weights_only=False,
        )
        validate_checkpoint_split_policy(payload)
        model = FourHeadForwardModel()
        model.load_state_dict(payload["model"])
        model.to(selected_device)
        model.eval()
        input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
        input_std = np.asarray(payload["input_std"], dtype=np.float32)
        target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
        target_std = np.asarray(payload["target_std"], dtype=np.float32)
        if (
            input_mean.shape != (20,)
            or input_std.shape != (20,)
            or target_mean.shape != (4, 1)
            or target_std.shape != (4, 1)
        ):
            raise ValueError("checkpoint normalization arrays have invalid shapes")
        return cls(
            model=model,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            device=selected_device,
        )

    def predict(self, vs: ArrayLike) -> NDArray[np.float32]:
        """Predict physical phase velocities from one model or a model batch."""
        values = np.asarray(vs, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 20:
            raise ValueError("vs must have shape (20,) or (batch, 20)")
        if not np.all(np.isfinite(values)):
            raise ValueError("vs values must be finite")
        if np.any(values < VS_MIN) or np.any(values > VS_MAX):
            raise ValueError("vs values must lie within 0.3–2.6 km/s")

        normalized = (values - self.input_mean) / self.input_std
        tensor = torch.from_numpy(normalized).to(self.device)
        with torch.inference_mode():
            prediction = self.model(tensor).cpu().numpy()
        physical = prediction * self.target_std[None, :, :] + self.target_mean[
            None, :, :
        ]
        result = np.asarray(physical, dtype=np.float32)
        return result[0] if single else result

    def predict_with_frequencies(
        self, vs: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        frequencies = np.arange(0.5, 60.0 + 0.25, 0.5, dtype=np.float64)
        return frequencies, self.predict(vs)
