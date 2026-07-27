"""Deterministic generation of normal and anomalous layered Earth models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .config import GeologyConfig
from .empirical import material_properties


class ModelKind(IntEnum):
    NORMAL = 0
    LOW_VELOCITY = 1
    HIGH_VELOCITY = 2
    COUPLED = 3


@dataclass(frozen=True)
class GeneratedModel:
    sample_id: int
    kind: ModelKind
    vs: NDArray[np.float64]
    vp: NDArray[np.float64]
    density: NDArray[np.float64]
    background_vs: NDArray[np.float64]
    retry_count: int


def _choose_kind(draw: float, config: GeologyConfig) -> ModelKind:
    low_edge = config.normal_fraction
    high_edge = low_edge + config.low_fraction
    coupled_edge = high_edge + config.high_fraction
    if draw < low_edge:
        return ModelKind.NORMAL
    if draw < high_edge:
        return ModelKind.LOW_VELOCITY
    if draw < coupled_edge:
        return ModelKind.HIGH_VELOCITY
    return ModelKind.COUPLED


def _background(
    rng: np.random.Generator, config: GeologyConfig
) -> NDArray[np.float64]:
    surface_high = min(0.65, config.vs_max - 0.75)
    surface = rng.uniform(config.vs_min, surface_high)
    halfspace_low = max(1.60, surface + 0.60)
    halfspace = rng.uniform(halfspace_low, config.vs_max)
    increments = rng.gamma(shape=2.0, scale=1.0, size=config.layers - 1)
    increments /= increments.sum()
    values = surface + np.r_[0.0, np.cumsum(increments)] * (
        halfspace - surface
    )
    padded = np.pad(values, (1, 1), mode="edge")
    values = np.convolve(padded, np.array([0.2, 0.6, 0.2]), mode="valid")
    return np.maximum.accumulate(values)


def _anomalous_candidate(
    background: NDArray[np.float64],
    kind: ModelKind,
    rng: np.random.Generator,
    config: GeologyConfig,
) -> NDArray[np.float64]:
    values = background.copy()
    first = config.anomaly_first_layer - 1
    last = config.anomaly_last_layer - 1
    contrast = rng.uniform(0.08, 0.35)

    if kind is ModelKind.LOW_VELOCITY:
        length = int(rng.integers(1, 4))
        start = int(rng.integers(first, last - length + 2))
        indexes = np.arange(start, start + length)
        values[indexes] = np.maximum(
            config.vs_min, background[indexes] * (1.0 - contrast)
        )
    elif kind is ModelKind.HIGH_VELOCITY:
        length = int(rng.integers(1, 3))
        start = int(rng.integers(first, last - length + 2))
        indexes = np.arange(start, start + length)
        values[indexes] = np.minimum(
            config.vs_max, background[indexes] * (1.0 + contrast)
        )
    elif kind is ModelKind.COUPLED:
        low_length = int(rng.integers(2, 4))
        start = int(rng.integers(first, last - low_length + 1))
        values[start] = min(
            config.vs_max, background[start] * (1.0 + contrast)
        )
        low_indexes = np.arange(start + 1, start + 1 + low_length)
        values[low_indexes] = np.maximum(
            config.vs_min, background[low_indexes] * (1.0 - contrast)
        )
    return values


def _valid_candidate(
    values: NDArray[np.float64],
    background: NDArray[np.float64],
    kind: ModelKind,
    config: GeologyConfig,
) -> bool:
    if values.shape != (config.layers,) or not np.all(np.isfinite(values)):
        return False
    if np.any(values < config.vs_min) or np.any(values > config.vs_max):
        return False
    delta = values - background
    changed = np.abs(delta) > 1e-12
    if kind is ModelKind.NORMAL:
        return not np.any(changed) and bool(np.all(np.diff(values) >= 0))
    if not np.any(changed):
        return False
    if np.any(np.abs(delta[changed]) < 0.05 * background[changed]):
        return False
    positive = np.flatnonzero(delta > 0)
    negative = np.flatnonzero(delta < 0)
    if kind is ModelKind.LOW_VELOCITY:
        return len(positive) == 0 and 1 <= len(negative) <= 3
    if kind is ModelKind.HIGH_VELOCITY:
        return len(negative) == 0 and 1 <= len(positive) <= 2
    return (
        len(positive) == 1
        and len(negative) in (2, 3)
        and negative[0] == positive[0] + 1
        and np.array_equal(
            negative, np.arange(negative[0], negative[0] + len(negative))
        )
    )


def generate_model(
    sample_id: int,
    config: GeologyConfig,
    global_seed: int,
    retry_count: int = 0,
) -> GeneratedModel:
    """Generate one reproducible model from its stable identity."""
    if sample_id < 0 or retry_count < 0:
        raise ValueError("sample_id and retry_count must be nonnegative")
    rng = np.random.default_rng(
        np.random.SeedSequence([global_seed, sample_id, retry_count])
    )
    kind = _choose_kind(float(rng.random()), config)
    for _ in range(128):
        background = _background(rng, config)
        values = _anomalous_candidate(background, kind, rng, config)
        if not _valid_candidate(values, background, kind, config):
            continue
        vp, density = material_properties(values, config.empirical_method)
        return GeneratedModel(
            sample_id=sample_id,
            kind=kind,
            vs=values,
            vp=vp,
            density=density,
            background_vs=background,
            retry_count=retry_count,
        )
    raise RuntimeError(
        f"unable to generate valid {kind.name} model for sample {sample_id}"
    )

