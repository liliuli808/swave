"""Validated configuration objects for physics, data, and training workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PhysicsConfig:
    """Numerical controls for the fixed Rayleigh-wave forward problem."""

    fmin: float = 0.5
    fmax: float = 60.0
    fstep: float = 0.5
    mode_count: int = 4
    epsilon: float = 0.5
    nfine: int = 2
    quadratic_iterations: int = 3
    root_tolerance: float = 1e-8
    dedup_tolerance: float = 1e-7
    strategy: str = "quadratic"

    def __post_init__(self) -> None:
        if self.fmin <= 0 or self.fmax < self.fmin or self.fstep <= 0:
            raise ValueError("frequency bounds and step must be positive and ordered")
        intervals = (self.fmax - self.fmin) / self.fstep
        if not math.isclose(intervals, round(intervals), abs_tol=1e-10):
            raise ValueError("frequency range must be divisible by fstep")
        if self.mode_count != 4:
            raise ValueError("mode_count must be exactly 4")
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if self.nfine < 0 or self.quadratic_iterations < 0:
            raise ValueError("refinement iteration counts must be nonnegative")
        if self.root_tolerance <= 0 or self.dedup_tolerance <= 0:
            raise ValueError("root tolerances must be positive")
        if self.strategy not in {"raw", "degraded", "quadratic"}:
            raise ValueError("strategy must be raw, degraded, or quadratic")

    @property
    def frequencies(self) -> np.ndarray:
        count = round((self.fmax - self.fmin) / self.fstep) + 1
        return self.fmin + np.arange(count, dtype=np.float64) * self.fstep


@dataclass(frozen=True)
class GeologyConfig:
    """Controls for deterministic layered-model sampling."""

    layers: int = 20
    thickness_km: float = 0.1
    vs_min: float = 0.30
    vs_max: float = 2.60
    anomaly_first_layer: int = 3
    anomaly_last_layer: int = 12
    normal_fraction: float = 0.25
    low_fraction: float = 0.15
    high_fraction: float = 0.10
    coupled_fraction: float = 0.50
    empirical_method: str = "brocher05"

    def __post_init__(self) -> None:
        if self.layers != 20:
            raise ValueError("layers must be exactly 20")
        if not math.isclose(self.thickness_km, 0.1, abs_tol=1e-12):
            raise ValueError("thickness_km must be exactly 0.1")
        if not 0 < self.vs_min < self.vs_max:
            raise ValueError("Vs bounds must be positive and ordered")
        if not (
            1
            <= self.anomaly_first_layer
            <= self.anomaly_last_layer
            <= self.layers
        ):
            raise ValueError("anomaly layer range is invalid")
        fractions = (
            self.normal_fraction,
            self.low_fraction,
            self.high_fraction,
            self.coupled_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("model-family fractions must be nonnegative")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
            raise ValueError("model-family fractions must sum to 1")
        if self.empirical_method not in {"brocher05", "gardner", "near_surface"}:
            raise ValueError(
                "empirical_method must be brocher05, gardner, or near_surface"
            )


@dataclass(frozen=True)
class DatasetConfig:
    """Complete deterministic dataset-generation configuration."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    samples: int = 1_000_000
    shard_size: int = 10_000
    seed: int = 20_260_727
    workers: int = 0
    output_dir: Path = Path("data/production")
    max_model_retries: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.max_model_retries < 0:
            raise ValueError("max_model_retries must be nonnegative")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DatasetConfig:
        physics_values = dict(mapping.get("physics", {}))
        geology_values = dict(mapping.get("geology", {}))
        dataset_values = dict(mapping.get("dataset", {}))
        dataset_keys = {
            "samples",
            "shard_size",
            "seed",
            "workers",
            "output_dir",
            "max_model_retries",
        }
        dataset_values.update(
            {key: value for key, value in mapping.items() if key in dataset_keys}
        )
        return cls(
            physics=PhysicsConfig(**physics_values),
            geology=GeologyConfig(**geology_values),
            **dataset_values,
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


@dataclass(frozen=True)
class TrainingConfig:
    """Training and checkpoint configuration."""

    dataset_dir: Path = Path("data/production")
    output_dir: Path = Path("runs/default")
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 20_260_727
    device: str = "auto"
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> TrainingConfig:
        values = dict(mapping.get("training", mapping))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_hash(config: DatasetConfig | TrainingConfig) -> str:
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_dataset_config(path: Path | str) -> DatasetConfig:
    with Path(path).open("rb") as handle:
        return DatasetConfig.from_mapping(tomllib.load(handle))


def load_training_config(path: Path | str) -> TrainingConfig:
    with Path(path).open("rb") as handle:
        return TrainingConfig.from_mapping(tomllib.load(handle))

