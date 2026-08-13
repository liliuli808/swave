"""Validated configuration objects for physics, data, and training workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

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
            1 <= self.anomaly_first_layer <= self.anomaly_last_layer <= self.layers
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
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
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


NoiseScenario = Literal["clean", "noise_1pct"]


@dataclass(frozen=True)
class InversionConfig:
    """Scientific and execution controls for production inversion runs."""

    dataset_config: Path = Path("configs/dataset.toml")
    dataset_dir: Path = Path("data/production")
    checkpoint: Path = Path("runs/production-48g/best.pt")
    output_dir: Path = Path("results/inversion")
    mode_weights: tuple[float, float, float, float] = (4.0, 1.0, 1.0, 1.0)
    regularization_lambda: float = 1e-2
    regularization_type: str = "adaptive"
    vs_min: float = 0.3
    vs_max: float = 2.6
    vs_width: float = 0.7
    max_iterations: int = 100
    relative_tolerance: float = 1e-5
    initial_models: int = 100
    minimum_valid_solutions: int = 20
    samples_per_kind: int = 100
    deep_samples_per_job: int = 10
    noise_scenarios: tuple[NoiseScenario, ...] = ("clean", "noise_1pct")
    seed: int = 20_260_727
    device: str = "auto"
    workers: int = 0
    threads_per_worker: int = 1
    task_index: int = 0
    task_count: int = 1

    def __post_init__(self) -> None:
        for name in ("dataset_config", "dataset_dir", "checkpoint", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not isinstance(self.mode_weights, tuple) or len(self.mode_weights) != 4:
            raise ValueError("mode_weights must contain four finite nonnegative values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in self.mode_weights
        ):
            raise ValueError("mode_weights must contain four finite nonnegative values")
        if not any(self.mode_weights):
            raise ValueError("at least one mode weight must be positive")
        if (
            isinstance(self.regularization_lambda, bool)
            or not isinstance(self.regularization_lambda, (int, float))
            or not math.isfinite(float(self.regularization_lambda))
            or self.regularization_lambda < 0
        ):
            raise ValueError("regularization_lambda must be finite and nonnegative")
        if self.regularization_type not in {"adaptive", "first_order"}:
            raise ValueError("regularization_type must be adaptive or first_order")
        bounds = (self.vs_min, self.vs_max, self.vs_width)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in bounds
        ):
            raise ValueError("Vs bounds and vs_width must be finite")
        if not 0.3 <= self.vs_min < self.vs_max <= 2.6:
            raise ValueError(
                "Vs bounds must stay within the supported [0.3, 2.6] range"
            )
        if not 0 < self.vs_width <= self.vs_max - self.vs_min:
            raise ValueError("vs_width is invalid for the configured Vs bounds")
        integer_fields = (
            "max_iterations",
            "initial_models",
            "minimum_valid_solutions",
            "samples_per_kind",
            "deep_samples_per_job",
            "seed",
            "workers",
            "threads_per_worker",
            "task_index",
            "task_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if (
            isinstance(self.relative_tolerance, bool)
            or not isinstance(self.relative_tolerance, (int, float))
            or not math.isfinite(float(self.relative_tolerance))
            or not 0 < self.relative_tolerance < 1
        ):
            raise ValueError("relative_tolerance must be finite and in (0, 1)")
        if (
            self.initial_models <= 0
            or not 1 <= self.minimum_valid_solutions <= self.initial_models
        ):
            raise ValueError("ensemble solution counts are invalid")
        if self.samples_per_kind <= 0:
            raise ValueError("samples_per_kind must be positive")
        if self.deep_samples_per_job <= 0:
            raise ValueError("deep_samples_per_job must be positive")
        if (
            not isinstance(self.noise_scenarios, tuple)
            or not self.noise_scenarios
            or any(
                value not in {"clean", "noise_1pct"} for value in self.noise_scenarios
            )
        ):
            raise ValueError("noise_scenarios are invalid")
        if len(set(self.noise_scenarios)) != len(self.noise_scenarios):
            raise ValueError("noise_scenarios must be unique")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.threads_per_worker <= 0:
            raise ValueError("threads_per_worker must be positive")
        if self.task_count <= 0 or not 0 <= self.task_index < self.task_count:
            raise ValueError("task_index must be in [0, task_count)")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> InversionConfig:
        values = dict(mapping.get("inversion", mapping))
        known_keys = {item.name for item in dataclasses.fields(cls)}
        unknown_keys = set(values) - known_keys
        if unknown_keys:
            raise ValueError(f"unknown inversion keys: {sorted(unknown_keys)}")
        for name in ("mode_weights", "noise_scenarios"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


@dataclass(frozen=True)
class HybridInversionConfig:
    """Scientific and execution controls for learning-prior inversion."""

    dataset_config: Path = Path("configs/dataset.toml")
    dataset_dir: Path = Path("data/production")
    forward_checkpoint: Path = Path("runs/production-48g/best.pt")
    supervised_dir: Path = Path("runs/supervised-inversion-v2")
    output_dir: Path = Path("results/hybrid-inversion")
    mode_weights: tuple[float, float, float, float] = (4.0, 1.0, 1.0, 1.0)
    smoothness_lambda: float = 1e-2
    regularization_type: str = "adaptive"
    prior_lambda_candidates: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    )
    sensitivity_epsilon_fraction: float = 1e-2
    sensitivity_phase_floor: float = 1e-12
    prior_weight_min: float = 0.25
    prior_weight_max: float = 4.0
    vs_min: float = 0.3
    vs_max: float = 2.6
    reference_width: float = 0.7
    max_iterations: int = 100
    relative_tolerance: float = 1e-5
    validation_samples_per_kind: int = 100
    noise_scenarios: tuple[NoiseScenario, ...] = ("clean", "noise_1pct")
    seed: int = 20_260_727
    device: str = "auto"
    workers: int = 0
    threads_per_worker: int = 1
    task_index: int = 0
    task_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "dataset_config",
            "dataset_dir",
            "forward_checkpoint",
            "supervised_dir",
            "output_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(
            self, "mode_weights", tuple(float(value) for value in self.mode_weights)
        )
        object.__setattr__(
            self,
            "prior_lambda_candidates",
            tuple(float(value) for value in self.prior_lambda_candidates),
        )
        object.__setattr__(self, "noise_scenarios", tuple(self.noise_scenarios))
        if len(self.mode_weights) != 4 or any(
            not math.isfinite(value) or value < 0 for value in self.mode_weights
        ):
            raise ValueError("mode_weights must contain four finite nonnegative values")
        if not any(self.mode_weights):
            raise ValueError("at least one mode weight must be positive")
        candidates = self.prior_lambda_candidates
        if not candidates or any(
            not math.isfinite(value) or value <= 0 for value in candidates
        ):
            raise ValueError("prior_lambda_candidates must be finite and positive")
        if any(left >= right for left, right in pairwise(candidates)):
            raise ValueError("prior_lambda_candidates must be strictly increasing")
        positive_floats = {
            "sensitivity_epsilon_fraction": self.sensitivity_epsilon_fraction,
            "sensitivity_phase_floor": self.sensitivity_phase_floor,
            "reference_width": self.reference_width,
        }
        for name, value in positive_floats.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.smoothness_lambda)
            or self.smoothness_lambda < 0
        ):
            raise ValueError("smoothness_lambda must be finite and nonnegative")
        if self.regularization_type not in {"adaptive", "first_order"}:
            raise ValueError("regularization_type must be adaptive or first_order")
        if (
            not math.isfinite(self.prior_weight_min)
            or not math.isfinite(self.prior_weight_max)
            or not 0 < self.prior_weight_min <= 1 <= self.prior_weight_max
        ):
            raise ValueError("prior weight bounds must contain one and be positive")
        if self.vs_min != 0.3 or self.vs_max != 2.6:
            raise ValueError("hybrid Vs bounds must be exactly [0.3, 2.6]")
        if self.reference_width > self.vs_max - self.vs_min:
            raise ValueError("reference_width is wider than the Vs range")
        integer_fields = (
            "max_iterations",
            "validation_samples_per_kind",
            "seed",
            "workers",
            "threads_per_worker",
            "task_index",
            "task_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.validation_samples_per_kind <= 0:
            raise ValueError("validation_samples_per_kind must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.threads_per_worker <= 0:
            raise ValueError("threads_per_worker must be positive")
        if self.task_count <= 0 or not 0 <= self.task_index < self.task_count:
            raise ValueError("task_index must be in [0, task_count)")
        if (
            isinstance(self.relative_tolerance, bool)
            or not isinstance(self.relative_tolerance, (int, float))
            or not math.isfinite(float(self.relative_tolerance))
            or not 0 < self.relative_tolerance < 1
        ):
            raise ValueError("relative_tolerance must be finite and in (0, 1)")
        if len(set(self.noise_scenarios)) != len(self.noise_scenarios):
            raise ValueError("noise_scenarios must be unique")
        if set(self.noise_scenarios) != {"clean", "noise_1pct"}:
            raise ValueError(
                "noise_scenarios must contain exactly clean and noise_1pct"
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        canonical_controls = {
            "mode_weights": ((4.0, 1.0, 1.0, 1.0), self.mode_weights),
            "smoothness_lambda": (1e-2, self.smoothness_lambda),
            "regularization_type": ("adaptive", self.regularization_type),
            "reference_width": (0.7, self.reference_width),
            "max_iterations": (100, self.max_iterations),
            "relative_tolerance": (1e-5, self.relative_tolerance),
            "seed": (20_260_727, self.seed),
        }
        for name, (expected, actual) in canonical_controls.items():
            if actual != expected:
                raise ValueError(
                    f"hybrid {name} must match the canonical baseline protocol"
                )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> HybridInversionConfig:
        values = dict(mapping.get("hybrid", mapping))
        known_keys = {item.name for item in dataclasses.fields(cls)}
        unknown_keys = set(values) - known_keys
        if unknown_keys:
            raise ValueError(f"unknown hybrid keys: {sorted(unknown_keys)}")
        for name in ("mode_weights", "prior_lambda_candidates", "noise_scenarios"):
            if name in values:
                values[name] = tuple(values[name])
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


def canonical_hash(
    config: DatasetConfig | TrainingConfig | InversionConfig | HybridInversionConfig,
) -> str:
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def inversion_identity_hash(config: InversionConfig) -> str:
    """Hash configuration while excluding per-task execution controls."""
    payload = config.to_dict()
    for name in (
        "device",
        "workers",
        "threads_per_worker",
        "deep_samples_per_job",
        "task_index",
        "task_count",
    ):
        del payload[name]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def hybrid_inversion_identity_hash(config: HybridInversionConfig) -> str:
    """Hash hybrid scientific controls while excluding execution placement."""
    payload = config.to_dict()
    for name in (
        "output_dir",
        "device",
        "workers",
        "threads_per_worker",
        "task_index",
        "task_count",
    ):
        del payload[name]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_dataset_config(path: Path | str) -> DatasetConfig:
    with Path(path).open("rb") as handle:
        return DatasetConfig.from_mapping(tomllib.load(handle))


def load_training_config(path: Path | str) -> TrainingConfig:
    with Path(path).open("rb") as handle:
        return TrainingConfig.from_mapping(tomllib.load(handle))


def load_inversion_config(path: Path | str) -> InversionConfig:
    with Path(path).open("rb") as handle:
        return InversionConfig.from_mapping(tomllib.load(handle))


def load_hybrid_inversion_config(path: Path | str) -> HybridInversionConfig:
    with Path(path).open("rb") as handle:
        return HybridInversionConfig.from_mapping(tomllib.load(handle))
