"""Multimode Rayleigh-wave root search with mode-kissing recovery strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import toms748

from .config import PhysicsConfig
from .sampling import (
    add_root_samples,
    augment_quadratic_samples,
    degraded_models,
    initial_phase_samples,
)
from .secular import LayeredModel, RayleighSecular, SecularNumericalError

RootStrategy = Literal["raw", "degraded", "quadratic"]
_QUADRATIC_ROOT_BIAS = 2


@dataclass(frozen=True)
class FrequencySolution:
    frequency: float
    roots: NDArray[np.float64]
    evaluations: int
    status: int = 0


@dataclass(frozen=True)
class DispersionResult:
    frequencies: NDArray[np.float64]
    phase_velocity: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    status: NDArray[np.uint8]
    evaluations: NDArray[np.int64]


def deduplicate_roots(
    roots: ArrayLike, tolerance: float
) -> NDArray[np.float64]:
    """Sort roots and merge numerical twins."""
    values = np.sort(np.asarray(roots, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return values
    return values[np.r_[True, np.diff(values) > tolerance]]


class _Evaluations:
    def __init__(self, model: LayeredModel, frequency: float) -> None:
        self.secular = RayleighSecular(model)
        self.frequency = frequency
        self.cache: dict[float, float] = {}
        self.failed = False

    def __call__(self, phase_velocity: float) -> float:
        key = float(phase_velocity)
        if key not in self.cache:
            try:
                self.cache[key] = self.secular.evaluate(self.frequency, key)
            except SecularNumericalError:
                self.failed = True
                self.cache[key] = float("nan")
        return self.cache[key]

    @property
    def count(self) -> int:
        return len(self.cache)


class DispersionSolver:
    """Calculate modes 0–3 independently at each requested frequency."""

    def __init__(
        self, model: LayeredModel, config: PhysicsConfig | None = None
    ) -> None:
        self.model = model
        self.config = config or PhysicsConfig()

    def _roots_from_samples(
        self,
        evaluator: _Evaluations,
        samples: NDArray[np.float64],
        known_values: dict[float, float] | None = None,
    ) -> NDArray[np.float64]:
        roots: list[float] = []
        if samples.size < 2:
            return np.asarray(roots, dtype=np.float64)
        left = float(samples[0])
        left_value = (
            known_values[left]
            if known_values is not None and left in known_values
            else evaluator(left)
        )
        for index in range(samples.size - 1):
            right = float(samples[index + 1])
            right_value = (
                known_values[right]
                if known_values is not None and right in known_values
                else evaluator(right)
            )
            if not np.isfinite(left_value) or not np.isfinite(right_value):
                if np.isfinite(right_value):
                    left = right
                    left_value = right_value
                continue
            if left_value == 0.0:
                roots.append(left)
            if right_value == 0.0:
                roots.append(right)
            if left_value * right_value < 0:
                try:
                    root = toms748(
                        evaluator,
                        left,
                        right,
                        xtol=self.config.root_tolerance,
                        rtol=4.0 * np.finfo(float).eps,
                        maxiter=100,
                    )
                except (ValueError, RuntimeError, SecularNumericalError):
                    evaluator.failed = True
                else:
                    if np.isfinite(root):
                        roots.append(float(root))
            roots = deduplicate_roots(
                roots, self.config.dedup_tolerance
            ).tolist()
            if len(roots) >= self.config.mode_count:
                break
            left = right
            left_value = right_value
        return np.asarray(roots[: self.config.mode_count], dtype=np.float64)

    @staticmethod
    def _coarse_prefix(
        evaluator: _Evaluations,
        samples: NDArray[np.float64],
        required_crossings: int,
    ) -> tuple[NDArray[np.float64], dict[float, float]]:
        """Evaluate only through a small root-biased prefix, as QEDispInv does."""
        if samples.size == 0:
            return samples, {}
        known = {float(samples[0]): evaluator(float(samples[0]))}
        previous_value = known[float(samples[0])]
        crossings = 0
        stop = samples.size
        for index in range(1, samples.size):
            current = float(samples[index])
            current_value = evaluator(current)
            known[current] = current_value
            if (
                np.isfinite(previous_value)
                and np.isfinite(current_value)
                and previous_value * current_value < 0
            ):
                crossings += 1
                if crossings >= required_crossings:
                    stop = index + 1
                    break
            if np.isfinite(current_value):
                previous_value = current_value
        return samples[:stop], known

    def _solve_model(
        self,
        model: LayeredModel,
        frequency: float,
        strategy: Literal["raw", "quadratic"],
        extra_samples: ArrayLike = (),
        upper_bound: float | None = None,
    ) -> tuple[NDArray[np.float64], int, bool]:
        evaluator = _Evaluations(model, frequency)
        samples = initial_phase_samples(model, frequency, self.config)
        if upper_bound is not None:
            samples = samples[samples < upper_bound]
        if len(extra_samples):
            samples = add_root_samples(
                samples,
                extra_samples,
                upper_bound or float(model.vs[-1]),
            )
        known_values: dict[float, float] | None = None
        if strategy == "quadratic":
            samples, known_values = self._coarse_prefix(
                evaluator,
                samples,
                self.config.mode_count + _QUADRATIC_ROOT_BIAS,
            )
            samples, known_values = augment_quadratic_samples(
                samples,
                evaluator,
                self.config.quadratic_iterations,
                self.config.dedup_tolerance,
            )
        roots = self._roots_from_samples(evaluator, samples, known_values)
        return roots, evaluator.count, evaluator.failed

    def solve_frequency(
        self, frequency: float, strategy: RootStrategy | None = None
    ) -> FrequencySolution:
        selected = strategy or self.config.strategy
        if selected not in {"raw", "degraded", "quadratic"}:
            raise ValueError("strategy must be raw, degraded, or quadratic")
        if frequency <= 0 or not np.isfinite(frequency):
            raise ValueError("frequency must be finite and positive")

        if selected in {"raw", "quadratic"}:
            roots, count, failed = self._solve_model(
                self.model, frequency, selected
            )
            return FrequencySolution(
                frequency,
                roots,
                count,
                status=1 if failed else 0,
            )

        prior_roots: list[float] = []
        final_roots = np.array([], dtype=np.float64)
        total_count = 0
        failed = False
        full_upper = float(self.model.vs[-1])
        for current in degraded_models(self.model):
            roots, count, current_failed = self._solve_model(
                current,
                frequency,
                "raw",
                extra_samples=prior_roots,
                upper_bound=full_upper,
            )
            prior_roots.extend(roots.tolist())
            final_roots = roots
            total_count += count
            failed |= current_failed
        return FrequencySolution(
            frequency,
            final_roots,
            total_count,
            status=1 if failed else 0,
        )

    def solve_grid(
        self,
        frequencies: ArrayLike | None = None,
        strategy: RootStrategy | None = None,
    ) -> DispersionResult:
        selected_frequencies = np.asarray(
            self.config.frequencies if frequencies is None else frequencies,
            dtype=np.float64,
        )
        if selected_frequencies.ndim != 1 or not np.all(
            np.isfinite(selected_frequencies)
        ):
            raise ValueError("frequencies must be a finite one-dimensional array")
        phase_velocity = np.full(
            (self.config.mode_count, selected_frequencies.size),
            np.nan,
            dtype=np.float64,
        )
        status = np.zeros(selected_frequencies.size, dtype=np.uint8)
        evaluations = np.zeros(selected_frequencies.size, dtype=np.int64)
        for index, frequency in enumerate(selected_frequencies):
            solution = self.solve_frequency(float(frequency), strategy)
            phase_velocity[: len(solution.roots), index] = solution.roots
            status[index] = solution.status
            evaluations[index] = solution.evaluations
        valid_mask = np.isfinite(phase_velocity)
        return DispersionResult(
            frequencies=selected_frequencies,
            phase_velocity=phase_velocity,
            valid_mask=valid_mask,
            status=status,
            evaluations=evaluations,
        )
