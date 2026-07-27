"""Quality classification and bounded recovery for dispersion curves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntFlag

import numpy as np
from numpy.typing import ArrayLike

from .config import PhysicsConfig
from .secular import LayeredModel
from .solver import DispersionResult, DispersionSolver


class QualityFlag(IntFlag):
    OK = 0
    INTERNAL_GAP = 1
    NONFINITE_VALID = 2
    ROOT_ORDER = 4
    NUMERICAL_FAILURE = 8
    RECOVERED = 16


@dataclass(frozen=True)
class QualityReport:
    flags: QualityFlag
    failing_frequency_indices: tuple[int, ...]

    @property
    def retry_required(self) -> bool:
        return bool(
            self.flags & (QualityFlag.INTERNAL_GAP | QualityFlag.NUMERICAL_FAILURE)
        )

    @property
    def hard_failure(self) -> bool:
        return bool(
            self.flags & (QualityFlag.NONFINITE_VALID | QualityFlag.ROOT_ORDER)
        )


@dataclass(frozen=True)
class RecoveryResult:
    dispersion: DispersionResult
    quality: QualityReport
    recovered: bool


def assess_arrays(
    phase_velocity: ArrayLike,
    valid_mask: ArrayLike,
    *,
    status: ArrayLike | None = None,
) -> QualityReport:
    """Classify stored arrays without interpolating or hiding missing values."""
    values = np.asarray(phase_velocity, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("phase_velocity and valid_mask must be equal 2D arrays")
    if values.shape[0] != 4:
        raise ValueError("phase_velocity must contain exactly four modal rows")
    status_value = (
        np.zeros(values.shape[1], dtype=np.uint8)
        if status is None
        else np.asarray(status, dtype=np.uint8)
    )
    if status_value.shape != (values.shape[1],):
        raise ValueError("status must have one entry per frequency")

    flags = QualityFlag.OK
    failing: set[int] = set()

    bad_valid = mask & ~np.isfinite(values)
    if np.any(bad_valid):
        flags |= QualityFlag.NONFINITE_VALID
        failing.update(np.flatnonzero(np.any(bad_valid, axis=0)).tolist())

    for row in mask:
        present = np.flatnonzero(row)
        if present.size < 2:
            continue
        internal = np.flatnonzero(~row[present[0] : present[-1] + 1]) + present[0]
        if internal.size:
            flags |= QualityFlag.INTERNAL_GAP
            failing.update(internal.tolist())

    for column in range(values.shape[1]):
        roots = values[mask[:, column], column]
        if roots.size >= 2 and np.any(np.diff(roots) <= 0):
            flags |= QualityFlag.ROOT_ORDER
            failing.add(column)

    numerical = np.flatnonzero(status_value != 0)
    if numerical.size:
        flags |= QualityFlag.NUMERICAL_FAILURE
        failing.update(numerical.tolist())

    return QualityReport(flags, tuple(sorted(failing)))


def assess_dispersion(result: DispersionResult) -> QualityReport:
    return assess_arrays(
        result.phase_velocity, result.valid_mask, status=result.status
    )


def solve_with_recovery(
    model: LayeredModel, physics: PhysicsConfig
) -> RecoveryResult:
    """Run one targeted refinement pass for internal gaps or numerical failures."""
    initial = DispersionSolver(model, physics).solve_grid()
    initial_report = assess_dispersion(initial)
    if not initial_report.retry_required or initial_report.hard_failure:
        return RecoveryResult(initial, initial_report, recovered=False)

    refined_config = replace(
        physics,
        nfine=physics.nfine + 1,
        quadratic_iterations=physics.quadratic_iterations + 2,
        strategy="quadratic",
    )
    refined_solver = DispersionSolver(model, refined_config)
    phase_velocity = initial.phase_velocity.copy()
    valid_mask = initial.valid_mask.copy()
    status = initial.status.copy()
    evaluations = initial.evaluations.copy()
    recovered = False

    for index in initial_report.failing_frequency_indices:
        solution = refined_solver.solve_frequency(
            float(initial.frequencies[index]), strategy="quadratic"
        )
        current_count = int(np.count_nonzero(valid_mask[:, index]))
        if solution.status != 0 or len(solution.roots) < current_count:
            evaluations[index] += solution.evaluations
            continue
        phase_velocity[:, index] = np.nan
        phase_velocity[: len(solution.roots), index] = solution.roots
        valid_mask[:, index] = False
        valid_mask[: len(solution.roots), index] = True
        status[index] = 0
        evaluations[index] += solution.evaluations
        recovered = True

    result = DispersionResult(
        frequencies=initial.frequencies,
        phase_velocity=phase_velocity,
        valid_mask=valid_mask,
        status=status,
        evaluations=evaluations,
    )
    report = assess_dispersion(result)
    if recovered:
        report = QualityReport(
            report.flags | QualityFlag.RECOVERED,
            report.failing_frequency_indices,
        )
    return RecoveryResult(result, report, recovered)

