"""Phase-velocity sampling for multimode Rayleigh-wave root searches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import PhysicsConfig
from .secular import LayeredModel

_SAMPLE_MARGIN = 1e-5


def rayleigh_reference_velocity(vp: float, vs: float) -> float:
    """Solve the homogeneous-half-space Rayleigh equation by Newton iteration."""
    root = 0.8 * vs
    for _ in range(20):
        inverse = 1.0 / root
        inverse2 = inverse * inverse
        vs_inverse2 = vs**-2
        vp_inverse2 = vp**-2
        xi = np.sqrt(inverse2 - vs_inverse2)
        eta = np.sqrt(inverse2 - vp_inverse2)
        function = (vs_inverse2 - 2.0 * inverse2) ** 2 - (
            4.0 * xi * eta * inverse2
        )
        derivative = inverse2 * (
            8.0 * inverse * (vs_inverse2 - 2.0 * inverse2)
            + 8.0 * inverse * xi * eta
            + 4.0 * inverse2 * inverse * (xi / eta + eta / xi)
        )
        step = function / derivative
        root -= step
        if abs(step) < 1e-12:
            break
    return float(root)


def estimate_mode_count(
    model: LayeredModel, frequency: float, phase_velocity: float
) -> float:
    """Estimate cumulative modes below a candidate phase velocity."""
    if frequency <= 0 or phase_velocity <= 0:
        raise ValueError("frequency and phase_velocity must be positive")
    inverse_c2 = phase_velocity**-2
    vs_terms = np.zeros(model.layers - 1, dtype=np.float64)
    vp_terms = np.zeros(model.layers - 1, dtype=np.float64)
    finite_vs = model.vs[:-1]
    finite_vp = model.vp[:-1]
    vs_mask = phase_velocity > finite_vs
    vp_mask = phase_velocity > finite_vp
    vs_terms[vs_mask] = np.sqrt(finite_vs[vs_mask] ** -2 - inverse_c2)
    vp_terms[vp_mask] = np.sqrt(finite_vp[vp_mask] ** -2 - inverse_c2)
    return float(
        2.0 * frequency * np.sum((vs_terms + vp_terms) * model.thickness)
    )


def _unique_sorted(values: ArrayLike, tolerance: float = 1e-12) -> NDArray[np.float64]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2:
        return ordered
    keep = np.r_[True, np.diff(ordered) > tolerance]
    return ordered[keep]


def initial_phase_samples(
    model: LayeredModel, frequency: float, config: PhysicsConfig
) -> NDArray[np.float64]:
    """Create adaptive samples from the Fan modal-count estimate."""
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    vs_min = float(np.min(model.vs))
    vs_halfspace = float(model.vs[-1])
    upper = vs_halfspace - _SAMPLE_MARGIN
    if upper <= 0:
        raise ValueError("half-space Vs is too small for a positive search range")

    predicted: list[float] = []
    if vs_halfspace > vs_min + _SAMPLE_MARGIN:
        estimated_max = estimate_mode_count(model, frequency, vs_halfspace)
        interval_count = max(1, int(np.floor(estimated_max)) + 1)
        step = (vs_halfspace - vs_min) / interval_count
        left = vs_min
        left_count = estimate_mode_count(model, frequency, left)
        for _ in range(100_000):
            if left >= vs_halfspace:
                break
            right = min(vs_halfspace, left + step)
            right_count = estimate_mode_count(model, frequency, right)
            while abs(right_count - left_count) > config.epsilon:
                right = left + 0.618 * (right - left)
                right_count = estimate_mode_count(model, frequency, right)
            if right <= left:
                break
            if right < vs_halfspace:
                predicted.append(right)
            left = right
            left_count = right_count

    predicted.extend([0.8 * vs_min, upper])
    samples = _unique_sorted(predicted)
    for _ in range(config.nfine):
        if samples.size < 2:
            break
        samples = _unique_sorted(
            np.concatenate([samples, (samples[:-1] + samples[1:]) / 2.0])
        )

    samples = np.concatenate(
        [
            samples,
            np.array(
                [
                    max(np.finfo(float).eps, vs_min - 10.0 * _SAMPLE_MARGIN),
                    min(upper, vs_min + 10.0 * _SAMPLE_MARGIN),
                    rayleigh_reference_velocity(model.vp[0], model.vs[0]),
                ]
            ),
        ]
    )
    samples = samples[(samples > 0) & (samples < vs_halfspace)]
    return _unique_sorted(samples)


def quadratic_vertex(x: ArrayLike, y: ArrayLike) -> float | None:
    """Return an in-window parabola vertex or ``None`` for an invalid fit."""
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    if x_value.shape != (3,) or y_value.shape != (3,):
        raise ValueError("quadratic interpolation requires three x and y values")
    if not np.all(np.isfinite(x_value)) or not np.all(np.isfinite(y_value)):
        return None
    if np.any(np.diff(x_value) <= 0):
        raise ValueError("quadratic interpolation x values must increase")

    x1, x2, x3 = x_value
    f1, f2, f3 = y_value
    denominator = (x1 - x2) * (x2 - x3) * (x3 - x1)
    if abs(denominator) <= np.finfo(float).eps:
        return None
    b = (
        (x2**2 - x3**2) * f1
        + (x3**2 - x1**2) * f2
        + (x1**2 - x2**2) * f3
    ) / denominator
    a = -(
        (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
    ) / denominator
    scale = max(1.0, float(np.max(np.abs(y_value))))
    if abs(a) <= np.finfo(float).eps * scale:
        return None
    vertex = float(-b / (2.0 * a))
    if not x1 < vertex < x3:
        return None
    return vertex


def augment_quadratic_samples(
    samples: NDArray[np.float64],
    evaluator: Callable[[float], float],
    iterations: int,
    dedup_tolerance: float,
) -> tuple[NDArray[np.float64], dict[float, float]]:
    """Iteratively add extrema inferred from consecutive sample triplets."""
    current = _unique_sorted(samples, dedup_tolerance)
    values = {float(sample): float(evaluator(float(sample))) for sample in current}
    for _ in range(iterations):
        candidates: list[float] = []
        for index in range(current.size - 2):
            window = current[index : index + 3]
            vertex = quadratic_vertex(
                window, np.array([values[float(item)] for item in window])
            )
            if vertex is None:
                continue
            if np.min(np.abs(current - vertex)) > dedup_tolerance:
                candidates.append(vertex)
        if not candidates:
            break
        additions = _unique_sorted(candidates, dedup_tolerance)
        for sample in additions:
            values[float(sample)] = float(evaluator(float(sample)))
        current = _unique_sorted(
            np.concatenate([current, additions]), dedup_tolerance
        )
    return current, values


def degraded_models(model: LayeredModel) -> list[LayeredModel]:
    """Return shallow truncations associated with every strict local Vs minimum."""
    layer_counts = [
        index
        for index in range(1, model.layers - 1)
        if model.vs[index] < model.vs[index - 1]
        and model.vs[index] < model.vs[index + 1]
    ]
    result = [
        LayeredModel(
            model.depth[:count],
            model.density[:count],
            model.vs[:count],
            model.vp[:count],
        )
        for count in layer_counts
        if count >= 2
    ]
    result.append(model)
    return result


def add_root_samples(
    samples: NDArray[np.float64],
    roots: ArrayLike,
    upper_bound: float,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Add each prior root, nearby probes, and neighboring midpoints."""
    base = _unique_sorted(samples)
    additions: list[float] = []
    for root in np.asarray(roots, dtype=np.float64):
        insertion = int(np.searchsorted(base, root))
        if insertion == 0 or insertion == len(base):
            continue
        additions.extend(
            [
                root - tolerance,
                root,
                root + tolerance,
                (root + base[insertion - 1]) / 2.0,
                (root + base[insertion]) / 2.0,
            ]
        )
    merged = _unique_sorted(np.concatenate([base, np.asarray(additions)]))
    return merged[(merged > 0) & (merged < upper_bound)]
