"""Empirical conversions from shear velocity to elastic material properties."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

EmpiricalMethod = Literal["brocher05", "gardner", "near_surface"]


def _validate_vs(vs: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(vs, dtype=np.float64)
    if value.ndim == 0:
        value = value.reshape(1)
    if not np.all(np.isfinite(value)):
        raise ValueError("Vs values must be finite")
    if np.any(value <= 0):
        raise ValueError("Vs values must be positive")
    return value


def _validate_output(
    vs: NDArray[np.float64],
    vp: NDArray[np.float64],
    density: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not np.all(np.isfinite(vp)) or not np.all(np.isfinite(density)):
        raise ValueError("empirical material properties must be finite")
    if np.any(vp <= vs):
        raise ValueError("empirical material properties must satisfy Vp > Vs")
    if np.any(density <= 0):
        raise ValueError("empirical density must be positive")
    return vp, density


def brocher05(vs: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Brocher's crustal velocity and density polynomials."""
    value = _validate_vs(vs)
    vp = (
        0.9409
        + 2.0947 * value
        - 0.8206 * value**2
        + 0.2683 * value**3
        - 0.0251 * value**4
    )
    density = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    return _validate_output(value, vp, density)


def gardner(vs: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a Poisson-solid Vp ratio and the Gardner density relation."""
    value = _validate_vs(vs)
    vp = 1.7321 * value
    density = 0.31 * (1000.0 * vp) ** 0.25
    return _validate_output(value, vp, density)


def near_surface(
    vs: ArrayLike, vp_vs_ratio: float = 1.7321
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the configurable shallow Vp ratio and quadratic density law."""
    if not np.isfinite(vp_vs_ratio) or vp_vs_ratio <= 1:
        raise ValueError("vp_vs_ratio must be finite and greater than 1")
    value = _validate_vs(vs)
    vp = vp_vs_ratio * value
    density = -0.22374079 * value**2 + 1.32248261 * value + 1.54840433
    return _validate_output(value, vp, density)


def material_properties(
    vs: ArrayLike,
    method: EmpiricalMethod | str = "brocher05",
    *,
    vp_vs_ratio: float = 1.7321,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ``Vs`` to ``Vp`` and density using a named relation."""
    if method == "brocher05":
        return brocher05(vs)
    if method == "gardner":
        return gardner(vs)
    if method == "near_surface":
        return near_surface(vs, vp_vs_ratio=vp_vs_ratio)
    raise ValueError(
        f"unknown empirical method {method!r}; expected "
        "brocher05, gardner, near_surface"
    )

