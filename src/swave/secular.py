"""Unnormalized Dunkin delta-matrix secular function for Rayleigh waves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .empirical import EmpiricalMethod, material_properties


class SecularNumericalError(ArithmeticError):
    """Raised when a valid-looking evaluation becomes numerically nonfinite."""

    def __init__(self, frequency: float, phase_velocity: float) -> None:
        super().__init__(
            "secular function failed at "
            f"frequency={frequency:g} Hz, phase_velocity={phase_velocity:g} km/s"
        )
        self.frequency = frequency
        self.phase_velocity = phase_velocity


@dataclass(frozen=True)
class LayeredModel:
    """One-dimensional elastic layers whose final row is a half-space."""

    depth: NDArray[np.float64]
    density: NDArray[np.float64]
    vs: NDArray[np.float64]
    vp: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = [
            np.array(value, dtype=np.float64, copy=True)
            for value in (self.depth, self.density, self.vs, self.vp)
        ]
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("model arrays must be one-dimensional")
        if len({len(value) for value in arrays}) != 1 or len(arrays[0]) < 2:
            raise ValueError("model arrays must have the same length of at least 2")
        depth, density, vs, vp = arrays
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("model arrays must be finite")
        if depth[0] != 0 or np.any(np.diff(depth) <= 0):
            raise ValueError("model depth must start at zero and be strictly increasing")
        if np.any(density <= 0) or np.any(vs <= 0) or np.any(vp <= vs):
            raise ValueError("model requires density > 0 and Vp > Vs > 0")
        for name, value in zip(
            ("depth", "density", "vs", "vp"), arrays, strict=True
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def from_vs(
        cls,
        vs: ArrayLike,
        empirical_method: EmpiricalMethod = "brocher05",
        thickness_km: float = 0.1,
    ) -> LayeredModel:
        value = np.asarray(vs, dtype=np.float64)
        if value.shape != (20,):
            raise ValueError("Vs input must contain exactly 20 layers")
        if thickness_km <= 0:
            raise ValueError("thickness_km must be positive")
        vp, density = material_properties(value, empirical_method)
        depth = np.arange(20, dtype=np.float64) * thickness_km
        return cls(depth=depth, density=density, vs=value, vp=vp)

    @property
    def layers(self) -> int:
        return len(self.depth)

    @property
    def thickness(self) -> NDArray[np.float64]:
        return np.diff(self.depth)


def _layer_variables(
    p: float,
    q: float,
    ra: float,
    rb: float,
    wavenumber: float,
    xka: float,
    xkb: float,
    thickness: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    if wavenumber < xka:
        sinp = np.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = np.cos(p)
        pex = 0.0
    elif wavenumber == xka:
        w = thickness
        x = 0.0
        cosp = 1.0
        pex = 0.0
    else:
        pex = p
        factor = np.exp(-2.0 * p) if p < 16.0 else 0.0
        cosp = 0.5 * (1.0 + factor)
        sinp = 0.5 * (1.0 - factor)
        w = sinp / ra
        x = ra * sinp

    if wavenumber < xkb:
        sinq = np.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = np.cos(q)
        sex = 0.0
    elif wavenumber == xkb:
        y = thickness
        z = 0.0
        cosq = 1.0
        sex = 0.0
    else:
        sex = q
        factor = np.exp(-2.0 * q) if q < 16.0 else 0.0
        cosq = 0.5 * (1.0 + factor)
        sinq = 0.5 * (1.0 - factor)
        y = sinq / rb
        z = rb * sinq

    exponent = pex + sex
    a0 = np.exp(-exponent) if exponent < 60.0 else 0.0
    return (
        w,
        cosp,
        a0,
        cosp * cosq,
        cosp * y,
        cosp * z,
        cosq * w,
        cosq * x,
        x * y,
        x * z,
        w * y,
        w * z,
    )


def _dunkin_matrix(
    wavenumber2: float,
    gam: float,
    gammk: float,
    density: float,
    a0: float,
    cpcq: float,
    cpy: float,
    cpz: float,
    cqw: float,
    cqx: float,
    xy: float,
    xz: float,
    wy: float,
    wz: float,
) -> NDArray[np.float64]:
    matrix = np.zeros((5, 5), dtype=np.float64)
    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1
    density2 = density * density
    a0pq = a0 - cpcq
    negative_two_k2 = -2.0 * wavenumber2

    matrix[0, 0] = (
        cpcq
        - 2.0 * gmgm1 * a0pq
        - gmgmk * xz
        - wavenumber2 * gm1sq * wy
    )
    matrix[0, 1] = (wavenumber2 * cpy - cqx) / density
    matrix[0, 2] = -(
        twgm1 * a0pq + gammk * xz + wavenumber2 * gamm1 * wy
    ) / density
    matrix[0, 3] = (cpz - wavenumber2 * cqw) / density
    matrix[0, 4] = -(
        2.0 * wavenumber2 * a0pq + xz + wavenumber2**2 * wy
    ) / density2

    matrix[1, 0] = (gmgmk * cpz - gm1sq * cqw) * density
    matrix[1, 1] = cpcq
    matrix[1, 2] = gammk * cpz - gamm1 * cqw
    matrix[1, 3] = -wz
    matrix[1, 4] = matrix[0, 3]

    matrix[3, 0] = (gm1sq * cpy - gmgmk * cqx) * density
    matrix[3, 1] = -xy
    matrix[3, 2] = gamm1 * cpy - gammk * cqx
    matrix[3, 3] = matrix[1, 1]
    matrix[3, 4] = matrix[0, 1]

    matrix[4, 0] = -(
        2.0 * gmgmk * gm1sq * a0pq
        + gmgmk**2 * xz
        + gm1sq**2 * wy
    ) * density2
    matrix[4, 1] = matrix[3, 0]
    matrix[4, 2] = -(
        gammk * gamm1 * twgm1 * a0pq
        + gam * gammk**2 * xz
        + gamm1 * gm1sq * wy
    ) * density
    matrix[4, 3] = matrix[1, 0]
    matrix[4, 4] = matrix[0, 0]

    matrix[2, 0] = negative_two_k2 * matrix[4, 2]
    matrix[2, 1] = negative_two_k2 * matrix[3, 2]
    matrix[2, 2] = a0 + 2.0 * (cpcq - matrix[0, 0])
    matrix[2, 3] = negative_two_k2 * matrix[1, 2]
    matrix[2, 4] = negative_two_k2 * matrix[0, 2]
    return matrix


class RayleighSecular:
    """Evaluate the unnormalized Rayleigh-wave dispersion determinant."""

    def __init__(self, model: LayeredModel) -> None:
        self.model = model

    def evaluate(self, frequency: float, phase_velocity: float) -> float:
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        if not np.isfinite(phase_velocity) or phase_velocity <= 0:
            raise ValueError("phase_velocity must be finite and positive")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = self._evaluate(float(frequency), float(phase_velocity))
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise SecularNumericalError(frequency, phase_velocity) from error
        if not np.isfinite(value):
            raise SecularNumericalError(frequency, phase_velocity)
        return value

    def _evaluate(self, frequency: float, phase_velocity: float) -> float:
        model = self.model
        omega = 2.0 * np.pi * frequency
        wavenumber = omega / phase_velocity
        wavenumber2 = wavenumber * wavenumber

        xka = omega / model.vp[-1]
        xkb = omega / model.vs[-1]
        ra = np.sqrt((wavenumber + xka) * abs(wavenumber - xka))
        rb = np.sqrt((wavenumber + xkb) * abs(wavenumber - xkb))
        t = model.vs[-1] / omega
        gammk = 2.0 * t * t
        gam = gammk * wavenumber2
        gamm1 = gam - 1.0
        density = model.density[-1]
        state = np.array(
            [
                density**2 * (gamm1**2 - gam * gammk * ra * rb),
                -density * ra,
                density * (gamm1 - gammk * ra * rb),
                density * rb,
                wavenumber2 - ra * rb,
            ],
            dtype=np.float64,
        )

        for layer in range(model.layers - 2, -1, -1):
            xka = omega / model.vp[layer]
            xkb = omega / model.vs[layer]
            t = model.vs[layer] / omega
            gammk = 2.0 * t * t
            gam = gammk * wavenumber2
            ra = np.sqrt((wavenumber + xka) * abs(wavenumber - xka))
            rb = np.sqrt((wavenumber + xkb) * abs(wavenumber - xkb))
            thickness = model.thickness[layer]
            variables = _layer_variables(
                ra * thickness,
                rb * thickness,
                ra,
                rb,
                wavenumber,
                xka,
                xkb,
                thickness,
            )
            matrix = _dunkin_matrix(
                wavenumber2,
                gam,
                gammk,
                model.density[layer],
                *variables[2:],
            )
            state = matrix.T @ state
        return float(state[0])

