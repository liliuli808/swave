"""Physical phase-velocity sensitivity kernels for the 20-layer Vs models.

The kernel ``dc/dVs_i`` is obtained from the Dunkin secular determinant by the
implicit-function theorem, ``dc/dm = -(dF/dm) / (dF/dc)``, evaluated at each
root already present in the dataset. This needs no extra root finding, so it is
exact to finite-difference precision and much cheaper than re-solving
perturbed models. ``coupled=True`` returns the total derivative used by the
neural surrogate, in which ``Vp`` and density follow ``Vs`` through Brocher
(2005); ``coupled=False`` holds ``Vp`` and density fixed, matching the ``vs``
component of QEDispInv's eigenfunction kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from .empirical import brocher05
from .secular import _evaluate_numba

THICKNESS_KM = 0.1
DEFAULT_STEP_KM_S = 1e-5
_RELATIVE_C_STEP = 1e-7


@njit(cache=True)
def _model_kernel(
    density: NDArray[np.float64],
    vs: NDArray[np.float64],
    vp: NDArray[np.float64],
    plus: NDArray[np.float64],
    minus: NDArray[np.float64],
    thickness: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    phase_velocity: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    step: float,
    output: NDArray[np.float64],
) -> None:
    layers = vs.size
    modes, count = phase_velocity.shape
    work_density = density.copy()
    work_vs = vs.copy()
    work_vp = vp.copy()
    for mode in range(modes):
        for index in range(count):
            if not valid_mask[mode, index]:
                continue
            frequency = frequencies[index]
            root = phase_velocity[mode, index]
            dc = _RELATIVE_C_STEP * root
            derivative_c = (
                _evaluate_numba(density, vs, vp, thickness, frequency, root + dc)
                - _evaluate_numba(density, vs, vp, thickness, frequency, root - dc)
            ) / (2.0 * dc)
            if not np.isfinite(derivative_c) or derivative_c == 0.0:
                continue
            for layer in range(layers):
                # plus/minus rows hold (vs, vp, density) for the perturbed layer.
                work_vs[layer] = plus[layer, 0]
                work_vp[layer] = plus[layer, 1]
                work_density[layer] = plus[layer, 2]
                upper = _evaluate_numba(
                    work_density, work_vs, work_vp, thickness, frequency, root
                )
                work_vs[layer] = minus[layer, 0]
                work_vp[layer] = minus[layer, 1]
                work_density[layer] = minus[layer, 2]
                lower = _evaluate_numba(
                    work_density, work_vs, work_vp, thickness, frequency, root
                )
                work_vs[layer] = vs[layer]
                work_vp[layer] = vp[layer]
                work_density[layer] = density[layer]
                output[mode, index, layer] = -(
                    (upper - lower) / (2.0 * step)
                ) / derivative_c


@njit(cache=True, parallel=True)
def _batch_kernels(
    density: NDArray[np.float64],
    vs: NDArray[np.float64],
    vp: NDArray[np.float64],
    plus: NDArray[np.float64],
    minus: NDArray[np.float64],
    thickness: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    phase_velocity: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    step: float,
    output: NDArray[np.float64],
) -> None:
    for sample in prange(vs.shape[0]):
        _model_kernel(
            density[sample],
            vs[sample],
            vp[sample],
            plus[sample],
            minus[sample],
            thickness,
            frequencies,
            phase_velocity[sample],
            valid_mask[sample],
            step,
            output[sample],
        )


def _perturbed_properties(
    vs: NDArray[np.float64], step: float, coupled: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vp, density = brocher05(vs.ravel())
    vp = vp.reshape(vs.shape)
    density = density.reshape(vs.shape)
    rows = []
    for sign in (1.0, -1.0):
        shifted = vs + sign * step
        if coupled:
            shifted_vp, shifted_density = brocher05(shifted.ravel())
            shifted_vp = shifted_vp.reshape(vs.shape)
            shifted_density = shifted_density.reshape(vs.shape)
        else:
            shifted_vp, shifted_density = vp, density
        rows.append(np.stack([shifted, shifted_vp, shifted_density], axis=-1))
    return rows[0], rows[1]


def sensitivity_kernels(
    vs: ArrayLike,
    phase_velocity: ArrayLike,
    valid_mask: ArrayLike,
    frequencies: ArrayLike,
    *,
    coupled: bool = True,
    step: float = DEFAULT_STEP_KM_S,
) -> NDArray[np.float64]:
    """Return ``dc/dVs`` with shape ``(batch, modes, frequencies, layers)``.

    Cells that are invalid, or whose determinant slope cannot be evaluated,
    are ``NaN``.
    """
    vs_values = np.atleast_2d(np.asarray(vs, dtype=np.float64))
    phase = np.asarray(phase_velocity, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if phase.ndim == 2:
        phase = phase[None]
        mask = mask[None]
    selected = np.asarray(frequencies, dtype=np.float64)
    if phase.shape != mask.shape or phase.shape[0] != vs_values.shape[0]:
        raise ValueError("phase_velocity and valid_mask must match the Vs batch")
    if phase.shape[2] != selected.size:
        raise ValueError("frequency count does not match phase_velocity")
    if step <= 0:
        raise ValueError("step must be positive")
    vp, density = brocher05(vs_values.ravel())
    vp = vp.reshape(vs_values.shape)
    density = density.reshape(vs_values.shape)
    plus, minus = _perturbed_properties(vs_values, step, coupled)
    thickness = np.full(vs_values.shape[1] - 1, THICKNESS_KM)
    output = np.full((*phase.shape, vs_values.shape[1]), np.nan)
    _batch_kernels(
        np.ascontiguousarray(density),
        np.ascontiguousarray(vs_values),
        np.ascontiguousarray(vp),
        np.ascontiguousarray(plus),
        np.ascontiguousarray(minus),
        thickness,
        selected,
        np.ascontiguousarray(phase),
        np.ascontiguousarray(mask & np.isfinite(phase)),
        float(step),
        output,
    )
    return output
