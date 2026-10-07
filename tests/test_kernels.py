import numpy as np
import torch
from scipy.optimize import brentq

from swave.config import PhysicsConfig
from swave.empirical import brocher05
from swave.kernels import sensitivity_kernels
from swave.network import FourHeadForwardModel, model_from_checkpoint
from swave.secular import LayeredModel, RayleighSecular
from swave.solver import DispersionSolver

FREQUENCIES = np.array([2.0, 7.5, 21.0])


def _vs() -> np.ndarray:
    vs = np.linspace(0.40, 2.15, 20)
    vs[5] *= 1.25
    vs[6:9] *= 0.75
    return vs


def _model(vs: np.ndarray, vp=None, density=None) -> LayeredModel:
    base_vp, base_density = brocher05(vs)
    return LayeredModel(
        np.arange(20) * 0.1,
        base_density if density is None else density,
        vs,
        base_vp if vp is None else vp,
    )


def _root(model: LayeredModel, frequency: float, guess: float) -> float:
    secular = RayleighSecular(model)
    return brentq(
        lambda value: secular.evaluate(frequency, value),
        guess - 2e-3,
        guess + 2e-3,
        xtol=1e-14,
    )


def test_kernels_match_finite_differences_of_roots() -> None:
    vs = _vs()
    result = DispersionSolver(_model(vs), PhysicsConfig()).solve_grid(FREQUENCIES)
    for coupled in (True, False):
        kernel = sensitivity_kernels(
            vs, result.phase_velocity, result.valid_mask, FREQUENCIES,
            coupled=coupled,
        )[0]
        assert kernel.shape == (4, FREQUENCIES.size, 20)
        for mode, index, layer in [(0, 0, 0), (1, 1, 6), (3, 2, 2), (2, 1, 12)]:
            roots = []
            for sign in (1.0, -1.0):
                shifted = vs.copy()
                shifted[layer] += sign * 1e-4
                if coupled:
                    model = _model(shifted)
                else:
                    vp, density = brocher05(vs)
                    model = _model(shifted, vp, density)
                roots.append(
                    _root(
                        model,
                        FREQUENCIES[index],
                        result.phase_velocity[mode, index],
                    )
                )
            expected = (roots[0] - roots[1]) / 2e-4
            assert abs(kernel[mode, index, layer] - expected) < 1e-5


def test_kernels_are_nan_for_invalid_cells() -> None:
    vs = _vs()
    phase = np.full((4, 3), 0.5)
    mask = np.zeros((4, 3), dtype=bool)
    kernel = sensitivity_kernels(vs, phase, mask, FREQUENCIES)
    assert np.isnan(kernel).all()


def test_model_from_checkpoint_honours_architecture() -> None:
    model = FourHeadForwardModel(width=32, blocks=1)
    payload = {"model": model.state_dict(), "architecture": {"width": 32, "blocks": 1}}
    loaded = model_from_checkpoint(payload)
    value = torch.rand(3, 20)
    torch.testing.assert_close(loaded(value), model(value))
    default = FourHeadForwardModel()
    assert isinstance(model_from_checkpoint({"model": default.state_dict()}), FourHeadForwardModel)
