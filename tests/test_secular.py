import numpy as np
import pytest

from swave.secular import LayeredModel, RayleighSecular


def _paper_model() -> LayeredModel:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    return LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4])


def test_layered_model_rejects_nonincreasing_depth() -> None:
    """Catches zero-thickness layers that invalidate matrix propagation."""
    with pytest.raises(ValueError, match="strictly increasing"):
        LayeredModel(
            depth=np.array([0.0, 0.1, 0.1]),
            density=np.ones(3) * 2,
            vs=np.ones(3),
            vp=np.ones(3) * 2,
        )


def test_from_vs_builds_regular_100m_layers_and_halfspace() -> None:
    """Catches a 20-layer interpretation that adds or loses a finite layer."""
    model = LayeredModel.from_vs(np.linspace(0.3, 2.0, 20))
    np.testing.assert_allclose(model.depth, np.arange(20) * 0.1)
    np.testing.assert_allclose(model.thickness, np.full(19, 0.1))
    assert model.layers == 20


def test_paper_model_is_finite_and_changes_sign_near_kissing_frequency() -> None:
    """Catches unstable propagation or a broken Dunkin matrix orientation."""
    secular = RayleighSecular(_paper_model())
    velocities = np.linspace(0.15, 0.599, 4000)
    values = np.array([secular.evaluate(19.7, value) for value in velocities])
    assert np.all(np.isfinite(values))
    assert np.count_nonzero(values[:-1] * values[1:] < 0) >= 2


def test_invalid_frequency_and_velocity_are_rejected() -> None:
    """Catches invalid numerical inputs entering square-root calculations."""
    secular = RayleighSecular(_paper_model())
    with pytest.raises(ValueError, match="frequency"):
        secular.evaluate(0.0, 0.3)
    with pytest.raises(ValueError, match="phase_velocity"):
        secular.evaluate(10.0, -0.3)


def test_numba_backend_matches_python_reference_path() -> None:
    """Catches JIT acceleration changing the physical determinant or its sign."""
    model = _paper_model()
    reference = RayleighSecular(model, backend="python")
    accelerated = RayleighSecular(model, backend="numba")
    for frequency, velocity in ((0.5, 0.17), (19.7, 0.267), (60.0, 0.55)):
        assert accelerated.evaluate(frequency, velocity) == pytest.approx(
            reference.evaluate(frequency, velocity), rel=1e-11, abs=1e-12
        )
