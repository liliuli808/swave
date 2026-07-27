import numpy as np
import pytest

from swave.config import PhysicsConfig
from swave.sampling import (
    estimate_mode_count,
    initial_phase_samples,
    quadratic_vertex,
)
from swave.secular import LayeredModel


def _paper_model() -> LayeredModel:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    return LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4])


def test_quadratic_vertex_recovers_known_extremum() -> None:
    """Catches an incorrect interpolation coefficient or vertex sign."""
    x = np.array([1.0, 2.0, 4.0])
    y = (x - 2.5) ** 2 - 0.25
    assert quadratic_vertex(x, y) == pytest.approx(2.5)


def test_quadratic_vertex_rejects_linear_and_outside_fit() -> None:
    """Catches spurious samples from flat/linear fits or extrapolation."""
    assert (
        quadratic_vertex(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        )
        is None
    )
    x = np.array([1.0, 2.0, 3.0])
    assert quadratic_vertex(x, (x - 8.0) ** 2) is None


def test_initial_samples_are_sorted_unique_and_inside_search_range() -> None:
    """Catches duplicate intervals or samples beyond the half-space bound."""
    model = _paper_model()
    samples = initial_phase_samples(model, 19.7, PhysicsConfig())
    assert np.all(np.diff(samples) > 0)
    assert samples[0] > 0
    assert samples[-1] < model.vs[-1]
    assert np.any(samples < np.min(model.vs))
    assert np.any(samples > np.min(model.vs))


def test_mode_count_estimate_increases_with_frequency() -> None:
    """Catches losing the frequency factor in the modal-count estimate."""
    model = _paper_model()
    low = estimate_mode_count(model, 10.0, 0.55)
    high = estimate_mode_count(model, 30.0, 0.55)
    assert low > 0
    assert high == pytest.approx(3 * low)
