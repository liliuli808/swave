import numpy as np

from swave.config import PhysicsConfig
from swave.secular import LayeredModel
from swave.solver import DispersionSolver, deduplicate_roots


def _paper_model() -> LayeredModel:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    return LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4])


def test_root_deduplication_preserves_ascending_distinct_values() -> None:
    """Catches false twin roots and unstable modal ordering."""
    roots = deduplicate_roots(
        [0.4, 0.3, 0.30000000001, 0.5], tolerance=1e-7
    )
    np.testing.assert_allclose(roots, [0.3, 0.4, 0.5])


def test_quadratic_finds_at_least_raw_roots_at_paper_kissing_frequency() -> None:
    """Catches the extrema pass failing to augment the mode-kissing gap."""
    solver = DispersionSolver(_paper_model(), PhysicsConfig())
    baseline = solver.solve_frequency(19.7, strategy="raw")
    improved = solver.solve_frequency(19.7, strategy="quadratic")
    assert len(improved.roots) >= len(baseline.roots)
    assert len(improved.roots) >= 2
    assert np.all(np.diff(improved.roots) > 0)


def test_degraded_strategy_returns_ordered_distinct_roots() -> None:
    """Catches degraded-model roots being appended without sorting or merging."""
    solution = DispersionSolver(
        _paper_model(), PhysicsConfig()
    ).solve_frequency(30.75, strategy="degraded")
    assert len(solution.roots) >= 2
    assert np.all(np.diff(solution.roots) > 1e-7)


def test_grid_has_fixed_four_by_frequency_shape_and_matching_mask() -> None:
    """Catches ragged modal output that cannot feed HDF5 or the network."""
    frequencies = np.array([10.0, 19.7, 30.75])
    result = DispersionSolver(_paper_model(), PhysicsConfig()).solve_grid(
        frequencies
    )
    assert result.phase_velocity.shape == (4, 3)
    assert result.valid_mask.shape == (4, 3)
    assert result.status.shape == (3,)
    assert np.array_equal(result.valid_mask, np.isfinite(result.phase_velocity))

