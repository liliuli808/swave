import numpy as np
import pytest

from swave.config import DatasetConfig, PhysicsConfig
from swave.geology import generate_model
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


def test_quadratic_search_stops_after_biased_coarse_root_budget() -> None:
    """Catches scanning/refining the entire high-frequency velocity range."""
    config = DatasetConfig()
    generated = generate_model(0, config.geology, config.seed)
    model = LayeredModel.from_vs(
        generated.vs,
        config.geology.empirical_method,
        config.geology.thickness_km,
    )
    solution = DispersionSolver(model, config.physics).solve_frequency(60.0)
    np.testing.assert_allclose(
        solution.roots,
        [0.61526340, 0.65003814, 0.65311292, 0.65826456],
        rtol=1e-7,
        atol=1e-8,
    )
    assert solution.evaluations < 1_000
    assert solution.status == pytest.approx(0)


def test_consensus_recovers_mode_kissing_pair_missed_by_quadratic() -> None:
    # Production test row 631489: at 5.5 Hz a kissing pair 1e-4 km/s apart was
    # skipped by the quadratic search, shifting modes 0-3 up by two roots.
    vs = np.array([
        0.642038, 0.664962, 0.73239, 0.567805, 0.614541, 0.927121, 1.012011,
        1.138163, 1.225484, 1.268235, 1.295506, 1.36333, 1.442533, 1.505713,
        1.637385, 1.818105, 1.940122, 2.029348, 2.109967, 2.163218,
    ])
    solver = DispersionSolver(LayeredModel.from_vs(vs), PhysicsConfig())
    consensus = solver.solve_frequency(5.5, "consensus").roots
    np.testing.assert_allclose(
        consensus, [0.6103, 0.6104, 0.6803, 0.7027], atol=1e-4
    )
    assert consensus[1] - consensus[0] < 5e-4
