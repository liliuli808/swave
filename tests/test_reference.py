import numpy as np

from swave.config import PhysicsConfig
from swave.secular import LayeredModel
from swave.solver import DispersionSolver


def _models(fixture: np.lib.npyio.NpzFile):
    for index, count_value in enumerate(fixture["layer_count"]):
        count = int(count_value)
        yield index, LayeredModel(
            fixture["depth"][index, :count],
            fixture["density"][index, :count],
            fixture["vs"][index, :count],
            fixture["vp"][index, :count],
        )


def _internal_gap_count(mask: np.ndarray) -> int:
    total = 0
    for row in mask:
        present = np.flatnonzero(row)
        if present.size:
            total += int(
                np.count_nonzero(~row[present[0] : present[-1] + 1])
            )
    return total


def test_fixture_records_exact_qedispinv_provenance() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    assert fixture["reference_repository"].item() == "pan3rock/QEDispInv"
    assert (
        fixture["reference_commit"].item()
        == "ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa"
    )
    assert (
        fixture["degraded_reference_commit"].item()
        == "c8a1d2c67e83c95ae42fab863bdbaf347465f732"
    )
    assert fixture["generated_date"].item() == "2026-07-27"


def test_python_roots_match_reference_within_1e_minus_5_km_s() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    physics = PhysicsConfig()
    for index, model in _models(fixture):
        result = DispersionSolver(model, physics).solve_grid(
            fixture["frequencies"]
        )
        reference_mask = fixture["valid_mask"][index]
        assert np.all(result.valid_mask[reference_mask])
        np.testing.assert_allclose(
            result.phase_velocity[reference_mask],
            fixture["phase_velocity"][index][reference_mask],
            atol=1e-5,
            rtol=0.0,
        )


def test_quadratic_never_has_more_internal_gaps_than_raw() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    physics = PhysicsConfig()
    for _, model in _models(fixture):
        solver = DispersionSolver(model, physics)
        raw = solver.solve_grid(fixture["frequencies"], strategy="raw")
        quadratic = solver.solve_grid(
            fixture["frequencies"], strategy="quadratic"
        )
        assert _internal_gap_count(quadratic.valid_mask) <= _internal_gap_count(
            raw.valid_mask
        )


def test_degraded_roots_match_mode_kissing_reference() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    physics = PhysicsConfig()
    for index, model in _models(fixture):
        result = DispersionSolver(model, physics).solve_grid(
            fixture["frequencies"], strategy="degraded"
        )
        reference_mask = fixture["degraded_valid_mask"][index]
        assert np.all(result.valid_mask[reference_mask])
        np.testing.assert_allclose(
            result.phase_velocity[reference_mask],
            fixture["degraded_phase_velocity"][index][reference_mask],
            atol=1e-5,
            rtol=0.0,
        )
