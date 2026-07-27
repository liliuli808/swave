import numpy as np

from swave.quality import QualityFlag, assess_arrays


def test_leading_missing_higher_mode_is_allowed() -> None:
    """Catches physical modal onset being mislabeled as an internal omission."""
    values = np.repeat(np.array([[0.2], [0.3], [0.4], [0.5]]), 8, axis=1)
    mask = np.ones((4, 8), dtype=bool)
    mask[3, :3] = False
    values[~mask] = np.nan
    report = assess_arrays(values, mask)
    assert not (report.flags & QualityFlag.INTERNAL_GAP)
    assert not report.retry_required


def test_internal_missing_cell_requests_recovery() -> None:
    """Catches a missing root inside an otherwise present modal branch."""
    values = np.repeat(np.array([[0.2], [0.3], [0.4], [0.5]]), 8, axis=1)
    mask = np.ones((4, 8), dtype=bool)
    mask[2, 4] = False
    values[~mask] = np.nan
    report = assess_arrays(values, mask)
    assert report.flags & QualityFlag.INTERNAL_GAP
    assert report.retry_required
    assert report.failing_frequency_indices == (4,)


def test_nonfinite_valid_value_is_hard_failure() -> None:
    """Catches a NaN target being admitted to HDF5 with a true mask."""
    values = np.repeat(np.array([[0.2], [0.3], [0.4], [0.5]]), 8, axis=1)
    mask = np.ones((4, 8), dtype=bool)
    values[0, 2] = np.nan
    report = assess_arrays(values, mask)
    assert report.flags & QualityFlag.NONFINITE_VALID
    assert report.hard_failure


def test_descending_modal_roots_are_rejected() -> None:
    """Catches mode rows being stored in the wrong velocity order."""
    values = np.array([[0.2], [0.4], [0.39], [0.5]])
    mask = np.ones_like(values, dtype=bool)
    report = assess_arrays(values, mask)
    assert report.flags & QualityFlag.ROOT_ORDER
    assert report.hard_failure


def test_numerical_status_requests_recovery_at_that_frequency() -> None:
    """Catches a secular numerical failure being ignored after partial roots."""
    values = np.repeat(np.array([[0.2], [0.3], [0.4], [0.5]]), 3, axis=1)
    mask = np.ones((4, 3), dtype=bool)
    report = assess_arrays(values, mask, status=np.array([0, 1, 0]))
    assert report.flags & QualityFlag.NUMERICAL_FAILURE
    assert report.failing_frequency_indices == (1,)
