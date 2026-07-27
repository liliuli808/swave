import numpy as np
import pytest

from swave.empirical import material_properties


def test_brocher05_matches_published_polynomial_examples() -> None:
    """Catches a wrong coefficient, power, or unit in either Brocher equation."""
    vp, rho = material_properties(np.array([0.3, 1.0, 2.6]), "brocher05")
    np.testing.assert_allclose(vp, [1.50249679, 2.4582, 4.40849504], atol=1e-8)
    np.testing.assert_allclose(
        rho, [1.63667637, 2.08000416, 2.44956763], atol=1e-8
    )


def test_gardner_matches_literal_examples_and_vp_exceeds_vs() -> None:
    """Catches applying Gardner to km/s without its required 1000 conversion."""
    vs = np.array([0.3, 1.0, 2.6])
    vp, rho = material_properties(vs, "gardner")
    np.testing.assert_allclose(vp, [0.51963, 1.7321, 4.50346], atol=1e-8)
    np.testing.assert_allclose(rho, [1.48008020, 1.99988459, 2.53950032], atol=1e-8)
    assert np.all(vp > vs)


def test_near_surface_uses_requested_vp_ratio() -> None:
    """Catches silently ignoring the saturation-dependent Vp/Vs ratio."""
    vp, rho = material_properties(
        np.array([0.4, 0.8]), "near_surface", vp_vs_ratio=2.5
    )
    np.testing.assert_allclose(vp, [1.0, 2.0])
    np.testing.assert_allclose(rho, [2.04159885, 2.46319631], atol=1e-8)


def test_nonfinite_or_nonpositive_vs_is_rejected() -> None:
    """Catches bad elastic models reaching the secular-function loop."""
    with pytest.raises(ValueError, match="finite"):
        material_properties(np.array([0.5, np.nan]), "brocher05")
    with pytest.raises(ValueError, match="positive"):
        material_properties(np.array([0.5, 0.0]), "brocher05")


def test_unknown_method_lists_supported_names() -> None:
    """Catches a typo silently selecting an unintended empirical law."""
    with pytest.raises(ValueError, match="brocher05, gardner, near_surface"):
        material_properties(np.array([1.0]), "unknown")
