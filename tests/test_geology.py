from collections import Counter
from dataclasses import replace

import numpy as np

from swave.config import GeologyConfig
from swave.geology import ModelKind, generate_model


def test_same_identity_generates_identical_model() -> None:
    """Catches worker scheduling or resume state leaking into random models."""
    cfg = GeologyConfig()
    left = generate_model(1234, cfg, global_seed=99, retry_count=2)
    right = generate_model(1234, cfg, global_seed=99, retry_count=2)
    np.testing.assert_array_equal(left.vs, right.vs)
    np.testing.assert_array_equal(left.vp, right.vp)
    assert left.kind == right.kind


def test_normal_model_is_nondecreasing_and_bounded() -> None:
    """Catches a background sampler that creates accidental anomalies."""
    cfg = replace(
        GeologyConfig(),
        normal_fraction=1.0,
        low_fraction=0.0,
        high_fraction=0.0,
        coupled_fraction=0.0,
    )
    model = generate_model(7, cfg, global_seed=10)
    assert model.kind is ModelKind.NORMAL
    assert np.all(np.diff(model.vs) >= 0)
    assert np.all((model.vs >= 0.30) & (model.vs <= 2.60))


def test_coupled_model_has_high_then_two_or_three_low_layers_in_target_zone() -> None:
    """Catches misplaced or wrongly ordered teacher-priority anomalies."""
    cfg = replace(
        GeologyConfig(),
        normal_fraction=0.0,
        low_fraction=0.0,
        high_fraction=0.0,
        coupled_fraction=1.0,
    )
    model = generate_model(12, cfg, global_seed=33)
    assert model.kind is ModelKind.COUPLED
    delta = model.vs - model.background_vs
    positive = np.flatnonzero(delta > 1e-12)
    negative = np.flatnonzero(delta < -1e-12)
    assert len(positive) == 1
    assert 2 <= positive[0] <= 11
    assert negative[0] == positive[0] + 1
    assert len(negative) in (2, 3)
    np.testing.assert_array_equal(
        negative, np.arange(positive[0] + 1, positive[0] + 1 + len(negative))
    )
    assert np.all(np.abs(delta[np.r_[positive, negative]]) >= 0.05 * model.background_vs[np.r_[positive, negative]])


def test_model_contains_consistent_positive_material_properties() -> None:
    """Catches failure to apply the selected empirical relation to final Vs."""
    model = generate_model(42, GeologyConfig(), global_seed=2026)
    assert model.vs.shape == model.vp.shape == model.density.shape == (20,)
    assert np.all(model.vp > model.vs)
    assert np.all(model.density > 0)
    assert np.all(np.isfinite(model.vp))


def test_family_draws_follow_configured_mixture() -> None:
    """Catches incorrect cumulative probability boundaries."""
    cfg = GeologyConfig()
    counts = Counter(
        generate_model(index, cfg, global_seed=123).kind for index in range(4000)
    )
    expected = {
        ModelKind.NORMAL: 0.25,
        ModelKind.LOW_VELOCITY: 0.15,
        ModelKind.HIGH_VELOCITY: 0.10,
        ModelKind.COUPLED: 0.50,
    }
    for kind, fraction in expected.items():
        assert abs(counts[kind] / 4000 - fraction) < 0.03

