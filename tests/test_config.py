import dataclasses
from pathlib import Path

import pytest

from swave.config import (
    DatasetConfig,
    GeologyConfig,
    HybridInversionConfig,
    InversionConfig,
    PhysicsConfig,
    canonical_hash,
    hybrid_inversion_identity_hash,
    inversion_identity_hash,
    load_dataset_config,
    load_hybrid_inversion_config,
    load_inversion_config,
)


def test_default_frequency_grid_has_120_exact_points() -> None:
    """Catches an off-by-one grid that drops 60 Hz or adds an extra point."""
    cfg = PhysicsConfig()
    assert cfg.frequencies[0] == pytest.approx(0.5)
    assert cfg.frequencies[-1] == pytest.approx(60.0)
    assert len(cfg.frequencies) == 120


def test_invalid_model_mixture_is_rejected() -> None:
    """Catches accepting model-family probabilities that cannot be sampled."""
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        GeologyConfig(
            normal_fraction=0.9,
            low_fraction=0.15,
            high_fraction=0.10,
            coupled_fraction=0.9,
        )


def test_config_hash_is_independent_of_round_trip() -> None:
    """Catches unstable hashes that would break safe dataset resume."""
    left = DatasetConfig()
    right = DatasetConfig.from_mapping(left.to_dict())
    assert canonical_hash(left) == canonical_hash(right)


def test_loads_production_config() -> None:
    """Catches incorrect TOML section merging or ignored production values."""
    cfg = load_dataset_config(Path("configs/dataset.toml"))
    assert cfg.samples == 1_000_000
    assert cfg.physics.mode_count == 4
    assert cfg.geology.layers == 20


def test_inversion_defaults_match_approved_experiment() -> None:
    """Catches an inversion run silently using unapproved scientific defaults."""
    config = InversionConfig()
    assert config.mode_weights == (4.0, 1.0, 1.0, 1.0)
    assert config.noise_scenarios == ("clean", "noise_1pct")
    assert config.initial_models == 100
    assert config.samples_per_kind == 100
    assert config.minimum_valid_solutions == 20
    assert config.deep_samples_per_job == 10
    assert config.threads_per_worker == 1


def test_inversion_config_rejects_invalid_cluster_and_unknown_key(tmp_path) -> None:
    """Catches invalid cluster partitions and misspelled TOML settings."""
    with pytest.raises(ValueError, match="task_index"):
        InversionConfig(task_index=2, task_count=2)
    path = tmp_path / "bad.toml"
    path.write_text("[inversion]\nunknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown inversion keys"):
        load_inversion_config(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"mode_weights": (4.0, 1.0, 1.0, float("nan"))}, "mode_weights"),
        ({"regularization_lambda": float("inf")}, "regularization_lambda"),
        ({"relative_tolerance": float("nan")}, "relative_tolerance"),
        ({"vs_min": 0.299}, "supported"),
        ({"vs_max": 2.601}, "supported"),
        ({"noise_scenarios": ("clean", "clean")}, "unique"),
        ({"max_iterations": True}, "max_iterations"),
        ({"workers": False}, "workers"),
        ({"task_count": 1.0}, "task_count"),
        ({"deep_samples_per_job": 0}, "deep_samples_per_job"),
        ({"threads_per_worker": True}, "threads_per_worker"),
    ],
)
def test_inversion_config_rejects_noncanonical_values_before_execution(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        InversionConfig(**changes)


def test_inversion_scientific_hash_excludes_new_operational_controls() -> None:
    base = InversionConfig()
    operational = dataclasses.replace(
        base,
        deep_samples_per_job=7,
        threads_per_worker=2,
        workers=3,
        task_index=2,
        task_count=4,
    )
    assert inversion_identity_hash(operational) == inversion_identity_hash(base)


def test_hybrid_defaults_encode_inverse_sensitivity_experiment() -> None:
    config = HybridInversionConfig()

    assert config.prior_lambda_candidates == (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    assert config.sensitivity_epsilon_fraction == pytest.approx(1e-2)
    assert config.prior_weight_min == pytest.approx(0.25)
    assert config.prior_weight_max == pytest.approx(4.0)
    assert config.validation_samples_per_kind == 100
    assert config.mode_weights == (4.0, 1.0, 1.0, 1.0)
    assert config.smoothness_lambda == pytest.approx(1e-2)
    assert (config.vs_min, config.vs_max) == pytest.approx((0.3, 2.6))


def test_loads_hybrid_toml_and_rejects_unknown_keys(tmp_path: Path) -> None:
    loaded = load_hybrid_inversion_config(Path("configs/hybrid-inversion.toml"))
    assert loaded.supervised_dir == Path("runs/supervised-inversion-v2")
    assert loaded.output_dir == Path("results/hybrid-inversion")

    path = tmp_path / "bad-hybrid.toml"
    path.write_text("[hybrid]\nmisspelled = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown hybrid keys"):
        load_hybrid_inversion_config(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prior_lambda_candidates": (0.1, 0.01)}, "increasing"),
        ({"prior_lambda_candidates": (0.0, 0.1)}, "positive"),
        ({"sensitivity_epsilon_fraction": 0.0}, "epsilon"),
        ({"prior_weight_min": 1.1}, "weight bounds"),
        ({"prior_weight_max": 0.9}, "weight bounds"),
        ({"validation_samples_per_kind": 0}, "validation_samples_per_kind"),
        ({"noise_scenarios": ("clean", "clean")}, "unique"),
        ({"task_index": 2, "task_count": 2}, "task_index"),
        ({"workers": False}, "workers"),
    ],
)
def test_hybrid_config_rejects_noncanonical_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        HybridInversionConfig(**changes)


def test_hybrid_identity_hash_excludes_only_execution_controls() -> None:
    base = HybridInversionConfig()
    operational = dataclasses.replace(
        base,
        device="cpu",
        workers=3,
        threads_per_worker=2,
        task_index=2,
        task_count=4,
    )
    scientific = dataclasses.replace(
        base, prior_lambda_candidates=(1e-3, 1e-2, 1e-1)
    )

    assert hybrid_inversion_identity_hash(operational) == (
        hybrid_inversion_identity_hash(base)
    )
    assert hybrid_inversion_identity_hash(scientific) != (
        hybrid_inversion_identity_hash(base)
    )
