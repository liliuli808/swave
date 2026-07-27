from pathlib import Path

import pytest

from swave.config import (
    DatasetConfig,
    GeologyConfig,
    PhysicsConfig,
    canonical_hash,
    load_dataset_config,
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
