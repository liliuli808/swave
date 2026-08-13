from pathlib import Path

import h5py
import pytest
import torch

from swave.dataset import validate_dataset_files
from swave.inference import ForwardPredictor
from swave.inversion_data import (
    iter_inversion_samples,
    iter_observation_samples,
    samples_by_source_shard,
    select_deep_samples,
    select_observation_samples_by_kind,
)
from swave.splits import SPLIT_POLICY


def test_tiny_complete_dataset_matches_shard_row_schema(
    tiny_complete_dataset: Path,
) -> None:
    """Fails if any declared sample field has a different row count."""
    with h5py.File(tiny_complete_dataset / "shard-00000.h5") as handle:
        assert handle["sample_id"].shape == (5,)
        assert handle["model_kind"].shape == (5,)
        assert handle["vs"].shape == (5, 20)
        assert handle["vp"].shape == (5, 20)
        assert handle["density"].shape == (5, 20)
        assert handle["phase_velocity"].shape == (5, 4, 120)
        assert handle["valid_mask"].shape == (5, 4, 120)
        assert handle["quality_flags"].shape == (5,)
        assert handle["retry_count"].shape == (5,)


def test_optimizer_samples_exclude_true_vs(tiny_complete_dataset: Path) -> None:
    """Fails if the reader exposes a ground-truth velocity field."""
    rows = list(iter_inversion_samples(tiny_complete_dataset))

    assert [row.sample_id for row in rows] == [90, 91, 92, 93]
    assert all(
        not hasattr(row, "vs") and not hasattr(row, "true_vs") for row in rows
    )
    assert all(row.phase_velocity.shape == (4, 120) for row in rows)


def test_observation_reader_supports_disjoint_final_splits_without_truth(
    tiny_complete_dataset: Path,
) -> None:
    test_rows = list(iter_observation_samples(tiny_complete_dataset, "test"))
    inversion_rows = list(
        iter_observation_samples(tiny_complete_dataset, "inversion")
    )

    assert [row.sample_id for row in test_rows] == [89]
    assert [row.sample_id for row in inversion_rows] == [90, 91, 92, 93]
    assert {row.sample_id for row in test_rows}.isdisjoint(
        row.sample_id for row in inversion_rows
    )
    assert all(not hasattr(row, "vs") for row in test_rows + inversion_rows)


def test_split_selection_is_smallest_id_per_family(
    tiny_complete_dataset: Path,
) -> None:
    selected = select_observation_samples_by_kind(
        tiny_complete_dataset, "inversion", per_kind=1
    )

    assert [(row.model_kind, row.sample_id) for row in selected] == [
        (0, 90),
        (1, 91),
        (2, 92),
        (3, 93),
    ]


def test_deep_selection_is_smallest_id_per_family(
    tiny_complete_dataset: Path,
) -> None:
    """Fails if selection does not deterministically retain each family."""
    selected = select_deep_samples(tiny_complete_dataset, per_kind=1)

    assert [(row.model_kind, row.sample_id) for row in selected] == [
        (0, 90),
        (1, 91),
        (2, 92),
        (3, 93),
    ]


def test_samples_by_source_shard_sorts_rows_by_sample_id(
    tiny_complete_dataset: Path,
) -> None:
    """Fails if a shard's rows are returned in an unstable sample order."""
    grouped = samples_by_source_shard(tiny_complete_dataset)

    assert list(grouped) == [0]
    assert [row.sample_id for row in grouped[0]] == [90, 91, 92, 93]


def test_deep_selection_names_each_deficient_family(
    tiny_complete_dataset: Path,
) -> None:
    """Fails if selection silently returns incomplete model-family coverage."""
    with pytest.raises(
        ValueError,
        match="NORMAL.*LOW_VELOCITY.*HIGH_VELOCITY.*COUPLED",
    ):
        select_deep_samples(tiny_complete_dataset, per_kind=2)


def test_tiny_checkpoint_matches_the_shared_dataset(
    tiny_complete_dataset: Path, tiny_checkpoint: Path
) -> None:
    """Fails if downstream fixture payloads stop matching supported loaders."""
    payload = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)

    assert payload["dataset_config_hash"] == validate_dataset_files(
        tiny_complete_dataset
    ).config_hash
    assert payload["split_policy"] == SPLIT_POLICY
    assert ForwardPredictor.load(tiny_checkpoint, device="cpu").predict(
        [0.4] * 20
    ).shape == (4, 120)
