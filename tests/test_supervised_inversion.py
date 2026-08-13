from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from swave.splits import SPLIT_POLICY
from swave.supervised_inversion import (
    InverseNet,
    SupervisedConfig,
    SupervisedHDF5Dataset,
    compute_supervised_normalization,
    train_supervised,
)


@pytest.fixture
def four_split_dataset(tmp_path: Path) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    path = directory / "shard-00000.h5"
    sample_ids = np.arange(100, dtype=np.uint64)
    kinds = (sample_ids % 4).astype(np.uint8)
    vs = np.repeat(sample_ids[:, None], 20, axis=1).astype(np.float32)
    phase = np.repeat(sample_ids[:, None, None], 4 * 120, axis=1).reshape(
        100, 4, 120
    ).astype(np.float32)
    valid = np.ones_like(phase, dtype=np.bool_)
    phase[0, 1, 1] = 7.0
    valid[1:80, 1, 1] = False
    phase[1:80, 1, 1] = np.nan
    phase[80:, 1, 1] = 999.0
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["config_hash"] = "supervised-fixture"
        handle.attrs["shard_id"] = 0
        handle.attrs["first_sample_id"] = 0
        handle.attrs["last_sample_id"] = 99
        handle.attrs["accepted_count"] = 100
        handle.attrs["sample_id_sha256"] = hashlib.sha256(
            sample_ids.tobytes()
        ).hexdigest()
        handle.attrs["accepted_by_kind"] = json.dumps(
            {"NORMAL": 25, "LOW_VELOCITY": 25, "HIGH_VELOCITY": 25,
             "COUPLED": 25},
            sort_keys=True,
        )
        handle.attrs["rejected_by_kind"] = "{}"
        handle.attrs["rejected_by_reason"] = "{}"
        handle.attrs["recovered_models"] = 0
        handle.create_dataset("sample_id", data=sample_ids, dtype="u8")
        handle.create_dataset("model_kind", data=kinds, dtype="u1")
        handle.create_dataset("vs", data=vs, dtype="f4")
        handle.create_dataset("vp", data=np.nan_to_num(vs) + 0.5, dtype="f4")
        handle.create_dataset(
            "density", data=np.ones((100, 20), dtype=np.float32) * 2, dtype="f4"
        )
        handle.create_dataset("phase_velocity", data=phase, dtype="f4")
        handle.create_dataset("valid_mask", data=valid, dtype="?")
        handle.create_dataset("quality_flags", data=np.zeros(100, dtype=np.uint16))
        handle.create_dataset("retry_count", data=np.zeros(100, dtype=np.uint8))

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": "supervised-fixture",
                "global_seed": 20260727,
                "expected_shards": 1,
                "completed_shards": [0],
                "accepted_by_kind": {
                    "NORMAL": 25,
                    "LOW_VELOCITY": 25,
                    "HIGH_VELOCITY": 25,
                    "COUPLED": 25,
                },
                "rejected_by_kind": {},
                "rejected_by_reason": {},
                "recovered_models": 0,
                "package_version": "0.1.0",
                "created_at": "2026-08-13T00:00:00+00:00",
                "shard_sha256": {"0": checksum},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_supervised_rows_obey_four_way_policy(four_split_dataset: Path) -> None:
    stats = compute_supervised_normalization(four_split_dataset)
    assert len(SupervisedHDF5Dataset(four_split_dataset, "train", stats)) == 80
    assert len(
        SupervisedHDF5Dataset(four_split_dataset, "validation", stats)
    ) == 5
    assert len(SupervisedHDF5Dataset(four_split_dataset, "test", stats)) == 5
    assert len(
        SupervisedHDF5Dataset(four_split_dataset, "inversion", stats)
    ) == 10


def test_normalization_uses_train_rows_only(
    four_split_dataset: Path,
) -> None:
    stats = compute_supervised_normalization(four_split_dataset)

    assert stats.target_mean[0] == pytest.approx(39.5)
    assert stats.input_mean[0] == pytest.approx(39.5)
    assert stats.fill_values[1, 0] == pytest.approx(7.0)
    assert stats.input_mean.reshape(4, 119)[1, 0] == pytest.approx(7.0)
    assert np.all(np.isfinite(stats.input_std))
    assert np.all(stats.input_std > 0)


def test_inverse_net_maps_476_values_to_twenty_layers() -> None:
    model = InverseNet(width=16, blocks=2)
    result = model(torch.zeros((3, 476)))
    assert result.shape == (3, 20)


def test_supervised_config_rejects_duplicate_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seeds"):
        SupervisedConfig(
            dataset_dir=tmp_path,
            output_dir=tmp_path / "run",
            seeds=(0, 0),
        )


def test_tiny_training_binds_identity_and_evaluates_final_holdouts(
    four_split_dataset: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    config = SupervisedConfig(
        dataset_dir=four_split_dataset,
        output_dir=output,
        seeds=(0,),
        width=8,
        blocks=1,
        batch_size=16,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        device="cpu",
        resume=True,
    )

    evaluation_path = train_supervised(config)

    payload = torch.load(
        output / "seed-0-best.pt", map_location="cpu", weights_only=False
    )
    assert payload["split_policy"] == SPLIT_POLICY
    assert payload["dataset_config_hash"] == "supervised-fixture"
    assert len(payload["dataset_manifest_sha256"]) == 64
    assert payload["train_sample_count"] == 80
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert report["splits"] == {
        "train": 80,
        "validation": 5,
        "test": 5,
        "inversion": 10,
    }
    assert report["test"]["sample_count"] == 5
    assert np.isfinite(report["test"]["overall"]["mae_km_s"])
    assert report["inversion_comparison"]["sample_count"] == 10
    assert np.isfinite(
        report["inversion_comparison"]["overall"]["mae_km_s"]
    )


def test_resume_rejects_changed_training_hyperparameters(
    four_split_dataset: Path,
    tmp_path: Path,
) -> None:
    config = SupervisedConfig(
        dataset_dir=four_split_dataset,
        output_dir=tmp_path / "run",
        seeds=(0,),
        width=8,
        blocks=1,
        batch_size=16,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        device="cpu",
        resume=True,
    )
    train_supervised(config)

    with pytest.raises(ValueError, match="training configuration"):
        train_supervised(replace(config, learning_rate=2e-3))
