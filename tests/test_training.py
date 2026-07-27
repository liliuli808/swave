import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from swave.config import TrainingConfig
from swave.inference import ForwardPredictor
from swave.training import (
    HDF5ShardDataset,
    compute_normalization,
    evaluate,
    train,
)


def _write_tiny_dataset(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sample_ids = np.array([0, 1, 2, 3, 90, 91, 95, 96], dtype=np.uint64)
    rng = np.random.default_rng(4)
    vs = rng.uniform(0.4, 2.0, size=(len(sample_ids), 20)).astype("f4")
    frequencies = np.arange(120, dtype=np.float32)
    phase = (
        vs.mean(axis=1)[:, None, None]
        + np.arange(4, dtype=np.float32)[None, :, None] * 0.1
        + frequencies[None, None, :] * 0.001
    )
    mask = np.ones_like(phase, dtype=np.bool_)
    phase[0, 3, :2] = np.nan
    mask[0, 3, :2] = False
    shard_path = directory / "shard-00000.h5"
    with h5py.File(shard_path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["config_hash"] = "tiny-fixture"
        handle.attrs["shard_id"] = 0
        handle.create_dataset("sample_id", data=sample_ids)
        handle.create_dataset("vs", data=vs)
        handle.create_dataset("phase_velocity", data=phase)
        handle.create_dataset("valid_mask", data=mask)
    shard_sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": "tiny-fixture",
                "global_seed": 4,
                "expected_shards": 1,
                "completed_shards": [0],
                "accepted_by_kind": {"NORMAL": 8},
                "rejected_by_reason": {},
                "recovered_models": 0,
                "package_version": "0.1.0",
                "created_at": "2026-07-27T00:00:00+00:00",
                "shard_sha256": {"0": shard_sha256},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )


def test_split_dataset_and_normalization_use_training_rows_only(
    tmp_path: Path,
) -> None:
    _write_tiny_dataset(tmp_path)
    training = HDF5ShardDataset(tmp_path, split="train")
    validation = HDF5ShardDataset(tmp_path, split="validation")
    testing = HDF5ShardDataset(tmp_path, split="test")
    assert (len(training), len(validation), len(testing)) == (4, 2, 2)
    _, target, mask = training[0]
    assert np.all(np.isfinite(target.numpy()))
    assert not mask[3, 0]
    normalization = compute_normalization(training)
    assert normalization.input_mean.shape == (20,)
    assert normalization.target_mean.shape == (4, 1)
    assert np.all(normalization.input_std > 0)
    assert np.all(normalization.target_std > 0)


def test_one_epoch_produces_loadable_checkpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    _write_tiny_dataset(data_dir)
    checkpoint = train(
        replace(
            TrainingConfig(),
            dataset_dir=data_dir,
            output_dir=run_dir,
            batch_size=2,
            epochs=1,
            num_workers=0,
            device="cpu",
        )
    )
    assert checkpoint == run_dir / "best.pt"
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history["epochs"]) == 1
    assert np.isfinite(history["epochs"][0]["training_loss"])
    with pytest.raises(ValueError, match="training configuration"):
        train(
            replace(
                TrainingConfig(),
                dataset_dir=data_dir,
                output_dir=run_dir,
                batch_size=2,
                epochs=2,
                num_workers=0,
                device="cpu",
            )
        )
    predictor = ForwardPredictor.load(checkpoint, device="cpu")
    output = predictor.predict(np.linspace(0.4, 2.0, 20))
    assert output.shape == (4, 120)
    assert np.all(np.isfinite(output))
    metrics = evaluate(checkpoint, data_dir, device="cpu")
    assert metrics["mode_0"]["valid_count"] == 240
    assert metrics["mode_0"]["mae_km_s"] >= 0


def test_training_rejects_incomplete_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_dataset(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        train(
            replace(
                TrainingConfig(),
                dataset_dir=data_dir,
                output_dir=tmp_path / "run",
                epochs=1,
                num_workers=0,
                device="cpu",
            )
        )


def test_training_rejects_manifest_whose_shard_was_deleted(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_dataset(data_dir)
    (data_dir / "shard-00000.h5").unlink()
    with pytest.raises(ValueError, match="shard files"):
        train(
            replace(
                TrainingConfig(),
                dataset_dir=data_dir,
                output_dir=tmp_path / "run",
                epochs=1,
                num_workers=0,
                device="cpu",
            )
        )


def test_evaluation_rejects_checkpoint_from_another_dataset(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    other_data_dir = tmp_path / "other-data"
    _write_tiny_dataset(data_dir)
    checkpoint = train(
        replace(
            TrainingConfig(),
            dataset_dir=data_dir,
            output_dir=tmp_path / "run",
            batch_size=2,
            epochs=1,
            num_workers=0,
            device="cpu",
        )
    )
    _write_tiny_dataset(other_data_dir)
    other_shard = other_data_dir / "shard-00000.h5"
    with h5py.File(other_shard, "r+") as handle:
        handle.attrs["config_hash"] = "different-dataset"
    manifest_path = other_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "different-dataset"
    manifest["shard_sha256"]["0"] = hashlib.sha256(
        other_shard.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration hash"):
        evaluate(checkpoint, other_data_dir, device="cpu")
