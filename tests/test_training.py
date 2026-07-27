import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

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
    with h5py.File(directory / "shard-00000.h5", "w") as handle:
        handle.create_dataset("sample_id", data=sample_ids)
        handle.create_dataset("vs", data=vs)
        handle.create_dataset("phase_velocity", data=phase)
        handle.create_dataset("valid_mask", data=mask)
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
    predictor = ForwardPredictor.load(checkpoint, device="cpu")
    output = predictor.predict(np.linspace(0.4, 2.0, 20))
    assert output.shape == (4, 120)
    assert np.all(np.isfinite(output))
    metrics = evaluate(checkpoint, data_dir, device="cpu")
    assert metrics["mode_0"]["valid_count"] == 240
    assert metrics["mode_0"]["mae_km_s"] >= 0
