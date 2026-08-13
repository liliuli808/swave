from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import swave.supervised_inversion as supervised_module
from swave.splits import SPLIT_POLICY
from swave.supervised_inversion import (
    EpochShuffleSampler,
    InverseNet,
    SupervisedConfig,
    SupervisedEnsemblePredictor,
    SupervisedHDF5BatchDataset,
    SupervisedHDF5Dataset,
    compute_supervised_normalization,
    train_supervised,
)


def _write_supervised_ensemble(
    output_dir: Path,
    *,
    second_seed: int = 1,
    second_fill_offset: float = 0.0,
) -> None:
    output_dir.mkdir()
    model_config = {
        "input_dim": 476,
        "output_dim": 20,
        "width": 4,
        "blocks": 0,
        "dropout": 0.0,
    }
    identity = {
        "schema_version": 1,
        "split_policy": SPLIT_POLICY,
        "dataset_config_hash": "fixture-config",
        "dataset_manifest_sha256": "a" * 64,
        "train_sample_count": 80,
        "train_sample_id_sha256": "b" * 64,
        "training_configuration_sha256": "c" * 64,
        "seed_ensemble": [0, 1],
        "epoch_randomness": "sha256-derived-from-seed-and-epoch-v1",
        "batch_order": "contiguous-hdf5-spans-epoch-shuffled-v1",
        "model_config": model_config,
    }
    (output_dir / "run-identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    for index, (seed, normalized_bias) in enumerate(((0, 1.0), (second_seed, 3.0))):
        model = InverseNet(**model_config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[1].bias.fill_(normalized_bias)
        fill = np.full((4, 119), 7.0, dtype=np.float32)
        if index == 1:
            fill += second_fill_offset
        payload = {
            **identity,
            "seed": seed,
            "fill_values": fill,
            "input_mean": np.full(476, 7.0, dtype=np.float32),
            "input_std": np.full(476, 2.0, dtype=np.float32),
            "target_mean": np.full(20, 10.0, dtype=np.float32),
            "target_std": np.full(20, 2.0, dtype=np.float32),
            "model": model.state_dict(),
        }
        torch.save(payload, output_dir / f"seed-{index}-best.pt")


def test_supervised_ensemble_predictor_applies_training_preprocessing_and_mean(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_supervised_ensemble(output_dir)
    observed = np.full((4, 120), 7.0, dtype=np.float64)
    valid = np.ones((4, 120), dtype=np.bool_)
    observed[2, 25] = np.nan
    valid[2, 25] = False

    predictor = SupervisedEnsemblePredictor.load(output_dir, device="cpu")
    prediction = predictor.predict(observed, valid)

    assert prediction.shape == (20,)
    np.testing.assert_allclose(prediction, 14.0)
    assert predictor.seeds == (0, 1)
    assert len(predictor.checkpoint_sha256) == 2
    assert all(len(value) == 64 for value in predictor.checkpoint_sha256)


def test_supervised_ensemble_predictor_requires_every_ordered_checkpoint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_supervised_ensemble(output_dir)
    (output_dir / "seed-1-best.pt").unlink()

    with pytest.raises(ValueError, match="seed-1-best.pt"):
        SupervisedEnsemblePredictor.load(output_dir, device="cpu")


def test_supervised_ensemble_predictor_rejects_checkpoint_seed_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_supervised_ensemble(output_dir, second_seed=7)

    with pytest.raises(ValueError, match="seed"):
        SupervisedEnsemblePredictor.load(output_dir, device="cpu")


def test_supervised_ensemble_predictor_rejects_normalization_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_supervised_ensemble(output_dir, second_fill_offset=0.5)

    with pytest.raises(ValueError, match="normalization"):
        SupervisedEnsemblePredictor.load(output_dir, device="cpu")


def test_supervised_ensemble_predictor_rejects_nonfinite_valid_observation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_supervised_ensemble(output_dir)
    predictor = SupervisedEnsemblePredictor.load(output_dir, device="cpu")
    observed = np.ones((4, 120), dtype=np.float64)
    valid = np.ones((4, 120), dtype=np.bool_)
    observed[0, 10] = np.nan

    with pytest.raises(ValueError, match="valid observations"):
        predictor.predict(observed, valid)


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


def test_supervised_batches_are_contiguous_bounded_and_complete(
    four_split_dataset: Path,
) -> None:
    stats = compute_supervised_normalization(four_split_dataset)
    dataset = SupervisedHDF5BatchDataset(
        four_split_dataset,
        "train",
        stats,
        batch_size=16,
    )

    batches = [dataset[index] for index in range(len(dataset))]
    sample_ids = np.concatenate([batch[2].numpy() for batch in batches])

    assert np.array_equal(np.sort(sample_ids), np.arange(80))
    assert max(len(batch[0]) for batch in batches) <= 16
    assert all(
        start < stop
        for _, spans in dataset.entries
        for start, stop in spans
    )


def test_epoch_sampler_is_history_independent() -> None:
    sampler = EpochShuffleSampler(size=12, seed=7)
    sampler.set_epoch(3)
    expected = list(sampler)
    sampler.set_epoch(1)
    list(sampler)
    sampler.set_epoch(3)

    assert list(sampler) == expected
    assert sorted(expected) == list(range(12))


def test_production_shard_becomes_one_contiguous_training_batch(
    four_split_dataset: Path,
    tmp_path: Path,
) -> None:
    stats = compute_supervised_normalization(four_split_dataset)
    directory = tmp_path / "production-shaped"
    directory.mkdir()
    path = directory / "shard-00000.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample_id", data=np.arange(10_000, dtype=np.uint64))

    dataset = SupervisedHDF5BatchDataset(
        directory,
        "train",
        stats,
        batch_size=8192,
    )

    assert len(dataset.entries) == 1
    entry_path, spans = dataset.entries[0]
    assert entry_path == path
    assert len(spans) == 100
    assert sum(stop - start for start, stop in spans) == 8000
    assert all(stop - start == 80 for start, stop in spans)


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
        dropout=0.2,
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
        dropout=0.2,
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


def test_resume_rejects_changed_seed_ensemble(
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
        num_workers=0,
        device="cpu",
    )
    train_supervised(config)

    with pytest.raises(ValueError, match="training configuration"):
        train_supervised(replace(config, seeds=(1,)))


def test_terminal_early_stop_is_idempotent(
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
        epochs=2,
        patience=1,
        num_workers=0,
        device="cpu",
    )
    train_supervised(config)
    last_path = config.output_dir / "seed-0-last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    payload["epoch"] = 0
    payload["bad_epochs"] = config.patience
    torch.save(payload, last_path)

    train_supervised(config)

    resumed = torch.load(last_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 0
    assert resumed["bad_epochs"] == config.patience


@pytest.mark.parametrize("failure_boundary", ["best", "history"])
def test_last_checkpoint_is_written_only_after_epoch_artifacts(
    four_split_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
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
        num_workers=0,
        device="cpu",
    )
    if failure_boundary == "best":
        original = supervised_module._atomic_torch_save

        def fail_before_best(path: Path, payload: dict[str, object]) -> None:
            if path.name == "seed-0-best.pt":
                raise RuntimeError("best boundary")
            original(path, payload)

        monkeypatch.setattr(
            supervised_module,
            "_atomic_torch_save",
            fail_before_best,
        )
    else:
        original_json = supervised_module._atomic_json_save

        def fail_before_history(path: Path, payload: dict[str, object]) -> None:
            if path.name == "seed-0-history.json":
                raise RuntimeError("history boundary")
            original_json(path, payload)

        monkeypatch.setattr(
            supervised_module,
            "_atomic_json_save",
            fail_before_history,
        )

    with pytest.raises(RuntimeError, match=failure_boundary):
        train_supervised(config)

    assert not (config.output_dir / "seed-0-last.pt").exists()


def test_interrupted_resume_matches_uninterrupted_training(
    four_split_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = {
        "dataset_dir": four_split_dataset,
        "seeds": (0,),
        "width": 8,
        "blocks": 1,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs": 2,
        "patience": 2,
        "num_workers": 0,
        "device": "cpu",
    }
    interrupted = SupervisedConfig(
        output_dir=tmp_path / "interrupted",
        **common,
    )
    original_json_save = supervised_module._atomic_json_save
    raised = False

    def interrupt_after_first_history(
        path: Path, payload: dict[str, object]
    ) -> None:
        nonlocal raised
        original_json_save(path, payload)
        if not raised and path.name == "seed-0-history.json":
            raised = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        supervised_module,
        "_atomic_json_save",
        interrupt_after_first_history,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_supervised(interrupted)
    monkeypatch.setattr(
        supervised_module,
        "_atomic_json_save",
        original_json_save,
    )

    train_supervised(interrupted)
    uninterrupted = replace(
        interrupted,
        output_dir=tmp_path / "uninterrupted",
    )
    train_supervised(uninterrupted)

    resumed_payload = torch.load(
        interrupted.output_dir / "seed-0-last.pt",
        map_location="cpu",
        weights_only=False,
    )
    continuous_payload = torch.load(
        uninterrupted.output_dir / "seed-0-last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_payload["history"] == continuous_payload["history"]
    for name, value in resumed_payload["model"].items():
        assert torch.equal(value, continuous_payload["model"][name]), name
