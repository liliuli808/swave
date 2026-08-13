from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from swave.config import HybridInversionConfig
from swave.hybrid_results import (
    HybridResultBatch,
    initialize_hybrid_manifest,
    mark_hybrid_job_complete,
    validate_complete_hybrid_results,
    validate_hybrid_result_shard,
    write_hybrid_result_shard,
)


def _batch() -> HybridResultBatch:
    count = 2
    weights = np.stack(
        [np.linspace(0.5, 1.5, 20), np.linspace(1.5, 0.5, 20)]
    ).astype(np.float64)
    data = np.array([0.2, 0.3], dtype=np.float64)
    smoothness = np.array([0.05, 0.04], dtype=np.float64)
    learning = np.array([0.1, 0.2], dtype=np.float64)
    return HybridResultBatch(
        sample_id=np.array([85, 86], dtype=np.uint64),
        model_kind=np.array([0, 1], dtype=np.uint8),
        valid_mask=np.ones((count, 4, 120), dtype=np.bool_),
        observed_phase_velocity=np.ones((count, 4, 120), dtype=np.float32),
        reference_vs=np.full((count, 20), 1.0, dtype=np.float32),
        supervised_vs=np.full((count, 20), 1.2, dtype=np.float32),
        sensitivity=np.full((count, 20), 0.1, dtype=np.float64),
        prior_weights=weights,
        control_success=np.ones(count, dtype=np.bool_),
        control_status=np.zeros(count, dtype=np.int32),
        control_failure_code=np.full(count, b"", dtype="S64"),
        control_iterations=np.full(count, 4, dtype=np.int32),
        control_evaluations=np.full(count, 5, dtype=np.int32),
        control_initial_objective=np.full(count, 1.0, dtype=np.float64),
        control_total=data + smoothness,
        control_data_misfit=data,
        control_smoothness=smoothness,
        control_vs=np.full((count, 20), 1.1, dtype=np.float32),
        control_prediction=np.ones((count, 4, 120), dtype=np.float32),
        hybrid_success=np.ones(count, dtype=np.bool_),
        hybrid_status=np.zeros(count, dtype=np.int32),
        hybrid_failure_code=np.full(count, b"", dtype="S64"),
        hybrid_iterations=np.full(count, 3, dtype=np.int32),
        hybrid_evaluations=np.full(count, 4, dtype=np.int32),
        hybrid_initial_objective=np.full(count, 1.1, dtype=np.float64),
        hybrid_total=data + smoothness + learning,
        hybrid_data_misfit=data,
        hybrid_smoothness=smoothness,
        hybrid_learning_prior=learning,
        hybrid_vs=np.full((count, 20), 1.15, dtype=np.float32),
        hybrid_prediction=np.ones((count, 4, 120), dtype=np.float32),
    )


def _manifest_context(tmp_path: Path):
    output = tmp_path / "results"
    forward = tmp_path / "forward.pt"
    forward.write_bytes(b"forward-checkpoint")
    config = replace(
        HybridInversionConfig(),
        forward_checkpoint=forward,
        output_dir=output,
        device="cpu",
        workers=1,
    )
    job = "hybrid-test-clean-shard-00000"
    manifest = initialize_hybrid_manifest(
        output,
        split="test",
        dataset_config_hash="a" * 64,
        dataset_manifest_sha256="b" * 64,
        forward_checkpoint=forward,
        supervised_checkpoint_sha256=("c" * 64, "d" * 64, "e" * 64),
        config=config,
        selected_prior_lambda=0.1,
        expected_sample_ids_by_job={
            job: np.array([85, 86], dtype=np.uint64),
        },
    )
    return output, job, manifest


def test_hybrid_result_round_trip_and_complete_manifest(tmp_path: Path) -> None:
    output, job, manifest = _manifest_context(tmp_path)
    path = write_hybrid_result_shard(output / f"{job}.h5", _batch(), manifest, job)

    loaded = validate_hybrid_result_shard(
        path,
        manifest=manifest,
        expected_sample_ids=np.array([85, 86], dtype=np.uint64),
    )
    np.testing.assert_array_equal(loaded.prior_weights, _batch().prior_weights)

    completed = mark_hybrid_job_complete(output, job, path)
    assert completed.complete
    assert validate_complete_hybrid_results(output).complete


def test_hybrid_result_detects_term_and_content_corruption(tmp_path: Path) -> None:
    invalid = replace(_batch(), hybrid_total=np.array([99.0, 99.0]))
    with pytest.raises(ValueError, match="hybrid_total"):
        invalid.validate()

    output, job, manifest = _manifest_context(tmp_path)
    path = write_hybrid_result_shard(output / f"{job}.h5", _batch(), manifest, job)
    with h5py.File(path, "r+") as handle:
        handle["hybrid_vs"][0, 0] = 2.0

    with pytest.raises(ValueError, match="content checksum"):
        validate_hybrid_result_shard(path, manifest=manifest)
