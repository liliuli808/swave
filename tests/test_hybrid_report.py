from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from swave.config import HybridInversionConfig
from swave.dataset import dataset_manifest_sha256, validate_dataset_files
from swave.hybrid_report import build_hybrid_report
from swave.hybrid_results import (
    HybridResultBatch,
    initialize_hybrid_manifest,
    mark_hybrid_job_complete,
    write_hybrid_result_shard,
)


def _report_batch(dataset_dir: Path) -> HybridResultBatch:
    with h5py.File(dataset_dir / "shard-00000.h5") as handle:
        truth = np.asarray(handle["vs"][1:5], dtype=np.float32)
    count = 4
    data = np.full(count, 0.2, dtype=np.float64)
    smoothness = np.full(count, 0.05, dtype=np.float64)
    learning = np.full(count, 0.1, dtype=np.float64)
    return HybridResultBatch(
        sample_id=np.array([90, 91, 92, 93], dtype=np.uint64),
        model_kind=np.arange(4, dtype=np.uint8),
        valid_mask=np.ones((count, 4, 120), dtype=np.bool_),
        observed_phase_velocity=np.ones((count, 4, 120), dtype=np.float32),
        reference_vs=np.full((count, 20), 1.0, dtype=np.float32),
        supervised_vs=np.asarray(truth + 0.03, dtype=np.float32),
        sensitivity=np.broadcast_to(
            np.linspace(0.01, 0.2, 20), (count, 20)
        ).copy(),
        prior_weights=np.ones((count, 20), dtype=np.float64),
        control_success=np.ones(count, dtype=np.bool_),
        control_status=np.zeros(count, dtype=np.int32),
        control_failure_code=np.full(count, b"", dtype="S64"),
        control_iterations=np.full(count, 5, dtype=np.int32),
        control_evaluations=np.full(count, 6, dtype=np.int32),
        control_initial_objective=np.full(count, 1.0, dtype=np.float64),
        control_total=data + smoothness,
        control_data_misfit=data,
        control_smoothness=smoothness,
        control_vs=np.asarray(truth + 0.2, dtype=np.float32),
        control_prediction=np.full((count, 4, 120), 1.1, dtype=np.float32),
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
        hybrid_vs=np.asarray(truth + 0.05, dtype=np.float32),
        hybrid_prediction=np.full((count, 4, 120), 1.02, dtype=np.float32),
    )


def _complete_hybrid_results(tmp_path: Path, dataset_dir: Path) -> Path:
    root = tmp_path / "hybrid-results"
    results = root / "inversion"
    forward = tmp_path / "forward.pt"
    forward.write_bytes(b"forward")
    config = replace(
        HybridInversionConfig(), forward_checkpoint=forward, output_dir=results
    )
    job = "hybrid-inversion-clean-shard-00000"
    manifest = initialize_hybrid_manifest(
        results,
        split="inversion",
        dataset_config_hash=validate_dataset_files(dataset_dir).config_hash,
        dataset_manifest_sha256=dataset_manifest_sha256(
            validate_dataset_files(dataset_dir)
        ),
        forward_checkpoint=forward,
        supervised_checkpoint_sha256=("a" * 64,),
        config=config,
        selected_prior_lambda=0.1,
        expected_sample_ids_by_job={
            job: np.array([90, 91, 92, 93], dtype=np.uint64)
        },
    )
    path = write_hybrid_result_shard(
        results / f"{job}.h5", _report_batch(dataset_dir), manifest, job
    )
    mark_hybrid_job_complete(results, job, path)
    return root


def test_hybrid_report_joins_truth_and_reports_control_hybrid_and_layers(
    tiny_complete_dataset: Path, tmp_path: Path
) -> None:
    results = _complete_hybrid_results(tmp_path, tiny_complete_dataset)
    output = tmp_path / "report"

    summary = build_hybrid_report(
        results,
        tiny_complete_dataset,
        output,
        baseline_summary=None,
        supervised_evaluation=None,
    )

    clean = summary["splits"]["inversion"]["by_noise"]["clean"]
    assert clean["control"]["overall"]["mae_km_s"] == pytest.approx(0.2)
    assert clean["hybrid"]["overall"]["mae_km_s"] == pytest.approx(0.05)
    assert clean["supervised_prior"]["overall"]["mae_km_s"] == pytest.approx(0.03)
    assert len(clean["hybrid"]["per_layer"]) == 20
    assert len(clean["mean_dimensionless_sensitivity_by_layer"]) == 20
    assert clean["mean_prior_weight_by_layer"] == pytest.approx([1.0] * 20)
    assert clean["optimization"]["control_mean_iterations"] == pytest.approx(5.0)
    assert clean["optimization"]["hybrid_mean_iterations"] == pytest.approx(3.0)
    assert summary["comparisons"]["baseline"]["available"] is False
    assert summary["comparisons"]["supervised_evaluation"]["available"] is False
    assert json.loads((output / "summary.json").read_text()) == summary
    assert (output / "sensitivity-and-prior-weight-by-depth.png").is_file()
