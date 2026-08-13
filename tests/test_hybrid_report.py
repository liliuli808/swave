from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from swave.config import (
    HybridInversionConfig,
    hybrid_inversion_identity_hash,
    inversion_identity_hash,
    load_inversion_config,
)
from swave.dataset import dataset_manifest_sha256, validate_dataset_files
from swave.hybrid_report import _population_identity, build_hybrid_report
from swave.hybrid_results import (
    HybridResultBatch,
    initialize_hybrid_manifest,
    mark_hybrid_job_complete,
    write_hybrid_result_shard,
)
from swave.inversion_results import (
    checkpoint_sha256,
    sample_id_sha256,
    software_sha256,
)
from swave.splits import SPLIT_POLICY


def _report_batch(
    dataset_dir: Path,
    sample_ids: np.ndarray | None = None,
) -> HybridResultBatch:
    selected_ids = (
        np.array([90, 91, 92, 93], dtype=np.uint64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.uint64)
    )
    with h5py.File(dataset_dir / "shard-00000.h5") as handle:
        stored_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
        indexes = np.searchsorted(stored_ids, selected_ids)
        truth = np.asarray(handle["vs"][indexes], dtype=np.float32)
        model_kind = np.asarray(handle["model_kind"][indexes], dtype=np.uint8)
    count = len(selected_ids)
    data = np.full(count, 0.2, dtype=np.float64)
    smoothness = np.full(count, 0.05, dtype=np.float64)
    learning = np.full(count, 0.1, dtype=np.float64)
    return HybridResultBatch(
        sample_id=selected_ids,
        model_kind=model_kind,
        valid_mask=np.ones((count, 4, 120), dtype=np.bool_),
        observed_phase_velocity=np.ones((count, 4, 120), dtype=np.float32),
        reference_vs=np.full((count, 20), 1.0, dtype=np.float32),
        supervised_vs=np.asarray(truth + 0.03, dtype=np.float32),
        sensitivity=np.broadcast_to(
            np.linspace(0.01, 0.2, 20), (count, 20)
        ).copy(),
        prior_weights=np.ones((count, 20), dtype=np.float64),
        preparation_success=np.ones(count, dtype=np.bool_),
        preparation_failure_code=np.full(count, b"", dtype="S64"),
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
    forward = tmp_path / "forward.pt"
    forward.write_bytes(b"forward")
    config = replace(
        HybridInversionConfig(), forward_checkpoint=forward, output_dir=root
    )
    validation_ids = np.array([80], dtype=np.uint64)
    tuning = {
        "schema_version": 1,
        "split_policy": SPLIT_POLICY,
        "usage": "validation_only_prior_lambda_selection",
        "dataset_config_hash": validate_dataset_files(dataset_dir).config_hash,
        "dataset_manifest_sha256": dataset_manifest_sha256(
            validate_dataset_files(dataset_dir)
        ),
        "forward_checkpoint_sha256": checkpoint_sha256(forward),
        "supervised_checkpoint_sha256": ["a" * 64, "d" * 64, "e" * 64],
        "supervised_seeds": [0, 1, 2],
        "supervised_run_identity_sha256": "b" * 64,
        "hybrid_config_hash": hybrid_inversion_identity_hash(config),
        "software_sha256": software_sha256(),
        "validation_sample_count": 1,
        "validation_sample_id_sha256": sample_id_sha256(validation_ids),
        "validation_sample_ids": [80],
        "prior_lambda_candidates": [0.1],
        "noise_scenarios": ["clean", "noise_1pct"],
        "candidate_mae_km_s": {"0.1": 0.05},
        "selected_prior_lambda": 0.1,
        "selection_metric": (
            "mean_vs_mae_across_validation_samples_layers_and_noise"
        ),
        "tie_break": "smallest_prior_lambda",
    }
    tuning_path = root / "tuning.json"
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    populations = {
        "test": np.array([89], dtype=np.uint64),
        "inversion": np.array([90, 91, 92, 93], dtype=np.uint64),
    }
    for split, sample_ids in populations.items():
        results = root / split
        jobs = {
            f"hybrid-{split}-clean-shard-00000": sample_ids,
            f"hybrid-{split}-noise_1pct-shard-00000": sample_ids,
        }
        manifest = initialize_hybrid_manifest(
            results,
            split=split,
            dataset_config_hash=validate_dataset_files(dataset_dir).config_hash,
            dataset_manifest_sha256=dataset_manifest_sha256(
                validate_dataset_files(dataset_dir)
            ),
            forward_checkpoint=forward,
            supervised_checkpoint_sha256=("a" * 64, "d" * 64, "e" * 64),
            supervised_seeds=(0, 1, 2),
            supervised_run_identity_sha256="b" * 64,
            tuning_sha256=checkpoint_sha256(tuning_path),
            config=config,
            selected_prior_lambda=0.1,
            expected_sample_ids_by_job=jobs,
        )
        for job in jobs:
            path = write_hybrid_result_shard(
                results / f"{job}.h5",
                _report_batch(dataset_dir, sample_ids),
                manifest,
                job,
            )
            manifest = mark_hybrid_job_complete(results, job, path)
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
    combined = summary["splits"]["inversion"]["combined_noise_scenarios"]
    assert clean["control"]["overall"]["mae_km_s"] == pytest.approx(0.2)
    assert clean["hybrid"]["overall"]["mae_km_s"] == pytest.approx(0.05)
    assert clean["supervised_prior"]["overall"]["mae_km_s"] == pytest.approx(0.03)
    assert len(clean["hybrid"]["per_layer"]) == 20
    assert len(clean["mean_dimensionless_sensitivity_by_layer"]) == 20
    assert clean["mean_prior_weight_by_layer"] == pytest.approx([1.0] * 20)
    assert clean["optimization"]["control_mean_iterations"] == pytest.approx(5.0)
    assert clean["optimization"]["hybrid_mean_iterations"] == pytest.approx(3.0)
    assert combined["sample_count"] == 8
    assert combined["hybrid"]["overall"]["mae_km_s"] == pytest.approx(0.05)
    assert summary["comparisons"]["baseline"]["available"] is False
    assert summary["comparisons"]["supervised_evaluation"]["available"] is False
    assert set(summary["same_population_methods"]["inversion"]) == {
        "narrow_bound_lbfgsb",
        "global_bound_control",
        "sensitivity_weighted_hybrid",
        "direct_supervised_prior",
    }
    assert json.loads((output / "summary.json").read_text()) == summary
    assert (output / "sensitivity-and-prior-weight-by-depth.png").is_file()


def test_hybrid_report_rejects_duplicate_sample_ids_across_shards(
    tiny_complete_dataset: Path,
) -> None:
    batch = _report_batch(tiny_complete_dataset)
    with pytest.raises(ValueError, match="duplicated across shards"):
        _population_identity({"clean": [batch, batch]})


def test_hybrid_report_rejects_cross_split_scientific_identity_mismatch(
    tiny_complete_dataset: Path, tmp_path: Path
) -> None:
    root = _complete_hybrid_results(tmp_path, tiny_complete_dataset)
    manifest_path = root / "test" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selected_prior_lambda"] = 0.2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific identity"):
        build_hybrid_report(root, tiny_complete_dataset, tmp_path / "report")


def test_hybrid_report_validates_exact_external_comparison_population(
    tiny_complete_dataset: Path, tmp_path: Path
) -> None:
    results = _complete_hybrid_results(tmp_path, tiny_complete_dataset)
    dataset_manifest = validate_dataset_files(tiny_complete_dataset)
    dataset_digest = dataset_manifest_sha256(dataset_manifest)
    population = {
        "sample_count": 4,
        "sample_id_sha256": sample_id_sha256(
            np.array([90, 91, 92, 93], dtype=np.uint64)
        ),
    }
    test_population = {
        "sample_count": 1,
        "sample_id_sha256": sample_id_sha256(
            np.array([89], dtype=np.uint64)
        ),
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "result_identity": {
                    "dataset_config_hash": dataset_manifest.config_hash,
                    "dataset_manifest_sha256": dataset_digest,
                    "split_policy": SPLIT_POLICY,
                    "checkpoint_sha256": checkpoint_sha256(
                        tmp_path / "forward.pt"
                    ),
                    "inversion_config_hash": inversion_identity_hash(
                        load_inversion_config("configs/inversion.toml")
                    ),
                    "experiment": "both",
                },
                "comparison_populations": {"inversion": population},
                "experiment_scopes": {
                    "full": {"groups": {"overall": {"sample_count": 4}}}
                },
            }
        ),
        encoding="utf-8",
    )
    supervised = tmp_path / "supervised.json"
    supervised.write_text(
        json.dumps(
            {
                "dataset_config_hash": dataset_manifest.config_hash,
                "dataset_manifest_sha256": dataset_digest,
                "split_policy": SPLIT_POLICY,
                "seed_ensemble": [0, 1, 2],
                "supervised_checkpoint_sha256": [
                    "a" * 64,
                    "d" * 64,
                    "e" * 64,
                ],
                "supervised_run_identity_sha256": "b" * 64,
                "split_sample_identity": {
                    "test": test_population,
                    "inversion": population,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_hybrid_report(
        results,
        tiny_complete_dataset,
        tmp_path / "report",
        baseline_summary=baseline,
        supervised_evaluation=supervised,
    )
    assert summary["comparisons"]["baseline"]["available"] is True
    assert summary["comparisons"]["supervised_evaluation"]["available"] is True
    assert summary["tuning"]["selected_prior_lambda"] == pytest.approx(0.1)

    changed = json.loads(supervised.read_text(encoding="utf-8"))
    changed["split_sample_identity"]["inversion"]["sample_count"] = 5
    supervised.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="sample population"):
        build_hybrid_report(
            results,
            tiny_complete_dataset,
            tmp_path / "bad-report",
            supervised_evaluation=supervised,
        )

    tuning = results / "tuning.json"
    tuning.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="tuning artifact identity"):
        build_hybrid_report(
            results,
            tiny_complete_dataset,
            tmp_path / "tampered-tuning-report",
        )
