from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

import swave.inversion_report as report_module
from swave.config import (
    DatasetConfig,
    InversionConfig,
    PhysicsConfig,
    canonical_hash,
)
from swave.dataset import dataset_manifest_sha256, load_manifest
from swave.inversion_report import (
    build_inversion_report,
    compute_frequency_metrics,
    compute_interval_metrics,
    compute_inversion_metrics,
    compute_vs_metrics,
)
from swave.inversion_results import (
    ResultBatch,
    initialize_result_manifest,
    mark_job_complete,
    sample_id_sha256,
    write_result_shard,
)

IDS = np.array([90, 91, 92, 93], dtype=np.uint64)
KINDS = np.arange(4, dtype=np.uint8)
REPORT_DATASET_CONFIG = DatasetConfig(
    physics=PhysicsConfig(fmin=1.0, fmax=120.0, fstep=1.0)
)
REPORT_DATASET_HASH = canonical_hash(REPORT_DATASET_CONFIG)


def _result_batch(
    truth: np.ndarray,
    *,
    offset: float,
    deep: bool,
    sample_ids: np.ndarray = IDS,
    model_kinds: np.ndarray = KINDS,
) -> ResultBatch:
    count = truth.shape[0]
    if sample_ids.shape != (count,) or model_kinds.shape != (count,):
        raise ValueError("fixture identities must align with truth")
    recovered = np.asarray(truth + offset, dtype=np.float32)
    observed = np.ones((count, 4, 120), dtype=np.float32)
    surrogate = np.asarray(observed + offset, dtype=np.float32)
    arguments: dict[str, np.ndarray | None] = {
        "sample_id": np.asarray(sample_ids, dtype=np.uint64),
        "model_kind": np.asarray(model_kinds, dtype=np.uint8),
        "success": np.ones(count, dtype=np.bool_),
        "status": np.zeros(count, dtype=np.int32),
        "iterations": np.arange(1, count + 1, dtype=np.int32),
        "evaluations": np.arange(2, count + 2, dtype=np.int32),
        "initial_objective": np.full(count, 2.0, dtype=np.float64),
        "final_objective": np.full(count, 1.0, dtype=np.float64),
        "data_misfit": np.full(count, 0.75, dtype=np.float64),
        "regularization": np.full(count, 0.25, dtype=np.float64),
        "reference_vs": np.asarray(truth, dtype=np.float32),
        "inverted_vs": recovered,
        "observed_phase_velocity": observed,
        "surrogate_phase_velocity": surrogate,
        "valid_mask": np.ones_like(observed, dtype=np.bool_),
        "failure_code": np.full(count, b"", dtype="S64"),
    }
    if deep:
        starts = 2
        arguments.update(
            ensemble_vs=np.broadcast_to(
                recovered[:, None, :], (count, starts, 20)
            ).copy(),
            ensemble_success=np.ones((count, starts), dtype=np.bool_),
            ensemble_status=np.zeros((count, starts), dtype=np.int32),
            ensemble_iterations=np.full((count, starts), 4, dtype=np.int32),
            ensemble_evaluations=np.full((count, starts), 5, dtype=np.int32),
            ensemble_initial_objective=np.full((count, starts), 2.0, dtype=np.float64),
            ensemble_objective=np.ones((count, starts), dtype=np.float64),
            ensemble_failure_code=np.full((count, starts), b"", dtype="S64"),
            ensemble_message=np.full((count, starts), b"converged", dtype="S256"),
            ensemble_inlier_mask=np.ones((count, starts), dtype=np.bool_),
            median_vs=recovered.copy(),
            p10_vs=np.asarray(recovered - 0.1, dtype=np.float32),
            p90_vs=np.asarray(recovered + 0.1, dtype=np.float32),
            physical_success=np.ones(count, dtype=np.bool_),
            physical_status=np.zeros(count, dtype=np.int32),
            physical_failure_code=np.full(count, b"", dtype="S64"),
            physical_phase_velocity=np.asarray(
                observed + offset + 0.05, dtype=np.float32
            ),
            physical_valid_mask=np.ones_like(observed, dtype=np.bool_),
        )
    return ResultBatch(**arguments)  # type: ignore[arg-type]


def _source_dataset(tmp_path: Path) -> tuple[Path, np.ndarray]:
    directory = tmp_path / "dataset"
    directory.mkdir()
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    with h5py.File(directory / "shard-00000.h5", "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["shard_id"] = 0
        handle.attrs["config_hash"] = REPORT_DATASET_HASH
        handle.create_dataset("sample_id", data=IDS, dtype="u8")
        handle.create_dataset("vs", data=truth, dtype="f4")
    shard_sha256 = hashlib.sha256(
        (directory / "shard-00000.h5").read_bytes()
    ).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": REPORT_DATASET_HASH,
                "global_seed": 20260727,
                "expected_shards": 1,
                "completed_shards": [0],
                "accepted_by_kind": {},
                "rejected_by_kind": {},
                "rejected_by_reason": {},
                "recovered_models": 0,
                "package_version": "0.1.0",
                "created_at": "2026-08-02T00:00:00+00:00",
                "shard_sha256": {"0": shard_sha256},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return directory, truth


def _complete_results(tmp_path: Path, checkpoint: Path, truth: np.ndarray) -> Path:
    results = tmp_path / "results"
    jobs = (
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
        (
            f"deep-clean-samples-{int(IDS[0]):020d}-{int(IDS[-1]):020d}-"
            f"{sample_id_sha256(IDS)[:12]}"
        ),
        (
            f"deep-noise_1pct-samples-{int(IDS[0]):020d}-{int(IDS[-1]):020d}-"
            f"{sample_id_sha256(IDS)[:12]}"
        ),
    )
    config = InversionConfig(initial_models=2, minimum_valid_solutions=1)
    manifest = initialize_result_manifest(
        results,
        dataset_config_hash=REPORT_DATASET_HASH,
        dataset_manifest_sha256=dataset_manifest_sha256(
            load_manifest(tmp_path / "dataset" / "manifest.json")
        ),
        checkpoint=checkpoint,
        config=config,
        experiment="both",
        expected_jobs=jobs,
        expected_sample_ids_by_job={job: IDS for job in jobs},
    )
    for job in jobs:
        deep = job.startswith("deep-")
        offset = 0.05 if "-clean-" in job else 0.15
        path = write_result_shard(
            results,
            job,
            _result_batch(truth, offset=offset, deep=deep),
            manifest,
        )
        mark_job_complete(results, job, path)
    return results


def test_compute_vs_metrics_reports_exact_scalar_and_layer_values() -> None:
    truth = np.array([[1.0] * 20, [2.0] * 20], dtype=np.float64)
    recovered = np.array([[1.1] * 20, [1.8] * 20], dtype=np.float64)

    metrics = compute_vs_metrics(truth, recovered)

    assert metrics["row_count"] == 2
    assert metrics["value_count"] == 40
    assert metrics["mae_km_s"] == pytest.approx(0.15)
    assert metrics["rmse_km_s"] == pytest.approx(np.sqrt(0.025))
    assert metrics["mean_relative_percent"] == pytest.approx(10.0)
    assert metrics["p95_absolute_error_km_s"] == pytest.approx(0.2)
    assert metrics["zero_truth_count"] == 0
    assert metrics["relative_denominator_count"] == 40
    assert metrics["per_layer"]["mae_km_s"] == pytest.approx([0.15] * 20)
    assert metrics["per_layer"]["bias_km_s"] == pytest.approx([-0.05] * 20)


def test_compute_vs_metrics_explicitly_excludes_zero_truth_denominators() -> None:
    truth = np.zeros((1, 20), dtype=np.float64)
    recovered = np.ones((1, 20), dtype=np.float64)

    metrics = compute_vs_metrics(truth, recovered)

    assert metrics["mean_relative_percent"] is None
    assert metrics["zero_truth_count"] == 20
    assert metrics["relative_denominator_count"] == 0


def test_frequency_metrics_ignore_noncanonical_infinity_outside_masks() -> None:
    observed = np.array(
        [[[[1.0, 2.0, np.inf], [3.0, 4.0, np.inf]]]], dtype=np.float64
    ).reshape(1, 2, 3)
    predicted = np.array(
        [[[[1.5, 1.5, -np.inf], [2.0, 6.0, np.inf]]]], dtype=np.float64
    ).reshape(1, 2, 3)
    observed_mask = np.array(
        [[[[True, True, False], [True, True, False]]]], dtype=np.bool_
    ).reshape(1, 2, 3)
    predicted_mask = np.array(
        [[[[True, False, False], [True, True, False]]]], dtype=np.bool_
    ).reshape(1, 2, 3)

    metrics = compute_frequency_metrics(
        observed,
        predicted,
        observed_mask,
        predicted_mask=predicted_mask,
    )

    assert metrics["overall"]["observed_count"] == 4
    assert metrics["overall"]["compared_count"] == 3
    assert metrics["overall"]["missing_fraction"] == pytest.approx(0.25)
    assert metrics["overall"]["mae_km_s"] == pytest.approx(7.0 / 6.0)
    assert metrics["mode_0"]["compared_count"] == 1
    assert metrics["mode_1"]["mae_km_s"] == pytest.approx(1.5)


def test_frequency_metrics_reject_nonfinite_values_inside_masks() -> None:
    observed = np.ones((1, 4, 3), dtype=np.float64)
    predicted = observed.copy()
    mask = np.ones_like(observed, dtype=np.bool_)
    predicted[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite.*active mask"):
        compute_frequency_metrics(observed, predicted, mask)


def test_interval_metrics_report_exact_coverage_and_width() -> None:
    truth = np.array([[1.0] * 20, [2.0] * 20], dtype=np.float64)
    p10 = np.array([[0.9] * 20, [2.1] * 20], dtype=np.float64)
    p90 = np.array([[1.1] * 20, [2.5] * 20], dtype=np.float64)

    metrics = compute_interval_metrics(truth, p10, p90)

    assert metrics["coverage_fraction"] == pytest.approx(0.5)
    assert metrics["mean_interval_width_km_s"] == pytest.approx(0.3)
    assert metrics["per_layer"]["coverage_fraction"] == pytest.approx([0.5] * 20)
    assert metrics["per_layer"]["mean_interval_width_km_s"] == pytest.approx([0.3] * 20)


def test_compute_inversion_metrics_keeps_failures_out_of_accuracy() -> None:
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    batch = _result_batch(truth, offset=0.1, deep=False)
    failed = replace(
        batch,
        success=np.array([True, True, True, False], dtype=np.bool_),
        failure_code=np.array([b"", b"", b"", b"optimizer_failure"], dtype="S64"),
    )

    metrics = compute_inversion_metrics(failed, truth)

    assert metrics["sample_count"] == 4
    assert metrics["successful_count"] == 3
    assert metrics["sample_outcomes"]["inversion"]["success_fraction"] == pytest.approx(
        0.75
    )
    assert metrics["sample_outcomes"]["inversion"]["failure_code_counts"] == {
        "optimizer_failure": 1
    }
    assert metrics["vs"]["row_count"] == 3
    assert metrics["vs"]["mae_km_s"] == pytest.approx(0.1)
    assert metrics["vs"]["per_layer"]["recovery_fraction"] == pytest.approx([0.75] * 20)
    assert metrics["surrogate_frequency"]["mode_0"]["mae_km_s"] == pytest.approx(0.1)


def test_compute_inversion_metrics_adds_physical_and_interval_metrics() -> None:
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    batch = _result_batch(truth, offset=0.05, deep=True)

    metrics = compute_inversion_metrics(batch, truth)

    assert metrics["physical_frequency"]["overall"]["mae_km_s"] == pytest.approx(0.1)
    assert metrics["uncertainty"]["coverage_fraction"] == pytest.approx(1.0)
    assert metrics["uncertainty"]["mean_interval_width_km_s"] == pytest.approx(0.2)


def test_physical_metrics_mask_invalid_infinities_without_contamination() -> None:
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    batch = _result_batch(truth, offset=0.05, deep=True)
    assert batch.physical_phase_velocity is not None
    assert batch.physical_valid_mask is not None
    physical = batch.physical_phase_velocity.copy()
    physical_mask = batch.physical_valid_mask.copy()
    physical[0, 0, 0] = np.nan
    physical_mask[0, 0, 0] = False
    masked = replace(
        batch,
        physical_phase_velocity=physical,
        physical_valid_mask=physical_mask,
    )

    metrics = compute_inversion_metrics(masked, truth)

    assert metrics["physical_frequency"]["overall"][
        "missing_fraction"
    ] == pytest.approx(1.0 / (4 * 4 * 120))
    assert metrics["physical_frequency"]["overall"]["mae_km_s"] == pytest.approx(0.1)


def test_physical_failure_retains_inversion_metrics_and_counts_missing_cells() -> None:
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    base = _result_batch(truth, offset=0.05, deep=True)
    assert base.physical_phase_velocity is not None
    assert base.physical_valid_mask is not None
    physical = base.physical_phase_velocity.copy()
    physical_mask = base.physical_valid_mask.copy()
    physical[0] = np.nan
    physical_mask[0] = False
    physical[1, 0, 0] = np.nan
    physical_mask[1, 0, 0] = False
    batch = replace(
        base,
        physical_phase_velocity=physical,
        physical_valid_mask=physical_mask,
        physical_success=np.array([False, True, True, True], dtype=np.bool_),
        physical_status=np.array([-1, 0, 0, 0], dtype=np.int32),
        physical_failure_code=np.array(
            [b"physical_solver_failure", b"", b"", b""], dtype="S64"
        ),
    )

    metrics = compute_inversion_metrics(batch, truth)

    assert metrics["successful_count"] == 4
    assert metrics["vs"]["row_count"] == 4
    assert metrics["sample_outcomes"]["physical"]["successful_count"] == 3
    assert metrics["sample_outcomes"]["physical"]["failure_code_counts"] == {
        "physical_solver_failure": 1
    }
    expected_missing = (4 * 120 + 1) / (4 * 4 * 120)
    assert metrics["physical_frequency"]["overall"][
        "missing_fraction"
    ] == pytest.approx(expected_missing)


def test_deep_metrics_report_start_convergence_rejection_and_effort() -> None:
    truth = np.linspace(0.5, 1.5, 80, dtype=np.float32).reshape(4, 20)
    base = _result_batch(truth, offset=0.05, deep=True)
    count, starts = 4, 2
    batch = replace(
        base,
        ensemble_success=np.array(
            [[True, False], [True, True], [True, True], [True, True]],
            dtype=np.bool_,
        ),
        ensemble_inlier_mask=np.array(
            [[True, False], [True, False], [True, True], [True, True]],
            dtype=np.bool_,
        ),
        ensemble_iterations=np.arange(count * starts, dtype=np.int32).reshape(
            count, starts
        ),
        ensemble_evaluations=(
            np.arange(count * starts, dtype=np.int32).reshape(count, starts) + 1
        ),
        ensemble_initial_objective=np.full((count, starts), 2.0, dtype=np.float64),
        ensemble_failure_code=np.array(
            [[b"", b"optimizer_failure"], [b"", b""], [b"", b""], [b"", b""]],
            dtype="S64",
        ),
        ensemble_message=np.full((count, starts), b"diagnostic", dtype="S256"),
        physical_success=np.ones(count, dtype=np.bool_),
        physical_status=np.zeros(count, dtype=np.int32),
        physical_failure_code=np.full(count, b"", dtype="S64"),
    )

    metrics = compute_inversion_metrics(batch, truth)
    starts_metrics = metrics["start_diagnostics"]

    assert "convergence" not in metrics
    assert starts_metrics["start_count"] == 8
    assert starts_metrics["convergence"]["successful_count"] == 7
    assert starts_metrics["convergence"]["success_fraction"] == pytest.approx(0.875)
    assert starts_metrics["iqr_rejection"]["rejected_successful_count"] == 1
    assert starts_metrics["failure_code_counts"] == {"optimizer_failure": 1}
    assert starts_metrics["effort"]["iterations"]["maximum"] == 7.0


def test_paired_deep_delta_reports_start_convergence_and_effort_changes() -> None:
    truth = np.ones((4, 20), dtype=np.float64)
    batch = _result_batch(truth, offset=0.1, deep=True)
    count, starts = 4, 2
    common = {
        "ensemble_evaluations": np.full((count, starts), 5, dtype=np.int32),
        "ensemble_initial_objective": np.full((count, starts), 2.0, dtype=np.float64),
        "ensemble_failure_code": np.full((count, starts), b"", dtype="S64"),
        "ensemble_message": np.full((count, starts), b"diagnostic", dtype="S256"),
        "physical_success": np.ones(count, dtype=np.bool_),
        "physical_status": np.zeros(count, dtype=np.int32),
        "physical_failure_code": np.full(count, b"", dtype="S64"),
    }
    clean_batch = replace(
        batch,
        ensemble_iterations=np.full((count, starts), 4, dtype=np.int32),
        **common,
    )
    noisy_success = np.ones((count, starts), dtype=np.bool_)
    noisy_success[0, 0] = False
    noisy_codes = np.full((count, starts), b"", dtype="S64")
    noisy_codes[0, 0] = b"optimizer_failure"
    noisy_batch = replace(
        batch,
        ensemble_success=noisy_success,
        ensemble_inlier_mask=noisy_success.copy(),
        ensemble_iterations=np.full((count, starts), 6, dtype=np.int32),
        **{**common, "ensemble_failure_code": noisy_codes},
    )
    clean = report_module._rows_from_batch(clean_batch, truth, "clean")
    noisy = report_module._rows_from_batch(noisy_batch, truth, "noise_1pct")

    delta = report_module._group_delta(clean, noisy)["start_diagnostics"]

    assert delta["paired_start_count"] == 8
    assert delta["convergence_fraction"] == pytest.approx(-0.125)
    assert delta["iterations_mean"] == pytest.approx(2.0)


def test_representative_plot_uses_validated_physical_frequency_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = np.linspace(0.5, 1.5, 20, dtype=np.float32).reshape(1, 20)
    rows = report_module._rows_from_batch(
        _result_batch(
            truth,
            offset=0.05,
            deep=True,
            sample_ids=np.array([90], dtype=np.uint64),
            model_kinds=np.array([0], dtype=np.uint8),
        ),
        truth,
        "clean",
    )
    frequencies = np.linspace(10.0, 129.0, 120)
    captured: dict[str, object] = {}

    def capture(figure, output_dir: Path, name: str) -> None:
        captured["xdata"] = figure.axes[1].lines[0].get_xdata().copy()
        captured["xlabel"] = figure.axes[1].get_xlabel()
        report_module.plt.close(figure)

    monkeypatch.setattr(report_module, "_save_figure", capture)

    report_module._plot_representative(rows, tmp_path, "normal", "clean", frequencies)

    np.testing.assert_array_equal(captured["xdata"], frequencies)
    assert captured["xlabel"] == "Frequency (Hz)"
    with pytest.raises(ValueError, match="frequencies.*120"):
        report_module._plot_representative(
            rows, tmp_path, "normal", "clean", frequencies[:-1]
        )


def test_clean_noisy_deltas_use_paired_successes_for_every_metric() -> None:
    truth = np.ones((4, 20), dtype=np.float64)
    base = report_module._rows_from_batch(
        _result_batch(truth, offset=0.1, deep=True), truth, "clean"
    )
    clean_errors = np.array([1.1, 0.1, 0.4, 0.4], dtype=np.float32)
    noisy_errors = np.array([0.4, 0.2, 0.2, 0.4], dtype=np.float32)
    clean_success = np.array([True, True, False, False], dtype=np.bool_)
    noisy_success = np.array([False, True, True, False], dtype=np.bool_)

    def scenario_rows(
        errors: np.ndarray,
        success: np.ndarray,
        noise: str,
        *,
        noisy_physical_gap: bool,
    ) -> report_module._MetricRows:
        recovered = np.asarray(truth + errors[:, None], dtype=np.float32)
        curves = np.asarray(
            np.ones((4, 4, 120), dtype=np.float32) + errors[:, None, None],
            dtype=np.float32,
        )
        physical = curves.copy()
        physical_mask = np.ones_like(physical, dtype=np.bool_)
        if noisy_physical_gap:
            physical[1, 0, 0] = np.inf
            physical_mask[1, 0, 0] = False
        p10 = np.asarray(recovered - 0.2, dtype=np.float32)
        p90 = np.asarray(recovered + 0.2, dtype=np.float32)
        if noise == "clean":
            p10[1] = np.float32(0.9)
            p90[1] = np.float32(1.1)
        else:
            p10[1] = np.float32(1.1)
            p90[1] = np.float32(1.5)
        failure_codes = np.where(success, b"", b"optimizer_failure").astype("S64")
        return replace(
            base,
            success=success,
            inverted_vs=recovered,
            surrogate_phase_velocity=curves,
            median_vs=recovered,
            p10_vs=p10,
            p90_vs=p90,
            physical_phase_velocity=physical,
            physical_valid_mask=physical_mask,
            failure_code=failure_codes,
            noise=np.full(4, noise, dtype="U10"),
        )

    clean = scenario_rows(
        clean_errors, clean_success, "clean", noisy_physical_gap=False
    )
    noisy = scenario_rows(
        noisy_errors, noisy_success, "noise_1pct", noisy_physical_gap=True
    )

    marginal_delta = (
        report_module._compute_rows_metrics(noisy)["vs"]["mae_km_s"]
        - report_module._compute_rows_metrics(clean)["vs"]["mae_km_s"]
    )
    delta = report_module._group_delta(clean, noisy)

    assert marginal_delta == pytest.approx(-0.4)
    assert delta["paired_sample_count"] == 4
    assert delta["paired_successful_count"] == 1
    assert delta["usable_counts"]["vs_rows"] == 1
    assert delta["usable_counts"]["surrogate_frequency_values"]["overall"] == 480
    assert delta["usable_counts"]["surrogate_frequency_rows"]["overall"] == 1
    assert delta["usable_counts"]["physical_frequency_values"]["overall"] == 479
    assert delta["usable_counts"]["physical_frequency_rows"]["overall"] == 1
    assert delta["usable_counts"]["interval_rows"] == 1
    assert delta["vs_mae_km_s"] == pytest.approx(0.1)
    assert delta["vs"]["rmse_km_s"] == pytest.approx(0.1)
    assert delta["surrogate_frequency_mae_km_s"] == pytest.approx(0.1)
    assert delta["surrogate_frequency"]["mode_0"]["mae_km_s"] == pytest.approx(0.1)
    assert delta["physical_frequency_mae_km_s"] == pytest.approx(0.1)
    assert delta["physical_frequency"]["mode_0"]["mae_km_s"] == pytest.approx(0.1)
    assert delta["interval_coverage_fraction"] == pytest.approx(-1.0)
    assert delta["interval_width_km_s"] == pytest.approx(0.2)
    assert delta["uncertainty"]["per_layer"]["coverage_fraction"] == pytest.approx(
        [-1.0] * 20
    )
    assert "success_fraction" not in delta


def test_clean_noisy_deltas_are_explicit_when_no_rows_succeed_in_both() -> None:
    truth = np.ones((4, 20), dtype=np.float64)
    base = report_module._rows_from_batch(
        _result_batch(truth, offset=0.1, deep=False), truth, "clean"
    )
    clean = replace(base, success=np.array([True, False, False, False], dtype=np.bool_))
    noisy = replace(
        base,
        success=np.array([False, True, False, False], dtype=np.bool_),
        noise=np.full(4, "noise_1pct", dtype="U10"),
    )

    delta = report_module._group_delta(clean, noisy)

    assert delta["paired_sample_count"] == 4
    assert delta["paired_successful_count"] == 0
    assert delta["usable_counts"]["vs_rows"] == 0
    assert delta["usable_counts"]["surrogate_frequency_values"]["overall"] == 0
    assert delta["usable_counts"]["surrogate_frequency_rows"]["overall"] == 0
    assert delta["vs_mae_km_s"] is None
    assert delta["surrogate_frequency_mae_km_s"] is None
    assert delta["vs"]["per_layer"]["mae_km_s"] == [None] * 20


def test_load_true_vs_reads_only_unique_requested_rows_in_requested_order(
    tmp_path: Path,
) -> None:
    directory, truth = _source_dataset(tmp_path)
    extra = directory / "shard-00001.h5"
    with h5py.File(extra, "w") as handle:
        handle.create_dataset("sample_id", data=np.array([94], dtype=np.uint64))
        handle.create_dataset("vs", data=np.full((1, 20), np.inf, dtype=np.float32))

    loaded = report_module._load_true_vs(directory, np.array([93, 90], dtype=np.uint64))

    assert loaded == pytest.approx(truth[[3, 0]])


def test_load_true_vs_rejects_duplicate_requests_and_missing_rows(
    tmp_path: Path,
) -> None:
    directory, _ = _source_dataset(tmp_path)

    with pytest.raises(ValueError, match="requested.*duplicate"):
        report_module._load_true_vs(directory, np.array([90, 90], dtype=np.uint64))
    with pytest.raises(ValueError, match="missing.*999"):
        report_module._load_true_vs(directory, np.array([999], dtype=np.uint64))


def test_load_true_vs_rejects_duplicate_source_rows(tmp_path: Path) -> None:
    directory, _ = _source_dataset(tmp_path)
    with h5py.File(directory / "shard-00001.h5", "w") as handle:
        handle.create_dataset("sample_id", data=np.array([90], dtype=np.uint64))
        handle.create_dataset("vs", data=np.ones((1, 20), dtype=np.float32))

    with pytest.raises(ValueError, match="duplicate requested sample_id 90"):
        report_module._load_true_vs(directory, np.array([90], dtype=np.uint64))


def test_load_true_vs_rejects_invalid_requested_truth_but_not_unrequested_truth(
    tmp_path: Path,
) -> None:
    directory, _ = _source_dataset(tmp_path)
    with h5py.File(directory / "shard-00000.h5", "r+") as handle:
        handle["vs"][1, 0] = np.float32(9.0)

    assert report_module._load_true_vs(directory, [90]).shape == (1, 20)
    with pytest.raises(ValueError, match="outside bounds"):
        report_module._load_true_vs(directory, [91])


def test_report_validates_complete_results_before_loading_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text("{}", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        report_module,
        "validate_complete_results",
        lambda path: (
            events.append("validated")
            or (_ for _ in ()).throw(ValueError("result manifest is incomplete"))
        ),
    )
    monkeypatch.setattr(
        report_module,
        "_load_true_vs",
        lambda path, ids: events.append("truth"),
    )

    with pytest.raises(ValueError, match="incomplete"):
        build_inversion_report(
            results,
            tmp_path / "dataset",
            tmp_path / "report",
            dataset_config=REPORT_DATASET_CONFIG,
        )

    assert events == ["validated"]


def test_report_checksum_validates_dataset_identity_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    events: list[str] = []
    manifest = type(
        "ManifestStub",
        (),
        {
            "dataset_config_hash": REPORT_DATASET_HASH,
            "dataset_manifest_sha256": "d" * 64,
        },
    )()
    monkeypatch.setattr(
        report_module,
        "validate_complete_results",
        lambda path: events.append("results") or manifest,
    )
    monkeypatch.setattr(
        report_module,
        "validate_dataset_files",
        lambda path: (
            events.append("dataset")
            or (_ for _ in ()).throw(
                ValueError("dataset shard checksum does not match")
            )
        ),
    )
    monkeypatch.setattr(
        report_module,
        "_load_true_vs",
        lambda *args: events.append("truth"),
    )

    with pytest.raises(ValueError, match="checksum"):
        build_inversion_report(
            results,
            dataset,
            tmp_path / "report",
            dataset_config=REPORT_DATASET_CONFIG,
        )

    assert events == ["results", "dataset"]


def test_report_rejects_corrupt_result_before_truth_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, truth = _source_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = _complete_results(tmp_path, checkpoint, truth)
    with h5py.File(results / "full-clean-shard-00000.h5", "r+") as handle:
        handle["sample_id"][0] = np.uint64(90_000)
    events: list[str] = []
    monkeypatch.setattr(
        report_module,
        "_load_true_vs",
        lambda path, ids: events.append("truth"),
    )

    with pytest.raises(ValueError, match="checksum|content|sample_id"):
        build_inversion_report(
            results,
            dataset,
            tmp_path / "report",
            dataset_config=REPORT_DATASET_CONFIG,
        )

    assert events == []


def test_report_rejects_clean_noisy_sample_mismatch_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, truth = _source_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = tmp_path / "mismatched-results"
    jobs = (
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    )
    manifest = initialize_result_manifest(
        results,
        dataset_config_hash=REPORT_DATASET_HASH,
        dataset_manifest_sha256=dataset_manifest_sha256(
            load_manifest(dataset / "manifest.json")
        ),
        checkpoint=checkpoint,
        config=InversionConfig(),
        experiment="full",
        expected_jobs=jobs,
        expected_sample_ids_by_job={jobs[0]: IDS, jobs[1]: IDS[:3]},
    )
    clean = _result_batch(truth, offset=0.05, deep=False)
    noisy = _result_batch(
        truth[:3],
        offset=0.15,
        deep=False,
        sample_ids=IDS[:3],
        model_kinds=KINDS[:3],
    )
    for job, batch in zip(jobs, (clean, noisy), strict=True):
        path = write_result_shard(results, job, batch, manifest)
        mark_job_complete(results, job, path)
    events: list[str] = []
    original = report_module._load_true_vs
    monkeypatch.setattr(
        report_module,
        "_load_true_vs",
        lambda path, ids: events.append("truth") or original(path, ids),
    )

    with pytest.raises(ValueError, match="clean and noisy.*align"):
        build_inversion_report(
            results,
            dataset,
            tmp_path / "report",
            dataset_config=REPORT_DATASET_CONFIG,
        )

    assert events == []


def test_report_rejects_a_plot_group_without_successful_rows(tmp_path: Path) -> None:
    dataset, truth = _source_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = tmp_path / "failed-kind-results"
    jobs = (
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    )
    manifest = initialize_result_manifest(
        results,
        dataset_config_hash=REPORT_DATASET_HASH,
        dataset_manifest_sha256=dataset_manifest_sha256(
            load_manifest(dataset / "manifest.json")
        ),
        checkpoint=checkpoint,
        config=InversionConfig(),
        experiment="full",
        expected_jobs=jobs,
        expected_sample_ids_by_job={job: IDS for job in jobs},
    )
    for job in jobs:
        batch = _result_batch(truth, offset=0.05, deep=False)
        success = batch.success.copy()
        success[3] = False
        failure_code = batch.failure_code.copy()
        failure_code[3] = b"optimizer_failure"
        failed_kind = replace(batch, success=success, failure_code=failure_code)
        path = write_result_shard(results, job, failed_kind, manifest)
        mark_job_complete(results, job, path)

    with pytest.raises(ValueError, match="empty successful group.*coupled"):
        build_inversion_report(
            results,
            dataset,
            tmp_path / "report",
            dataset_config=REPORT_DATASET_CONFIG,
        )


def test_report_requires_the_matching_dataset_configuration(
    tmp_path: Path,
) -> None:
    dataset, truth = _source_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = _complete_results(tmp_path, checkpoint, truth)

    with pytest.raises(TypeError, match="dataset_config"):
        build_inversion_report(results, dataset, tmp_path / "missing-config")
    with pytest.raises(ValueError, match="configuration.*does not match"):
        build_inversion_report(
            results,
            dataset,
            tmp_path / "wrong-config",
            dataset_config=DatasetConfig(),
        )


def test_report_separates_scopes_and_writes_deterministic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, truth = _source_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = _complete_results(tmp_path, checkpoint, truth)
    first_output = tmp_path / "report-a"
    second_output = tmp_path / "report-b"

    plotted_frequencies: list[np.ndarray] = []
    plot_representative = report_module._plot_representative

    def capture_frequencies(*args, **kwargs):
        plotted_frequencies.append(np.asarray(args[4]).copy())
        return plot_representative(*args, **kwargs)

    monkeypatch.setattr(report_module, "_plot_representative", capture_frequencies)

    first = build_inversion_report(
        results,
        dataset,
        first_output,
        dataset_config=REPORT_DATASET_CONFIG,
    )
    second = build_inversion_report(
        results,
        dataset,
        second_output,
        dataset_config=REPORT_DATASET_CONFIG,
    )

    assert first == second
    assert set(first["experiment_scopes"]) == {"full", "deep"}
    assert first["experiment_scopes"]["full"]["scope_label"] == (
        "full single-start population"
    )
    assert first["experiment_scopes"]["deep"]["scope_label"] == (
        "deep multi-start uncertainty"
    )
    assert first["experiment_scopes"]["full"]["groups"]["overall"]["sample_count"] == 8
    assert first["experiment_scopes"]["deep"]["groups"]["overall"]["sample_count"] == 8
    assert first["experiment_scopes"]["full"]["clean_to_noise_1pct_delta"]["overall"][
        "vs_mae_km_s"
    ] == pytest.approx(0.1)
    assert first["representatives"]["normal"]["clean"]["sample_id"] == 90
    assert len(plotted_frequencies) == 16
    for frequencies in plotted_frequencies:
        np.testing.assert_array_equal(
            frequencies, REPORT_DATASET_CONFIG.physics.frequencies
        )

    expected_pngs = {
        "vs-error-by-depth.png",
        "vs-error-by-kind-and-noise.png",
        "optimization-diagnostics.png",
        *{
            f"representative-{kind}-{noise}.png"
            for kind in ("normal", "low_velocity", "high_velocity", "coupled")
            for noise in ("clean", "noise_1pct")
        },
    }
    first_pngs = {path.name for path in first_output.glob("*.png")}
    assert first_pngs == expected_pngs
    assert all((first_output / name).stat().st_size > 0 for name in expected_pngs)
    assert (first_output / "summary.json").stat().st_size > 0
    assert not list(first_output.glob("summary.json.tmp-*"))
    for name in expected_pngs:
        first_hash = hashlib.sha256((first_output / name).read_bytes()).hexdigest()
        second_hash = hashlib.sha256((second_output / name).read_bytes()).hexdigest()
        assert first_hash == second_hash


@pytest.mark.parametrize(
    ("truth", "recovered", "message"),
    [
        (np.ones((0, 20)), np.ones((0, 20)), "nonempty"),
        (np.ones((1, 19)), np.ones((1, 19)), "20 layers"),
        (np.ones((1, 20)), np.full((1, 20), np.nan), "finite"),
    ],
)
def test_compute_vs_metrics_rejects_invalid_groups(
    truth: np.ndarray, recovered: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_vs_metrics(truth, recovered)
