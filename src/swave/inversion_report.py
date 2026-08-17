"""Truth-isolated metrics and figures for completed inversion experiments."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from numpy.typing import NDArray

from .config import DatasetConfig, canonical_hash
from .dataset import dataset_manifest_sha256, validate_dataset_files
from .geology import ModelKind
from .inversion_results import (
    ResultBatch,
    ResultManifest,
    sample_id_sha256,
    software_sha256,
    validate_complete_results,
    validate_result_shard,
)

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

_JOB_PATTERN = re.compile(
    r"(?:(?P<full>full)-(?P<full_noise>clean|noise_1pct)-shard-[0-9]+|"
    r"(?P<deep>deep)-(?P<deep_noise>clean|noise_1pct)-samples-"
    r"[0-9]{20}-[0-9]{20}-[0-9a-f]{12})"
)
_KIND_NAMES = {int(kind): kind.name.lower() for kind in ModelKind}
_SCOPE_LABELS = {
    "full": "full single-start population",
    "deep": "deep multi-start uncertainty",
}
_VS_MIN_KM_S = 0.3
_VS_MAX_KM_S = 2.6


@dataclass(frozen=True)
class _MetricRows:
    sample_id: NDArray[np.uint64]
    model_kind: NDArray[np.uint8]
    success: NDArray[np.bool_]
    status: NDArray[np.int32]
    iterations: NDArray[np.int32]
    evaluations: NDArray[np.int32]
    initial_objective: NDArray[np.float64]
    final_objective: NDArray[np.float64]
    inverted_vs: NDArray[np.float32]
    observed_phase_velocity: NDArray[np.float32]
    surrogate_phase_velocity: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    failure_code: NDArray[np.bytes_]
    reference_vs: NDArray[np.float32]
    ensemble_success: NDArray[np.bool_] | None
    ensemble_status: NDArray[np.int32] | None
    ensemble_iterations: NDArray[np.int32] | None
    ensemble_evaluations: NDArray[np.int32] | None
    ensemble_initial_objective: NDArray[np.float64] | None
    ensemble_objective: NDArray[np.float64] | None
    ensemble_failure_code: NDArray[np.bytes_] | None
    ensemble_message: NDArray[np.bytes_] | None
    ensemble_inlier_mask: NDArray[np.bool_] | None
    median_vs: NDArray[np.float32] | None
    p10_vs: NDArray[np.float32] | None
    p90_vs: NDArray[np.float32] | None
    physical_success: NDArray[np.bool_] | None
    physical_status: NDArray[np.int32] | None
    physical_failure_code: NDArray[np.bytes_] | None
    physical_phase_velocity: NDArray[np.float32] | None
    physical_valid_mask: NDArray[np.bool_] | None
    truth_vs: NDArray[np.float64]
    noise: NDArray[np.str_]


def _rows_from_batch(
    batch: ResultBatch,
    truth: NDArray[np.floating[Any]],
    noise: str = "",
) -> _MetricRows:
    true_values = np.asarray(truth, dtype=np.float64)
    if true_values.shape != (batch.sample_id.size, 20):
        raise ValueError("truth Vs rows must align exactly with result sample IDs")
    return _MetricRows(
        sample_id=batch.sample_id,
        model_kind=batch.model_kind,
        success=batch.success,
        status=batch.status,
        iterations=batch.iterations,
        evaluations=batch.evaluations,
        initial_objective=batch.initial_objective,
        final_objective=batch.final_objective,
        inverted_vs=batch.inverted_vs,
        observed_phase_velocity=batch.observed_phase_velocity,
        surrogate_phase_velocity=batch.surrogate_phase_velocity,
        valid_mask=batch.valid_mask,
        failure_code=batch.failure_code,
        reference_vs=batch.reference_vs,
        ensemble_success=batch.ensemble_success,
        ensemble_status=batch.ensemble_status,
        ensemble_iterations=batch.ensemble_iterations,
        ensemble_evaluations=batch.ensemble_evaluations,
        ensemble_initial_objective=batch.ensemble_initial_objective,
        ensemble_objective=batch.ensemble_objective,
        ensemble_failure_code=batch.ensemble_failure_code,
        ensemble_message=batch.ensemble_message,
        ensemble_inlier_mask=batch.ensemble_inlier_mask,
        median_vs=batch.median_vs,
        p10_vs=batch.p10_vs,
        p90_vs=batch.p90_vs,
        physical_success=batch.physical_success,
        physical_status=batch.physical_status,
        physical_failure_code=batch.physical_failure_code,
        physical_phase_velocity=batch.physical_phase_velocity,
        physical_valid_mask=batch.physical_valid_mask,
        truth_vs=true_values,
        noise=np.full(batch.sample_id.size, noise, dtype="U10"),
    )


def _select_rows(rows: _MetricRows, mask: NDArray[np.bool_]) -> _MetricRows:
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != rows.sample_id.shape:
        raise ValueError("row selection mask has an invalid shape")
    values: dict[str, Any] = {}
    for field in fields(rows):
        value = getattr(rows, field.name)
        values[field.name] = None if value is None else value[selected]
    return _MetricRows(**values)


def _take_rows(rows: _MetricRows, indexes: NDArray[np.intp]) -> _MetricRows:
    order = np.asarray(indexes, dtype=np.intp)
    values: dict[str, Any] = {}
    for field in fields(rows):
        value = getattr(rows, field.name)
        values[field.name] = None if value is None else value[order]
    return _MetricRows(**values)


def _concatenate_rows(groups: list[_MetricRows]) -> _MetricRows:
    if not groups:
        raise ValueError("cannot concatenate an empty result group")
    values: dict[str, Any] = {}
    for field in fields(groups[0]):
        items = [getattr(group, field.name) for group in groups]
        if all(item is None for item in items):
            values[field.name] = None
        elif any(item is None for item in items):
            raise ValueError("result groups mix full and deep schemas")
        else:
            values[field.name] = np.concatenate(items, axis=0)
    return _MetricRows(**values)


def _profile_pair(
    truth: NDArray[np.floating[Any]],
    recovered: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true_values = np.asarray(truth, dtype=np.float64)
    recovered_values = np.asarray(recovered, dtype=np.float64)
    if true_values.shape != recovered_values.shape or true_values.ndim != 2:
        raise ValueError(
            "truth and recovered Vs must have the same two-dimensional shape"
        )
    if true_values.shape[0] == 0:
        raise ValueError("Vs metric groups must be nonempty")
    if true_values.shape[1] != 20:
        raise ValueError("Vs profiles must contain exactly 20 layers")
    if not np.all(np.isfinite(true_values)) or not np.all(
        np.isfinite(recovered_values)
    ):
        raise ValueError("Vs metric inputs must be finite")
    return true_values, recovered_values


def compute_vs_metrics(
    truth: NDArray[np.floating[Any]],
    recovered: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Compute exact scalar and per-layer Vs metrics in km/s."""
    true_values, recovered_values = _profile_pair(truth, recovered)
    error = recovered_values - true_values
    absolute_error = np.abs(error)
    nonzero = true_values != 0.0
    relative_count = int(np.count_nonzero(nonzero))
    mean_relative = (
        float(np.mean(absolute_error[nonzero] / np.abs(true_values[nonzero])) * 100.0)
        if relative_count
        else None
    )
    return {
        "row_count": int(true_values.shape[0]),
        "value_count": int(true_values.size),
        "mae_km_s": float(np.mean(absolute_error)),
        "rmse_km_s": float(np.sqrt(np.mean(np.square(error)))),
        "mean_relative_percent": mean_relative,
        "p95_absolute_error_km_s": float(np.percentile(absolute_error, 95)),
        "zero_truth_count": int(true_values.size - relative_count),
        "relative_denominator_count": relative_count,
        "per_layer": {
            "mae_km_s": np.mean(absolute_error, axis=0).tolist(),
            "bias_km_s": np.mean(error, axis=0).tolist(),
        },
    }


def _curve_arrays(
    observed: NDArray[np.floating[Any]],
    predicted: NDArray[np.floating[Any]],
    observed_mask: NDArray[np.bool_],
    predicted_mask: NDArray[np.bool_] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    observed_values = np.asarray(observed, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=np.bool_)
    if (
        observed_values.shape != predicted_values.shape
        or observed_values.shape != mask.shape
        or observed_values.ndim != 3
    ):
        raise ValueError("frequency arrays and masks must share shape (N, modes, F)")
    if observed_values.shape[0] == 0 or observed_values.shape[1] == 0:
        raise ValueError("frequency metric groups must be nonempty")
    prediction_mask = (
        np.ones_like(mask)
        if predicted_mask is None
        else np.asarray(predicted_mask, dtype=np.bool_)
    )
    if prediction_mask.shape != mask.shape:
        raise ValueError("predicted_mask must match the curve shape")
    if not np.all(np.isfinite(observed_values[mask])) or not np.all(
        np.isfinite(predicted_values[mask & prediction_mask])
    ):
        raise ValueError("non-finite frequency values occur under an active mask")
    return observed_values, predicted_values, mask, prediction_mask


def _frequency_group(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
    compared_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    observed_count = int(np.count_nonzero(observed_mask))
    compared_count = int(np.count_nonzero(compared_mask))
    missing_fraction = (
        float(1.0 - compared_count / observed_count) if observed_count else None
    )
    if compared_count == 0:
        return {
            "observed_count": observed_count,
            "compared_count": 0,
            "missing_fraction": missing_fraction,
            "mae_km_s": None,
            "rmse_km_s": None,
            "p95_absolute_error_km_s": None,
        }
    error = predicted[compared_mask] - observed[compared_mask]
    absolute = np.abs(error)
    return {
        "observed_count": observed_count,
        "compared_count": compared_count,
        "missing_fraction": missing_fraction,
        "mae_km_s": float(np.mean(absolute)),
        "rmse_km_s": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_error_km_s": float(np.percentile(absolute, 95)),
    }


def compute_frequency_metrics(
    observed: NDArray[np.floating[Any]],
    predicted: NDArray[np.floating[Any]],
    observed_mask: NDArray[np.bool_],
    *,
    predicted_mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Compute mask-safe overall and per-mode phase-velocity errors in km/s."""
    observed_values, predicted_values, mask, prediction_mask = _curve_arrays(
        observed, predicted, observed_mask, predicted_mask
    )
    compared = mask & prediction_mask
    metrics = {
        "overall": _frequency_group(observed_values, predicted_values, mask, compared)
    }
    for mode in range(observed_values.shape[1]):
        metrics[f"mode_{mode}"] = _frequency_group(
            observed_values[:, mode],
            predicted_values[:, mode],
            mask[:, mode],
            compared[:, mode],
        )
    return metrics


def compute_interval_metrics(
    truth: NDArray[np.floating[Any]],
    p10: NDArray[np.floating[Any]],
    p90: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Compute P10--P90 truth coverage and interval width in km/s."""
    true_values, lower = _profile_pair(truth, p10)
    _, upper = _profile_pair(truth, p90)
    if np.any(lower > upper):
        raise ValueError("P10 profiles must not exceed P90 profiles")
    covered = (lower <= true_values) & (true_values <= upper)
    width = upper - lower
    return {
        "row_count": int(true_values.shape[0]),
        "coverage_fraction": float(np.mean(covered)),
        "mean_interval_width_km_s": float(np.mean(width)),
        "per_layer": {
            "coverage_fraction": np.mean(covered, axis=0).tolist(),
            "mean_interval_width_km_s": np.mean(width, axis=0).tolist(),
        },
    }


def _distribution(values: NDArray[Any]) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }


def _decode_code(value: np.bytes_) -> str:
    try:
        return bytes(value).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("failure_code contains non-ASCII bytes") from error


def _decode_message(value: np.bytes_) -> str:
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ensemble_message contains non-UTF-8 bytes") from error


def _start_diagnostics(rows: _MetricRows) -> dict[str, Any] | None:
    if rows.ensemble_success is None:
        return None
    assert rows.ensemble_status is not None
    assert rows.ensemble_iterations is not None
    assert rows.ensemble_evaluations is not None
    assert rows.ensemble_initial_objective is not None
    assert rows.ensemble_objective is not None
    assert rows.ensemble_failure_code is not None
    assert rows.ensemble_message is not None
    assert rows.ensemble_inlier_mask is not None
    success = rows.ensemble_success
    inlier = rows.ensemble_inlier_mask
    total = int(success.size)
    successful = int(np.count_nonzero(success))
    rejected = success & ~inlier
    failed_codes = [
        _decode_code(value) for value in rows.ensemble_failure_code[~success]
    ]
    failed_messages = [
        _decode_message(value) for value in rows.ensemble_message[~success]
    ]
    return {
        "start_count": total,
        "convergence": {
            "successful_count": successful,
            "success_fraction": float(successful / total),
        },
        "failure_code_counts": dict(sorted(Counter(failed_codes).items())),
        "failure_message_counts": dict(sorted(Counter(failed_messages).items())),
        "status_counts": {
            str(int(status)): int(count)
            for status, count in sorted(
                Counter(int(value) for value in rows.ensemble_status.ravel()).items()
            )
        },
        "iqr_rejection": {
            "inlier_count": int(np.count_nonzero(inlier)),
            "rejected_successful_count": int(np.count_nonzero(rejected)),
            "rejected_fraction_of_successful": (
                float(np.count_nonzero(rejected) / successful) if successful else None
            ),
            "rejected_starts_per_sample": _distribution(
                np.count_nonzero(rejected, axis=1)
            ),
            "policy": "inclusive 1.5-IQR fences, including zero-IQR samples",
        },
        "effort": {
            "iterations": _distribution(rows.ensemble_iterations),
            "evaluations": _distribution(rows.ensemble_evaluations),
        },
        "objective": {
            "initial": _distribution(rows.ensemble_initial_objective),
            "final": _distribution(rows.ensemble_objective),
        },
    }


def _empty_accuracy(total_rows: int) -> dict[str, Any]:
    return {
        "row_count": 0,
        "value_count": 0,
        "mae_km_s": None,
        "rmse_km_s": None,
        "mean_relative_percent": None,
        "p95_absolute_error_km_s": None,
        "zero_truth_count": 0,
        "relative_denominator_count": 0,
        "per_layer": {
            "mae_km_s": [None] * 20,
            "bias_km_s": [None] * 20,
            "recovery_fraction": [0.0] * 20,
        },
        "total_row_count": total_rows,
    }


def _empty_frequency(mode_count: int) -> dict[str, Any]:
    metric = {
        "observed_count": 0,
        "compared_count": 0,
        "missing_fraction": None,
        "mae_km_s": None,
        "rmse_km_s": None,
        "p95_absolute_error_km_s": None,
    }
    return {
        "overall": dict(metric),
        **{f"mode_{mode}": dict(metric) for mode in range(mode_count)},
    }


def _compute_rows_metrics(rows: _MetricRows) -> dict[str, Any]:
    count = int(rows.sample_id.size)
    if count == 0:
        raise ValueError("inversion metric groups must be nonempty")
    successful = rows.success
    success_count = int(np.count_nonzero(successful))
    codes = [_decode_code(code) for code in rows.failure_code[~successful]]
    sample_outcomes: dict[str, Any] = {
        "inversion": {
            "successful_count": success_count,
            "success_fraction": float(success_count / count),
            "failure_code_counts": dict(sorted(Counter(codes).items())),
            "status_counts": {
                str(int(status)): int(total)
                for status, total in sorted(
                    Counter(int(value) for value in rows.status).items()
                )
            },
        }
    }
    if rows.physical_success is not None:
        assert rows.physical_status is not None
        assert rows.physical_failure_code is not None
        attempted = successful
        physical_success_count = int(
            np.count_nonzero(rows.physical_success & attempted)
        )
        physical_failure_codes = [
            _decode_code(code)
            for code in rows.physical_failure_code[attempted & ~rows.physical_success]
        ]
        sample_outcomes["physical"] = {
            "attempted_count": int(np.count_nonzero(attempted)),
            "successful_count": physical_success_count,
            "success_fraction_of_attempted": (
                float(physical_success_count / np.count_nonzero(attempted))
                if np.any(attempted)
                else None
            ),
            "failure_code_counts": dict(
                sorted(Counter(physical_failure_codes).items())
            ),
            "status_counts": {
                str(int(status)): int(total)
                for status, total in sorted(
                    Counter(
                        int(value) for value in rows.physical_status[attempted]
                    ).items()
                )
            },
            "not_attempted_count": int(np.count_nonzero(~attempted)),
        }
    result: dict[str, Any] = {
        "sample_count": count,
        "successful_count": success_count,
        "sample_outcomes": sample_outcomes,
        "optimization": {
            "iterations": _distribution(rows.iterations),
            "evaluations": _distribution(rows.evaluations),
            "initial_objective": _distribution(rows.initial_objective),
            "final_objective": _distribution(rows.final_objective),
        },
    }
    start_metrics = _start_diagnostics(rows)
    if start_metrics is not None:
        result["start_diagnostics"] = start_metrics
    if success_count == 0:
        result["vs"] = _empty_accuracy(count)
        result["surrogate_frequency"] = _empty_frequency(
            rows.observed_phase_velocity.shape[1]
        )
        if rows.physical_phase_velocity is not None:
            result["physical_frequency"] = _empty_frequency(
                rows.observed_phase_velocity.shape[1]
            )
            result["uncertainty"] = None
        return result

    vs_metrics = compute_vs_metrics(
        rows.truth_vs[successful], rows.inverted_vs[successful]
    )
    vs_metrics["total_row_count"] = count
    vs_metrics["per_layer"]["recovery_fraction"] = [float(success_count / count)] * 20
    result["vs"] = vs_metrics
    result["surrogate_frequency"] = compute_frequency_metrics(
        rows.observed_phase_velocity[successful],
        rows.surrogate_phase_velocity[successful],
        rows.valid_mask[successful],
    )
    if rows.physical_phase_velocity is not None:
        assert rows.physical_valid_mask is not None
        assert rows.p10_vs is not None
        assert rows.p90_vs is not None
        result["physical_frequency"] = compute_frequency_metrics(
            rows.observed_phase_velocity[successful],
            rows.physical_phase_velocity[successful],
            rows.valid_mask[successful],
            predicted_mask=rows.physical_valid_mask[successful],
        )
        result["uncertainty"] = compute_interval_metrics(
            rows.truth_vs[successful],
            rows.p10_vs[successful],
            rows.p90_vs[successful],
        )
    return result


def compute_inversion_metrics(
    batch: ResultBatch,
    truth: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Compute metrics for one validated full or deep result batch."""
    if not isinstance(batch, ResultBatch):
        raise TypeError("batch must be a validated ResultBatch")
    return _compute_rows_metrics(_rows_from_batch(batch, truth))


def _load_true_vs(
    dataset_dir: Path | str,
    requested_ids: NDArray[np.uint64] | Sequence[int],
) -> NDArray[np.float64]:
    """Load only requested truth rows and return them in requested-ID order."""
    directory = Path(dataset_dir)
    raw_requested = np.asarray(requested_ids)
    if raw_requested.ndim != 1 or raw_requested.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "requested sample IDs must be a one-dimensional integer vector"
        )
    if np.any(raw_requested < 0):
        raise ValueError("requested sample IDs must be nonnegative")
    requested = np.asarray(raw_requested, dtype=np.uint64)
    if requested.size == 0:
        raise ValueError("requested sample IDs must be nonempty")
    if np.unique(requested).size != requested.size:
        raise ValueError("requested sample IDs contain duplicates")
    wanted = {int(value) for value in requested}
    found: dict[int, NDArray[np.float64]] = {}
    paths = sorted(directory.glob("shard-*.h5"))
    if not paths:
        raise ValueError("dataset directory contains no source HDF5 shards")
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                if "sample_id" not in handle or "vs" not in handle:
                    raise ValueError(
                        f"source shard {path.name} is missing sample_id or vs"
                    )
                sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
                if sample_ids.ndim != 1:
                    raise ValueError(f"source shard {path.name} sample_id is invalid")
                rows = np.flatnonzero(np.isin(sample_ids, requested))
                if rows.size == 0:
                    continue
                profiles = np.asarray(handle["vs"][rows], dtype=np.float64)
        except OSError as error:
            raise ValueError(f"source shard {path.name} is not readable") from error
        if profiles.shape != (rows.size, 20):
            raise ValueError(f"source shard {path.name} requested Vs rows are invalid")
        if not np.all(np.isfinite(profiles)):
            raise ValueError(
                f"source shard {path.name} requested Vs rows are non-finite"
            )
        if np.any(profiles < _VS_MIN_KM_S) or np.any(profiles > _VS_MAX_KM_S):
            raise ValueError(
                f"source shard {path.name} requested Vs rows are outside bounds"
            )
        for sample_id, profile in zip(sample_ids[rows], profiles, strict=True):
            key = int(sample_id)
            if key not in wanted:
                raise AssertionError("truth selection returned an unrequested row")
            if key in found:
                raise ValueError(
                    f"duplicate requested sample_id {key} in source shards"
                )
            found[key] = profile
    missing = sorted(wanted - set(found))
    if missing:
        rendered = ", ".join(str(value) for value in missing[:10])
        raise ValueError(f"missing requested truth sample IDs: {rendered}")
    return np.stack([found[int(value)] for value in requested], axis=0)


def _validate_dataset_identity(
    dataset_dir: Path, result_manifest: ResultManifest
) -> None:
    manifest = validate_dataset_files(dataset_dir)
    if manifest.config_hash != result_manifest.dataset_config_hash:
        raise ValueError("dataset identity does not match inversion results")
    if dataset_manifest_sha256(manifest) != result_manifest.dataset_manifest_sha256:
        raise ValueError("dataset manifest digest does not match inversion results")


def _load_result_groups(
    results_dir: Path,
    manifest: ResultManifest,
) -> dict[str, dict[str, list[ResultBatch]]]:
    grouped: dict[str, dict[str, list[ResultBatch]]] = {}
    for job in manifest.expected_jobs:
        match = _JOB_PATTERN.fullmatch(job)
        if match is None:
            raise ValueError(f"result job {job!r} does not have a reportable identity")
        experiment = match.group("full") or match.group("deep")
        noise = match.group("full_noise") or match.group("deep_noise")
        assert experiment is not None and noise is not None
        batch = validate_result_shard(
            results_dir / f"{job}.h5",
            manifest=manifest,
            expected_sha256=manifest.job_sha256[job],
        )
        grouped.setdefault(experiment, {}).setdefault(noise, []).append(batch)
    return grouped


def _validate_result_alignment(
    grouped: dict[str, dict[str, list[ResultBatch]]],
) -> NDArray[np.uint64]:
    """Validate cross-shard/noise row identity before truth becomes available."""
    all_ids: set[int] = set()
    kind_by_id: dict[int, int] = {}
    for experiment, by_noise in grouped.items():
        expected_ids: NDArray[np.uint64] | None = None
        expected_kinds: NDArray[np.uint8] | None = None
        for noise, batches in by_noise.items():
            if not batches:
                raise ValueError(f"result group {experiment}/{noise} is empty")
            sample_ids = np.concatenate([batch.sample_id for batch in batches])
            model_kinds = np.concatenate([batch.model_kind for batch in batches])
            if sample_ids.size > 1 and np.any(sample_ids[1:] <= sample_ids[:-1]):
                raise ValueError(
                    f"duplicate or unordered result rows in {experiment}/{noise}"
                )
            if expected_ids is None:
                expected_ids = sample_ids
                expected_kinds = model_kinds
            elif not np.array_equal(sample_ids, expected_ids) or not np.array_equal(
                model_kinds, expected_kinds
            ):
                raise ValueError(
                    f"clean and noisy result rows do not align in {experiment} "
                    "by sample ID and kind"
                )
            for sample_id, model_kind in zip(sample_ids, model_kinds, strict=True):
                key = int(sample_id)
                kind = int(model_kind)
                previous = kind_by_id.setdefault(key, kind)
                if previous != kind:
                    raise ValueError(
                        f"sample_id {key} has conflicting model_kind values"
                    )
                all_ids.add(key)
    if not all_ids:
        raise ValueError("complete result set contains no result rows")
    requested_ids = np.asarray(sorted(all_ids), dtype=np.uint64)
    if np.any(requested_ids % 100 < 90):
        raise ValueError("result rows contain IDs outside the inversion split")
    return requested_ids


def _truth_lookup(
    requested_ids: NDArray[np.uint64], truth: NDArray[np.float64]
) -> dict[int, NDArray[np.float64]]:
    return {
        int(sample_id): profile
        for sample_id, profile in zip(requested_ids, truth, strict=True)
    }


def _scope_rows(
    batches_by_noise: dict[str, list[ResultBatch]],
    truth_by_id: dict[int, NDArray[np.float64]],
) -> _MetricRows:
    noise_rows: list[_MetricRows] = []
    expected_ids: NDArray[np.uint64] | None = None
    expected_kinds: NDArray[np.uint8] | None = None
    for noise, batches in sorted(batches_by_noise.items()):
        batch_rows: list[_MetricRows] = []
        for batch in batches:
            truth = np.stack([truth_by_id[int(value)] for value in batch.sample_id])
            batch_rows.append(_rows_from_batch(batch, truth, noise))
        rows = _concatenate_rows(batch_rows)
        order = np.argsort(rows.sample_id, kind="stable")
        rows = _take_rows(rows, order)
        if rows.sample_id.size > 1 and np.any(
            rows.sample_id[1:] <= rows.sample_id[:-1]
        ):
            raise ValueError(
                f"duplicate or unordered result rows in noise group {noise}"
            )
        if expected_ids is None:
            expected_ids = rows.sample_id
            expected_kinds = rows.model_kind
        elif not np.array_equal(rows.sample_id, expected_ids) or not np.array_equal(
            rows.model_kind, expected_kinds
        ):
            raise ValueError(
                "clean and noisy result rows do not align by sample ID and kind"
            )
        noise_rows.append(rows)
    return _concatenate_rows(noise_rows)


def _delta_value(noisy: Any, clean: Any) -> float | None:
    if noisy is None or clean is None:
        return None
    return float(noisy - clean)


def _list_delta(noisy: list[Any], clean: list[Any]) -> list[float | None]:
    if len(noisy) != len(clean):
        raise ValueError("paired metric vectors have different lengths")
    return [
        _delta_value(noisy_value, clean_value)
        for noisy_value, clean_value in zip(noisy, clean, strict=True)
    ]


def _vs_delta(clean: _MetricRows, noisy: _MetricRows) -> tuple[dict[str, Any], float]:
    clean_metrics = compute_vs_metrics(clean.truth_vs, clean.inverted_vs)
    noisy_metrics = compute_vs_metrics(noisy.truth_vs, noisy.inverted_vs)
    scalar_names = (
        "mae_km_s",
        "rmse_km_s",
        "mean_relative_percent",
        "p95_absolute_error_km_s",
    )
    result = {
        name: _delta_value(noisy_metrics[name], clean_metrics[name])
        for name in scalar_names
    }
    result["per_layer"] = {
        name: _list_delta(
            noisy_metrics["per_layer"][name], clean_metrics["per_layer"][name]
        )
        for name in ("mae_km_s", "bias_km_s")
    }
    clean_row_mae = np.mean(np.abs(clean.inverted_vs - clean.truth_vs), axis=1)
    noisy_row_mae = np.mean(np.abs(noisy.inverted_vs - noisy.truth_vs), axis=1)
    return result, float(np.mean(noisy_row_mae - clean_row_mae))


def _frequency_usable_counts(mask: NDArray[np.bool_]) -> dict[str, int]:
    counts = {"overall": int(np.count_nonzero(mask))}
    counts.update(
        {
            f"mode_{mode}": int(np.count_nonzero(mask[:, mode]))
            for mode in range(mask.shape[1])
        }
    )
    return counts


def _frequency_usable_row_counts(mask: NDArray[np.bool_]) -> dict[str, int]:
    counts = {
        "overall": int(
            np.count_nonzero(np.any(mask.reshape(mask.shape[0], -1), axis=1))
        )
    }
    counts.update(
        {
            f"mode_{mode}": int(np.count_nonzero(np.any(mask[:, mode], axis=1)))
            for mode in range(mask.shape[1])
        }
    )
    return counts


def _frequency_delta(
    clean_observed: NDArray[np.float32],
    clean_predicted: NDArray[np.float32],
    noisy_observed: NDArray[np.float32],
    noisy_predicted: NDArray[np.float32],
    common_mask: NDArray[np.bool_],
) -> tuple[dict[str, Any], dict[str, int]]:
    clean_metrics = compute_frequency_metrics(
        clean_observed, clean_predicted, common_mask
    )
    noisy_metrics = compute_frequency_metrics(
        noisy_observed, noisy_predicted, common_mask
    )
    result: dict[str, Any] = {}
    for group in clean_metrics:
        result[group] = {
            name: _delta_value(noisy_metrics[group][name], clean_metrics[group][name])
            for name in (
                "mae_km_s",
                "rmse_km_s",
                "p95_absolute_error_km_s",
            )
        }
    return result, _frequency_usable_counts(common_mask)


def _interval_delta(clean: _MetricRows, noisy: _MetricRows) -> dict[str, Any]:
    assert clean.p10_vs is not None
    assert clean.p90_vs is not None
    assert noisy.p10_vs is not None
    assert noisy.p90_vs is not None
    clean_metrics = compute_interval_metrics(clean.truth_vs, clean.p10_vs, clean.p90_vs)
    noisy_metrics = compute_interval_metrics(noisy.truth_vs, noisy.p10_vs, noisy.p90_vs)
    return {
        "coverage_fraction": _delta_value(
            noisy_metrics["coverage_fraction"], clean_metrics["coverage_fraction"]
        ),
        "mean_interval_width_km_s": _delta_value(
            noisy_metrics["mean_interval_width_km_s"],
            clean_metrics["mean_interval_width_km_s"],
        ),
        "per_layer": {
            name: _list_delta(
                noisy_metrics["per_layer"][name],
                clean_metrics["per_layer"][name],
            )
            for name in ("coverage_fraction", "mean_interval_width_km_s")
        },
    }


def _null_frequency_delta(mode_count: int) -> dict[str, Any]:
    values = {
        "mae_km_s": None,
        "rmse_km_s": None,
        "p95_absolute_error_km_s": None,
    }
    return {
        "overall": dict(values),
        **{f"mode_{mode}": dict(values) for mode in range(mode_count)},
    }


def _null_vs_delta() -> dict[str, Any]:
    return {
        "mae_km_s": None,
        "rmse_km_s": None,
        "mean_relative_percent": None,
        "p95_absolute_error_km_s": None,
        "per_layer": {
            "mae_km_s": [None] * 20,
            "bias_km_s": [None] * 20,
        },
    }


def _paired_start_delta(clean: _MetricRows, noisy: _MetricRows) -> dict[str, Any]:
    assert clean.ensemble_success is not None
    assert noisy.ensemble_success is not None
    assert clean.ensemble_iterations is not None
    assert noisy.ensemble_iterations is not None
    assert clean.ensemble_evaluations is not None
    assert noisy.ensemble_evaluations is not None
    assert clean.ensemble_initial_objective is not None
    assert noisy.ensemble_initial_objective is not None
    assert clean.ensemble_objective is not None
    assert noisy.ensemble_objective is not None
    if clean.ensemble_success.shape != noisy.ensemble_success.shape:
        raise ValueError("clean and noisy start diagnostics are not aligned")

    def paired_finite_mean_delta(
        clean_values: NDArray[Any], noisy_values: NDArray[Any]
    ) -> tuple[int, float | None]:
        usable = np.isfinite(clean_values) & np.isfinite(noisy_values)
        count = int(np.count_nonzero(usable))
        return (
            count,
            float(np.mean(noisy_values[usable] - clean_values[usable]))
            if count
            else None,
        )

    initial_count, initial_delta = paired_finite_mean_delta(
        clean.ensemble_initial_objective, noisy.ensemble_initial_objective
    )
    final_count, final_delta = paired_finite_mean_delta(
        clean.ensemble_objective, noisy.ensemble_objective
    )
    return {
        "paired_start_count": int(clean.ensemble_success.size),
        "paired_successful_start_count": int(
            np.count_nonzero(clean.ensemble_success & noisy.ensemble_success)
        ),
        "convergence_fraction": float(
            np.mean(noisy.ensemble_success) - np.mean(clean.ensemble_success)
        ),
        "iterations_mean": float(
            np.mean(noisy.ensemble_iterations - clean.ensemble_iterations)
        ),
        "evaluations_mean": float(
            np.mean(noisy.ensemble_evaluations - clean.ensemble_evaluations)
        ),
        "initial_objective": {
            "paired_finite_count": initial_count,
            "mean": initial_delta,
        },
        "final_objective": {
            "paired_finite_count": final_count,
            "mean": final_delta,
        },
        "policy": "noisy minus clean on aligned sample/start identities",
    }


def _group_delta(clean: _MetricRows, noisy: _MetricRows) -> dict[str, Any]:
    if not np.array_equal(clean.sample_id, noisy.sample_id):
        raise ValueError("clean and noisy rows are not paired by sample ID")
    paired = clean.success & noisy.success
    paired_count = int(np.count_nonzero(paired))
    mode_count = clean.observed_phase_velocity.shape[1]
    result: dict[str, Any] = {
        "paired_sample_count": int(clean.sample_id.size),
        "paired_successful_count": paired_count,
        "paired_policy": (
            "noisy minus clean on rows successful in both scenarios; frequency "
            "deltas additionally use cells valid in both scenarios"
        ),
    }
    if clean.ensemble_success is not None:
        result["start_diagnostics"] = _paired_start_delta(clean, noisy)
    if clean.physical_success is not None:
        assert noisy.physical_success is not None
        result["physical_success_fraction"] = float(
            np.mean(noisy.physical_success) - np.mean(clean.physical_success)
        )
    if paired_count == 0:
        result.update(
            usable_counts={
                "vs_rows": 0,
                "surrogate_frequency_values": {
                    "overall": 0,
                    **{f"mode_{mode}": 0 for mode in range(mode_count)},
                },
                "surrogate_frequency_rows": {
                    "overall": 0,
                    **{f"mode_{mode}": 0 for mode in range(mode_count)},
                },
                **(
                    {
                        "physical_frequency_values": {
                            "overall": 0,
                            **{f"mode_{mode}": 0 for mode in range(mode_count)},
                        },
                        "physical_frequency_rows": {
                            "overall": 0,
                            **{f"mode_{mode}": 0 for mode in range(mode_count)},
                        },
                        "interval_rows": 0,
                    }
                    if clean.physical_phase_velocity is not None
                    else {}
                ),
            },
            mean_paired_sample_vs_mae_delta_km_s=None,
            vs_mae_km_s=None,
            vs_rmse_km_s=None,
            surrogate_frequency_mae_km_s=None,
            vs=_null_vs_delta(),
            surrogate_frequency=_null_frequency_delta(mode_count),
        )
        if clean.physical_phase_velocity is not None:
            result.update(
                physical_frequency_mae_km_s=None,
                interval_coverage_fraction=None,
                interval_width_km_s=None,
                physical_frequency=_null_frequency_delta(mode_count),
                uncertainty=None,
            )
        return result

    paired_clean = _select_rows(clean, paired)
    paired_noisy = _select_rows(noisy, paired)
    vs_delta, paired_sample_delta = _vs_delta(paired_clean, paired_noisy)
    common_surrogate_mask = paired_clean.valid_mask & paired_noisy.valid_mask
    surrogate_delta, surrogate_counts = _frequency_delta(
        paired_clean.observed_phase_velocity,
        paired_clean.surrogate_phase_velocity,
        paired_noisy.observed_phase_velocity,
        paired_noisy.surrogate_phase_velocity,
        common_surrogate_mask,
    )
    result.update(
        usable_counts={
            "vs_rows": paired_count,
            "surrogate_frequency_values": surrogate_counts,
            "surrogate_frequency_rows": _frequency_usable_row_counts(
                common_surrogate_mask
            ),
        },
        mean_paired_sample_vs_mae_delta_km_s=paired_sample_delta,
        vs_mae_km_s=vs_delta["mae_km_s"],
        vs_rmse_km_s=vs_delta["rmse_km_s"],
        surrogate_frequency_mae_km_s=surrogate_delta["overall"]["mae_km_s"],
        vs=vs_delta,
        surrogate_frequency=surrogate_delta,
    )
    if paired_clean.physical_phase_velocity is not None:
        assert paired_clean.physical_valid_mask is not None
        assert paired_noisy.physical_phase_velocity is not None
        assert paired_noisy.physical_valid_mask is not None
        common_physical_mask = (
            common_surrogate_mask
            & paired_clean.physical_valid_mask
            & paired_noisy.physical_valid_mask
        )
        physical_delta, physical_counts = _frequency_delta(
            paired_clean.observed_phase_velocity,
            paired_clean.physical_phase_velocity,
            paired_noisy.observed_phase_velocity,
            paired_noisy.physical_phase_velocity,
            common_physical_mask,
        )
        uncertainty_delta = _interval_delta(paired_clean, paired_noisy)
        result["usable_counts"].update(
            physical_frequency_values=physical_counts,
            physical_frequency_rows=_frequency_usable_row_counts(common_physical_mask),
            interval_rows=paired_count,
        )
        result.update(
            physical_frequency_mae_km_s=physical_delta["overall"]["mae_km_s"],
            interval_coverage_fraction=uncertainty_delta["coverage_fraction"],
            interval_width_km_s=uncertainty_delta["mean_interval_width_km_s"],
            physical_frequency=physical_delta,
            uncertainty=uncertainty_delta,
        )
    return result


def _scope_summary(rows: _MetricRows) -> dict[str, Any]:
    groups: dict[str, Any] = {
        "overall": _compute_rows_metrics(rows),
        "noise": {},
        "model_kind": {},
        "noise_model_kind": {},
    }
    for noise in sorted(set(rows.noise.tolist())):
        noise_rows = _select_rows(rows, rows.noise == noise)
        groups["noise"][noise] = _compute_rows_metrics(noise_rows)
        groups["noise_model_kind"][noise] = {}
        for kind_value, kind_name in _KIND_NAMES.items():
            kind_rows = _select_rows(noise_rows, noise_rows.model_kind == kind_value)
            if kind_rows.sample_id.size:
                groups["noise_model_kind"][noise][kind_name] = _compute_rows_metrics(
                    kind_rows
                )
    for kind_value, kind_name in _KIND_NAMES.items():
        kind_rows = _select_rows(rows, rows.model_kind == kind_value)
        if kind_rows.sample_id.size:
            groups["model_kind"][kind_name] = _compute_rows_metrics(kind_rows)

    deltas: dict[str, Any] = {}
    noises = set(rows.noise.tolist())
    if {"clean", "noise_1pct"}.issubset(noises):
        clean = _select_rows(rows, rows.noise == "clean")
        noisy = _select_rows(rows, rows.noise == "noise_1pct")
        deltas["overall"] = _group_delta(clean, noisy)
        deltas["model_kind"] = {}
        for kind_value, kind_name in _KIND_NAMES.items():
            clean_kind = _select_rows(clean, clean.model_kind == kind_value)
            noisy_kind = _select_rows(noisy, noisy.model_kind == kind_value)
            if clean_kind.sample_id.size:
                deltas["model_kind"][kind_name] = _group_delta(clean_kind, noisy_kind)
    return {"groups": groups, "clean_to_noise_1pct_delta": deltas}


def _require_plot_rows(rows: _MetricRows, description: str) -> NDArray[np.bool_]:
    successful = np.asarray(rows.success, dtype=np.bool_)
    if not np.any(successful):
        raise ValueError(f"cannot plot empty successful group: {description}")
    return successful


def _save_figure(figure: Any, output_dir: Path, name: str) -> None:
    temporary = output_dir / f".{name}.tmp-{os.getpid()}"
    figure.savefig(
        temporary,
        format="png",
        dpi=120,
        bbox_inches="tight",
        metadata={"Software": "swave-forward"},
    )
    plt.close(figure)
    if temporary.stat().st_size == 0:
        temporary.unlink()
        raise ValueError(f"plot {name} is empty")
    temporary.replace(output_dir / name)


def _plot_vs_depth(rows: _MetricRows, output_dir: Path, scope_label: str) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    depth = np.arange(20, dtype=np.float64) * 0.1
    for noise in sorted(set(rows.noise.tolist())):
        selected = _select_rows(rows, rows.noise == noise)
        successful = _require_plot_rows(selected, f"{scope_label}/{noise}")
        metrics = compute_vs_metrics(
            selected.truth_vs[successful], selected.inverted_vs[successful]
        )
        axes[0].plot(metrics["per_layer"]["mae_km_s"], depth, label=noise)
        axes[1].plot(metrics["per_layer"]["bias_km_s"], depth, label=noise)
    axes[0].set_xlabel("Vs MAE (km/s)")
    axes[1].set_xlabel("Vs bias (km/s)")
    axes[0].set_ylabel("Layer-top depth (km)")
    axes[0].invert_yaxis()
    axes[0].legend()
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    figure.suptitle(f"Vs error by depth — {scope_label}")
    figure.tight_layout()
    name = "vs-error-by-depth.png"
    _save_figure(figure, output_dir, name)
    return name


def _plot_kind_noise(rows: _MetricRows, output_dir: Path, scope_label: str) -> str:
    data: list[NDArray[np.float64]] = []
    labels: list[str] = []
    noises = sorted(set(rows.noise.tolist()))
    for kind_value, kind_name in _KIND_NAMES.items():
        for noise in noises:
            selected = _select_rows(
                rows, (rows.model_kind == kind_value) & (rows.noise == noise)
            )
            if selected.sample_id.size == 0:
                raise ValueError(
                    f"cannot plot empty kind/noise group: "
                    f"{scope_label}/{kind_name}/{noise}"
                )
            successful = _require_plot_rows(
                selected, f"{scope_label}/{kind_name}/{noise}"
            )
            row_mae = np.mean(
                np.abs(
                    selected.inverted_vs[successful] - selected.truth_vs[successful]
                ),
                axis=1,
            )
            data.append(row_mae)
            labels.append(f"{kind_name}\n{noise}")
    if not data:
        raise ValueError(f"cannot plot empty kind/noise group: {scope_label}")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(data, tick_labels=labels, showfliers=True)
    axis.set_ylabel("Per-sample Vs MAE (km/s)")
    axis.set_title(f"Vs error by model kind and noise — {scope_label}")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    name = "vs-error-by-kind-and-noise.png"
    _save_figure(figure, output_dir, name)
    return name


def _plot_optimization(rows: _MetricRows, output_dir: Path, scope_label: str) -> str:
    if rows.sample_id.size == 0:
        raise ValueError(f"cannot plot empty optimization group: {scope_label}")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    is_deep = rows.ensemble_success is not None
    if is_deep:
        assert rows.ensemble_initial_objective is not None
        assert rows.ensemble_objective is not None
        assert rows.ensemble_iterations is not None
        finite_initial = rows.ensemble_initial_objective[
            np.isfinite(rows.ensemble_initial_objective)
        ]
        finite_final = rows.ensemble_objective[np.isfinite(rows.ensemble_objective)]
    else:
        finite_initial = rows.initial_objective[np.isfinite(rows.initial_objective)]
        finite_final = rows.final_objective[np.isfinite(rows.final_objective)]
    if finite_initial.size == 0 or finite_final.size == 0:
        raise ValueError(f"cannot plot empty objective group: {scope_label}")
    axes[0].boxplot([finite_initial, finite_final], tick_labels=["initial", "final"])
    axes[0].set_ylabel("Objective")
    noises = sorted(set(rows.noise.tolist()))
    success_rates = []
    for noise in noises:
        selected = _select_rows(rows, rows.noise == noise)
        if selected.ensemble_success is None:
            success_rates.append(float(np.mean(selected.success)))
        else:
            success_rates.append(float(np.mean(selected.ensemble_success)))
    axes[1].bar(noises, success_rates)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel(
        "Start convergence fraction" if is_deep else "Inversion success fraction"
    )
    if is_deep:
        assert rows.ensemble_success is not None
        assert rows.ensemble_iterations is not None
        iterations = rows.ensemble_iterations[rows.ensemble_success]
        if iterations.size == 0:
            raise ValueError(f"cannot plot empty successful starts: {scope_label}")
        axes[2].hist(iterations, bins="auto")
        axes[2].set_ylabel("Successful starts")
    else:
        successful = _require_plot_rows(rows, scope_label)
        axes[2].hist(rows.iterations[successful], bins="auto")
        axes[2].set_ylabel("Successful inversions")
    axes[2].set_xlabel("Iterations")
    figure.suptitle(f"Optimization diagnostics — {scope_label}")
    figure.tight_layout()
    name = "optimization-diagnostics.png"
    _save_figure(figure, output_dir, name)
    return name


def _plot_representative(
    row: _MetricRows,
    output_dir: Path,
    kind_name: str,
    noise: str,
    frequencies: Sequence[float],
) -> str:
    if row.sample_id.size != 1 or not bool(row.success[0]):
        raise ValueError("representative plotting requires one successful deep row")
    assert row.median_vs is not None
    assert row.p10_vs is not None
    assert row.p90_vs is not None
    assert row.physical_phase_velocity is not None
    assert row.physical_valid_mask is not None
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    frequency_count = row.observed_phase_velocity.shape[2]
    if frequency_values.shape != (frequency_count,):
        raise ValueError(f"frequencies must contain exactly {frequency_count} values")
    if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values <= 0):
        raise ValueError("frequencies must be finite and positive")
    if np.any(np.diff(frequency_values) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    figure, axes = plt.subplots(1, 5, figsize=(19, 5))
    depth = np.arange(20, dtype=np.float64) * 0.1
    profile_axis = axes[0]
    profile_axis.step(row.truth_vs[0], depth, where="post", label="true")
    profile_axis.step(row.reference_vs[0], depth, where="post", label="reference")
    profile_axis.step(row.median_vs[0], depth, where="post", label="median")
    profile_axis.step(row.p10_vs[0], depth, where="post", linestyle="--", label="P10")
    profile_axis.step(row.p90_vs[0], depth, where="post", linestyle="--", label="P90")
    profile_axis.fill_betweenx(
        depth, row.p10_vs[0], row.p90_vs[0], alpha=0.25, label="P10–P90"
    )
    profile_axis.set_xlabel("Vs (km/s)")
    profile_axis.set_ylabel("Layer-top depth (km)")
    profile_axis.invert_yaxis()
    profile_axis.legend(fontsize=7)
    for mode, axis in enumerate(axes[1:]):
        observed_mask = row.valid_mask[0, mode]
        physical_mask = observed_mask & row.physical_valid_mask[0, mode]
        axis.plot(
            frequency_values[observed_mask],
            row.observed_phase_velocity[0, mode, observed_mask],
            label="observed",
        )
        axis.plot(
            frequency_values[observed_mask],
            row.surrogate_phase_velocity[0, mode, observed_mask],
            label="surrogate",
        )
        axis.plot(
            frequency_values[physical_mask],
            row.physical_phase_velocity[0, mode, physical_mask],
            label="physical",
        )
        axis.set_title(f"Mode {mode}")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Phase velocity (km/s)")
        if mode == 0:
            axis.legend(fontsize=7)
    figure.suptitle(
        f"Representative {kind_name}/{noise} — sample {int(row.sample_id[0])}"
    )
    figure.tight_layout()
    name = f"representative-{kind_name}-{noise}.png"
    _save_figure(figure, output_dir, name)
    return name


def _write_summary_atomic(output_dir: Path, summary: dict[str, Any]) -> None:
    target = output_dir / "summary.json"
    temporary = output_dir / f"summary.json.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    descriptor = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_inversion_report(
    results_dir: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    dataset_config: DatasetConfig,
    allow_archived_software: bool = False,
) -> dict[str, Any]:
    """Build a deterministic report after validating the complete result set."""
    results_path = Path(results_dir)
    dataset_path = Path(dataset_dir)

    # This must remain the first external-data action: truth is unavailable until the
    # entire immutable result identity and every result shard have passed validation.
    manifest = validate_complete_results(
        results_path,
        allow_archived_software=allow_archived_software,
    )
    _validate_dataset_identity(dataset_path, manifest)
    grouped_batches = _load_result_groups(results_path, manifest)
    requested_ids = _validate_result_alignment(grouped_batches)
    if not isinstance(dataset_config, DatasetConfig):
        raise TypeError("dataset_config must be a DatasetConfig")
    if canonical_hash(dataset_config) != manifest.dataset_config_hash:
        raise ValueError(
            "report dataset configuration does not match inversion results"
        )

    # The only source-HDF5 Vs access is here, for the exact unique requested IDs.
    truth = _load_true_vs(dataset_path, requested_ids)
    truth_by_id = _truth_lookup(requested_ids, truth)

    rows_by_scope = {
        experiment: _scope_rows(batches_by_noise, truth_by_id)
        for experiment, batches_by_noise in sorted(grouped_batches.items())
    }
    frequencies = dataset_config.physics.frequencies
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        raise ValueError("report output path is not a directory")

    summaries: dict[str, Any] = {}
    for experiment, rows in rows_by_scope.items():
        scope_summary = _scope_summary(rows)
        summaries[experiment] = {
            "scope_label": _SCOPE_LABELS[experiment],
            "optimization_diagnostic_semantics": (
                "one optimizer run per result row"
                if experiment == "full"
                else "start diagnostics are reported per optimizer start; row totals "
                "are retained only as sample-level effort summaries"
            ),
            "result_row_count": int(rows.sample_id.size),
            "unique_sample_count": int(np.unique(rows.sample_id).size),
            "noise_scenarios": sorted(set(rows.noise.tolist())),
            **scope_summary,
        }

    population_scope = "full" if "full" in rows_by_scope else "deep"
    population_rows = rows_by_scope[population_scope]
    figures = [
        _plot_vs_depth(population_rows, output_path, _SCOPE_LABELS[population_scope]),
        _plot_kind_noise(population_rows, output_path, _SCOPE_LABELS[population_scope]),
        _plot_optimization(
            population_rows, output_path, _SCOPE_LABELS[population_scope]
        ),
    ]

    representatives: dict[str, dict[str, Any]] = {}
    if "deep" in rows_by_scope:
        deep_rows = rows_by_scope["deep"]
        for kind_value, kind_name in _KIND_NAMES.items():
            representatives[kind_name] = {}
            for noise in sorted(set(deep_rows.noise.tolist())):
                group = _select_rows(
                    deep_rows,
                    (deep_rows.model_kind == kind_value) & (deep_rows.noise == noise),
                )
                if group.sample_id.size == 0:
                    raise ValueError(
                        f"cannot select representative from empty group: {kind_name}/{noise}"
                    )
                successful = np.flatnonzero(group.success)
                if successful.size == 0:
                    raise ValueError(
                        "cannot select representative from empty successful group: "
                        f"{kind_name}/{noise}"
                    )
                candidate_ids = group.sample_id[successful]
                index = int(successful[np.argmin(candidate_ids)])
                representative = _take_rows(group, np.asarray([index], dtype=np.intp))
                figure_name = _plot_representative(
                    representative,
                    output_path,
                    kind_name,
                    noise,
                    frequencies,
                )
                figures.append(figure_name)
                representatives[kind_name][noise] = {
                    "sample_id": int(representative.sample_id[0]),
                    "selection": "smallest successful deep sample_id",
                    "figure": figure_name,
                }

    reporting_software_sha256 = software_sha256()
    summary: dict[str, Any] = {
        "schema_version": 2,
        "units": {
            "velocity": "km/s",
            "depth": "km",
            "relative_error": "percent",
        },
        "result_identity": {
            "dataset_config_hash": manifest.dataset_config_hash,
            "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "inversion_config_hash": manifest.inversion_config_hash,
            "split_policy": manifest.split_policy,
            "experiment": manifest.experiment,
            "software_sha256": manifest.software_sha256,
        },
        "report_generation": {
            "software_sha256": reporting_software_sha256,
            "archived_results": (
                manifest.software_sha256 != reporting_software_sha256
            ),
        },
        "scope_policy": (
            "full and deep rows are reported independently; figures never pool "
            "single-start population rows with multi-start uncertainty rows"
        ),
        "population_figure_scope": population_scope,
        "experiment_scopes": summaries,
        "comparison_populations": {
            "inversion": {
                "sample_count": len(np.unique(rows_by_scope["full"].sample_id)),
                "sample_id_sha256": sample_id_sha256(
                    np.asarray(
                        np.unique(rows_by_scope["full"].sample_id),
                        dtype=np.uint64,
                    )
                ),
            }
        }
        if "full" in rows_by_scope
        else {},
        "representatives": representatives,
        "figures": sorted(figures),
    }
    _write_summary_atomic(output_path, summary)
    return summary
