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

from .geology import ModelKind
from .inversion_results import (
    ResultBatch,
    ResultManifest,
    validate_complete_results,
    validate_result_shard,
)

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

_JOB_PATTERN = re.compile(
    r"(?P<experiment>full|deep)-(?P<noise>clean|noise_1pct)-shard-[0-9]+"
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
    median_vs: NDArray[np.float32] | None
    p10_vs: NDArray[np.float32] | None
    p90_vs: NDArray[np.float32] | None
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
        median_vs=batch.median_vs,
        p10_vs=batch.p10_vs,
        p90_vs=batch.p90_vs,
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
    result: dict[str, Any] = {
        "sample_count": count,
        "successful_count": success_count,
        "convergence": {
            "success_fraction": float(success_count / count),
            "failure_code_counts": dict(sorted(Counter(codes).items())),
            "status_counts": {
                str(int(status)): int(total)
                for status, total in sorted(
                    Counter(int(value) for value in rows.status).items()
                )
            },
        },
        "optimization": {
            "iterations": _distribution(rows.iterations),
            "evaluations": _distribution(rows.evaluations),
            "initial_objective": _distribution(rows.initial_objective),
            "final_objective": _distribution(rows.final_objective),
        },
    }
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


def _dataset_identity(dataset_dir: Path, expected_hash: str) -> None:
    manifest_path = dataset_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is not readable JSON") from error
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("dataset manifest is incomplete")
    if payload.get("config_hash") != expected_hash:
        raise ValueError("dataset identity does not match inversion results")


def _load_result_groups(
    results_dir: Path,
    manifest: ResultManifest,
) -> dict[str, dict[str, list[ResultBatch]]]:
    grouped: dict[str, dict[str, list[ResultBatch]]] = {}
    for job in manifest.expected_jobs:
        match = _JOB_PATTERN.fullmatch(job)
        if match is None:
            raise ValueError(f"result job {job!r} does not have a reportable identity")
        experiment = match.group("experiment")
        noise = match.group("noise")
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


def _group_delta(clean: _MetricRows, noisy: _MetricRows) -> dict[str, Any]:
    if not np.array_equal(clean.sample_id, noisy.sample_id):
        raise ValueError("clean and noisy rows are not paired by sample ID")
    clean_metrics = _compute_rows_metrics(clean)
    noisy_metrics = _compute_rows_metrics(noisy)
    paired = clean.success & noisy.success
    paired_count = int(np.count_nonzero(paired))
    paired_delta = None
    if paired_count:
        clean_mae = np.mean(
            np.abs(clean.inverted_vs[paired] - clean.truth_vs[paired]), axis=1
        )
        noisy_mae = np.mean(
            np.abs(noisy.inverted_vs[paired] - noisy.truth_vs[paired]), axis=1
        )
        paired_delta = float(np.mean(noisy_mae - clean_mae))
    result = {
        "paired_sample_count": int(clean.sample_id.size),
        "paired_successful_count": paired_count,
        "mean_paired_sample_vs_mae_delta_km_s": paired_delta,
        "success_fraction": _delta_value(
            noisy_metrics["convergence"]["success_fraction"],
            clean_metrics["convergence"]["success_fraction"],
        ),
        "vs_mae_km_s": _delta_value(
            noisy_metrics["vs"]["mae_km_s"], clean_metrics["vs"]["mae_km_s"]
        ),
        "vs_rmse_km_s": _delta_value(
            noisy_metrics["vs"]["rmse_km_s"], clean_metrics["vs"]["rmse_km_s"]
        ),
        "surrogate_frequency_mae_km_s": _delta_value(
            noisy_metrics["surrogate_frequency"]["overall"]["mae_km_s"],
            clean_metrics["surrogate_frequency"]["overall"]["mae_km_s"],
        ),
    }
    if "physical_frequency" in clean_metrics:
        result.update(
            physical_frequency_mae_km_s=_delta_value(
                noisy_metrics["physical_frequency"]["overall"]["mae_km_s"],
                clean_metrics["physical_frequency"]["overall"]["mae_km_s"],
            ),
            interval_coverage_fraction=_delta_value(
                None
                if noisy_metrics["uncertainty"] is None
                else noisy_metrics["uncertainty"]["coverage_fraction"],
                None
                if clean_metrics["uncertainty"] is None
                else clean_metrics["uncertainty"]["coverage_fraction"],
            ),
            interval_width_km_s=_delta_value(
                None
                if noisy_metrics["uncertainty"] is None
                else noisy_metrics["uncertainty"]["mean_interval_width_km_s"],
                None
                if clean_metrics["uncertainty"] is None
                else clean_metrics["uncertainty"]["mean_interval_width_km_s"],
            ),
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
    finite_initial = rows.initial_objective[np.isfinite(rows.initial_objective)]
    finite_final = rows.final_objective[np.isfinite(rows.final_objective)]
    if finite_initial.size == 0 or finite_final.size == 0:
        raise ValueError(f"cannot plot empty objective group: {scope_label}")
    axes[0].boxplot([finite_initial, finite_final], tick_labels=["initial", "final"])
    axes[0].set_ylabel("Objective")
    noises = sorted(set(rows.noise.tolist()))
    success_rates = [
        float(np.mean(rows.success[rows.noise == noise])) for noise in noises
    ]
    axes[1].bar(noises, success_rates)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Convergence fraction")
    successful = _require_plot_rows(rows, scope_label)
    axes[2].hist(rows.iterations[successful], bins="auto")
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Successful rows")
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
) -> str:
    if row.sample_id.size != 1 or not bool(row.success[0]):
        raise ValueError("representative plotting requires one successful deep row")
    assert row.median_vs is not None
    assert row.p10_vs is not None
    assert row.p90_vs is not None
    assert row.physical_phase_velocity is not None
    assert row.physical_valid_mask is not None
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
    frequency_index = np.arange(row.observed_phase_velocity.shape[2])
    for mode, axis in enumerate(axes[1:]):
        observed_mask = row.valid_mask[0, mode]
        physical_mask = observed_mask & row.physical_valid_mask[0, mode]
        axis.plot(
            frequency_index[observed_mask],
            row.observed_phase_velocity[0, mode, observed_mask],
            label="observed",
        )
        axis.plot(
            frequency_index[observed_mask],
            row.surrogate_phase_velocity[0, mode, observed_mask],
            label="surrogate",
        )
        axis.plot(
            frequency_index[physical_mask],
            row.physical_phase_velocity[0, mode, physical_mask],
            label="physical",
        )
        axis.set_title(f"Mode {mode}")
        axis.set_xlabel("Frequency-grid index")
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
) -> dict[str, Any]:
    """Build a deterministic report after validating the complete result set."""
    results_path = Path(results_dir)
    dataset_path = Path(dataset_dir)

    # This must remain the first external-data action: truth is unavailable until the
    # entire immutable result identity and every result shard have passed validation.
    manifest = validate_complete_results(results_path)
    grouped_batches = _load_result_groups(results_path, manifest)
    requested_ids = _validate_result_alignment(grouped_batches)
    _dataset_identity(dataset_path, manifest.dataset_config_hash)

    # The only source-HDF5 Vs access is here, for the exact unique requested IDs.
    truth = _load_true_vs(dataset_path, requested_ids)
    truth_by_id = _truth_lookup(requested_ids, truth)

    rows_by_scope = {
        experiment: _scope_rows(batches_by_noise, truth_by_id)
        for experiment, batches_by_noise in sorted(grouped_batches.items())
    }
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
                else "stored iterations and evaluations are totals across ensemble starts"
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
                    representative, output_path, kind_name, noise
                )
                figures.append(figure_name)
                representatives[kind_name][noise] = {
                    "sample_id": int(representative.sample_id[0]),
                    "selection": "smallest successful deep sample_id",
                    "figure": figure_name,
                }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "units": {
            "velocity": "km/s",
            "depth": "km",
            "relative_error": "percent",
        },
        "result_identity": {
            "dataset_config_hash": manifest.dataset_config_hash,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "inversion_config_hash": manifest.inversion_config_hash,
            "split_policy": manifest.split_policy,
            "experiment": manifest.experiment,
        },
        "scope_policy": (
            "full and deep rows are reported independently; figures never pool "
            "single-start population rows with multi-start uncertainty rows"
        ),
        "population_figure_scope": population_scope,
        "experiment_scopes": summaries,
        "representatives": representatives,
        "figures": sorted(figures),
    }
    _write_summary_atomic(output_path, summary)
    return summary
