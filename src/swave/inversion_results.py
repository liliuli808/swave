"""Strict, atomic, and resumable storage for inversion result shards."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from . import __version__
from .config import InversionConfig, inversion_identity_hash
from .splits import SPLIT_POLICY

SCHEMA_VERSION = 3
VS_MIN = 0.3
VS_MAX = 2.6
CURVE_SHAPE = (4, 120)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_JOB_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_FAILURE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_:-]{0,127}")

_REQUIRED_DATASETS = (
    "sample_id",
    "model_kind",
    "success",
    "status",
    "iterations",
    "evaluations",
    "initial_objective",
    "final_objective",
    "data_misfit",
    "regularization",
    "reference_vs",
    "inverted_vs",
    "observed_phase_velocity",
    "surrogate_phase_velocity",
    "valid_mask",
    "failure_code",
)
_OPTIONAL_DATASETS = (
    "ensemble_vs",
    "ensemble_success",
    "ensemble_status",
    "ensemble_iterations",
    "ensemble_evaluations",
    "ensemble_initial_objective",
    "ensemble_objective",
    "ensemble_failure_code",
    "ensemble_message",
    "ensemble_inlier_mask",
    "median_vs",
    "p10_vs",
    "p90_vs",
    "physical_success",
    "physical_status",
    "physical_failure_code",
    "physical_phase_velocity",
    "physical_valid_mask",
)
_ALL_DATASETS = _REQUIRED_DATASETS + _OPTIONAL_DATASETS
_IDENTITY_ATTRS = (
    "schema_version",
    "dataset_config_hash",
    "dataset_manifest_sha256",
    "checkpoint_sha256",
    "split_policy",
    "inversion_config_hash",
    "experiment",
    "package_version",
    "software_sha256",
)


def _require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_job_name(value: object) -> str:
    if not isinstance(value, str) or _JOB_PATTERN.fullmatch(value) is None:
        raise ValueError("job name contains unsafe characters")
    if value.endswith(".h5"):
        raise ValueError("job name must not include the .h5 suffix")
    return value


def _job_experiment(job: str) -> str:
    if job.startswith("full-"):
        return "full"
    if job.startswith("deep-"):
        return "deep"
    raise ValueError("job name must start with full- or deep-")


def _require_array(
    value: object,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
) -> NDArray[Any]:
    expected_dtype = np.dtype(dtype)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != expected_dtype:
        raise ValueError(f"{name} dtype must be {expected_dtype}, not {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {value.shape}")
    return value


def _require_bytes_array(
    value: object,
    name: str,
    shape: tuple[int, ...],
    *,
    maximum_width: int,
) -> NDArray[np.bytes_]:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype.kind != "S"
        or not 1 <= value.dtype.itemsize <= maximum_width
    ):
        raise ValueError(
            f"{name} must have shape {shape} and a fixed-width bytes dtype"
        )
    return value


def _row_is_all_nan(values: NDArray[Any]) -> NDArray[np.bool_]:
    flattened = values.reshape(values.shape[0], -1)
    return np.asarray(np.all(np.isnan(flattened), axis=1), dtype=np.bool_)


def _row_is_all_finite(values: NDArray[Any]) -> NDArray[np.bool_]:
    flattened = values.reshape(values.shape[0], -1)
    return np.asarray(np.all(np.isfinite(flattened), axis=1), dtype=np.bool_)


def _validate_profile_rows(values: NDArray[Any], name: str) -> NDArray[np.bool_]:
    finite = _row_is_all_finite(values)
    missing = _row_is_all_nan(values)
    if not np.all(finite | missing):
        raise ValueError(f"{name} rows must be entirely finite or entirely NaN")
    if np.any((values[finite] < VS_MIN) | (values[finite] > VS_MAX)):
        raise ValueError(f"{name} contains values outside global Vs bounds")
    return finite


def _decode_failure_codes(values: NDArray[np.bytes_]) -> tuple[str, ...]:
    decoded: list[str] = []
    for raw in values.reshape(-1):
        try:
            text = bytes(raw).decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("failure_code must contain ASCII bytes") from error
        if text and _FAILURE_PATTERN.fullmatch(text) is None:
            raise ValueError("failure_code contains unsafe characters")
        decoded.append(text)
    return tuple(decoded)


def _decode_messages(values: NDArray[np.bytes_]) -> tuple[str, ...]:
    decoded: list[str] = []
    for raw in values.reshape(-1):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ensemble_message must contain UTF-8 bytes") from error
        if not text or any(ord(character) < 32 for character in text):
            raise ValueError("ensemble_message must contain nonempty printable text")
        decoded.append(text)
    return tuple(decoded)


def sample_id_sha256(values: NDArray[np.uint64]) -> str:
    """Hash one nonempty, strictly ordered uint64 sample population."""
    sample_ids = np.asarray(values)
    if sample_ids.dtype != np.dtype(np.uint64) or sample_ids.ndim != 1:
        raise ValueError("sample IDs must be a uint64 vector")
    if sample_ids.size == 0:
        raise ValueError("sample IDs must be nonempty")
    if sample_ids.size > 1 and not np.all(sample_ids[1:] > sample_ids[:-1]):
        raise ValueError("sample IDs must be unique and strictly ordered")
    canonical = np.asarray(sample_ids, dtype="<u8")
    digest = hashlib.sha256()
    digest.update(int(canonical.size).to_bytes(8, "big"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def software_sha256(
    package_root: Path | str | None = None,
    *,
    package_version: str | None = None,
) -> str:
    """Hash installed swave Python sources by canonical relative name and bytes."""
    root = (
        Path(__file__).resolve().parent if package_root is None else Path(package_root)
    )
    version = __version__ if package_version is None else package_version
    if not isinstance(version, str) or not version:
        raise ValueError("package_version must be a nonempty string")
    paths = sorted(
        (path for path in root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError("software package contains no Python source files")
    digest = hashlib.sha256()
    version_bytes = version.encode("utf-8")
    digest.update(len(version_bytes).to_bytes(8, "big"))
    digest.update(version_bytes)
    for path in paths:
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResultManifest:
    """Immutable identity and completion index for one inversion run."""

    schema_version: int
    dataset_config_hash: str
    dataset_manifest_sha256: str
    checkpoint_sha256: str
    split_policy: str
    inversion_config_hash: str
    minimum_valid_solutions: int | None
    experiment: str
    expected_jobs: tuple[str, ...]
    expected_job_sample_count: dict[str, int]
    expected_job_sample_id_sha256: dict[str, str]
    completed_jobs: tuple[str, ...]
    job_sha256: dict[str, str]
    package_version: str
    software_sha256: str
    created_at: str
    complete: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("result manifest schema version is unsupported")
        _require_hash(self.dataset_config_hash, "dataset_config_hash")
        _require_hash(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_hash(self.checkpoint_sha256, "checkpoint_sha256")
        _require_hash(self.inversion_config_hash, "inversion_config_hash")
        _require_hash(self.software_sha256, "software_sha256")
        if self.split_policy != SPLIT_POLICY:
            raise ValueError("result manifest split policy does not match")
        if self.experiment not in {"full", "deep", "both"}:
            raise ValueError("experiment must be full, deep, or both")
        if self.minimum_valid_solutions is not None and (
            type(self.minimum_valid_solutions) is not int
            or self.minimum_valid_solutions <= 0
        ):
            raise ValueError("minimum_valid_solutions must be a positive integer")
        if self.experiment == "full" and self.minimum_valid_solutions is not None:
            raise ValueError(
                "full result identity requires minimum_valid_solutions to be None"
            )
        if self.experiment in {"deep", "both"} and self.minimum_valid_solutions is None:
            raise ValueError("deep result identity requires minimum_valid_solutions")
        if not isinstance(self.expected_jobs, tuple) or not self.expected_jobs:
            raise ValueError("expected_jobs must be a nonempty tuple")
        if not isinstance(self.completed_jobs, tuple):
            raise TypeError("completed_jobs must be a tuple")
        expected = tuple(_require_job_name(job) for job in self.expected_jobs)
        completed = tuple(_require_job_name(job) for job in self.completed_jobs)
        if len(set(expected)) != len(expected):
            raise ValueError("expected_jobs contains duplicate jobs")
        if not isinstance(self.expected_job_sample_count, dict) or set(
            self.expected_job_sample_count
        ) != set(expected):
            raise ValueError(
                "expected_job_sample_count keys must exactly match expected_jobs"
            )
        if not isinstance(self.expected_job_sample_id_sha256, dict) or set(
            self.expected_job_sample_id_sha256
        ) != set(expected):
            raise ValueError(
                "expected_job_sample_id_sha256 keys must exactly match expected_jobs"
            )
        for job in expected:
            count = self.expected_job_sample_count[job]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("expected job sample counts must be positive integers")
            _require_hash(
                self.expected_job_sample_id_sha256[job],
                f"expected_job_sample_id_sha256[{job!r}]",
            )
        job_experiments = {_job_experiment(job) for job in expected}
        if self.experiment == "both" and job_experiments != {"full", "deep"}:
            raise ValueError(
                "both result manifest requires full and deep expected jobs"
            )
        if self.experiment != "both" and job_experiments != {self.experiment}:
            raise ValueError(
                "result manifest experiment and expected job prefixes disagree"
            )
        if len(set(completed)) != len(completed):
            raise ValueError("completed_jobs contains duplicate jobs")
        if any(job not in expected for job in completed):
            raise ValueError("completed_jobs contains an unexpected job")
        expected_completed_order = tuple(job for job in expected if job in completed)
        if completed != expected_completed_order:
            raise ValueError("completed_jobs must follow expected_jobs order")
        if not isinstance(self.job_sha256, dict):
            raise TypeError("job_sha256 must be a dictionary")
        if set(self.job_sha256) != set(completed):
            raise ValueError("job_sha256 keys must exactly match completed_jobs")
        for job, digest in self.job_sha256.items():
            _require_job_name(job)
            _require_hash(digest, f"job_sha256[{job!r}]")
        if not isinstance(self.package_version, str) or not self.package_version:
            raise ValueError("package_version must be a nonempty string")
        if not isinstance(self.created_at, str):
            raise TypeError("created_at must be an ISO-8601 string")
        try:
            created = datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise ValueError("created_at must be an ISO-8601 timestamp") from error
        if created.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if type(self.complete) is not bool:
            raise ValueError("complete must be a boolean")
        actually_complete = completed == expected
        if self.complete != actually_complete:
            raise ValueError("manifest complete flag disagrees with completed_jobs")
        object.__setattr__(self, "job_sha256", dict(self.job_sha256))
        object.__setattr__(
            self, "expected_job_sample_count", dict(self.expected_job_sample_count)
        )
        object.__setattr__(
            self,
            "expected_job_sample_id_sha256",
            dict(self.expected_job_sample_id_sha256),
        )


@dataclass(frozen=True)
class ResultBatch:
    """One strictly validated, row-oriented inversion result batch."""

    sample_id: NDArray[np.uint64]
    model_kind: NDArray[np.uint8]
    success: NDArray[np.bool_]
    status: NDArray[np.int32]
    iterations: NDArray[np.int32]
    evaluations: NDArray[np.int32]
    initial_objective: NDArray[np.float64]
    final_objective: NDArray[np.float64]
    data_misfit: NDArray[np.float64]
    regularization: NDArray[np.float64]
    reference_vs: NDArray[np.float32]
    inverted_vs: NDArray[np.float32]
    observed_phase_velocity: NDArray[np.float32]
    surrogate_phase_velocity: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    failure_code: NDArray[np.bytes_]
    ensemble_vs: NDArray[np.float32] | None = None
    ensemble_success: NDArray[np.bool_] | None = None
    ensemble_status: NDArray[np.int32] | None = None
    ensemble_iterations: NDArray[np.int32] | None = None
    ensemble_evaluations: NDArray[np.int32] | None = None
    ensemble_initial_objective: NDArray[np.float64] | None = None
    ensemble_objective: NDArray[np.float64] | None = None
    ensemble_failure_code: NDArray[np.bytes_] | None = None
    ensemble_message: NDArray[np.bytes_] | None = None
    ensemble_inlier_mask: NDArray[np.bool_] | None = None
    median_vs: NDArray[np.float32] | None = None
    p10_vs: NDArray[np.float32] | None = None
    p90_vs: NDArray[np.float32] | None = None
    physical_success: NDArray[np.bool_] | None = None
    physical_status: NDArray[np.int32] | None = None
    physical_failure_code: NDArray[np.bytes_] | None = None
    physical_phase_velocity: NDArray[np.float32] | None = None
    physical_valid_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, np.ndarray) or self.sample_id.ndim != 1:
            raise ValueError("sample_id must be a one-dimensional NumPy array")
        count = self.sample_id.shape[0]
        if count == 0:
            raise ValueError("result batches must contain at least one row")
        _require_array(self.sample_id, "sample_id", np.uint64, (count,))
        _require_array(self.model_kind, "model_kind", np.uint8, (count,))
        _require_array(self.success, "success", np.bool_, (count,))
        _require_array(self.status, "status", np.int32, (count,))
        _require_array(self.iterations, "iterations", np.int32, (count,))
        _require_array(self.evaluations, "evaluations", np.int32, (count,))
        for name in (
            "initial_objective",
            "final_objective",
            "data_misfit",
            "regularization",
        ):
            _require_array(getattr(self, name), name, np.float64, (count,))
        for name in ("reference_vs", "inverted_vs"):
            _require_array(getattr(self, name), name, np.float32, (count, 20))
        for name in ("observed_phase_velocity", "surrogate_phase_velocity"):
            _require_array(getattr(self, name), name, np.float32, (count, *CURVE_SHAPE))
        _require_array(self.valid_mask, "valid_mask", np.bool_, (count, *CURVE_SHAPE))
        if (
            not isinstance(self.failure_code, np.ndarray)
            or self.failure_code.shape != (count,)
            or self.failure_code.dtype.kind != "S"
            or not 1 <= self.failure_code.dtype.itemsize <= 128
        ):
            raise ValueError(
                "failure_code must have shape (N,) and a fixed-width bytes dtype"
            )
        self._validate_values()
        self._validate_optional_fields(count)

    def _validate_values(self) -> None:
        count = self.sample_id.shape[0]
        if count > 1 and not np.all(self.sample_id[1:] > self.sample_id[:-1]):
            raise ValueError("sample_id values must be unique and strictly ordered")
        if np.any(self.model_kind > 3):
            raise ValueError("model_kind contains an unknown model family")
        if np.any(self.iterations < 0) or np.any(self.evaluations < 0):
            raise ValueError("iterations and evaluations must be nonnegative")
        if not np.array_equal(
            self.valid_mask, np.isfinite(self.observed_phase_velocity)
        ):
            raise ValueError("observed_phase_velocity and valid_mask disagree")
        if not np.all(np.isnan(self.observed_phase_velocity[~self.valid_mask])):
            raise ValueError("invalid observed phase velocities must be NaN")
        if np.any(self.observed_phase_velocity[self.valid_mask] <= 0):
            raise ValueError("valid observed phase velocities must be positive")

        codes = _decode_failure_codes(self.failure_code)
        for row, (success, code) in enumerate(zip(self.success, codes, strict=True)):
            if bool(success) != (code == ""):
                raise ValueError(
                    f"failure_code must be empty exactly for successful row {row}"
                )

        reference_finite = _validate_profile_rows(self.reference_vs, "reference_vs")
        inverted_finite = _validate_profile_rows(self.inverted_vs, "inverted_vs")
        surrogate_finite = _row_is_all_finite(self.surrogate_phase_velocity)
        surrogate_nan = _row_is_all_nan(self.surrogate_phase_velocity)
        if not np.all(surrogate_finite | surrogate_nan):
            raise ValueError(
                "surrogate_phase_velocity rows must be entirely finite or entirely NaN"
            )

        diagnostics = np.column_stack(
            (
                self.initial_objective,
                self.final_objective,
                self.data_misfit,
                self.regularization,
            )
        )
        diagnostics_finite = np.all(np.isfinite(diagnostics), axis=1)
        diagnostics_nan = np.all(np.isnan(diagnostics), axis=1)
        if not np.all(diagnostics_finite | diagnostics_nan):
            raise ValueError(
                "objective diagnostic rows must be entirely finite or entirely NaN"
            )
        if np.any(diagnostics[diagnostics_finite] < 0):
            raise ValueError("objective diagnostics must be nonnegative")
        finite_rows = np.flatnonzero(diagnostics_finite)
        if finite_rows.size and not np.allclose(
            self.final_objective[finite_rows],
            self.data_misfit[finite_rows] + self.regularization[finite_rows],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "final_objective must equal the sum of data_misfit and regularization"
            )

        successful = self.success
        valid_success = (
            reference_finite & inverted_finite & surrogate_finite & diagnostics_finite
        )
        if not np.all(valid_success[successful]):
            raise ValueError("successful rows must contain finite bounded diagnostics")
        finite_failure = inverted_finite & surrogate_finite & diagnostics_finite
        missing_failure = ~inverted_finite & ~surrogate_finite & diagnostics_nan
        if not np.all((finite_failure | missing_failure)[~successful]):
            raise ValueError(
                "failed rows must retain complete finite diagnostics or complete NaNs"
            )

    def _validate_optional_fields(self, count: int) -> None:
        optional = [getattr(self, name) for name in _OPTIONAL_DATASETS]
        if all(value is None for value in optional):
            return
        if any(value is None for value in optional):
            raise ValueError(
                "deep optional fields must either all be present or all absent"
            )

        ensemble_vs = self.ensemble_vs
        assert ensemble_vs is not None
        if not isinstance(ensemble_vs, np.ndarray) or ensemble_vs.ndim != 3:
            raise ValueError("ensemble_vs must have shape (N, starts, 20)")
        starts = ensemble_vs.shape[1]
        if starts <= 0:
            raise ValueError("ensemble_vs must contain at least one start")
        _require_array(ensemble_vs, "ensemble_vs", np.float32, (count, starts, 20))
        expected_ensemble = {
            "ensemble_success": np.bool_,
            "ensemble_status": np.int32,
            "ensemble_iterations": np.int32,
            "ensemble_evaluations": np.int32,
            "ensemble_initial_objective": np.float64,
            "ensemble_objective": np.float64,
            "ensemble_inlier_mask": np.bool_,
        }
        for name, dtype in expected_ensemble.items():
            _require_array(getattr(self, name), name, dtype, (count, starts))
        _require_bytes_array(
            self.ensemble_failure_code,
            "ensemble_failure_code",
            (count, starts),
            maximum_width=128,
        )
        _require_bytes_array(
            self.ensemble_message,
            "ensemble_message",
            (count, starts),
            maximum_width=512,
        )
        for name in ("median_vs", "p10_vs", "p90_vs"):
            _require_array(getattr(self, name), name, np.float32, (count, 20))
        _require_array(
            self.physical_success,
            "physical_success",
            np.bool_,
            (count,),
        )
        _require_array(
            self.physical_status,
            "physical_status",
            np.int32,
            (count,),
        )
        _require_bytes_array(
            self.physical_failure_code,
            "physical_failure_code",
            (count,),
            maximum_width=128,
        )
        _require_array(
            self.physical_phase_velocity,
            "physical_phase_velocity",
            np.float32,
            (count, *CURVE_SHAPE),
        )
        _require_array(
            self.physical_valid_mask,
            "physical_valid_mask",
            np.bool_,
            (count, *CURVE_SHAPE),
        )

        assert self.ensemble_success is not None
        assert self.ensemble_iterations is not None
        assert self.ensemble_evaluations is not None
        assert self.ensemble_initial_objective is not None
        assert self.ensemble_objective is not None
        assert self.ensemble_failure_code is not None
        assert self.ensemble_message is not None
        assert self.ensemble_inlier_mask is not None
        if np.any(self.ensemble_iterations < 0):
            raise ValueError("ensemble_iterations must be nonnegative")
        if np.any(self.ensemble_evaluations < 0):
            raise ValueError("ensemble_evaluations must be nonnegative")
        initial_finite = np.isfinite(self.ensemble_initial_objective)
        initial_nan = np.isnan(self.ensemble_initial_objective)
        if not np.all(initial_finite | initial_nan) or np.any(
            self.ensemble_initial_objective[initial_finite] < 0
        ):
            raise ValueError(
                "ensemble_initial_objective must contain nonnegative finite values or NaN"
            )
        start_codes = np.asarray(
            _decode_failure_codes(self.ensemble_failure_code)
        ).reshape(count, starts)
        _decode_messages(self.ensemble_message)
        if np.any(self.ensemble_success != (start_codes == "")):
            raise ValueError(
                "ensemble_failure_code must be empty exactly for successful starts"
            )
        flat_vs = ensemble_vs.reshape(count * starts, 20)
        vs_finite = _validate_profile_rows(flat_vs, "ensemble_vs").reshape(
            count, starts
        )
        objective_finite = np.isfinite(self.ensemble_objective)
        objective_nan = np.isnan(self.ensemble_objective)
        if np.any(self.ensemble_objective[objective_finite] < 0):
            raise ValueError("ensemble_objective must be nonnegative")
        if not np.all(objective_finite | objective_nan):
            raise ValueError("ensemble_objective must not contain infinity")
        if not np.all(vs_finite[self.ensemble_success]):
            raise ValueError("successful ensemble starts must contain bounded Vs")
        if not np.all(objective_finite[self.ensemble_success]):
            raise ValueError("successful ensemble starts need finite objectives")
        if not np.all(initial_finite[self.ensemble_success]):
            raise ValueError(
                "successful ensemble starts need finite initial objectives"
            )
        failed = ~self.ensemble_success
        failed_consistent = (vs_finite & objective_finite) | (
            ~vs_finite & objective_nan
        )
        if not np.all(failed_consistent[failed]):
            raise ValueError("failed ensemble start diagnostics are inconsistent")
        if np.any(self.ensemble_inlier_mask & ~self.ensemble_success):
            raise ValueError("ensemble inliers must be successful starts")
        inlier_counts = np.count_nonzero(self.ensemble_inlier_mask, axis=1)
        if np.any(self.success & (inlier_counts == 0)):
            raise ValueError(
                "successful deep rows require a successful ensemble inlier"
            )

        assert self.median_vs is not None
        assert self.p10_vs is not None
        assert self.p90_vs is not None
        median_finite = _validate_profile_rows(self.median_vs, "median_vs")
        p10_finite = _validate_profile_rows(self.p10_vs, "p10_vs")
        p90_finite = _validate_profile_rows(self.p90_vs, "p90_vs")
        if not np.array_equal(median_finite, p10_finite) or not np.array_equal(
            median_finite, p90_finite
        ):
            raise ValueError("median_vs, p10_vs, and p90_vs must be present together")
        if np.any(self.p10_vs[median_finite] > self.median_vs[median_finite]) or np.any(
            self.median_vs[median_finite] > self.p90_vs[median_finite]
        ):
            raise ValueError("deep percentile profiles are not ordered")
        if not np.all(median_finite[self.success]):
            raise ValueError("successful deep rows require finite percentile profiles")

        assert self.physical_phase_velocity is not None
        assert self.physical_valid_mask is not None
        assert self.physical_success is not None
        assert self.physical_status is not None
        assert self.physical_failure_code is not None
        if not np.array_equal(
            self.physical_valid_mask,
            np.isfinite(self.physical_phase_velocity),
        ):
            raise ValueError("physical_phase_velocity and physical_valid_mask disagree")
        if not np.all(
            np.isnan(self.physical_phase_velocity[~self.physical_valid_mask])
        ):
            raise ValueError("invalid physical phase velocities must be NaN")
        if np.any(self.physical_phase_velocity[self.physical_valid_mask] <= 0):
            raise ValueError("valid physical phase velocities must be positive")
        physical_rows = np.any(self.physical_valid_mask.reshape(count, -1), axis=1)
        physical_codes = np.asarray(_decode_failure_codes(self.physical_failure_code))
        if np.any(self.physical_success != (physical_codes == "")):
            raise ValueError(
                "physical_failure_code must be empty exactly for successful physical rows"
            )
        if np.any(self.physical_success & ~self.success):
            raise ValueError("physical success requires inversion success")
        if not np.all(physical_rows[self.physical_success]):
            raise ValueError(
                "successful physical rows require at least one reconstructed cell"
            )
        if np.any(physical_rows[~self.physical_success]):
            raise ValueError("failed physical rows must not publish physical cells")
        if np.any(self.physical_status[self.physical_success] != 0):
            raise ValueError("successful physical rows require status zero")
        if np.any(self.physical_status[~self.physical_success] >= 0):
            raise ValueError("failed physical rows require a negative status")
        not_attempted = ~self.success
        if np.any(physical_codes[not_attempted] != "not_attempted") or np.any(
            self.physical_status[not_attempted] != -2
        ):
            raise ValueError(
                "failed inversions require a canonical not-attempted physical outcome"
            )
        attempted_failure = self.success & ~self.physical_success
        if np.any(physical_codes[attempted_failure] == "not_attempted"):
            raise ValueError(
                "successful inversions with physical failure require a failure code"
            )


def _reject_symlink(path: Path, description: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link")


def _reject_symlinked_directory_components(directory: Path) -> None:
    current = Path(directory.anchor) if directory.is_absolute() else Path.cwd()
    parts = directory.parts[1:] if directory.is_absolute() else directory.parts
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        if current.is_symlink():
            raise ValueError("result directory path must not contain a symbolic link")


def _result_directory(path: Path | str, *, create: bool) -> Path:
    directory = Path(path)
    _reject_symlinked_directory_components(directory)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("result directory does not exist or is not a directory")
    return directory


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _locked(directory: Path, name: str) -> Iterator[None]:
    path = directory / name
    _reject_symlink(path, "lock file")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("lock file changed while it was opened")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _file_sha256(path: Path | str) -> str:
    source = Path(path)
    _reject_symlink(source, "checksummed file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("checksummed path must be a regular file")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def checkpoint_sha256(path: Path | str) -> str:
    """Return the immutable SHA-256 identity of a regular checkpoint file."""
    try:
        return _file_sha256(path)
    except OSError as error:
        raise ValueError("checkpoint is not a readable regular file") from error


def _manifest_payload(manifest: ResultManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["expected_jobs"] = list(manifest.expected_jobs)
    payload["completed_jobs"] = list(manifest.completed_jobs)
    return payload


def _write_manifest(path: Path, manifest: ResultManifest) -> None:
    _reject_symlink(path, "manifest")
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    _reject_symlink(temporary, "temporary manifest")
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(
                _manifest_payload(manifest),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> ResultManifest:
    _reject_symlink(path, "manifest")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(
                handle, object_pairs_hook=_object_without_duplicate_keys
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("result manifest is not readable JSON") from error
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        raise ValueError(
            "result manifest schema version 2 is unsupported; create a new "
            "schema v3 result directory"
        )
    expected_fields = {field.name for field in fields(ResultManifest)}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("result manifest fields do not match the schema")
    if not isinstance(payload["expected_jobs"], list) or not isinstance(
        payload["completed_jobs"], list
    ):
        raise ValueError(  # noqa: TRY004 - malformed external JSON
            "result manifest job collections must be JSON arrays"
        )
    if not isinstance(payload["job_sha256"], dict):
        raise ValueError(  # noqa: TRY004 - malformed external JSON
            "result manifest job_sha256 must be a JSON object"
        )
    try:
        return ResultManifest(
            schema_version=payload["schema_version"],
            dataset_config_hash=payload["dataset_config_hash"],
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            split_policy=payload["split_policy"],
            inversion_config_hash=payload["inversion_config_hash"],
            minimum_valid_solutions=payload["minimum_valid_solutions"],
            experiment=payload["experiment"],
            expected_jobs=tuple(payload["expected_jobs"]),
            expected_job_sample_count=dict(payload["expected_job_sample_count"]),
            expected_job_sample_id_sha256=dict(
                payload["expected_job_sample_id_sha256"]
            ),
            completed_jobs=tuple(payload["completed_jobs"]),
            job_sha256=dict(payload["job_sha256"]),
            package_version=payload["package_version"],
            software_sha256=payload["software_sha256"],
            created_at=payload["created_at"],
            complete=payload["complete"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"result manifest is invalid: {error}") from error


def _identity(manifest: ResultManifest) -> tuple[object, ...]:
    return (
        manifest.schema_version,
        manifest.dataset_config_hash,
        manifest.dataset_manifest_sha256,
        manifest.checkpoint_sha256,
        manifest.split_policy,
        manifest.inversion_config_hash,
        manifest.minimum_valid_solutions,
        manifest.experiment,
        manifest.expected_jobs,
        tuple(sorted(manifest.expected_job_sample_count.items())),
        tuple(sorted(manifest.expected_job_sample_id_sha256.items())),
        manifest.software_sha256,
    )


def _expected_population_identity(
    expected_jobs: tuple[str, ...],
    expected_sample_ids_by_job: Mapping[str, NDArray[np.uint64]],
) -> tuple[dict[str, int], dict[str, str]]:
    if not isinstance(expected_sample_ids_by_job, Mapping) or set(
        expected_sample_ids_by_job
    ) != set(expected_jobs):
        raise ValueError(
            "expected_sample_ids_by_job keys must exactly match expected_jobs"
        )
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for job in expected_jobs:
        values = np.asarray(expected_sample_ids_by_job[job])
        digest = sample_id_sha256(values)
        counts[job] = int(values.size)
        digests[job] = digest
    return counts, digests


def initialize_result_manifest(
    results_dir: Path | str,
    *,
    dataset_config_hash: str,
    dataset_manifest_sha256: str,
    checkpoint: Path | str,
    inversion_config_hash: str | None = None,
    config: InversionConfig | None = None,
    minimum_valid_solutions: int | None = None,
    experiment: str,
    expected_jobs: tuple[str, ...],
    expected_sample_ids_by_job: Mapping[str, NDArray[np.uint64]],
) -> ResultManifest:
    """Create or safely resume the exact scientific result identity."""
    directory = _result_directory(results_dir, create=True)
    _require_hash(dataset_config_hash, "dataset_config_hash")
    _require_hash(dataset_manifest_sha256, "dataset_manifest_sha256")
    if experiment not in {"full", "deep", "both"}:
        raise ValueError("experiment must be full, deep, or both")
    if experiment in {"deep", "both"}:
        if config is None or inversion_config_hash is not None:
            raise ValueError(
                "config is required for deep-capable result identity; "
                "a precomputed hash cannot verify minimum_valid_solutions"
            )
        config_digest = inversion_identity_hash(config)
        if (
            minimum_valid_solutions is not None
            and minimum_valid_solutions != config.minimum_valid_solutions
        ):
            raise ValueError("minimum_valid_solutions does not match inversion config")
        configured_minimum = config.minimum_valid_solutions
    else:
        if (inversion_config_hash is None) == (config is None):
            raise ValueError("provide exactly one of inversion_config_hash or config")
        if minimum_valid_solutions is not None:
            raise ValueError(
                "full result identity requires minimum_valid_solutions to be None"
            )
        config_digest = (
            inversion_identity_hash(config)
            if config is not None
            else _require_hash(inversion_config_hash, "inversion_config_hash")
        )
        configured_minimum = None
    assert config_digest is not None
    expected_counts, expected_digests = _expected_population_identity(
        expected_jobs, expected_sample_ids_by_job
    )
    checkpoint_digest = checkpoint_sha256(checkpoint)
    candidate = ResultManifest(
        schema_version=SCHEMA_VERSION,
        dataset_config_hash=dataset_config_hash,
        dataset_manifest_sha256=dataset_manifest_sha256,
        checkpoint_sha256=checkpoint_digest,
        split_policy=SPLIT_POLICY,
        inversion_config_hash=config_digest,
        minimum_valid_solutions=configured_minimum,
        experiment=experiment,
        expected_jobs=expected_jobs,
        expected_job_sample_count=expected_counts,
        expected_job_sample_id_sha256=expected_digests,
        completed_jobs=(),
        job_sha256={},
        package_version=__version__,
        software_sha256=software_sha256(),
        created_at=datetime.now(UTC).isoformat(),
        complete=False,
    )
    manifest_path = directory / "manifest.json"
    with _locked(directory, "manifest.lock"):
        if manifest_path.exists() or manifest_path.is_symlink():
            existing = _load_manifest(manifest_path)
            if _identity(existing) != _identity(candidate):
                raise ValueError("existing result manifest identity does not match")
            return existing
        _write_manifest(manifest_path, candidate)
    return candidate


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _content_sha256(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in sorted(handle.keys()):
        link = handle.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(  # noqa: TRY004 - malformed shard content
                f"result dataset {name} must not be a symbolic link"
            )
        dataset = handle[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(  # noqa: TRY004 - malformed shard content
                f"result member {name} is not a dataset"
            )
        values = np.asarray(dataset)
        metadata = json.dumps(
            [name, values.dtype.str, list(values.shape)],
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _write_batch(handle: h5py.File, batch: ResultBatch) -> None:
    for name in _REQUIRED_DATASETS:
        values = getattr(batch, name)
        handle.create_dataset(name, data=values, dtype=values.dtype)
    if batch.ensemble_vs is not None:
        for name in _OPTIONAL_DATASETS:
            values = getattr(batch, name)
            assert values is not None
            handle.create_dataset(name, data=values, dtype=values.dtype)


def _validate_shard_identity(
    handle: h5py.File, manifest: ResultManifest | None
) -> tuple[str, int | None, int, str]:
    if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("result shard schema version is invalid")
    for name in _IDENTITY_ATTRS[1:]:
        if name not in handle.attrs or not _attribute_text(handle.attrs[name]):
            raise ValueError(f"result shard is missing identity attribute {name}")
    if _attribute_text(handle.attrs["split_policy"]) != SPLIT_POLICY:
        raise ValueError("result shard split policy does not match")
    _require_hash(
        _attribute_text(handle.attrs["dataset_config_hash"]),
        "dataset_config_hash",
    )
    _require_hash(
        _attribute_text(handle.attrs["dataset_manifest_sha256"]),
        "dataset_manifest_sha256",
    )
    _require_hash(
        _attribute_text(handle.attrs["checkpoint_sha256"]),
        "checkpoint_sha256",
    )
    _require_hash(
        _attribute_text(handle.attrs["inversion_config_hash"]),
        "inversion_config_hash",
    )
    _require_hash(
        _attribute_text(handle.attrs["software_sha256"]),
        "software_sha256",
    )
    job = _require_job_name(_attribute_text(handle.attrs.get("job_name", "")))
    raw_expected_count = handle.attrs.get("expected_sample_count")
    if (
        isinstance(raw_expected_count, (bool, np.bool_))
        or not isinstance(raw_expected_count, (int, np.integer))
        or int(raw_expected_count) <= 0
    ):
        raise ValueError("result shard expected_sample_count is invalid")
    expected_count = int(raw_expected_count)
    expected_sample_digest = _require_hash(
        _attribute_text(handle.attrs.get("expected_sample_id_sha256", "")),
        "expected_sample_id_sha256",
    )
    raw_minimum = handle.attrs.get("minimum_valid_solutions")
    if raw_minimum is not None and (
        isinstance(raw_minimum, (bool, np.bool_))
        or not isinstance(raw_minimum, (int, np.integer))
    ):
        raise ValueError("result shard minimum_valid_solutions must be an integer")
    shard_minimum = None if raw_minimum is None else int(raw_minimum)
    if shard_minimum is not None and shard_minimum <= 0:
        raise ValueError("result shard minimum_valid_solutions must be positive")
    if _job_requires_deep_schema(job, manifest) and shard_minimum is None:
        raise ValueError(
            "deep result shard is missing minimum_valid_solutions identity"
        )
    if manifest is None:
        return job, shard_minimum, expected_count, expected_sample_digest
    expected_attrs = {
        "schema_version": manifest.schema_version,
        "dataset_config_hash": manifest.dataset_config_hash,
        "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "split_policy": manifest.split_policy,
        "inversion_config_hash": manifest.inversion_config_hash,
        "experiment": manifest.experiment,
        "package_version": manifest.package_version,
        "software_sha256": manifest.software_sha256,
    }
    for name, expected in expected_attrs.items():
        actual: object = handle.attrs.get(name, "")
        if isinstance(expected, str):
            actual = _attribute_text(actual)
        else:
            actual = int(actual)
        if actual != expected:
            raise ValueError(f"result shard identity attribute {name} does not match")
    if job not in manifest.expected_jobs:
        raise ValueError("result shard job is not expected by the manifest")
    if shard_minimum != manifest.minimum_valid_solutions:
        raise ValueError("result shard minimum_valid_solutions identity does not match")
    if expected_count != manifest.expected_job_sample_count[job]:
        raise ValueError("result shard expected sample count does not match")
    if expected_sample_digest != manifest.expected_job_sample_id_sha256[job]:
        raise ValueError("result shard expected sample identity does not match")
    return job, shard_minimum, expected_count, expected_sample_digest


def _job_requires_deep_schema(job: str, manifest: ResultManifest | None) -> bool:
    job_experiment = _job_experiment(job)
    if manifest is not None:
        allowed = (
            {"full", "deep"} if manifest.experiment == "both" else {manifest.experiment}
        )
        if job_experiment not in allowed:
            raise ValueError("result manifest experiment and job schema disagree")
    return job_experiment == "deep"


def _validate_batch_schema_for_job(
    job: str,
    batch: ResultBatch,
    manifest: ResultManifest | None,
    minimum_valid_solutions: int | None,
) -> None:
    requires_deep = _job_requires_deep_schema(job, manifest)
    has_deep = batch.ensemble_vs is not None
    if requires_deep and not has_deep:
        raise ValueError("deep result jobs require all deep optional fields")
    if not requires_deep and has_deep:
        raise ValueError("full result jobs must not contain deep optional fields")
    if not requires_deep:
        return
    if minimum_valid_solutions is None:
        raise ValueError("deep result identity requires minimum_valid_solutions")
    assert batch.ensemble_inlier_mask is not None
    starts = batch.ensemble_inlier_mask.shape[1]
    if minimum_valid_solutions > starts:
        raise ValueError(
            "minimum_valid_solutions exceeds the stored ensemble start count"
        )
    inlier_counts = np.count_nonzero(batch.ensemble_inlier_mask, axis=1)
    if np.any(batch.success & (inlier_counts < minimum_valid_solutions)):
        raise ValueError("successful deep rows do not meet minimum_valid_solutions")
    codes = _decode_failure_codes(batch.failure_code)
    for row, code in enumerate(codes):
        if code != "insufficient_valid_solutions":
            continue
        if inlier_counts[row] >= minimum_valid_solutions:
            raise ValueError(
                "insufficient_valid_solutions disagrees with ensemble inliers"
            )
        assert batch.median_vs is not None
        assert batch.p10_vs is not None
        assert batch.p90_vs is not None
        assert batch.physical_valid_mask is not None
        if (
            np.any(np.isfinite(batch.median_vs[row]))
            or np.any(np.isfinite(batch.p10_vs[row]))
            or np.any(np.isfinite(batch.p90_vs[row]))
            or np.any(batch.physical_valid_mask[row])
        ):
            raise ValueError(
                "insufficient_valid_solutions must not publish a deep summary"
            )


def validate_result_shard(
    path: Path | str,
    *,
    expected_sample_ids: NDArray[np.uint64] | None = None,
    manifest: ResultManifest | None = None,
    expected_sha256: str | None = None,
) -> ResultBatch:
    """Validate one shard's identity, schema, content digest, and diagnostics."""
    source = Path(path)
    _reject_symlink(source, "result shard")
    try:
        with h5py.File(source, "r") as handle:
            (
                job,
                shard_minimum,
                expected_sample_count,
                expected_sample_digest,
            ) = _validate_shard_identity(handle, manifest)
            published_name = f"{job}.h5"
            temporary_pattern = rf"{re.escape(published_name)}\.tmp-[0-9]+"
            if (
                source.name != published_name
                and re.fullmatch(temporary_pattern, source.name) is None
            ):
                raise ValueError("result shard job name does not match its file name")
            actual_names = set(handle.keys())
            required = set(_REQUIRED_DATASETS)
            optional = set(_OPTIONAL_DATASETS)
            if not required.issubset(actual_names):
                missing = sorted(required - actual_names)
                raise ValueError(f"result shard is missing datasets: {missing}")
            extra = actual_names - required - optional
            present_optional = actual_names & optional
            if extra or (present_optional and present_optional != optional):
                raise ValueError("result shard dataset fields do not match the schema")
            sample_ids = np.asarray(handle["sample_id"])
            stored_sample_hash = _attribute_text(
                handle.attrs.get("sample_id_sha256", "")
            )
            actual_sample_hash = hashlib.sha256(
                np.ascontiguousarray(sample_ids).tobytes()
            ).hexdigest()
            if stored_sample_hash != actual_sample_hash:
                raise ValueError("result shard sample_id checksum is invalid")
            stored_content_hash = _attribute_text(
                handle.attrs.get("content_sha256", "")
            )
            actual_content_hash = _content_sha256(handle)
            if stored_content_hash != actual_content_hash:
                raise ValueError("result shard content checksum is invalid")
            values = {name: np.asarray(handle[name]) for name in actual_names}
    except OSError as error:
        raise ValueError("result shard is not a readable HDF5 file") from error

    try:
        batch = ResultBatch(
            **{name: values.get(name) for name in _ALL_DATASETS}  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"result shard content is invalid: {error}") from error
    _validate_batch_schema_for_job(job, batch, manifest, shard_minimum)
    if batch.sample_id.size != expected_sample_count:
        raise ValueError("result shard sample count does not match its identity")
    if sample_id_sha256(batch.sample_id) != expected_sample_digest:
        raise ValueError("result shard sample identity does not match")
    if expected_sample_ids is not None:
        expected = np.asarray(expected_sample_ids)
        if expected.dtype != np.dtype(np.uint64) or expected.ndim != 1:
            raise ValueError("expected_sample_ids must be a uint64 vector")
        if not np.array_equal(batch.sample_id, expected):
            raise ValueError("result shard sample_id values do not match")
    if expected_sha256 is not None:
        _require_hash(expected_sha256, "expected_sha256")
        if _file_sha256(source) != expected_sha256:
            raise ValueError("result shard file checksum does not match")
    return batch


def _manifest_matches_disk(directory: Path, manifest: ResultManifest) -> None:
    current = _load_manifest(directory / "manifest.json")
    if _identity(current) != _identity(manifest):
        raise ValueError("provided result manifest identity does not match disk")


def _require_current_software(manifest: ResultManifest) -> None:
    if manifest.software_sha256 != software_sha256():
        raise ValueError(
            "result manifest software identity does not match this checkout"
        )


def write_result_shard(
    results_dir: Path | str,
    job_name: str,
    batch: ResultBatch,
    manifest: ResultManifest,
) -> Path:
    """Validate, atomically publish, and idempotently resume one HDF5 shard."""
    directory = _result_directory(results_dir, create=False)
    job = _require_job_name(job_name)
    if job not in manifest.expected_jobs:
        raise ValueError("job is not expected by the result manifest")
    _require_current_software(manifest)
    if not isinstance(batch, ResultBatch):
        raise TypeError("batch must be a ResultBatch")
    _validate_batch_schema_for_job(
        job, batch, manifest, manifest.minimum_valid_solutions
    )
    target = directory / f"{job}.h5"
    temporary = directory / f"{target.name}.tmp-{os.getpid()}"
    with _locked(directory, f".{job}.lock"):
        _manifest_matches_disk(directory, manifest)
        _reject_symlink(target, "result shard")
        _reject_symlink(temporary, "temporary result shard")
        if temporary.exists():
            if not temporary.is_file():
                raise ValueError("temporary result path is not a regular file")
            temporary.unlink()
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["schema_version"] = manifest.schema_version
                handle.attrs["dataset_config_hash"] = manifest.dataset_config_hash
                handle.attrs["dataset_manifest_sha256"] = (
                    manifest.dataset_manifest_sha256
                )
                handle.attrs["checkpoint_sha256"] = manifest.checkpoint_sha256
                handle.attrs["split_policy"] = manifest.split_policy
                handle.attrs["inversion_config_hash"] = manifest.inversion_config_hash
                if manifest.minimum_valid_solutions is not None:
                    handle.attrs["minimum_valid_solutions"] = (
                        manifest.minimum_valid_solutions
                    )
                handle.attrs["experiment"] = manifest.experiment
                handle.attrs["package_version"] = manifest.package_version
                handle.attrs["software_sha256"] = manifest.software_sha256
                handle.attrs["job_name"] = job
                handle.attrs["expected_sample_count"] = (
                    manifest.expected_job_sample_count[job]
                )
                handle.attrs["expected_sample_id_sha256"] = (
                    manifest.expected_job_sample_id_sha256[job]
                )
                _write_batch(handle, batch)
                handle.attrs["sample_id_sha256"] = hashlib.sha256(
                    batch.sample_id.tobytes()
                ).hexdigest()
                handle.attrs["content_sha256"] = _content_sha256(handle)
                handle.flush()
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            validate_result_shard(
                temporary,
                expected_sample_ids=batch.sample_id,
                manifest=manifest,
            )
            with h5py.File(temporary, "r") as handle:
                new_content_hash = _attribute_text(handle.attrs["content_sha256"])
            if target.exists():
                validate_result_shard(
                    target,
                    expected_sample_ids=batch.sample_id,
                    manifest=manifest,
                )
                with h5py.File(target, "r") as handle:
                    old_content_hash = _attribute_text(handle.attrs["content_sha256"])
                if old_content_hash != new_content_hash:
                    raise ValueError("existing result shard has conflicting content")
                temporary.unlink()
                return target
            temporary.replace(target)
            _fsync_directory(directory)
            return target
        except BaseException:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
            raise


def _shard_path(directory: Path, job: str, path: Path | str | None) -> Path:
    expected = directory / f"{job}.h5"
    if path is None:
        return expected
    supplied = Path(path)
    if (
        supplied.parent.resolve() != directory.resolve()
        or supplied.name != expected.name
    ):
        raise ValueError("result shard path does not match its job name")
    return supplied


def mark_job_complete(
    results_dir: Path | str,
    job_name: str,
    shard_path: Path | str | None = None,
) -> ResultManifest:
    """Add one validated job/checksum under an exclusive manifest lock."""
    directory = _result_directory(results_dir, create=False)
    job = _require_job_name(job_name)
    path = _shard_path(directory, job, shard_path)
    with _locked(directory, "manifest.lock"):
        manifest = _load_manifest(directory / "manifest.json")
        _require_current_software(manifest)
        if job not in manifest.expected_jobs:
            raise ValueError("job is not expected by the result manifest")
        validate_result_shard(path, manifest=manifest)
        digest = _file_sha256(path)
        if job in manifest.completed_jobs:
            if manifest.job_sha256[job] != digest:
                raise ValueError("completed job checksum conflicts with manifest")
            return manifest
        checksums = dict(manifest.job_sha256)
        checksums[job] = digest
        completed = tuple(
            expected for expected in manifest.expected_jobs if expected in checksums
        )
        updated = ResultManifest(
            schema_version=manifest.schema_version,
            dataset_config_hash=manifest.dataset_config_hash,
            dataset_manifest_sha256=manifest.dataset_manifest_sha256,
            checkpoint_sha256=manifest.checkpoint_sha256,
            split_policy=manifest.split_policy,
            inversion_config_hash=manifest.inversion_config_hash,
            minimum_valid_solutions=manifest.minimum_valid_solutions,
            experiment=manifest.experiment,
            expected_jobs=manifest.expected_jobs,
            expected_job_sample_count=manifest.expected_job_sample_count,
            expected_job_sample_id_sha256=(manifest.expected_job_sample_id_sha256),
            completed_jobs=completed,
            job_sha256={name: checksums[name] for name in completed},
            package_version=manifest.package_version,
            software_sha256=manifest.software_sha256,
            created_at=manifest.created_at,
            complete=completed == manifest.expected_jobs,
        )
        _write_manifest(directory / "manifest.json", updated)
        return updated


def _canonical_job_group(job: str) -> str:
    match = re.fullmatch(
        r"(full)-(clean|noise_1pct)-shard-[0-9]+"
        r"|(deep)-(clean|noise_1pct)-samples-[0-9]{20}-[0-9]{20}-[0-9a-f]{12}",
        job,
    )
    if match is None:
        return "__generic__"
    experiment = match.group(1) or match.group(3)
    noise = match.group(2) or match.group(4)
    assert experiment is not None and noise is not None
    return f"{experiment}-{noise}"


def validate_complete_results(results_dir: Path | str) -> ResultManifest:
    """Verify the exact complete manifest, file set, hashes, and shard contents."""
    directory = _result_directory(results_dir, create=False)
    manifest = _load_manifest(directory / "manifest.json")
    _require_current_software(manifest)
    if not manifest.complete:
        raise ValueError("result manifest is incomplete")
    if manifest.completed_jobs != manifest.expected_jobs:
        raise ValueError("result manifest has an incomplete job index")
    expected_paths = {directory / f"{job}.h5" for job in manifest.expected_jobs}
    actual_paths = set(directory.glob("*.h5"))
    if actual_paths != expected_paths:
        raise ValueError("result HDF5 files include missing or unexpected jobs")

    sample_ids_by_group: dict[str, set[int]] = {}
    for job in manifest.expected_jobs:
        path = directory / f"{job}.h5"
        _reject_symlink(path, "result shard")
        digest = manifest.job_sha256.get(job)
        if digest is None:
            raise ValueError("result manifest has incomplete job checksums")
        batch = validate_result_shard(
            path,
            manifest=manifest,
            expected_sha256=digest,
        )
        group = _canonical_job_group(job)
        seen = sample_ids_by_group.setdefault(group, set())
        current = {int(value) for value in batch.sample_id}
        if seen & current:
            raise ValueError(f"duplicate sample_id values in result group {group}")
        seen.update(current)
    return manifest
