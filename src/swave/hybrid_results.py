"""Strict, independently versioned artifacts for hybrid inversion runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from . import __version__
from .config import HybridInversionConfig, hybrid_inversion_identity_hash
from .inversion_results import checkpoint_sha256, sample_id_sha256, software_sha256
from .splits import SPLIT_POLICY

HybridSplit = Literal["test", "inversion"]


def _require_array(
    value: object,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
) -> NDArray[Any]:
    expected = np.dtype(dtype)
    if not isinstance(value, np.ndarray) or value.dtype != expected or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype {expected}")
    return value


def _decode_codes(values: NDArray[np.bytes_]) -> NDArray[np.str_]:
    decoded: list[str] = []
    for raw in values:
        try:
            text = bytes(raw).decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("failure codes must contain ASCII") from error
        if any(not (character.isalnum() or character == "_") for character in text):
            raise ValueError("failure codes contain unsafe characters")
        decoded.append(text)
    return np.asarray(decoded)


@dataclass(frozen=True)
class HybridResultBatch:
    """One source-shard/noise batch with paired control and hybrid outcomes."""

    sample_id: NDArray[np.uint64]
    model_kind: NDArray[np.uint8]
    valid_mask: NDArray[np.bool_]
    observed_phase_velocity: NDArray[np.float32]
    reference_vs: NDArray[np.float32]
    supervised_vs: NDArray[np.float32]
    sensitivity: NDArray[np.float64]
    prior_weights: NDArray[np.float64]
    control_success: NDArray[np.bool_]
    control_status: NDArray[np.int32]
    control_failure_code: NDArray[np.bytes_]
    control_iterations: NDArray[np.int32]
    control_evaluations: NDArray[np.int32]
    control_initial_objective: NDArray[np.float64]
    control_total: NDArray[np.float64]
    control_data_misfit: NDArray[np.float64]
    control_smoothness: NDArray[np.float64]
    control_vs: NDArray[np.float32]
    control_prediction: NDArray[np.float32]
    hybrid_success: NDArray[np.bool_]
    hybrid_status: NDArray[np.int32]
    hybrid_failure_code: NDArray[np.bytes_]
    hybrid_iterations: NDArray[np.int32]
    hybrid_evaluations: NDArray[np.int32]
    hybrid_initial_objective: NDArray[np.float64]
    hybrid_total: NDArray[np.float64]
    hybrid_data_misfit: NDArray[np.float64]
    hybrid_smoothness: NDArray[np.float64]
    hybrid_learning_prior: NDArray[np.float64]
    hybrid_vs: NDArray[np.float32]
    hybrid_prediction: NDArray[np.float32]

    def validate(self) -> None:
        if not isinstance(self.sample_id, np.ndarray) or self.sample_id.dtype != np.uint64:
            raise ValueError("sample_id must be a uint64 vector")
        count = len(self.sample_id)
        if self.sample_id.shape != (count,) or count == 0:
            raise ValueError("sample_id must be a nonempty vector")
        sample_id_sha256(self.sample_id)
        _require_array(self.model_kind, "model_kind", np.uint8, (count,))
        if np.any(self.model_kind > 3):
            raise ValueError("model_kind contains an unsupported value")
        _require_array(self.valid_mask, "valid_mask", np.bool_, (count, 4, 120))
        _require_array(
            self.observed_phase_velocity,
            "observed_phase_velocity",
            np.float32,
            (count, 4, 120),
        )
        if not np.all(np.isfinite(self.observed_phase_velocity[self.valid_mask])):
            raise ValueError("valid observations must be finite")
        if np.any(self.observed_phase_velocity[self.valid_mask] <= 0):
            raise ValueError("valid observations must be positive")
        for name in ("reference_vs", "supervised_vs"):
            values = _require_array(getattr(self, name), name, np.float32, (count, 20))
            if not np.all(np.isfinite(values)) or np.any((values < 0.3) | (values > 2.6)):
                raise ValueError(f"{name} must contain finite globally bounded profiles")
        sensitivity = _require_array(
            self.sensitivity, "sensitivity", np.float64, (count, 20)
        )
        weights = _require_array(
            self.prior_weights, "prior_weights", np.float64, (count, 20)
        )
        if (
            not np.all(np.isfinite(sensitivity))
            or np.any(sensitivity < 0)
            or np.any(np.all(sensitivity == 0, axis=1))
        ):
            raise ValueError("sensitivity rows must be finite and nonzero")
        if not np.all(np.isfinite(weights)) or np.any(
            (weights < 0.25) | (weights > 4.0)
        ):
            raise ValueError("prior_weights must stay within [0.25, 4.0]")
        if not np.allclose(weights.mean(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("prior_weights must have unit row means")
        self._validate_outcome("control", has_learning_prior=False)
        self._validate_outcome("hybrid", has_learning_prior=True)

    def _validate_outcome(self, prefix: str, *, has_learning_prior: bool) -> None:
        count = len(self.sample_id)
        success = _require_array(
            getattr(self, f"{prefix}_success"),
            f"{prefix}_success",
            np.bool_,
            (count,),
        )
        _require_array(
            getattr(self, f"{prefix}_status"),
            f"{prefix}_status",
            np.int32,
            (count,),
        )
        codes = getattr(self, f"{prefix}_failure_code")
        if (
            not isinstance(codes, np.ndarray)
            or codes.shape != (count,)
            or codes.dtype.kind != "S"
            or codes.dtype.itemsize > 64
        ):
            raise ValueError(f"{prefix}_failure_code has an invalid schema")
        decoded = _decode_codes(codes)
        if np.any(success != (decoded == "")):
            raise ValueError(f"{prefix}_failure_code disagrees with success")
        for suffix in ("iterations", "evaluations"):
            values = _require_array(
                getattr(self, f"{prefix}_{suffix}"),
                f"{prefix}_{suffix}",
                np.int32,
                (count,),
            )
            if np.any(values < 0):
                raise ValueError(f"{prefix}_{suffix} must be nonnegative")
        scalar_names = [
            "initial_objective",
            "total",
            "data_misfit",
            "smoothness",
        ]
        if has_learning_prior:
            scalar_names.append("learning_prior")
        scalars: dict[str, NDArray[np.float64]] = {}
        for suffix in scalar_names:
            scalars[suffix] = _require_array(
                getattr(self, f"{prefix}_{suffix}"),
                f"{prefix}_{suffix}",
                np.float64,
                (count,),
            )
        scientific_finite = success
        for suffix, values in scalars.items():
            if not np.all(np.isfinite(values[scientific_finite])) or np.any(
                values[scientific_finite] < 0
            ):
                raise ValueError(f"{prefix}_{suffix} is invalid for successful rows")
        expected_total = scalars["data_misfit"] + scalars["smoothness"]
        if has_learning_prior:
            expected_total = expected_total + scalars["learning_prior"]
        if not np.allclose(
            scalars["total"][success],
            expected_total[success],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"{prefix}_total does not equal its objective terms")
        profile = _require_array(
            getattr(self, f"{prefix}_vs"),
            f"{prefix}_vs",
            np.float32,
            (count, 20),
        )
        prediction = _require_array(
            getattr(self, f"{prefix}_prediction"),
            f"{prefix}_prediction",
            np.float32,
            (count, 4, 120),
        )
        if not np.all(np.isfinite(profile[success])) or np.any(
            (profile[success] < 0.3) | (profile[success] > 2.6)
        ):
            raise ValueError(f"{prefix}_vs is invalid for successful rows")
        if not np.all(np.isfinite(prediction[success])):
            raise ValueError(f"{prefix}_prediction is invalid for successful rows")


@dataclass(frozen=True)
class HybridManifest:
    schema_version: int
    split: HybridSplit
    dataset_config_hash: str
    dataset_manifest_sha256: str
    forward_checkpoint_sha256: str
    supervised_checkpoint_sha256: tuple[str, ...]
    split_policy: str
    hybrid_config_hash: str
    selected_prior_lambda: float
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
        if self.schema_version != 1:
            raise ValueError("hybrid manifest schema_version must be one")
        if self.split not in {"test", "inversion"}:
            raise ValueError("hybrid manifest split is invalid")
        if not np.isfinite(self.selected_prior_lambda) or self.selected_prior_lambda <= 0:
            raise ValueError("selected_prior_lambda must be finite and positive")
        if not self.expected_jobs or len(set(self.expected_jobs)) != len(self.expected_jobs):
            raise ValueError("expected_jobs must be nonempty and unique")
        if set(self.completed_jobs) - set(self.expected_jobs):
            raise ValueError("completed_jobs contains an unexpected job")
        if self.complete != (set(self.completed_jobs) == set(self.expected_jobs)):
            raise ValueError("hybrid manifest complete flag is inconsistent")


def _manifest_payload(manifest: HybridManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload["supervised_checkpoint_sha256"] = list(
        manifest.supervised_checkpoint_sha256
    )
    payload["expected_jobs"] = list(manifest.expected_jobs)
    payload["completed_jobs"] = list(manifest.completed_jobs)
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_manifest(path: Path, manifest: HybridManifest) -> None:
    _atomic_json(path, _manifest_payload(manifest))


def _load_manifest(path: Path) -> HybridManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("hybrid manifest is not readable JSON") from error
    expected = {field.name for field in fields(HybridManifest)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("hybrid manifest fields do not match the schema")
    payload["supervised_checkpoint_sha256"] = tuple(
        payload["supervised_checkpoint_sha256"]
    )
    payload["expected_jobs"] = tuple(payload["expected_jobs"])
    payload["completed_jobs"] = tuple(payload["completed_jobs"])
    return HybridManifest(**payload)


def load_hybrid_manifest(output_dir: Path | str) -> HybridManifest:
    """Load and validate the hybrid manifest in one result directory."""
    return _load_manifest(Path(output_dir) / "manifest.json")


def _manifest_identity(manifest: HybridManifest) -> tuple[object, ...]:
    return (
        manifest.schema_version,
        manifest.split,
        manifest.dataset_config_hash,
        manifest.dataset_manifest_sha256,
        manifest.forward_checkpoint_sha256,
        manifest.supervised_checkpoint_sha256,
        manifest.split_policy,
        manifest.hybrid_config_hash,
        manifest.selected_prior_lambda,
        manifest.expected_jobs,
        tuple(sorted(manifest.expected_job_sample_count.items())),
        tuple(sorted(manifest.expected_job_sample_id_sha256.items())),
        manifest.software_sha256,
    )


def initialize_hybrid_manifest(
    output_dir: Path | str,
    *,
    split: HybridSplit,
    dataset_config_hash: str,
    dataset_manifest_sha256: str,
    forward_checkpoint: Path | str,
    supervised_checkpoint_sha256: tuple[str, ...],
    config: HybridInversionConfig,
    selected_prior_lambda: float,
    expected_sample_ids_by_job: dict[str, NDArray[np.uint64]],
) -> HybridManifest:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    jobs = tuple(expected_sample_ids_by_job)
    counts = {name: len(ids) for name, ids in expected_sample_ids_by_job.items()}
    digests = {
        name: sample_id_sha256(ids)
        for name, ids in expected_sample_ids_by_job.items()
    }
    proposed = HybridManifest(
        schema_version=1,
        split=split,
        dataset_config_hash=dataset_config_hash,
        dataset_manifest_sha256=dataset_manifest_sha256,
        forward_checkpoint_sha256=checkpoint_sha256(forward_checkpoint),
        supervised_checkpoint_sha256=tuple(supervised_checkpoint_sha256),
        split_policy=SPLIT_POLICY,
        hybrid_config_hash=hybrid_inversion_identity_hash(config),
        selected_prior_lambda=float(selected_prior_lambda),
        expected_jobs=jobs,
        expected_job_sample_count=counts,
        expected_job_sample_id_sha256=digests,
        completed_jobs=(),
        job_sha256={},
        package_version=__version__,
        software_sha256=software_sha256(),
        created_at=datetime.now(UTC).isoformat(),
        complete=False,
    )
    path = directory / "manifest.json"
    if path.exists():
        stored = _load_manifest(path)
        if _manifest_identity(stored) != _manifest_identity(proposed):
            raise ValueError("hybrid result manifest identity does not match")
        return stored
    _write_manifest(path, proposed)
    return proposed


def _content_sha256(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in sorted(handle.keys()):
        dataset = handle[name]
        name_bytes = name.encode()
        dtype = dataset.dtype.str.encode()
        values = np.ascontiguousarray(dataset[...])
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def write_hybrid_result_shard(
    path: Path | str,
    batch: HybridResultBatch,
    manifest: HybridManifest,
    job: str,
) -> Path:
    batch.validate()
    if job not in manifest.expected_jobs:
        raise ValueError("hybrid result job is not expected")
    if len(batch.sample_id) != manifest.expected_job_sample_count[job]:
        raise ValueError("hybrid result sample count does not match")
    if sample_id_sha256(batch.sample_id) != manifest.expected_job_sample_id_sha256[job]:
        raise ValueError("hybrid result sample identity does not match")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        for field in fields(HybridResultBatch):
            handle.create_dataset(field.name, data=getattr(batch, field.name))
        handle.attrs["schema_version"] = manifest.schema_version
        handle.attrs["job"] = job
        handle.attrs["split"] = manifest.split
        handle.attrs["hybrid_config_hash"] = manifest.hybrid_config_hash
        handle.attrs["software_sha256"] = manifest.software_sha256
        handle.attrs["sample_id_sha256"] = sample_id_sha256(batch.sample_id)
        handle.attrs["content_sha256"] = _content_sha256(handle)
        handle.flush()
    temporary.replace(destination)
    return destination


def validate_hybrid_result_shard(
    path: Path | str,
    *,
    manifest: HybridManifest | None = None,
    expected_sample_ids: NDArray[np.uint64] | None = None,
) -> HybridResultBatch:
    source = Path(path)
    try:
        with h5py.File(source, "r") as handle:
            names = {field.name for field in fields(HybridResultBatch)}
            if set(handle.keys()) != names:
                raise ValueError("hybrid result datasets do not match the schema")
            stored_content = str(handle.attrs.get("content_sha256", ""))
            if stored_content != _content_sha256(handle):
                raise ValueError("hybrid result content checksum is invalid")
            payload = {name: np.asarray(handle[name]) for name in names}
            job = str(handle.attrs.get("job", ""))
            if manifest is not None:
                if job not in manifest.expected_jobs:
                    raise ValueError("hybrid result job is not in the manifest")
                if str(handle.attrs.get("split", "")) != manifest.split:
                    raise ValueError("hybrid result split identity does not match")
                if str(handle.attrs.get("hybrid_config_hash", "")) != manifest.hybrid_config_hash:
                    raise ValueError("hybrid result configuration identity does not match")
                if str(handle.attrs.get("software_sha256", "")) != manifest.software_sha256:
                    raise ValueError("hybrid result software identity does not match")
    except OSError as error:
        raise ValueError("hybrid result shard is not readable HDF5") from error
    batch = HybridResultBatch(**payload)
    batch.validate()
    if expected_sample_ids is not None and not np.array_equal(
        batch.sample_id, expected_sample_ids
    ):
        raise ValueError("hybrid result sample IDs do not match")
    return batch


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _manifest_lock(directory: Path) -> Iterator[None]:
    path = directory / ".manifest.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def mark_hybrid_job_complete(
    output_dir: Path | str, job: str, shard_path: Path | str
) -> HybridManifest:
    directory = Path(output_dir)
    with _manifest_lock(directory):
        manifest = _load_manifest(directory / "manifest.json")
        path = Path(shard_path)
        validate_hybrid_result_shard(path, manifest=manifest)
        completed = tuple(
            name for name in manifest.expected_jobs if name in {*manifest.completed_jobs, job}
        )
        hashes = dict(manifest.job_sha256)
        hashes[job] = _file_sha256(path)
        updated = HybridManifest(
            **{
                **asdict(manifest),
                "supervised_checkpoint_sha256": manifest.supervised_checkpoint_sha256,
                "expected_jobs": manifest.expected_jobs,
                "completed_jobs": completed,
                "job_sha256": hashes,
                "complete": len(completed) == len(manifest.expected_jobs),
            }
        )
        _write_manifest(directory / "manifest.json", updated)
        return updated


def validate_complete_hybrid_results(output_dir: Path | str) -> HybridManifest:
    directory = Path(output_dir)
    manifest = _load_manifest(directory / "manifest.json")
    if not manifest.complete:
        raise ValueError("hybrid result manifest is incomplete")
    for job in manifest.expected_jobs:
        path = directory / f"{job}.h5"
        if _file_sha256(path) != manifest.job_sha256.get(job):
            raise ValueError("hybrid result shard checksum does not match")
        validate_hybrid_result_shard(path, manifest=manifest)
    return manifest
