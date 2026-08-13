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
from .splits import SPLIT_POLICY, mask_for_split

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
    preparation_success: NDArray[np.bool_]
    preparation_failure_code: NDArray[np.bytes_]
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

    def validate(
        self,
        *,
        vs_min: float = 0.3,
        vs_max: float = 2.6,
        prior_weight_min: float = 0.25,
        prior_weight_max: float = 4.0,
    ) -> None:
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
        preparation_success = _require_array(
            self.preparation_success,
            "preparation_success",
            np.bool_,
            (count,),
        )
        preparation_codes = self.preparation_failure_code
        if (
            not isinstance(preparation_codes, np.ndarray)
            or preparation_codes.shape != (count,)
            or preparation_codes.dtype.kind != "S"
            or preparation_codes.dtype.itemsize > 64
        ):
            raise ValueError("preparation_failure_code has an invalid schema")
        decoded_preparation_codes = _decode_codes(preparation_codes)
        if np.any(preparation_success != (decoded_preparation_codes == "")):
            raise ValueError(
                "preparation_failure_code disagrees with preparation_success"
            )
        active_observations = preparation_success[:, None, None] & self.valid_mask
        if not np.all(np.isfinite(self.observed_phase_velocity[active_observations])):
            raise ValueError("valid observations must be finite")
        if np.any(self.observed_phase_velocity[active_observations] <= 0):
            raise ValueError("valid observations must be positive")
        if not np.all(np.isnan(self.observed_phase_velocity[~active_observations])):
            raise ValueError("inactive observations must be NaN")
        for name in ("reference_vs", "supervised_vs"):
            values = _require_array(getattr(self, name), name, np.float32, (count, 20))
            if not np.all(np.isfinite(values[preparation_success])) or np.any(
                (values[preparation_success] < vs_min)
                | (values[preparation_success] > vs_max)
            ):
                raise ValueError(f"{name} must contain finite globally bounded profiles")
            if not np.all(np.isnan(values[~preparation_success])):
                raise ValueError(f"{name} must be NaN when preparation failed")
        sensitivity = _require_array(
            self.sensitivity, "sensitivity", np.float64, (count, 20)
        )
        weights = _require_array(
            self.prior_weights, "prior_weights", np.float64, (count, 20)
        )
        if (
            not np.all(np.isfinite(sensitivity[preparation_success]))
            or np.any(sensitivity[preparation_success] < 0)
            or np.any(np.all(sensitivity[preparation_success] == 0, axis=1))
        ):
            raise ValueError("sensitivity rows must be finite and nonzero")
        if not np.all(np.isfinite(weights[preparation_success])) or np.any(
            (weights[preparation_success] < prior_weight_min)
            | (weights[preparation_success] > prior_weight_max)
        ):
            raise ValueError("prior_weights exceed the configured bounds")
        if not np.allclose(
            weights[preparation_success].mean(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("prior_weights must have unit row means")
        if not np.all(np.isnan(sensitivity[~preparation_success])) or not np.all(
            np.isnan(weights[~preparation_success])
        ):
            raise ValueError("prior arrays must be NaN when preparation failed")
        self._validate_outcome(
            "control",
            has_learning_prior=False,
            preparation_success=preparation_success,
            preparation_codes=decoded_preparation_codes,
            vs_min=vs_min,
            vs_max=vs_max,
        )
        self._validate_outcome(
            "hybrid",
            has_learning_prior=True,
            preparation_success=preparation_success,
            preparation_codes=decoded_preparation_codes,
            vs_min=vs_min,
            vs_max=vs_max,
        )

    def _validate_outcome(
        self,
        prefix: str,
        *,
        has_learning_prior: bool,
        preparation_success: NDArray[np.bool_],
        preparation_codes: NDArray[np.str_],
        vs_min: float,
        vs_max: float,
    ) -> None:
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
        if np.any(success & ~preparation_success) or np.any(
            decoded[~preparation_success] != preparation_codes[~preparation_success]
        ):
            raise ValueError(f"{prefix} outcome disagrees with preparation failure")
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
            if not np.all(np.isnan(values[~success])):
                raise ValueError(
                    f"{prefix}_{suffix} must be NaN for failed rows"
                )
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
            (profile[success] < vs_min) | (profile[success] > vs_max)
        ):
            raise ValueError(f"{prefix}_vs is invalid for successful rows")
        if not np.all(np.isfinite(prediction[success])):
            raise ValueError(f"{prefix}_prediction is invalid for successful rows")
        if not np.all(np.isnan(profile[~success])) or not np.all(
            np.isnan(prediction[~success])
        ):
            raise ValueError(
                f"{prefix} scientific arrays must be NaN for failed rows"
            )


@dataclass(frozen=True)
class HybridManifest:
    schema_version: int
    split: HybridSplit
    dataset_config_hash: str
    dataset_manifest_sha256: str
    forward_checkpoint_sha256: str
    supervised_checkpoint_sha256: tuple[str, ...]
    supervised_seeds: tuple[int, ...]
    supervised_run_identity_sha256: str
    tuning_sha256: str
    split_policy: str
    hybrid_config_hash: str
    selected_prior_lambda: float
    vs_min: float
    vs_max: float
    prior_weight_min: float
    prior_weight_max: float
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
        if self.supervised_seeds != (0, 1, 2):
            raise ValueError("hybrid manifest requires supervised seeds 0, 1, and 2")
        if len(self.supervised_checkpoint_sha256) != len(self.supervised_seeds):
            raise ValueError("hybrid manifest supervised checkpoint count is invalid")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in self.supervised_checkpoint_sha256
        ):
            raise ValueError("hybrid manifest supervised checkpoint digest is invalid")
        for name in (
            "dataset_manifest_sha256",
            "forward_checkpoint_sha256",
            "supervised_run_identity_sha256",
            "tuning_sha256",
            "software_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not 0.3 <= self.vs_min < self.vs_max <= 2.6:
            raise ValueError("hybrid manifest Vs bounds are invalid")
        if not 0 < self.prior_weight_min <= 1 <= self.prior_weight_max:
            raise ValueError("hybrid manifest prior-weight bounds are invalid")
        if not self.expected_jobs or len(set(self.expected_jobs)) != len(self.expected_jobs):
            raise ValueError("expected_jobs must be nonempty and unique")
        noise_scenarios = {
            noise
            for job in self.expected_jobs
            for noise in ("clean", "noise_1pct")
            if f"-{noise}-" in job
        }
        if noise_scenarios != {"clean", "noise_1pct"}:
            raise ValueError("expected_jobs must cover clean and noise_1pct")
        if any(
            not job.startswith(f"hybrid-{self.split}-")
            or not any(f"-{noise}-" in job for noise in noise_scenarios)
            for job in self.expected_jobs
        ):
            raise ValueError("expected_jobs do not match the hybrid split protocol")
        if set(self.completed_jobs) - set(self.expected_jobs):
            raise ValueError("completed_jobs contains an unexpected job")
        if self.complete != (set(self.completed_jobs) == set(self.expected_jobs)):
            raise ValueError("hybrid manifest complete flag is inconsistent")


def _manifest_payload(manifest: HybridManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload["supervised_checkpoint_sha256"] = list(
        manifest.supervised_checkpoint_sha256
    )
    payload["supervised_seeds"] = list(manifest.supervised_seeds)
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
    _fsync_directory(path.parent)


def _write_manifest(path: Path, manifest: HybridManifest) -> None:
    _atomic_json(path, _manifest_payload(manifest))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    payload["supervised_seeds"] = tuple(payload["supervised_seeds"])
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
        manifest.supervised_seeds,
        manifest.supervised_run_identity_sha256,
        manifest.tuning_sha256,
        manifest.split_policy,
        manifest.hybrid_config_hash,
        manifest.selected_prior_lambda,
        manifest.vs_min,
        manifest.vs_max,
        manifest.prior_weight_min,
        manifest.prior_weight_max,
        manifest.expected_jobs,
        tuple(sorted(manifest.expected_job_sample_count.items())),
        tuple(sorted(manifest.expected_job_sample_id_sha256.items())),
        manifest.software_sha256,
    )


def _scientific_identity_sha256(manifest: HybridManifest) -> str:
    payload = json.dumps(
        _manifest_identity(manifest),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def initialize_hybrid_manifest(
    output_dir: Path | str,
    *,
    split: HybridSplit,
    dataset_config_hash: str,
    dataset_manifest_sha256: str,
    forward_checkpoint: Path | str,
    supervised_checkpoint_sha256: tuple[str, ...],
    supervised_seeds: tuple[int, ...],
    supervised_run_identity_sha256: str,
    tuning_sha256: str,
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
        supervised_seeds=tuple(supervised_seeds),
        supervised_run_identity_sha256=supervised_run_identity_sha256,
        tuning_sha256=tuning_sha256,
        split_policy=SPLIT_POLICY,
        hybrid_config_hash=hybrid_inversion_identity_hash(config),
        selected_prior_lambda=float(selected_prior_lambda),
        vs_min=config.vs_min,
        vs_max=config.vs_max,
        prior_weight_min=config.prior_weight_min,
        prior_weight_max=config.prior_weight_max,
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
    with _manifest_lock(directory):
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
    batch.validate(
        vs_min=manifest.vs_min,
        vs_max=manifest.vs_max,
        prior_weight_min=manifest.prior_weight_min,
        prior_weight_max=manifest.prior_weight_max,
    )
    if job not in manifest.expected_jobs:
        raise ValueError("hybrid result job is not expected")
    if len(batch.sample_id) != manifest.expected_job_sample_count[job]:
        raise ValueError("hybrid result sample count does not match")
    if sample_id_sha256(batch.sample_id) != manifest.expected_job_sample_id_sha256[job]:
        raise ValueError("hybrid result sample identity does not match")
    destination = Path(path)
    if destination.name != f"{job}.h5":
        raise ValueError("hybrid result path does not match the job name")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    try:
        with h5py.File(temporary, "w") as handle:
            for field in fields(HybridResultBatch):
                handle.create_dataset(field.name, data=getattr(batch, field.name))
            handle.attrs["schema_version"] = manifest.schema_version
            handle.attrs["job"] = job
            handle.attrs["split"] = manifest.split
            handle.attrs["hybrid_config_hash"] = manifest.hybrid_config_hash
            handle.attrs["software_sha256"] = manifest.software_sha256
            handle.attrs["scientific_identity_sha256"] = (
                _scientific_identity_sha256(manifest)
            )
            handle.attrs["sample_id_sha256"] = sample_id_sha256(batch.sample_id)
            handle.attrs["content_sha256"] = _content_sha256(handle)
            handle.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        validate_hybrid_result_shard(
            temporary, manifest=manifest, expected_job=job
        )
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def validate_hybrid_result_shard(
    path: Path | str,
    *,
    manifest: HybridManifest | None = None,
    expected_sample_ids: NDArray[np.uint64] | None = None,
    expected_job: str | None = None,
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
            if expected_job is not None and job != expected_job:
                raise ValueError("hybrid result job identity does not match")
            stored_sample_digest = str(handle.attrs.get("sample_id_sha256", ""))
            if manifest is not None:
                if job not in manifest.expected_jobs:
                    raise ValueError("hybrid result job is not in the manifest")
                if str(handle.attrs.get("split", "")) != manifest.split:
                    raise ValueError("hybrid result split identity does not match")
                if str(handle.attrs.get("hybrid_config_hash", "")) != manifest.hybrid_config_hash:
                    raise ValueError("hybrid result configuration identity does not match")
                if str(handle.attrs.get("software_sha256", "")) != manifest.software_sha256:
                    raise ValueError("hybrid result software identity does not match")
                if str(
                    handle.attrs.get("scientific_identity_sha256", "")
                ) != _scientific_identity_sha256(manifest):
                    raise ValueError("hybrid result scientific identity does not match")
    except OSError as error:
        raise ValueError("hybrid result shard is not readable HDF5") from error
    batch = HybridResultBatch(**payload)
    if manifest is None:
        batch.validate()
    else:
        batch.validate(
            vs_min=manifest.vs_min,
            vs_max=manifest.vs_max,
            prior_weight_min=manifest.prior_weight_min,
            prior_weight_max=manifest.prior_weight_max,
        )
        if not np.all(mask_for_split(batch.sample_id, manifest.split)):
            raise ValueError("hybrid result sample IDs do not belong to the split")
    actual_sample_digest = sample_id_sha256(batch.sample_id)
    if stored_sample_digest != actual_sample_digest:
        raise ValueError("hybrid result stored sample identity does not match")
    if manifest is not None and (
        len(batch.sample_id) != manifest.expected_job_sample_count[job]
        or actual_sample_digest != manifest.expected_job_sample_id_sha256[job]
    ):
        raise ValueError("hybrid result manifest sample identity does not match")
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
        if path != directory / f"{job}.h5":
            raise ValueError("hybrid result path does not match the completed job")
        validate_hybrid_result_shard(
            path, manifest=manifest, expected_job=job
        )
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
    if manifest.software_sha256 != software_sha256():
        raise ValueError("hybrid result software identity does not match this checkout")
    if not manifest.complete:
        raise ValueError("hybrid result manifest is incomplete")
    for job in manifest.expected_jobs:
        path = directory / f"{job}.h5"
        if _file_sha256(path) != manifest.job_sha256.get(job):
            raise ValueError("hybrid result shard checksum does not match")
        validate_hybrid_result_shard(
            path, manifest=manifest, expected_job=job
        )
    return manifest
