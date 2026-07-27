"""Deterministic, atomic HDF5 dataset generation and resume support."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .config import DatasetConfig, canonical_hash
from .geology import GeneratedModel, generate_model
from .quality import QualityFlag, solve_with_recovery
from .secular import LayeredModel

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    config_hash: str
    global_seed: int
    expected_shards: int
    completed_shards: tuple[int, ...]
    accepted_by_kind: dict[str, int]
    rejected_by_reason: dict[str, int]
    recovered_models: int
    complete: bool


@dataclass(frozen=True)
class ShardResult:
    shard_id: int
    path: Path
    accepted_by_kind: dict[str, int]
    rejected_by_reason: dict[str, int]
    recovered_models: int
    file_sha256: str


def load_manifest(path: Path | str) -> Manifest:
    """Load and validate the public JSON manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return Manifest(
        schema_version=int(payload["schema_version"]),
        config_hash=str(payload["config_hash"]),
        global_seed=int(payload["global_seed"]),
        expected_shards=int(payload["expected_shards"]),
        completed_shards=tuple(int(value) for value in payload["completed_shards"]),
        accepted_by_kind={
            str(key): int(value)
            for key, value in payload["accepted_by_kind"].items()
        },
        rejected_by_reason={
            str(key): int(value)
            for key, value in payload["rejected_by_reason"].items()
        },
        recovered_models=int(payload["recovered_models"]),
        complete=bool(payload["complete"]),
    )


def _write_manifest(path: Path, manifest: Manifest) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    payload = asdict(manifest)
    payload["completed_shards"] = list(manifest.completed_shards)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _quality_reason(flags: QualityFlag) -> str:
    reasons = [
        member.name
        for member in QualityFlag
        if member is not QualityFlag.OK and flags & member
    ]
    return "|".join(reasons) or "UNKNOWN"


def _solve_sample(
    sample_id: int, config: DatasetConfig
) -> tuple[GeneratedModel, NDArray[np.float64], NDArray[np.bool_], int, Counter[str]]:
    rejected: Counter[str] = Counter()
    for retry_count in range(config.max_model_retries + 1):
        generated = generate_model(
            sample_id,
            config.geology,
            config.seed,
            retry_count=retry_count,
        )
        depth = (
            np.arange(config.geology.layers, dtype=np.float64)
            * config.geology.thickness_km
        )
        model = LayeredModel(
            depth=depth,
            density=generated.density,
            vs=generated.vs,
            vp=generated.vp,
        )
        recovered = solve_with_recovery(model, config.physics)
        report = recovered.quality
        if report.hard_failure or report.retry_required:
            rejected[_quality_reason(report.flags)] += 1
            continue
        flags = int(report.flags)
        return (
            generated,
            recovered.dispersion.phase_velocity,
            recovered.dispersion.valid_mask,
            flags,
            rejected,
        )
    raise RuntimeError(
        f"sample {sample_id} failed after "
        f"{config.max_model_retries + 1} deterministic attempts: {dict(rejected)}"
    )


def _create_compressed(
    handle: h5py.File,
    name: str,
    data: NDArray[Any],
    dtype: str,
    chunks: tuple[int, ...],
) -> None:
    handle.create_dataset(
        name,
        data=data,
        dtype=dtype,
        chunks=chunks,
        compression="gzip",
        shuffle=True,
    )


def generate_shard(shard_id: int, config: DatasetConfig) -> ShardResult:
    """Generate and atomically publish one independently reproducible shard."""
    expected_shards = math.ceil(config.samples / config.shard_size)
    if not 0 <= shard_id < expected_shards:
        raise ValueError("shard_id is outside the configured shard range")

    first = shard_id * config.shard_size
    stop = min(first + config.shard_size, config.samples)
    count = stop - first
    sample_ids = np.arange(first, stop, dtype=np.uint64)
    kinds = np.empty(count, dtype=np.uint8)
    vs = np.empty((count, config.geology.layers), dtype=np.float32)
    vp = np.empty_like(vs)
    density = np.empty_like(vs)
    phase = np.empty(
        (count, config.physics.mode_count, config.physics.frequencies.size),
        dtype=np.float32,
    )
    mask = np.empty(phase.shape, dtype=np.bool_)
    flags = np.empty(count, dtype=np.uint16)
    retries = np.empty(count, dtype=np.uint8)
    accepted_by_kind: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    recovered_models = 0

    for row, sample_id_value in enumerate(sample_ids):
        generated, curve, valid, quality, rejected = _solve_sample(
            int(sample_id_value), config
        )
        kinds[row] = int(generated.kind)
        vs[row] = generated.vs
        vp[row] = generated.vp
        density[row] = generated.density
        phase[row] = curve
        mask[row] = valid
        flags[row] = quality
        retries[row] = generated.retry_count
        accepted_by_kind[generated.kind.name] += 1
        rejected_by_reason.update(rejected)
        if quality & int(QualityFlag.RECOVERED):
            recovered_models += 1

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"shard-{shard_id:05d}.h5"
    temporary = output_dir / f"{target.name}.tmp-{os.getpid()}"
    config_digest = canonical_hash(config)
    id_digest = hashlib.sha256(sample_ids.tobytes()).hexdigest()
    chunk_rows = min(256, count)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["config_hash"] = config_digest
        handle.attrs["shard_id"] = shard_id
        handle.attrs["first_sample_id"] = first
        handle.attrs["last_sample_id"] = stop - 1
        handle.attrs["accepted_count"] = count
        handle.attrs["sample_id_sha256"] = id_digest
        handle.attrs["accepted_by_kind"] = json.dumps(
            dict(accepted_by_kind), sort_keys=True
        )
        handle.attrs["rejected_by_reason"] = json.dumps(
            dict(rejected_by_reason), sort_keys=True
        )
        handle.attrs["recovered_models"] = recovered_models
        handle.create_dataset("sample_id", data=sample_ids, dtype="u8")
        handle.create_dataset("model_kind", data=kinds, dtype="u1")
        _create_compressed(
            handle, "vs", vs, "f4", (chunk_rows, config.geology.layers)
        )
        _create_compressed(
            handle, "vp", vp, "f4", (chunk_rows, config.geology.layers)
        )
        _create_compressed(
            handle,
            "density",
            density,
            "f4",
            (chunk_rows, config.geology.layers),
        )
        _create_compressed(
            handle,
            "phase_velocity",
            phase,
            "f4",
            (
                min(32, count),
                config.physics.mode_count,
                config.physics.frequencies.size,
            ),
        )
        _create_compressed(
            handle,
            "valid_mask",
            mask,
            "?",
            (
                min(64, count),
                config.physics.mode_count,
                config.physics.frequencies.size,
            ),
        )
        handle.create_dataset("quality_flags", data=flags, dtype="u2")
        handle.create_dataset("retry_count", data=retries, dtype="u1")
        handle.flush()

    file_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(target)
    return ShardResult(
        shard_id=shard_id,
        path=target,
        accepted_by_kind=dict(accepted_by_kind),
        rejected_by_reason=dict(rejected_by_reason),
        recovered_models=recovered_models,
        file_sha256=file_sha256,
    )


def _inspect_shard(shard_id: int, config: DatasetConfig) -> ShardResult | None:
    path = config.output_dir / f"shard-{shard_id:05d}.h5"
    if not path.exists():
        return None
    expected_first = shard_id * config.shard_size
    expected_stop = min(expected_first + config.shard_size, config.samples)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("config_hash", "")) != canonical_hash(config):
            raise ValueError(f"shard {shard_id} configuration hash does not match")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "shard_id": shard_id,
            "first_sample_id": expected_first,
            "last_sample_id": expected_stop - 1,
            "accepted_count": expected_stop - expected_first,
        }
        if any(int(handle.attrs.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError(f"shard {shard_id} metadata is incomplete or inconsistent")
        accepted = json.loads(str(handle.attrs["accepted_by_kind"]))
        rejected = json.loads(str(handle.attrs["rejected_by_reason"]))
        recovered = int(handle.attrs["recovered_models"])
    return ShardResult(
        shard_id=shard_id,
        path=path,
        accepted_by_kind={str(key): int(value) for key, value in accepted.items()},
        rejected_by_reason={
            str(key): int(value) for key, value in rejected.items()
        },
        recovered_models=recovered,
        file_sha256="",
    )


def _manifest_from_results(
    config: DatasetConfig,
    expected_shards: int,
    results: dict[int, ShardResult],
) -> Manifest:
    kinds: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    recovered = 0
    for result in results.values():
        kinds.update(result.accepted_by_kind)
        rejected.update(result.rejected_by_reason)
        recovered += result.recovered_models
    completed = tuple(sorted(results))
    return Manifest(
        schema_version=SCHEMA_VERSION,
        config_hash=canonical_hash(config),
        global_seed=config.seed,
        expected_shards=expected_shards,
        completed_shards=completed,
        accepted_by_kind=dict(sorted(kinds.items())),
        rejected_by_reason=dict(sorted(rejected.items())),
        recovered_models=recovered,
        complete=len(completed) == expected_shards,
    )


def generate_dataset(config: DatasetConfig) -> Manifest:
    """Generate missing shards and deterministically resume an existing run."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.json"
    digest = canonical_hash(config)
    expected_shards = math.ceil(config.samples / config.shard_size)
    if manifest_path.exists():
        previous = load_manifest(manifest_path)
        if previous.config_hash != digest:
            raise ValueError("existing manifest configuration hash does not match")
        if previous.schema_version != SCHEMA_VERSION:
            raise ValueError("existing manifest schema version is unsupported")

    results: dict[int, ShardResult] = {}
    missing: list[int] = []
    for shard_id in range(expected_shards):
        result = _inspect_shard(shard_id, config)
        if result is None:
            missing.append(shard_id)
        else:
            results[shard_id] = result

    manifest = _manifest_from_results(config, expected_shards, results)
    _write_manifest(manifest_path, manifest)
    if not missing:
        return manifest

    workers = (
        max(1, (os.cpu_count() or 2) - 1) if config.workers == 0 else config.workers
    )
    if workers == 1:
        for shard_id in missing:
            results[shard_id] = generate_shard(shard_id, config)
            manifest = _manifest_from_results(config, expected_shards, results)
            _write_manifest(manifest_path, manifest)
        return manifest

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_shard, shard_id, config): shard_id
            for shard_id in missing
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.shard_id] = result
            manifest = _manifest_from_results(config, expected_shards, results)
            _write_manifest(manifest_path, manifest)
    return manifest
