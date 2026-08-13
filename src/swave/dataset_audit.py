"""Auditable geology and exact-duplicate checks for production HDF5 data."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .config import DatasetConfig, canonical_hash
from .dataset import (
    dataset_manifest_sha256,
    load_manifest,
    validate_dataset_files,
)
from .geology import ModelKind, generate_model
from .splits import mask_for_split, split_for_sample_id

_DIGEST_DTYPE = np.dtype("S32")
_FLOAT32_ROUNDING_TOLERANCE = 5e-7


def _duplicate_digest_groups(
    digests: NDArray[np.bytes_], sample_ids: NDArray[np.uint64]
) -> list[list[int]]:
    values = np.asarray(digests, dtype=_DIGEST_DTYPE)
    ids = np.asarray(sample_ids, dtype=np.uint64)
    if values.ndim != 1 or ids.ndim != 1 or values.shape != ids.shape:
        raise ValueError("digests and sample_ids must be equal one-dimensional arrays")
    if not values.size:
        return []

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(ordered)]
    groups: list[list[int]] = []
    for start, stop in zip(starts, stops, strict=True):
        if stop - start < 2:
            continue
        group = sorted(int(value) for value in ids[order[start:stop]])
        groups.append(group)
    return groups


def summarize_duplicate_digests(
    digests: NDArray[np.bytes_],
    sample_ids: NDArray[np.uint64],
    *,
    raw_rows: Mapping[int, bytes] | None = None,
) -> dict[str, object]:
    """Summarize digest candidates, optionally confirming raw-byte equality."""
    groups = _duplicate_digest_groups(digests, sample_ids)
    if raw_rows is not None:
        confirmed: list[list[int]] = []
        for group in groups:
            by_payload: dict[bytes, list[int]] = defaultdict(list)
            for sample_id in group:
                try:
                    payload = raw_rows[sample_id]
                except KeyError as error:
                    raise ValueError(
                        f"missing raw bytes for candidate sample {sample_id}"
                    ) from error
                by_payload[payload].append(sample_id)
            confirmed.extend(
                sorted(values) for values in by_payload.values() if len(values) > 1
            )
        groups = sorted(confirmed)
    duplicate_rows = sum(len(group) for group in groups)
    cross_split_groups = sum(
        len({split_for_sample_id(value) for value in group}) > 1
        for group in groups
    )
    return {
        "duplicate_groups": len(groups),
        "duplicate_rows": duplicate_rows,
        "extra_duplicate_rows": duplicate_rows - len(groups),
        "cross_split_groups": cross_split_groups,
        "examples": groups[:10],
    }


def _candidate_ids(
    digests: NDArray[np.bytes_], sample_ids: NDArray[np.uint64]
) -> set[int]:
    return {
        sample_id
        for group in _duplicate_digest_groups(digests, sample_ids)
        for sample_id in group
    }


def _raw_row_bytes(handle: h5py.File, row: int, names: tuple[str, ...]) -> bytes:
    return b"".join(
        np.ascontiguousarray(np.asarray(handle[name][row])).tobytes()
        for name in names
    )


def _load_candidate_raw_rows(
    directory: Path,
    vs_ids: set[int],
    record_ids: set[int],
) -> tuple[dict[int, bytes], dict[int, bytes]]:
    vs_rows: dict[int, bytes] = {}
    record_rows: dict[int, bytes] = {}
    wanted = vs_ids | record_ids
    if not wanted:
        return vs_rows, record_rows
    record_names = (
        "model_kind",
        "vs",
        "vp",
        "density",
        "phase_velocity",
        "valid_mask",
        "quality_flags",
        "retry_count",
    )
    for path in sorted(directory.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            for row, value in enumerate(sample_ids):
                sample_id = int(value)
                if sample_id in vs_ids:
                    vs_rows[sample_id] = _raw_row_bytes(handle, row, ("vs",))
                if sample_id in record_ids:
                    record_rows[sample_id] = _raw_row_bytes(
                        handle, row, record_names
                    )
    return vs_rows, record_rows


def check_geology_rule(
    kind: ModelKind,
    values: NDArray[np.float64],
    background: NDArray[np.float64],
    config: Any,
) -> dict[str, object]:
    """Check one model against the deterministic four-family geology contract."""
    vs = np.asarray(values, dtype=np.float64)
    base = np.asarray(background, dtype=np.float64)
    if vs.shape != (config.layers,) or base.shape != (config.layers,):
        return {"valid": False, "reason": "invalid_shape"}
    if (
        not np.all(np.isfinite(vs))
        or not np.all(np.isfinite(base))
        or np.any(vs < config.vs_min)
        or np.any(vs > config.vs_max)
    ):
        return {"valid": False, "reason": "invalid_velocity_bounds"}

    delta = vs - base
    changed = np.abs(delta) > _FLOAT32_ROUNDING_TOLERANCE
    positive = np.flatnonzero(delta > _FLOAT32_ROUNDING_TOLERANCE)
    negative = np.flatnonzero(delta < -_FLOAT32_ROUNDING_TOLERANCE)
    if kind is ModelKind.NORMAL:
        if np.any(np.diff(vs) < 0):
            return {"valid": False, "reason": "normal_not_nondecreasing"}
        if np.any(changed):
            return {"valid": False, "reason": "normal_changed_from_background"}
        return {
            "valid": True,
            "reason": "ok",
            "anomaly_layers": [],
            "contrast_min": None,
            "contrast_max": None,
        }

    indexes = np.flatnonzero(changed)
    first = config.anomaly_first_layer - 1
    last = config.anomaly_last_layer - 1
    if not len(indexes) or indexes[0] < first or indexes[-1] > last:
        return {"valid": False, "reason": "anomaly_outside_target_zone"}
    contrast = np.abs(delta[indexes]) / base[indexes]
    if np.any(contrast < 0.05 - 1e-12):
        return {"valid": False, "reason": "anomaly_contrast_below_five_percent"}

    contiguous = bool(
        np.array_equal(indexes, np.arange(indexes[0], indexes[-1] + 1))
    )
    valid = False
    reason = "family_pattern_mismatch"
    if kind is ModelKind.LOW_VELOCITY:
        valid = len(positive) == 0 and 1 <= len(negative) <= 3 and contiguous
    elif kind is ModelKind.HIGH_VELOCITY:
        valid = len(negative) == 0 and 1 <= len(positive) <= 2 and contiguous
    elif kind is ModelKind.COUPLED:
        valid = (
            len(positive) == 1
            and len(negative) in (2, 3)
            and negative[0] == positive[0] + 1
            and np.array_equal(
                negative,
                np.arange(negative[0], negative[0] + len(negative)),
            )
        )
    return {
        "valid": valid,
        "reason": "ok" if valid else reason,
        "anomaly_layers": [int(value) for value in indexes],
        "contrast_min": float(contrast.min()),
        "contrast_max": float(contrast.max()),
    }


def _row_digests(*arrays: NDArray[Any]) -> NDArray[np.bytes_]:
    if not arrays:
        raise ValueError("at least one array is required")
    count = len(arrays[0])
    if any(len(array) != count for array in arrays):
        raise ValueError("all digest arrays must have the same row count")
    digests = np.empty(count, dtype=_DIGEST_DTYPE)
    for row in range(count):
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(np.ascontiguousarray(array[row]).tobytes())
        digests[row] = digest.digest()
    return digests


def _empty_kind_counts() -> dict[str, int]:
    return {kind.name: 0 for kind in ModelKind}


def audit_dataset(
    dataset_dir: Path | str,
    dataset_config: DatasetConfig,
    *,
    validate_checksums: bool = True,
) -> dict[str, object]:
    """Audit one complete dataset without retaining physical arrays in memory."""
    directory = Path(dataset_dir)
    manifest = (
        validate_dataset_files(directory)
        if validate_checksums
        else load_manifest(directory / "manifest.json")
    )
    expected_config_hash = canonical_hash(dataset_config)
    if manifest.config_hash != expected_config_hash:
        raise ValueError("dataset configuration hash does not match the audit config")

    all_ids: list[NDArray[np.uint64]] = []
    vs_digests: list[NDArray[np.bytes_]] = []
    record_digests: list[NDArray[np.bytes_]] = []
    by_kind = Counter[str]()
    violation_reasons = Counter[str]()
    anomaly_layer_counts: dict[str, Counter[int]] = defaultdict(Counter)
    anomaly_starts: dict[str, list[int]] = defaultdict(list)
    anomaly_ends: dict[str, list[int]] = defaultdict(list)
    contrasts: dict[str, list[float]] = defaultdict(list)
    reconstruction_mismatches = 0

    for path in sorted(directory.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            kinds = np.asarray(handle["model_kind"], dtype=np.uint8)
            vs = np.asarray(handle["vs"], dtype=np.float32)
            vp = np.asarray(handle["vp"], dtype=np.float32)
            density = np.asarray(handle["density"], dtype=np.float32)
            phase = np.asarray(handle["phase_velocity"], dtype=np.float32)
            valid = np.asarray(handle["valid_mask"], dtype=np.bool_)
            quality = np.asarray(handle["quality_flags"], dtype=np.uint16)
            retries = np.asarray(handle["retry_count"], dtype=np.uint8)

        all_ids.append(sample_ids)
        vs_digests.append(_row_digests(vs))
        record_digests.append(
            _row_digests(kinds, vs, vp, density, phase, valid, quality, retries)
        )
        for row, sample_id_value in enumerate(sample_ids):
            kind = ModelKind(int(kinds[row]))
            by_kind[kind.name] += 1
            generated = generate_model(
                int(sample_id_value),
                dataset_config.geology,
                dataset_config.seed,
                retry_count=int(retries[row]),
            )
            matches = (
                generated.kind is kind
                and np.array_equal(vs[row], generated.vs.astype(np.float32))
                and np.array_equal(vp[row], generated.vp.astype(np.float32))
                and np.array_equal(
                    density[row], generated.density.astype(np.float32)
                )
            )
            if not matches:
                reconstruction_mismatches += 1
                violation_reasons["deterministic_reconstruction_mismatch"] += 1
            rule = check_geology_rule(
                kind,
                vs[row],
                generated.background_vs,
                dataset_config.geology,
            )
            if not bool(rule["valid"]):
                violation_reasons[str(rule["reason"])] += 1
                continue
            indexes = [int(value) for value in rule["anomaly_layers"]]
            if indexes:
                anomaly_layer_counts[kind.name][len(indexes)] += 1
                anomaly_starts[kind.name].append(indexes[0])
                anomaly_ends[kind.name].append(indexes[-1])
                contrasts[kind.name].extend(
                    [float(rule["contrast_min"]), float(rule["contrast_max"])]
                )

    ids = np.concatenate(all_ids) if all_ids else np.array([], dtype=np.uint64)
    unique_ids = np.unique(ids)
    contiguous = np.array_equal(
        np.sort(ids), np.arange(dataset_config.samples, dtype=np.uint64)
    )
    split_counts = {
        split: int(mask_for_split(ids, split).sum())
        for split in ("train", "validation", "test", "inversion")
    }
    kind_counts = _empty_kind_counts()
    kind_counts.update({key: int(value) for key, value in by_kind.items()})
    all_vs_digests = np.concatenate(vs_digests)
    all_record_digests = np.concatenate(record_digests)
    vs_rows, record_rows = _load_candidate_raw_rows(
        directory,
        _candidate_ids(all_vs_digests, ids),
        _candidate_ids(all_record_digests, ids),
    )

    rule_statistics: dict[str, object] = {}
    for kind in ModelKind:
        name = kind.name
        rule_statistics[name] = {
            "anomaly_layer_count": {
                str(key): int(value)
                for key, value in sorted(anomaly_layer_counts[name].items())
            },
            "anomaly_start_layer_index_range": (
                [min(anomaly_starts[name]), max(anomaly_starts[name])]
                if anomaly_starts[name]
                else None
            ),
            "anomaly_end_layer_index_range": (
                [min(anomaly_ends[name]), max(anomaly_ends[name])]
                if anomaly_ends[name]
                else None
            ),
            "contrast_range": (
                [min(contrasts[name]), max(contrasts[name])]
                if contrasts[name]
                else None
            ),
        }

    return {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_manifest_sha256(manifest),
        "dataset_config_hash": manifest.config_hash,
        "identity": {
            "sample_count": int(ids.size),
            "sample_ids_unique": bool(unique_ids.size == ids.size),
            "sample_ids_contiguous": bool(contiguous),
            "split_counts": split_counts,
        },
        "duplicates": {
            "definition": (
                "exact fixed-schema raw array byte equality; SHA-256 is used "
                "only to locate candidates"
            ),
            "vs": summarize_duplicate_digests(
                all_vs_digests, ids, raw_rows=vs_rows
            ),
            "full_record": summarize_duplicate_digests(
                all_record_digests, ids, raw_rows=record_rows
            ),
        },
        "geology": {
            "by_kind": kind_counts,
            "violations": int(sum(violation_reasons.values())),
            "violation_reasons": dict(sorted(violation_reasons.items())),
            "deterministic_reconstruction_mismatches": (
                reconstruction_mismatches
            ),
            "rule_statistics": rule_statistics,
            "interpretation_limit": (
                "The checks validate the declared synthetic one-dimensional "
                "geology rules; they do not establish agreement with field geology."
            ),
        },
    }


def write_audit_report(path: Path | str, report: dict[str, object]) -> Path:
    """Atomically write a strict JSON audit report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target
