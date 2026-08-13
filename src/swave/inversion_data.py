"""Validated observation-only access to the deterministic inversion split."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .dataset import validate_dataset_files
from .geology import ModelKind
from .splits import Split, mask_for_split


@dataclass(frozen=True)
class InversionSample:
    """One observed inversion curve and its immutable source provenance."""

    sample_id: int
    model_kind: int
    phase_velocity: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    source_path: Path
    source_shard_id: int
    source_row: int


def iter_observation_samples(
    dataset_dir: Path | str, split: Split
) -> Iterator[InversionSample]:
    """Yield one validated split without exposing target Vs profiles."""
    directory = Path(dataset_dir)
    validate_dataset_files(directory)
    for path in sorted(directory.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            rows = np.flatnonzero(mask_for_split(sample_ids, split))
            shard_id = int(handle.attrs["shard_id"])
            for row in rows:
                yield InversionSample(
                    sample_id=int(sample_ids[row]),
                    model_kind=int(handle["model_kind"][row]),
                    phase_velocity=np.asarray(
                        handle["phase_velocity"][row], dtype=np.float32
                    ),
                    valid_mask=np.asarray(handle["valid_mask"][row], dtype=np.bool_),
                    source_path=path,
                    source_shard_id=shard_id,
                    source_row=int(row),
                )


def iter_inversion_samples(
    dataset_dir: Path | str,
) -> Iterator[InversionSample]:
    """Yield validated inversion rows without exposing their target Vs profiles."""
    yield from iter_observation_samples(dataset_dir, "inversion")


def samples_by_source_shard(
    dataset_dir: Path | str,
) -> dict[int, list[InversionSample]]:
    """Group inversion rows by immutable shard source in stable sample order."""
    grouped: dict[int, list[InversionSample]] = {}
    for sample in iter_inversion_samples(dataset_dir):
        grouped.setdefault(sample.source_shard_id, []).append(sample)
    for samples in grouped.values():
        samples.sort(key=lambda sample: sample.sample_id)
    return dict(sorted(grouped.items()))


def select_deep_samples(
    dataset_dir: Path | str, per_kind: int
) -> list[InversionSample]:
    """Select the lowest-ID inversion samples from every model family."""
    return select_observation_samples_by_kind(dataset_dir, "inversion", per_kind)


def select_observation_samples_by_kind(
    dataset_dir: Path | str, split: Split, per_kind: int
) -> list[InversionSample]:
    """Select the lowest-ID observation-only rows per family from one split."""
    if per_kind <= 0:
        raise ValueError("per_kind must be positive")

    samples_by_kind: dict[ModelKind, list[InversionSample]] = {
        kind: [] for kind in ModelKind
    }
    for sample in iter_observation_samples(dataset_dir, split):
        samples_by_kind[ModelKind(sample.model_kind)].append(sample)

    deficient: list[str] = []
    selected: list[InversionSample] = []
    for kind in ModelKind:
        samples = sorted(samples_by_kind[kind], key=lambda sample: sample.sample_id)
        if len(samples) < per_kind:
            deficient.append(f"{kind.name} ({len(samples)}/{per_kind})")
        selected.extend(samples[:per_kind])
    if deficient:
        raise ValueError(
            f"insufficient {split} samples for model families: "
            + ", ".join(deficient)
        )
    return selected
