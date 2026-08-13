"""Leakage-safe supervised inversion training and ensemble evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import (
    dataset_manifest_sha256,
    validate_dataset_files,
)
from .geology import ModelKind
from .inference import resolve_device
from .splits import SPLIT_POLICY, Split, mask_for_split

DROP_FREQUENCY_COLUMNS = 1
INPUT_DIMENSION = 4 * (120 - DROP_FREQUENCY_COLUMNS)
OUTPUT_DIMENSION = 20


@dataclass(frozen=True)
class SupervisedConfig:
    """Validated controls for the direct supervised inversion baseline."""

    dataset_dir: Path = Path("data/production")
    output_dir: Path = Path("runs/supervised-inversion-v2")
    seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 1024
    blocks: int = 4
    dropout: float = 0.0
    batch_size: int = 8192
    epochs: int = 150
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 16
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        integer_fields = (
            "width",
            "blocks",
            "batch_size",
            "epochs",
            "patience",
            "num_workers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.width <= 0 or self.blocks < 0:
            raise ValueError("width must be positive and blocks nonnegative")
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.seeds
            )
        ):
            raise ValueError("seeds must be unique nonnegative integers")
        floats = (self.dropout, self.learning_rate, self.weight_decay)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in floats
        ):
            raise ValueError("dropout and optimizer values must be finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> SupervisedConfig:
        values = dict(mapping.get("supervised", mapping))
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown supervised keys: {sorted(unknown)}")
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dataset_dir"] = self.dataset_dir.as_posix()
        payload["output_dir"] = self.output_dir.as_posix()
        payload["seeds"] = list(self.seeds)
        return payload


def load_supervised_config(path: Path | str) -> SupervisedConfig:
    with Path(path).open("rb") as handle:
        return SupervisedConfig.from_mapping(tomllib.load(handle))


@dataclass(frozen=True)
class SupervisedNormalization:
    """Train-only fill and z-score statistics in physical units."""

    fill_values: NDArray[np.float32]
    input_mean: NDArray[np.float32]
    input_std: NDArray[np.float32]
    target_mean: NDArray[np.float32]
    target_std: NDArray[np.float32]
    train_sample_count: int
    train_sample_id_sha256: str


class PreNormResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(self.norm(value))


class InverseNet(nn.Module):
    """Map normalized four-mode dispersion to normalized twenty-layer Vs."""

    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIMENSION,
        output_dim: int = OUTPUT_DIMENSION,
        width: int = 1024,
        blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or width <= 0 or blocks < 0:
            raise ValueError("network dimensions must be positive")
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "width": width,
            "blocks": blocks,
            "dropout": dropout,
        }
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.backbone = nn.Sequential(
            *(PreNormResidualBlock(width, dropout) for _ in range(blocks))
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 2 or value.shape[1] != self.config["input_dim"]:
            raise ValueError(
                f"input must have shape (batch, {self.config['input_dim']})"
            )
        return self.head(self.backbone(self.stem(value)))


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _normalization_from_checkpoint(
    payload: dict[str, object],
) -> SupervisedNormalization:
    arrays = {
        "fill_values": np.asarray(payload.get("fill_values"), dtype=np.float32),
        "input_mean": np.asarray(payload.get("input_mean"), dtype=np.float32),
        "input_std": np.asarray(payload.get("input_std"), dtype=np.float32),
        "target_mean": np.asarray(payload.get("target_mean"), dtype=np.float32),
        "target_std": np.asarray(payload.get("target_std"), dtype=np.float32),
    }
    expected_shapes = {
        "fill_values": (4, 119),
        "input_mean": (INPUT_DIMENSION,),
        "input_std": (INPUT_DIMENSION,),
        "target_mean": (OUTPUT_DIMENSION,),
        "target_std": (OUTPUT_DIMENSION,),
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name] or not np.all(np.isfinite(values)):
            raise ValueError(f"supervised checkpoint {name} is invalid")
    if np.any(arrays["input_std"] <= 0) or np.any(arrays["target_std"] <= 0):
        raise ValueError("supervised checkpoint normalization scales must be positive")
    train_sample_count = payload.get("train_sample_count")
    train_sample_digest = payload.get("train_sample_id_sha256")
    if (
        isinstance(train_sample_count, bool)
        or not isinstance(train_sample_count, int)
        or train_sample_count <= 0
        or not isinstance(train_sample_digest, str)
        or len(train_sample_digest) != 64
    ):
        raise ValueError("supervised checkpoint training identity is invalid")
    return SupervisedNormalization(
        fill_values=arrays["fill_values"].copy(),
        input_mean=arrays["input_mean"].copy(),
        input_std=arrays["input_std"].copy(),
        target_mean=arrays["target_mean"].copy(),
        target_std=arrays["target_std"].copy(),
        train_sample_count=train_sample_count,
        train_sample_id_sha256=train_sample_digest,
    )


def _same_normalization(
    left: SupervisedNormalization, right: SupervisedNormalization
) -> bool:
    return (
        left.train_sample_count == right.train_sample_count
        and left.train_sample_id_sha256 == right.train_sample_id_sha256
        and np.array_equal(left.fill_values, right.fill_values)
        and np.array_equal(left.input_mean, right.input_mean)
        and np.array_equal(left.input_std, right.input_std)
        and np.array_equal(left.target_mean, right.target_mean)
        and np.array_equal(left.target_std, right.target_std)
    )


@dataclass
class SupervisedEnsemblePredictor:
    """Validated equal-weight inference over fixed supervised best checkpoints."""

    models: tuple[InverseNet, ...]
    normalization: SupervisedNormalization
    seeds: tuple[int, ...]
    device: torch.device
    checkpoint_sha256: tuple[str, ...]

    @classmethod
    def load(
        cls, output_dir: Path | str, device: str = "auto"
    ) -> SupervisedEnsemblePredictor:
        directory = Path(output_dir)
        identity_path = directory / "run-identity.json"
        try:
            with identity_path.open(encoding="utf-8") as handle:
                identity = json.load(
                    handle, object_pairs_hook=_json_object_without_duplicate_keys
                )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("supervised run identity is not readable JSON") from error
        if not isinstance(identity, dict):
            raise TypeError("supervised run identity must be a JSON object")
        raw_seeds = identity.get("seed_ensemble")
        if (
            not isinstance(raw_seeds, list)
            or not raw_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in raw_seeds
            )
            or len(set(raw_seeds)) != len(raw_seeds)
        ):
            raise ValueError("supervised run seed ensemble is invalid")
        seeds = tuple(raw_seeds)
        selected_device = resolve_device(device)
        models: list[InverseNet] = []
        digests: list[str] = []
        common_normalization: SupervisedNormalization | None = None
        for seed in seeds:
            path = directory / f"seed-{seed}-best.pt"
            if not path.is_file():
                raise ValueError(f"supervised checkpoint {path.name} is missing")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError("supervised checkpoint payload must be a mapping")
            _validate_checkpoint_identity(payload, identity)
            if payload.get("seed") != seed:
                raise ValueError("supervised checkpoint seed does not match")
            normalization = _normalization_from_checkpoint(payload)
            if common_normalization is None:
                common_normalization = normalization
            elif not _same_normalization(common_normalization, normalization):
                raise ValueError("supervised checkpoint normalization does not match")
            model_config = payload.get("model_config")
            if not isinstance(model_config, dict):
                raise TypeError("supervised checkpoint model configuration is invalid")
            model = InverseNet(**model_config)
            model.load_state_dict(payload["model"])
            model.to(selected_device)
            model.eval()
            models.append(model)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        assert common_normalization is not None
        return cls(
            models=tuple(models),
            normalization=common_normalization,
            seeds=seeds,
            device=selected_device,
            checkpoint_sha256=tuple(digests),
        )

    def predict(
        self, observed: ArrayLike, valid_mask: ArrayLike
    ) -> NDArray[np.float64]:
        """Return one physical twenty-layer equal-weight ensemble prediction."""
        values = np.asarray(observed, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=np.bool_)
        if values.shape != (4, 120) or mask.shape != (4, 120):
            raise ValueError("observed and valid_mask must have shape (4, 120)")
        if not np.all(np.isfinite(values[mask])):
            raise ValueError("valid observations must be finite")
        phase = values[:, DROP_FREQUENCY_COLUMNS:]
        active = mask[:, DROP_FREQUENCY_COLUMNS:]
        filled = np.where(active, phase, self.normalization.fill_values).reshape(-1)
        normalized = (
            filled - self.normalization.input_mean
        ) / self.normalization.input_std
        tensor = torch.from_numpy(normalized.astype(np.float32)).to(self.device)
        target_mean = torch.as_tensor(
            self.normalization.target_mean, device=self.device
        )
        target_std = torch.as_tensor(
            self.normalization.target_std, device=self.device
        )
        predictions: list[Tensor] = []
        with torch.no_grad():
            for model in self.models:
                predictions.append(model(tensor.unsqueeze(0))[0] * target_std + target_mean)
        ensemble = torch.stack(predictions).mean(dim=0)
        if not bool(torch.isfinite(ensemble).all()):
            raise ArithmeticError("supervised ensemble prediction is non-finite")
        return np.asarray(ensemble.cpu().numpy(), dtype=np.float64).copy()


def compute_supervised_normalization(
    dataset_dir: Path | str,
) -> SupervisedNormalization:
    """Compute fill and normalization arrays from train rows only."""
    directory = Path(dataset_dir)
    input_sum = np.zeros((4, 119), dtype=np.float64)
    input_square_sum = np.zeros((4, 119), dtype=np.float64)
    input_valid_count = np.zeros((4, 119), dtype=np.int64)
    target_sum = np.zeros(20, dtype=np.float64)
    target_square_sum = np.zeros(20, dtype=np.float64)
    train_count = 0
    id_digest = hashlib.sha256()

    for path in sorted(directory.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            selected = mask_for_split(sample_ids, "train")
            if not np.any(selected):
                continue
            ids = np.ascontiguousarray(sample_ids[selected])
            phase = np.asarray(
                handle["phase_velocity"][selected, :, DROP_FREQUENCY_COLUMNS:],
                dtype=np.float64,
            )
            valid = np.asarray(
                handle["valid_mask"][selected, :, DROP_FREQUENCY_COLUMNS:],
                dtype=np.bool_,
            )
            target = np.asarray(handle["vs"][selected], dtype=np.float64)
        safe_phase = np.where(valid, phase, 0.0)
        input_sum += safe_phase.sum(axis=0)
        input_square_sum += np.square(safe_phase).sum(axis=0)
        input_valid_count += valid.sum(axis=0)
        target_sum += target.sum(axis=0)
        target_square_sum += np.square(target).sum(axis=0)
        train_count += len(ids)
        id_digest.update(ids.tobytes())

    if train_count == 0:
        raise ValueError("training split is empty")
    mode_sum = input_sum.sum(axis=1)
    mode_count = input_valid_count.sum(axis=1)
    if np.any(mode_count == 0):
        raise ValueError("a complete mode has no valid training values")
    mode_mean = mode_sum / mode_count
    fill = np.divide(
        input_sum,
        input_valid_count,
        out=np.broadcast_to(mode_mean[:, None], input_sum.shape).copy(),
        where=input_valid_count > 0,
    )
    centered_square_sum = (
        input_square_sum
        - 2.0 * fill * input_sum
        + input_valid_count * np.square(fill)
    )
    input_variance = np.maximum(centered_square_sum / train_count, 0.0)
    input_std = np.sqrt(input_variance)
    input_std[input_std < 1e-8] = 1.0

    target_mean = target_sum / train_count
    target_variance = np.maximum(
        target_square_sum / train_count - np.square(target_mean), 0.0
    )
    target_std = np.sqrt(target_variance)
    target_std[target_std < 1e-8] = 1.0
    return SupervisedNormalization(
        fill_values=fill.astype(np.float32),
        input_mean=fill.reshape(-1).astype(np.float32),
        input_std=input_std.reshape(-1).astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        train_sample_count=train_count,
        train_sample_id_sha256=id_digest.hexdigest(),
    )


class SupervisedHDF5Dataset(
    Dataset[tuple[Tensor, Tensor, int, int]]
):
    """Map-style four-way split reader with train-derived preprocessing."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: Split,
        normalization: SupervisedNormalization,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.normalization = normalization
        self.entries: list[tuple[Path, int, int, int]] = []
        self._handles: dict[Path, h5py.File] = {}
        for path in sorted(self.dataset_dir.glob("shard-*.h5")):
            with h5py.File(path, "r") as handle:
                sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
                kinds = np.asarray(handle["model_kind"], dtype=np.uint8)
            for row in np.flatnonzero(mask_for_split(sample_ids, split)):
                self.entries.append(
                    (path, int(row), int(sample_ids[row]), int(kinds[row]))
                )

    def __len__(self) -> int:
        return len(self.entries)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int, int]:
        path, row, sample_id, kind = self.entries[index]
        handle = self._handle(path)
        phase = np.asarray(
            handle["phase_velocity"][row, :, DROP_FREQUENCY_COLUMNS:],
            dtype=np.float32,
        )
        valid = np.asarray(
            handle["valid_mask"][row, :, DROP_FREQUENCY_COLUMNS:],
            dtype=np.bool_,
        )
        filled = np.where(valid, phase, self.normalization.fill_values).reshape(-1)
        normalized_input = (
            filled - self.normalization.input_mean
        ) / self.normalization.input_std
        target = np.asarray(handle["vs"][row], dtype=np.float32)
        normalized_target = (
            target - self.normalization.target_mean
        ) / self.normalization.target_std
        return (
            torch.from_numpy(normalized_input.astype(np.float32, copy=False)),
            torch.from_numpy(normalized_target.astype(np.float32, copy=False)),
            sample_id,
            kind,
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


class SupervisedHDF5BatchDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]
):
    """Read contiguous HDF5 spans and return one preprocessed tensor batch."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: Split,
        normalization: SupervisedNormalization,
        *,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.normalization = normalization
        self.entries: list[tuple[Path, tuple[tuple[int, int], ...]]] = []
        self._handles: dict[Path, h5py.File] = {}
        for path in sorted(self.dataset_dir.glob("shard-*.h5")):
            with h5py.File(path, "r") as handle:
                sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            selected_rows = np.flatnonzero(mask_for_split(sample_ids, split))
            if not len(selected_rows):
                continue
            breaks = np.flatnonzero(np.diff(selected_rows) > 1) + 1
            runs = np.split(selected_rows, breaks)
            spans: list[tuple[int, int]] = []
            count = 0
            for run in runs:
                position = int(run[0])
                run_stop = int(run[-1]) + 1
                while position < run_stop:
                    take = min(batch_size - count, run_stop - position)
                    spans.append((position, position + take))
                    position += take
                    count += take
                    if count == batch_size:
                        self.entries.append((path, tuple(spans)))
                        spans = []
                        count = 0
            if spans:
                self.entries.append((path, tuple(spans)))

    def __len__(self) -> int:
        return len(self.entries)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        path, spans = self.entries[index]
        handle = self._handle(path)
        sample_ids = np.concatenate(
            [
                np.asarray(handle["sample_id"][start:stop], dtype=np.int64)
                for start, stop in spans
            ]
        )
        phase = np.concatenate(
            [
                np.asarray(
                    handle["phase_velocity"][
                        start:stop, :, DROP_FREQUENCY_COLUMNS:
                    ],
                    dtype=np.float32,
                )
                for start, stop in spans
            ]
        )
        valid = np.concatenate(
            [
                np.asarray(
                    handle["valid_mask"][
                        start:stop, :, DROP_FREQUENCY_COLUMNS:
                    ],
                    dtype=np.bool_,
                )
                for start, stop in spans
            ]
        )
        target = np.concatenate(
            [
                np.asarray(handle["vs"][start:stop], dtype=np.float32)
                for start, stop in spans
            ]
        )
        kinds = np.concatenate(
            [
                np.asarray(handle["model_kind"][start:stop], dtype=np.int64)
                for start, stop in spans
            ]
        )
        filled = np.where(valid, phase, self.normalization.fill_values).reshape(
            len(target), -1
        )
        normalized_input = (
            filled - self.normalization.input_mean
        ) / self.normalization.input_std
        normalized_target = (
            target - self.normalization.target_mean
        ) / self.normalization.target_std
        return (
            torch.from_numpy(normalized_input.astype(np.float32, copy=False)),
            torch.from_numpy(normalized_target.astype(np.float32, copy=False)),
            torch.from_numpy(sample_ids),
            torch.from_numpy(kinds),
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


class EpochShuffleSampler(Sampler[int]):
    """Deterministically shuffle batch indexes from only seed and epoch."""

    def __init__(self, *, size: int, seed: int) -> None:
        if size < 0 or seed < 0:
            raise ValueError("size and seed must be nonnegative")
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            _derived_seed(self.seed, self.epoch, "batch-order")
        )
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _normalization_payload(
    normalization: SupervisedNormalization,
) -> dict[str, object]:
    return {
        "fill_values": normalization.fill_values,
        "input_mean": normalization.input_mean,
        "input_std": normalization.input_std,
        "target_mean": normalization.target_mean,
        "target_std": normalization.target_std,
    }


def _checkpoint_identity(
    config: SupervisedConfig,
    normalization: SupervisedNormalization,
    dataset_config_hash: str,
    manifest_sha256: str,
) -> dict[str, object]:
    training_identity = {
        "seeds": list(config.seeds),
        "width": config.width,
        "blocks": config.blocks,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    training_configuration_sha256 = hashlib.sha256(
        json.dumps(
            training_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "split_policy": SPLIT_POLICY,
        "dataset_config_hash": dataset_config_hash,
        "dataset_manifest_sha256": manifest_sha256,
        "train_sample_count": normalization.train_sample_count,
        "train_sample_id_sha256": normalization.train_sample_id_sha256,
        "training_configuration_sha256": training_configuration_sha256,
        "seed_ensemble": list(config.seeds),
        "epoch_randomness": "sha256-derived-from-seed-and-epoch-v1",
        "batch_order": "contiguous-hdf5-spans-epoch-shuffled-v1",
        "model_config": {
            "input_dim": INPUT_DIMENSION,
            "output_dim": OUTPUT_DIMENSION,
            "width": config.width,
            "blocks": config.blocks,
            "dropout": config.dropout,
        },
    }


def _validate_checkpoint_identity(
    payload: dict[str, object], identity: dict[str, object]
) -> None:
    for key, value in identity.items():
        if payload.get(key) != value:
            if key == "training_configuration_sha256":
                raise ValueError(
                    "supervised checkpoint training configuration does not match"
                )
            raise ValueError(f"supervised checkpoint {key} does not match")


def _physical_validation_mae(
    model: InverseNet,
    loader: DataLoader[tuple[Tensor, Tensor, int, int]],
    normalization: SupervisedNormalization,
    device: torch.device,
) -> float:
    target_mean = torch.as_tensor(normalization.target_mean, device=device)
    target_std = torch.as_tensor(normalization.target_std, device=device)
    absolute_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for values, targets, _, _ in loader:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(values) * target_std + target_mean
            physical_targets = targets * target_std + target_mean
            absolute_sum += (predictions - physical_targets).abs().sum().item()
            count += targets.numel()
    if count == 0:
        raise ValueError("validation split has no target cells")
    return absolute_sum / count


def _loader_worker_options(num_workers: int) -> dict[str, object]:
    if num_workers == 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 1}


def _checkpoint_payload(
    *,
    model: InverseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_mae: float,
    bad_epochs: int,
    history: list[dict[str, float | int]],
    config: SupervisedConfig,
    normalization: SupervisedNormalization,
    identity: dict[str, object],
    seed: int,
) -> dict[str, object]:
    return {
        **identity,
        **_normalization_payload(normalization),
        "seed": seed,
        "training_config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_mae_km_s": best_validation_mae,
        "bad_epochs": bad_epochs,
        "history": history,
    }


def _derived_seed(seed: int, epoch: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_seed(
    config: SupervisedConfig,
    seed: int,
    normalization: SupervisedNormalization,
    identity: dict[str, object],
    device: torch.device,
) -> Path:
    _seed_everything(seed)
    model = InverseNet(
        width=config.width,
        blocks=config.blocks,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    loss_function = nn.HuberLoss(delta=1.0)
    best_path = config.output_dir / f"seed-{seed}-best.pt"
    last_path = config.output_dir / f"seed-{seed}-last.pt"
    history_path = config.output_dir / f"seed-{seed}-history.json"
    start_epoch = 0
    best_validation_mae = float("inf")
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    if config.resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_identity(payload, identity)
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("supervised checkpoint seed does not match")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation_mae = float(payload["best_validation_mae_km_s"])
        bad_epochs = int(payload["bad_epochs"])
        history = list(payload["history"])
        if bad_epochs >= config.patience:
            if not best_path.exists():
                raise ValueError("terminal supervised run has no best checkpoint")
            return best_path

    train_dataset = SupervisedHDF5BatchDataset(
        config.dataset_dir,
        "train",
        normalization,
        batch_size=config.batch_size,
    )
    validation_dataset = SupervisedHDF5BatchDataset(
        config.dataset_dir,
        "validation",
        normalization,
        batch_size=config.batch_size,
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("train and validation splits must be nonempty")
    train_sampler = EpochShuffleSampler(size=len(train_dataset), seed=seed)
    loader_generator = torch.Generator().manual_seed(
        _derived_seed(seed, 0, "data-loader-workers")
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        **_loader_worker_options(config.num_workers),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        **_loader_worker_options(config.num_workers),
    )
    for epoch in range(start_epoch, config.epochs):
        _seed_everything(_derived_seed(seed, epoch, "training"))
        train_sampler.set_epoch(epoch)
        model.train()
        training_loss_sum = 0.0
        training_rows = 0
        for values, targets, _, _ in train_loader:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = loss_function(model(values), targets)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item() * len(values)
            training_rows += len(values)
        validation_mae = _physical_validation_mae(
            model, validation_loader, normalization, device
        )
        improved = validation_mae < best_validation_mae
        if improved:
            best_validation_mae = validation_mae
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "training_huber_loss": training_loss_sum / training_rows,
                "validation_mae_km_s": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_mae=best_validation_mae,
            bad_epochs=bad_epochs,
            history=history,
            config=config,
            normalization=normalization,
            identity=identity,
            seed=seed,
        )
        if improved:
            _atomic_torch_save(best_path, payload)
        _atomic_json_save(
            history_path,
            {"seed": seed, "epochs": history},
        )
        # Publish the resumable checkpoint last: its presence commits the
        # epoch only after every artifact needed to describe it is durable.
        _atomic_torch_save(last_path, payload)
        print(
            json.dumps(
                {
                    "event": "supervised_epoch",
                    "seed": seed,
                    **history[-1],
                    "best_validation_mae_km_s": best_validation_mae,
                    "bad_epochs": bad_epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if bad_epochs >= config.patience:
            break
    if not best_path.exists():
        raise ValueError(f"seed {seed} has no best checkpoint")
    return best_path


def _model_from_checkpoint(
    path: Path,
    identity: dict[str, object],
    device: torch.device,
) -> InverseNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_identity(payload, identity)
    model = InverseNet(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def _collect_predictions(
    dataset_dir: Path,
    split: Split,
    normalization: SupervisedNormalization,
    models: list[InverseNet],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]], NDArray[np.int64]]:
    dataset = SupervisedHDF5BatchDataset(
        dataset_dir,
        split,
        normalization,
        batch_size=batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **_loader_worker_options(num_workers),
    )
    target_mean = torch.as_tensor(normalization.target_mean, device=device)
    target_std = torch.as_tensor(normalization.target_std, device=device)
    truth_parts: list[NDArray[np.float32]] = []
    kind_parts: list[NDArray[np.int64]] = []
    prediction_parts: list[list[NDArray[np.float32]]] = [
        [] for _ in models
    ]
    with torch.no_grad():
        for values, targets, _, kinds in loader:
            values = values.to(device, non_blocking=True)
            truth = targets.to(device, non_blocking=True) * target_std + target_mean
            truth_parts.append(truth.cpu().numpy())
            kind_parts.append(np.asarray(kinds, dtype=np.int64))
            for index, model in enumerate(models):
                prediction = model(values) * target_std + target_mean
                prediction_parts[index].append(prediction.cpu().numpy())
    return (
        np.concatenate(truth_parts).astype(np.float32, copy=False),
        [
            np.concatenate(parts).astype(np.float32, copy=False)
            for parts in prediction_parts
        ],
        np.concatenate(kind_parts),
    )


def _r2_by_layer(
    truth: NDArray[np.float32], prediction: NDArray[np.float32]
) -> list[float | None]:
    result: list[float | None] = []
    for layer in range(truth.shape[1]):
        residual = float(np.square(prediction[:, layer] - truth[:, layer]).sum())
        centered = truth[:, layer] - truth[:, layer].mean()
        total = float(np.square(centered).sum())
        result.append(None if total == 0 else 1.0 - residual / total)
    return result


def _metrics(
    truth: NDArray[np.float32], prediction: NDArray[np.float32]
) -> dict[str, object]:
    difference = prediction - truth
    absolute = np.abs(difference)
    r2_layers = _r2_by_layer(truth, prediction)
    finite_r2 = [value for value in r2_layers if value is not None]
    return {
        "mae_km_s": float(absolute.mean()),
        "rmse_km_s": float(np.sqrt(np.square(difference).mean())),
        "p95_absolute_error_km_s": float(np.percentile(absolute, 95)),
        "r2_mean_across_layers": (
            float(np.mean(finite_r2)) if finite_r2 else None
        ),
    }


def _test_metrics(
    truth: NDArray[np.float32],
    prediction: NDArray[np.float32],
    kinds: NDArray[np.int64],
) -> dict[str, object]:
    by_kind: dict[str, object] = {}
    for kind in ModelKind:
        selected = kinds == int(kind)
        if np.any(selected):
            by_kind[kind.name] = {
                "sample_count": int(selected.sum()),
                **_metrics(truth[selected], prediction[selected]),
            }
    r2_layers = _r2_by_layer(truth, prediction)
    per_layer = []
    for layer in range(truth.shape[1]):
        difference = prediction[:, layer] - truth[:, layer]
        absolute = np.abs(difference)
        per_layer.append(
            {
                "layer_index": layer,
                "depth_km": layer * 0.1,
                "mae_km_s": float(absolute.mean()),
                "rmse_km_s": float(np.sqrt(np.square(difference).mean())),
                "bias_km_s": float(difference.mean()),
                "p95_absolute_error_km_s": float(np.percentile(absolute, 95)),
                "r2": r2_layers[layer],
            }
        )
    return {
        "sample_count": len(truth),
        "overall": _metrics(truth, prediction),
        "by_model_kind": by_kind,
        "per_layer": per_layer,
    }


def _split_counts(dataset_dir: Path) -> dict[str, int]:
    counts = {split: 0 for split in ("train", "validation", "test", "inversion")}
    for path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
        for split in counts:
            counts[split] += int(mask_for_split(sample_ids, split).sum())
    return counts


def train_supervised(config: SupervisedConfig) -> Path:
    """Train all configured seeds and write one final ensemble evaluation."""
    manifest = validate_dataset_files(config.dataset_dir)
    manifest_digest = dataset_manifest_sha256(manifest)
    normalization = compute_supervised_normalization(config.dataset_dir)
    identity = _checkpoint_identity(
        config,
        normalization,
        manifest.config_hash,
        manifest_digest,
    )
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_identity_path = config.output_dir / "run-identity.json"
    if run_identity_path.exists():
        with run_identity_path.open(encoding="utf-8") as handle:
            stored_identity = json.load(handle)
        _validate_checkpoint_identity(stored_identity, identity)
    else:
        existing_outputs = list(config.output_dir.glob("seed-*-*.pt"))
        if existing_outputs or (config.output_dir / "evaluation.json").exists():
            raise ValueError(
                "supervised output exists without a run training configuration"
            )
        _atomic_json_save(run_identity_path, identity)
    best_paths = [
        _train_seed(config, seed, normalization, identity, device)
        for seed in config.seeds
    ]
    models = [
        _model_from_checkpoint(path, identity, device) for path in best_paths
    ]
    validation_truth, validation_predictions, _ = _collect_predictions(
        config.dataset_dir,
        "validation",
        normalization,
        models,
        config.batch_size,
        config.num_workers,
        device,
    )
    validation_ensemble = np.mean(validation_predictions, axis=0)
    test_truth, test_predictions, test_kinds = _collect_predictions(
        config.dataset_dir,
        "test",
        normalization,
        models,
        config.batch_size,
        config.num_workers,
        device,
    )
    test_ensemble = np.mean(test_predictions, axis=0)
    inversion_truth, inversion_predictions, inversion_kinds = _collect_predictions(
        config.dataset_dir,
        "inversion",
        normalization,
        models,
        config.batch_size,
        config.num_workers,
        device,
    )
    inversion_ensemble = np.mean(inversion_predictions, axis=0)
    report = {
        **identity,
        "seeds": list(config.seeds),
        "splits": _split_counts(config.dataset_dir),
        "validation": {
            "sample_count": len(validation_truth),
            "per_seed_mae_km_s": {
                str(seed): float(np.abs(prediction - validation_truth).mean())
                for seed, prediction in zip(
                    config.seeds, validation_predictions, strict=True
                )
            },
            "ensemble": _metrics(validation_truth, validation_ensemble),
        },
        "test": _test_metrics(test_truth, test_ensemble, test_kinds),
        "inversion_comparison": {
            "usage": "post_training_same_sample_comparison_only",
            **_test_metrics(
                inversion_truth,
                inversion_ensemble,
                inversion_kinds,
            ),
        },
    }
    output = config.output_dir / "evaluation.json"
    _atomic_json_save(output, report)
    return output
