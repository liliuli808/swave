"""Streaming HDF5 training, checkpointing, and physical error metrics."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .dataset import validate_dataset_files
from .inference import ForwardPredictor, resolve_device
from .network import FourHeadForwardModel, masked_smooth_l1
from .splits import SPLIT_POLICY, Split, mask_for_split


@dataclass(frozen=True)
class Normalization:
    input_mean: NDArray[np.float32]
    input_std: NDArray[np.float32]
    target_mean: NDArray[np.float32]
    target_std: NDArray[np.float32]


class HDF5ShardDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Map-style row index over deterministic dataset shards."""

    def __init__(self, dataset_dir: Path | str, split: Split = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.entries: list[tuple[Path, int]] = []
        self._handles: dict[Path, h5py.File] = {}
        for path in sorted(self.dataset_dir.glob("shard-*.h5")):
            with h5py.File(path, "r") as handle:
                sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            selected = mask_for_split(sample_ids, split)
            self.entries.extend(
                (path, int(row)) for row in np.flatnonzero(selected)
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, row = self.entries[index]
        handle = self._handle(path)
        vs = np.asarray(handle["vs"][row], dtype=np.float32)
        target = np.asarray(handle["phase_velocity"][row], dtype=np.float32)
        mask = np.asarray(handle["valid_mask"][row], dtype=np.bool_)
        target = np.where(mask, target, 0.0).astype(np.float32, copy=False)
        return (
            torch.from_numpy(vs.copy()),
            torch.from_numpy(target.copy()),
            torch.from_numpy(mask.copy()),
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


def compute_normalization(dataset: HDF5ShardDataset) -> Normalization:
    """Compute train-only input and valid-target moments in shard batches."""
    if not dataset.entries:
        raise ValueError("cannot normalize an empty dataset split")
    rows_by_path: dict[Path, list[int]] = defaultdict(list)
    for path, row in dataset.entries:
        rows_by_path[path].append(row)

    input_sum = np.zeros(20, dtype=np.float64)
    input_square_sum = np.zeros(20, dtype=np.float64)
    input_count = 0
    target_sum = np.zeros(4, dtype=np.float64)
    target_square_sum = np.zeros(4, dtype=np.float64)
    target_count = np.zeros(4, dtype=np.int64)
    for path, selected_rows in rows_by_path.items():
        with h5py.File(path, "r") as handle:
            for offset in range(0, len(selected_rows), 8192):
                rows = selected_rows[offset : offset + 8192]
                vs = np.asarray(handle["vs"][rows], dtype=np.float64)
                target = np.asarray(
                    handle["phase_velocity"][rows], dtype=np.float64
                )
                mask = np.asarray(handle["valid_mask"][rows], dtype=np.bool_)
                input_sum += vs.sum(axis=0)
                input_square_sum += np.square(vs).sum(axis=0)
                input_count += vs.shape[0]
                safe_target = np.where(mask, target, 0.0)
                target_sum += safe_target.sum(axis=(0, 2))
                target_square_sum += np.square(safe_target).sum(axis=(0, 2))
                target_count += mask.sum(axis=(0, 2))

    input_mean = input_sum / input_count
    input_variance = np.maximum(
        input_square_sum / input_count - np.square(input_mean), 0.0
    )
    input_std = np.sqrt(input_variance)
    input_std[input_std < 1e-8] = 1.0

    target_mean = np.divide(
        target_sum,
        target_count,
        out=np.zeros_like(target_sum),
        where=target_count > 0,
    )
    target_variance = np.divide(
        target_square_sum,
        target_count,
        out=np.zeros_like(target_square_sum),
        where=target_count > 0,
    ) - np.square(target_mean)
    target_std = np.sqrt(np.maximum(target_variance, 0.0))
    target_std[(target_std < 1e-8) | (target_count == 0)] = 1.0
    return Normalization(
        input_mean=input_mean.astype(np.float32),
        input_std=input_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32)[:, None],
        target_std=target_std.astype(np.float32)[:, None],
    )


def _normalization_from_checkpoint(payload: dict[str, object]) -> Normalization:
    return Normalization(
        input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
        input_std=np.asarray(payload["input_std"], dtype=np.float32),
        target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
        target_std=np.asarray(payload["target_std"], dtype=np.float32),
    )


def _validation_mae(
    model: FourHeadForwardModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    normalization: Normalization,
    device: torch.device,
) -> float:
    input_mean = torch.as_tensor(normalization.input_mean, device=device)
    input_std = torch.as_tensor(normalization.input_std, device=device)
    target_mean = torch.as_tensor(normalization.target_mean, device=device)
    target_std = torch.as_tensor(normalization.target_std, device=device)
    absolute_sum = 0.0
    valid_count = 0
    model.eval()
    with torch.inference_mode():
        for vs, target, mask in loader:
            vs = vs.to(device)
            target = target.to(device)
            mask = mask.to(device)
            prediction = model((vs - input_mean) / input_std)
            prediction = prediction * target_std + target_mean
            absolute_sum += torch.abs(prediction - target)[mask].sum().item()
            valid_count += int(mask.sum().item())
    if valid_count == 0:
        raise ValueError("validation split has no valid targets")
    return absolute_sum / valid_count


def _checkpoint_payload(
    *,
    model: FourHeadForwardModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_mae: float,
    normalization: Normalization,
    dataset_hash: str,
    config: TrainingConfig,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_mae": best_mae,
        "input_mean": normalization.input_mean,
        "input_std": normalization.input_std,
        "target_mean": normalization.target_mean,
        "target_std": normalization.target_std,
        "dataset_config_hash": dataset_hash,
        "split_policy": SPLIT_POLICY,
        "training_config": config.to_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_history(path: Path, history: dict[str, list[dict[str, float | int]]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_resume_config(
    saved: dict[str, object], current: TrainingConfig
) -> None:
    """Allow execution-location changes but reject optimization changes."""
    allowed_differences = {"device", "num_workers", "resume"}
    current_values = current.to_dict()
    incompatible = [
        key
        for key in sorted(set(saved) | set(current_values))
        if key not in allowed_differences and saved.get(key) != current_values.get(key)
    ]
    if incompatible:
        raise ValueError(
            "checkpoint training configuration does not match: "
            + ", ".join(incompatible)
        )


def train(config: TrainingConfig) -> Path:
    """Train or resume the four-head model and return the best checkpoint."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    manifest = validate_dataset_files(config.dataset_dir)
    training = HDF5ShardDataset(config.dataset_dir, "train")
    validation = HDF5ShardDataset(config.dataset_dir, "validation")
    if not training or not validation:
        raise ValueError("dataset must contain nonempty train and validation splits")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = config.output_dir / "last.pt"
    best_path = config.output_dir / "best.pt"
    history_path = config.output_dir / "history.json"

    model = FourHeadForwardModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    normalization = compute_normalization(training)
    start_epoch = 0
    best_mae = float("inf")
    if config.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload["dataset_config_hash"] != manifest.config_hash:
            raise ValueError("checkpoint dataset configuration hash does not match")
        _validate_resume_config(payload["training_config"], config)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        normalization = _normalization_from_checkpoint(payload)
        start_epoch = int(payload["epoch"]) + 1
        best_mae = float(payload["best_validation_mae"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        np.random.set_state(payload["numpy_rng_state"])
    history: dict[str, list[dict[str, float | int]]] = {"epochs": []}
    if config.resume and history_path.exists():
        with history_path.open(encoding="utf-8") as handle:
            loaded_history = json.load(handle)
        history["epochs"] = [
            item
            for item in loaded_history.get("epochs", [])
            if int(item["epoch"]) < start_epoch
        ]

    validation_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    input_mean = torch.as_tensor(normalization.input_mean, device=device)
    input_std = torch.as_tensor(normalization.input_std, device=device)
    target_mean = torch.as_tensor(normalization.target_mean, device=device)
    target_std = torch.as_tensor(normalization.target_std, device=device)

    for epoch in range(start_epoch, config.epochs):
        generator = torch.Generator()
        generator.manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            training,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=generator,
        )
        model.train()
        training_loss_sum = 0.0
        training_batches = 0
        for vs, target, mask in train_loader:
            vs = vs.to(device)
            target = target.to(device)
            mask = mask.to(device)
            normalized_vs = (vs - input_mean) / input_std
            normalized_target = (target - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            prediction = model(normalized_vs)
            loss = masked_smooth_l1(prediction, normalized_target, mask)
            loss.backward()
            optimizer.step()
            training_loss_sum += float(loss.detach().item())
            training_batches += 1
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        validation_mae = _validation_mae(
            model, validation_loader, normalization, device
        )
        improved = validation_mae < best_mae
        if improved:
            best_mae = validation_mae
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_mae=best_mae,
            normalization=normalization,
            dataset_hash=manifest.config_hash,
            config=config,
        )
        _save_checkpoint(last_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        history["epochs"].append(
            {
                "epoch": epoch,
                "training_loss": training_loss_sum / training_batches,
                "validation_mae_km_s": validation_mae,
                "learning_rate": learning_rate,
            }
        )
        _save_history(history_path, history)

    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    raise ValueError("configured epoch count is already complete but no checkpoint exists")


def evaluate(
    checkpoint: Path | str,
    dataset_dir: Path | str,
    *,
    device: str = "auto",
) -> dict[str, dict[str, float | int]]:
    """Evaluate a checkpoint on the deterministic test split in physical units."""
    dataset_path = Path(dataset_dir)
    manifest = validate_dataset_files(dataset_path)
    payload = torch.load(
        Path(checkpoint), map_location="cpu", weights_only=False
    )
    if payload.get("dataset_config_hash") != manifest.config_hash:
        raise ValueError("checkpoint dataset configuration hash does not match")
    if payload.get("split_policy") != SPLIT_POLICY:
        raise ValueError("checkpoint split policy does not match")
    predictor = ForwardPredictor.load(checkpoint, device=device)
    dataset = HDF5ShardDataset(dataset_path, "test")
    if not dataset:
        raise ValueError("test split is empty")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    absolute: list[list[NDArray[np.float32]]] = [[] for _ in range(4)]
    squared_sum = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    for vs, target, mask in loader:
        prediction = predictor.predict(vs.numpy())
        target_values = target.numpy()
        mask_values = mask.numpy()
        difference = prediction - target_values
        for mode in range(4):
            valid_difference = difference[:, mode][mask_values[:, mode]]
            absolute[mode].append(np.abs(valid_difference))
            squared_sum[mode] += np.square(valid_difference).sum(dtype=np.float64)
            counts[mode] += valid_difference.size

    metrics: dict[str, dict[str, float | int]] = {}
    for mode in range(4):
        errors = (
            np.concatenate(absolute[mode])
            if absolute[mode]
            else np.array([], dtype=np.float32)
        )
        metrics[f"mode_{mode}"] = {
            "mae_km_s": float(errors.mean()) if errors.size else float("nan"),
            "rmse_km_s": (
                float(np.sqrt(squared_sum[mode] / counts[mode]))
                if counts[mode]
                else float("nan")
            ),
            "p95_absolute_error_km_s": (
                float(np.percentile(errors, 95)) if errors.size else float("nan")
            ),
            "valid_count": int(counts[mode]),
        }
    return metrics
