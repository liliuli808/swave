"""Sobolev fine-tuning of the four-head surrogate against physical kernels.

The base surrogate was trained on phase velocities alone. Its curves agree with
the Dunkin solver to about 1e-4 km/s on average, but the worst cells near modal
cut-offs and mode-kissing bends reach several percent, and its autograd
Jacobian ``dc/dVs`` drifts from the physical sensitivity kernel for higher
modes. This stage starts from that checkpoint and minimizes

* the squared *relative* phase-velocity error of every valid cell, and
* the squared relative error of random Jacobian-vector products
  ``J v`` against ``K v`` for physical kernels ``K``
  (``E_v |(J - K) v|^2 = ||J - K||_F^2`` for ``v ~ N(0, I)``),

so that the network agrees with the solver both in value and in derivative.
Normalization statistics and the split policy of the base checkpoint are kept,
so the fine-tuned checkpoint is a drop-in replacement for ``ForwardPredictor``
and the inversion code.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.func import jvp

from .kernels import sensitivity_kernels
from .network import FourHeadForwardModel
from .splits import Split, mask_for_split, validate_checkpoint_split_policy

VALUE_SCALE = 1e-3  # relative value error expressed in per mille
KERNEL_SCALE = 1e-2  # relative kernel error expressed in percent
KERNEL_FLOOR = 1e-2  # km/s per km/s; keeps tiny kernel rows from dominating


@dataclass
class KernelTrainingConfig:
    base_checkpoint: Path
    dataset_dir: Path
    kernel_dir: Path
    output_dir: Path
    cache_dir: Path
    epochs: int = 30
    batch_size: int = 1024
    kernel_batch_size: int = 512
    kernel_directions: int = 2
    kernel_weight: float = 1.0
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    final_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    hard_example_power: float = 1.0
    kernel_hard_example_power: float = 0.0
    threads: int = 6
    device: str = "cpu"
    seed: int = 20261007
    width: int = 256
    blocks: int = 4
    steps_per_epoch: int | None = None
    corrections: Path | None = None
    kernel_train_split: Split = "train"

    def to_dict(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class Normalizer:
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Normalizer:
        def tensor(key: str) -> Tensor:
            return torch.as_tensor(np.asarray(payload[key], dtype=np.float32))

        return cls(
            tensor("input_mean"),
            tensor("input_std"),
            tensor("target_mean"),
            tensor("target_std"),
        )

    def physical(self, model: torch.nn.Module) -> callable:
        """Return ``vs (km/s) -> phase velocity (km/s)`` through ``model``.

        Inputs are moved to the model's device; outputs stay on it.
        """
        device = next(model.parameters()).device
        input_mean = self.input_mean.to(device)
        input_std = self.input_std.to(device)
        target_mean = self.target_mean.to(device)
        target_std = self.target_std.to(device)

        def forward(vs: Tensor) -> Tensor:
            output = model((vs.to(device) - input_mean) / input_std)
            return output * target_std + target_mean

        forward.device = device
        return forward


def load_split_rows(
    dataset_dir: Path, split: Split, cache_dir: Path
) -> dict[str, NDArray]:
    """Load every row of ``split`` once and cache it as an uncompressed npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{split}.npz"
    if cache.exists():
        with np.load(cache) as loaded:
            return {key: loaded[key] for key in loaded.files}
    parts: dict[str, list[NDArray]] = {
        "sample_id": [],
        "vs": [],
        "phase_velocity": [],
        "valid_mask": [],
        "model_kind": [],
    }
    for shard in sorted(Path(dataset_dir).glob("shard-*.h5")):
        with h5py.File(shard, "r") as handle:
            sample_id = np.asarray(handle["sample_id"], dtype=np.uint64)
            selected = mask_for_split(sample_id, split)
            for key, values in parts.items():
                values.append(np.asarray(handle[key])[selected])
    rows = {key: np.concatenate(value) for key, value in parts.items()}
    rows["phase_velocity"] = np.where(
        rows["valid_mask"], rows["phase_velocity"], 1.0
    ).astype(np.float32)
    temporary = cache.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez(temporary, **rows)
    temporary.replace(cache)
    return rows


def apply_corrections(
    rows: dict[str, NDArray], corrections: Path | None
) -> list[int]:
    """Overwrite re-solved cells in place; return the affected row indices."""
    if corrections is None:
        return []
    with np.load(corrections) as loaded:
        sample_ids = loaded["sample_id"]
        indices = loaded["frequency_index"]
        values = loaded["phase_velocity"]
    position = {int(sample): row for row, sample in enumerate(rows["sample_id"])}
    touched = []
    for sample, index, roots in zip(sample_ids, indices, values, strict=True):
        row = position.get(int(sample))
        if row is None:
            continue
        valid = np.isfinite(roots)
        rows["valid_mask"][row, :, index] = valid
        rows["phase_velocity"][row, :, index] = np.where(valid, roots, 1.0)
        touched.append(row)
    return sorted(set(touched))


def load_kernel_rows(
    kernel_dir: Path, split: Split, corrections: Path | None = None
) -> dict[str, NDArray]:
    """Load kernel labels; kernels are held as float16 to bound memory.

    Labels are written file by file into one preallocated array, so peak memory
    stays close to the final ``rows x 4 x 120 x 20`` float16 array.
    """
    files = sorted((Path(kernel_dir) / split).glob("kernels-*.h5"))
    if not files:
        raise FileNotFoundError(f"no kernel files for split {split!r}")
    counts = []
    for path in files:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["sample_id"].shape[0]))
    total = sum(counts)
    with h5py.File(files[0], "r") as handle:
        kernel_shape = handle["kernel"].shape[1:]
    rows: dict[str, NDArray] = {
        "sample_id": np.empty(total, dtype=np.uint64),
        "vs": np.empty((total, 20), dtype=np.float32),
        "phase_velocity": np.empty((total, *kernel_shape[:2]), dtype=np.float32),
        "valid_mask": np.empty((total, *kernel_shape[:2]), dtype=np.bool_),
        "kernel": np.empty((total, *kernel_shape), dtype=np.float16),
        "kernel_mask": np.empty((total, *kernel_shape[:2]), dtype=np.bool_),
    }
    offset = 0
    for path, count in zip(files, counts, strict=True):
        part = slice(offset, offset + count)
        with h5py.File(path, "r") as handle:
            for key in ("sample_id", "vs", "phase_velocity", "valid_mask"):
                rows[key][part] = np.asarray(handle[key])
            kernel = np.asarray(handle["kernel"], dtype=np.float32)
        finite = np.isfinite(kernel).all(axis=-1)
        kernel[~finite] = 0.0
        rows["kernel"][part] = kernel
        rows["kernel_mask"][part] = finite
        offset += count
    rows["phase_velocity"][~rows["valid_mask"]] = 1.0
    touched = apply_corrections(rows, corrections)
    if touched:
        kernel = sensitivity_kernels(
            rows["vs"][touched].astype(np.float64),
            rows["phase_velocity"][touched],
            rows["valid_mask"][touched],
            np.arange(0.5, 60.0 + 0.25, 0.5),
        )
        finite = np.isfinite(kernel).all(axis=-1)
        kernel[~finite] = 0.0
        rows["kernel"][touched] = kernel.astype(np.float16)
        rows["kernel_mask"][touched] = finite
    rows["kernel_mask"] &= rows["valid_mask"]
    return rows


def relative_value_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, weight: Tensor | None = None
) -> Tensor:
    """Mode-balanced mean squared relative error in per mille."""
    error = ((prediction - target) / target / VALUE_SCALE) ** 2
    if weight is not None:
        error = error * weight[:, None, None]
    maskf = mask.to(error.dtype)
    per_mode = (error * maskf).sum(dim=(0, 2)) / maskf.sum(dim=(0, 2)).clamp_min(1)
    present = maskf.sum(dim=(0, 2)) > 0
    return per_mode[present].mean()


def kernel_loss(
    forward: callable,
    vs: Tensor,
    kernel: Tensor,
    mask: Tensor,
    directions: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return value prediction, mode-balanced relative JVP error (percent^2),
    and each sample's worst kernel-row error estimate in percent."""
    batch = vs.shape[0]
    tangent = torch.randn(
        (directions, batch, vs.shape[1]), generator=generator, dtype=vs.dtype
    ).to(vs.device)
    repeated = vs.unsqueeze(0).expand(directions, -1, -1).reshape(-1, vs.shape[1])
    prediction, product = jvp(forward, (repeated,), (tangent.reshape(-1, vs.shape[1]),))
    product = product.reshape(directions, batch, *product.shape[1:])
    reference = torch.einsum("bmfl,dbl->dbmf", kernel, tangent)
    row_norm2 = (kernel**2).sum(dim=-1) + KERNEL_FLOOR**2
    error = ((product - reference) ** 2).mean(dim=0) / row_norm2 / KERNEL_SCALE**2
    maskf = mask.to(error.dtype)
    per_mode = (error * maskf).sum(dim=(0, 2)) / maskf.sum(dim=(0, 2)).clamp_min(1)
    present = maskf.sum(dim=(0, 2)) > 0
    worst = torch.sqrt(
        torch.where(mask, error, torch.zeros_like(error)).detach().amax(dim=(1, 2))
    )
    return prediction[:batch], per_mode[present].mean(), worst


def full_jacobian(forward: callable, vs: Tensor) -> Tensor:
    """Exact ``(batch, 4, F, 20)`` Jacobian through 20 forward-mode passes."""
    columns = []
    for layer in range(vs.shape[1]):
        tangent = torch.zeros_like(vs)
        tangent[:, layer] = 1.0
        _, column = jvp(forward, (vs,), (tangent,))
        columns.append(column)
    return torch.stack(columns, dim=-1)


def value_metrics(
    prediction: NDArray, target: NDArray, mask: NDArray
) -> dict[str, object]:
    relative = np.where(mask, np.abs(prediction - target) / np.abs(target), 0.0)
    curve_max = relative.max(axis=2)  # (N, 4)
    sample_max = curve_max.max(axis=1)
    metrics: dict[str, object] = {
        "points_within_1pct": float((relative[mask] < 0.01).mean()),
        "points_within_0.1pct": float((relative[mask] < 0.001).mean()),
        "mean_relative_error": float(relative[mask].mean()),
        "max_relative_error": float(relative[mask].max()),
        "samples_all_within_1pct": float((sample_max < 0.01).mean()),
        "samples_all_within_0.5pct": float((sample_max < 0.005).mean()),
        "mae_km_s": float(np.abs(prediction - target)[mask].mean()),
    }
    for mode in range(4):
        valid = mask[:, mode]
        has = valid.any(axis=1)
        metrics[f"mode_{mode}"] = {
            "mae_km_s": float(np.abs(prediction - target)[:, mode][valid].mean()),
            "mean_relative_error": float(relative[:, mode][valid].mean()),
            "max_relative_error": float(relative[:, mode][valid].max()),
            "points_within_1pct": float((relative[:, mode][valid] < 0.01).mean()),
            "curves_all_within_1pct": float((curve_max[has, mode] < 0.01).mean()),
        }
    return metrics


def kernel_metrics(
    network_kernel: NDArray, physical_kernel: NDArray, mask: NDArray
) -> dict[str, object]:
    error = np.linalg.norm(network_kernel - physical_kernel, axis=-1)
    norm = np.linalg.norm(physical_kernel, axis=-1)
    relative = error / np.maximum(norm, 1e-12)
    metrics: dict[str, object] = {
        "rows": int(mask.sum()),
        "median_relative_l2": float(np.median(relative[mask])),
        "rows_within_1pct": float((relative[mask] < 0.01).mean()),
        "rows_within_5pct": float((relative[mask] < 0.05).mean()),
        "rows_within_10pct": float((relative[mask] < 0.10).mean()),
    }
    for mode in range(4):
        values = relative[:, mode][mask[:, mode]]
        metrics[f"mode_{mode}"] = {
            "median_relative_l2": float(np.median(values)),
            "mean_relative_l2": float(values.mean()),
            "rows_within_1pct": float((values < 0.01).mean()),
            "rows_within_5pct": float((values < 0.05).mean()),
        }
    return metrics


def predict(
    forward: callable, vs: NDArray, batch_size: int = 4096
) -> NDArray[np.float32]:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(vs), batch_size):
            outputs.append(
                forward(torch.as_tensor(vs[start : start + batch_size]))
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def network_kernels(
    forward: callable, vs: NDArray, batch_size: int = 512
) -> NDArray[np.float32]:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(vs), batch_size):
            chunk = torch.as_tensor(vs[start : start + batch_size]).to(
                getattr(forward, "device", "cpu")
            )
            outputs.append(full_jacobian(forward, chunk).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_rows(
    forward: callable, rows: dict[str, NDArray]
) -> dict[str, object]:
    """Value metrics on all rows; kernel metrics when ``kernel`` is present."""
    vs = rows["vs"].astype(np.float32)
    prediction = predict(forward, vs)
    result: dict[str, object] = {
        "value": value_metrics(prediction, rows["phase_velocity"], rows["valid_mask"])
    }
    if "kernel" in rows:
        kernels = network_kernels(forward, vs)
        result["kernel"] = kernel_metrics(
            kernels, rows["kernel"].astype(np.float32), rows["kernel_mask"]
        )
    return result


def _selection_score(metrics: dict[str, object]) -> float:
    """Lower is better: weakest curve agreement plus kernel misfit."""
    value = metrics["value"]
    kernel = metrics["kernel"]
    return (1.0 - value["samples_all_within_1pct"]) + (
        1.0 - kernel["rows_within_5pct"]
    )


def _save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_with_kernels(config: KernelTrainingConfig) -> Path:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    base = torch.load(config.base_checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_split_policy(base)
    normalizer = Normalizer.from_payload(base)
    device = torch.device(config.device)
    model = FourHeadForwardModel(width=config.width, blocks=config.blocks)
    if (config.width, config.blocks) == (256, 4):
        model.load_state_dict(base["model"])
    model.to(device)
    forward = normalizer.physical(model)

    train_rows = load_split_rows(config.dataset_dir, "train", config.cache_dir)
    apply_corrections(train_rows, config.corrections)
    kernel_rows = load_kernel_rows(
        config.kernel_dir, config.kernel_train_split, config.corrections
    )
    validation_rows = load_kernel_rows(
        config.kernel_dir, "validation", config.corrections
    )
    train_vs = torch.as_tensor(train_rows["vs"].astype(np.float32))
    train_target = torch.as_tensor(train_rows["phase_velocity"])
    train_mask = torch.as_tensor(train_rows["valid_mask"])
    kernel_vs = torch.as_tensor(kernel_rows["vs"].astype(np.float32))
    kernel_target = torch.as_tensor(kernel_rows["phase_velocity"])
    kernel_value_mask = torch.as_tensor(kernel_rows["valid_mask"])
    kernel_mask = torch.as_tensor(kernel_rows["kernel_mask"])
    kernel_labels = torch.from_numpy(kernel_rows["kernel"])
    # Everything fits in a 48 GB GPU (~6 GB); keep it there to avoid transfers.
    (
        train_vs,
        train_target,
        train_mask,
        kernel_vs,
        kernel_target,
        kernel_value_mask,
        kernel_mask,
        kernel_labels,
    ) = (
        tensor.to(device)
        for tensor in (
            train_vs,
            train_target,
            train_mask,
            kernel_vs,
            kernel_target,
            kernel_value_mask,
            kernel_mask,
            kernel_labels,
        )
    )

    steps_per_epoch = config.steps_per_epoch or math.ceil(
        len(train_vs) / config.batch_size
    )
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    floor = config.final_learning_rate / config.learning_rate
    warmup = max(config.warmup_steps, 1)

    def schedule(step: int) -> float:
        cosine = floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(step, total_steps) / total_steps)
        )
        return cosine * min(1.0, (step + 1) / warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = config.output_dir / "last.pt"
    best_path = config.output_dir / "best.pt"
    history_path = config.output_dir / "history.json"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_score = float("inf")
    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        rng = np.random.default_rng(config.seed + start_epoch)
        if history_path.exists():
            history = json.loads(history_path.read_text())["epochs"][:start_epoch]
    else:
        model.eval()
        initial = evaluate_rows(forward, validation_rows)
        history.append({"epoch": -1, "validation": initial})
        best_score = _selection_score(initial)
        print(json.dumps({"epoch": -1, "score": best_score}), flush=True)

    # Per-sample hard-example weights from the latest relative error.
    sample_weight = torch.ones(len(train_vs), device=device)
    kernel_weight = torch.ones(len(kernel_vs), device=device)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        started = time.time()
        order = torch.as_tensor(rng.permutation(len(train_vs)), device=device)
        if config.hard_example_power > 0:
            probability = sample_weight / sample_weight.sum()
            order = torch.multinomial(probability, len(train_vs), replacement=True)
        value_total = 0.0
        kernel_total = 0.0
        for step in range(steps_per_epoch):
            index = order[step * config.batch_size : (step + 1) * config.batch_size]
            if config.kernel_hard_example_power > 0:
                kernel_index = torch.multinomial(
                    kernel_weight, config.kernel_batch_size, replacement=True
                )
            else:
                kernel_index = torch.as_tensor(
                    rng.integers(0, len(kernel_vs), config.kernel_batch_size),
                    device=device,
                )
            prediction = forward(train_vs[index])
            value = relative_value_loss(
                prediction, train_target[index], train_mask[index]
            )
            kernel_prediction, derivative, worst_row = kernel_loss(
                forward,
                kernel_vs[kernel_index],
                kernel_labels[kernel_index].to(torch.float32),
                kernel_mask[kernel_index],
                config.kernel_directions,
            )
            value = value + relative_value_loss(
                kernel_prediction,
                kernel_target[kernel_index],
                kernel_value_mask[kernel_index],
            )
            loss = value + config.kernel_weight * derivative
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            value_total += float(value.detach())
            kernel_total += float(derivative.detach())
            if config.kernel_hard_example_power > 0:
                # Weight 1 for a perfect sample, 2 at a 5 % worst row; capped.
                kernel_weight[kernel_index] = (
                    (1.0 + worst_row / 5.0) ** config.kernel_hard_example_power
                ).clamp(max=100.0)
            if config.hard_example_power > 0:
                with torch.no_grad():
                    relative = torch.where(
                        train_mask[index],
                        (prediction.detach() - train_target[index]).abs()
                        / train_target[index],
                        torch.zeros((), device=device),
                    ).amax(dim=(1, 2))
                    # Weight 1 below 0.2 %, rising with the worst-cell error.
                    sample_weight[index] = (
                        1.0 + relative / 2e-3
                    ) ** config.hard_example_power
        model.eval()
        metrics = evaluate_rows(forward, validation_rows)
        score = _selection_score(metrics)
        record = {
            "epoch": epoch,
            "value_loss": value_total / steps_per_epoch,
            "kernel_loss": kernel_total / steps_per_epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
            "score": score,
            "validation": metrics,
        }
        history.append(record)
        payload = {
            **{key: base[key] for key in base if key not in {"model", "optimizer", "scheduler"}},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": min(best_score, score),
            "architecture": {"width": config.width, "blocks": config.blocks},
            "kernel_training_config": config.to_dict(),
            "base_checkpoint": str(config.base_checkpoint),
        }
        _save(last_path, payload)
        if score < best_score:
            best_score = score
            _save(best_path, payload)
        history_path.write_text(json.dumps({"epochs": history}, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in ("epoch", "value_loss", "kernel_loss", "seconds", "score")
                }
                | {
                    "samples_all_within_1pct": metrics["value"]["samples_all_within_1pct"],
                    "kernel_rows_within_5pct": metrics["kernel"]["rows_within_5pct"],
                    "kernel_median": metrics["kernel"]["median_relative_l2"],
                }
            ),
            flush=True,
        )
    return best_path if best_path.exists() else last_path
