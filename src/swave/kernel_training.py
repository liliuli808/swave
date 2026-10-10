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

import hashlib
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

from .kernel_mining import exact_row_kernel_loss, mine_kernel_rows
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
    max_grad_norm: float = 10.0
    hard_example_power: float = 1.0
    kernel_hard_example_power: float = 0.0
    kernel_row_weight: float = 0.0
    kernel_row_batch_size: int = 256
    kernel_mining_samples: int = 8192
    kernel_mining_interval: int = 5
    threads: int = 6
    device: str = "cpu"
    seed: int = 20261007
    width: int = 256
    blocks: int = 4
    profile_features: bool = False
    steps_per_epoch: int | None = None
    corrections: Path | None = None
    kernel_train_split: Split = "train"
    require_warm_start: bool = False
    max_initial_score: float = 2.0
    # <1 trains kernels on a fixed subset of kernel-labelled models (data-scaling
    # tests); value rows and validation are unchanged.
    kernel_train_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.kernel_train_fraction)
                and 0 < self.kernel_train_fraction <= 1):
            raise ValueError("kernel_train_fraction must be in (0, 1]")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if (not math.isfinite(self.max_initial_score)
                or not 0 <= self.max_initial_score <= 2):
            raise ValueError("max_initial_score must be finite and between 0 and 2")
        if not math.isfinite(self.kernel_row_weight) or self.kernel_row_weight < 0:
            raise ValueError("kernel_row_weight must be finite and nonnegative")
        if min(self.kernel_row_batch_size, self.kernel_mining_samples,
               self.kernel_mining_interval) < 1:
            raise ValueError("kernel row batch, mining samples and interval must be positive")
        if self.kernel_row_weight > 0 and self.kernel_train_split != "train":
            raise ValueError("kernel row mining requires kernel_train_split='train'")

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
    kernel_dir: Path, split: Split, corrections: Path | None = None,
    *, sample_ids: NDArray | None = None,
) -> dict[str, NDArray]:
    """Load kernel labels; kernels are held as float16 to bound memory.

    Labels are written file by file into one preallocated array, so peak memory
    stays close to the final ``rows x 4 x 120 x 20`` float16 array. An explicit
    sample-ID subset reads only those models, in shard order, for small audits.
    """
    files = sorted((Path(kernel_dir) / split).glob("kernels-*.h5"))
    if not files:
        raise FileNotFoundError(f"no kernel files for split {split!r}")
    requested = None if sample_ids is None else np.asarray(sample_ids)
    if requested is not None and (
        requested.ndim != 1 or not len(requested) or np.any(requested < 0)
        or len(np.unique(requested)) != len(requested)
    ):
        raise ValueError("sample_ids must be a nonempty vector of unique nonnegative IDs")
    counts, selections, found = [], [], []
    for path in files:
        with h5py.File(path, "r") as handle:
            if requested is None:
                counts.append(int(handle["sample_id"].shape[0]))
                selections.append(None)
            else:
                ids = np.asarray(handle["sample_id"], dtype=np.uint64)
                index = np.flatnonzero(np.isin(ids, requested))
                selections.append(index)
                counts.append(len(index))
                found.extend(ids[index].tolist())
    if requested is not None and (
        len(found) != len(requested) or set(found) != set(requested.tolist())
    ):
        raise ValueError("requested kernel sample IDs are missing or duplicated in the split")
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
    for path, count, index in zip(files, counts, selections, strict=True):
        if count == 0:
            continue
        part = slice(offset, offset + count)
        with h5py.File(path, "r") as handle:
            for key in ("sample_id", "vs", "phase_velocity", "valid_mask"):
                rows[key][part] = np.asarray(
                    handle[key] if index is None else handle[key][index]
                )
            kernel = np.asarray(
                handle["kernel"] if index is None else handle["kernel"][index],
                dtype=np.float32,
            )
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


def select_kernel_fraction(
    rows: dict[str, NDArray], fraction: float, seed: int
) -> dict[str, NDArray]:
    """Keep a seeded fraction of models, chosen by sample ID.

    The choice is a prefix of one permutation of the sorted IDs, so it does not
    depend on shard read order and smaller fractions are nested in larger ones.
    """
    if fraction >= 1:
        return rows
    ids = np.asarray(rows["sample_id"])
    order = np.argsort(ids, kind="stable")
    keep = order[np.random.default_rng(seed).permutation(len(ids))]
    keep = np.sort(keep[:max(1, round(fraction * len(ids)))])
    return {key: value[keep] for key, value in rows.items()}


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
    and each sample's worst kernel-row error estimate in percent.

    ``directions <= 0`` uses the unit vectors of every layer instead of random
    directions, i.e. the exact Frobenius error of the full Jacobian."""
    batch, layers = vs.shape
    exact = directions <= 0
    if exact:
        directions = layers
        tangent = torch.eye(layers, dtype=vs.dtype, device=vs.device)
        tangent = tangent[:, None, :].expand(-1, batch, -1)
    else:
        tangent = torch.randn(
            (directions, batch, layers), generator=generator, dtype=vs.dtype
        ).to(vs.device)
    repeated = vs.unsqueeze(0).expand(directions, -1, -1).reshape(-1, layers)
    prediction, product = jvp(forward, (repeated,), (tangent.reshape(-1, layers),))
    product = product.reshape(directions, batch, *product.shape[1:])
    reference = torch.einsum("bmfl,dbl->dbmf", kernel, tangent)
    row_norm2 = (kernel**2).sum(dim=-1) + KERNEL_FLOOR**2
    squared = (product - reference) ** 2
    # Unit vectors sum to ||J - K||^2; Gaussian directions estimate it by the mean.
    squared = squared.sum(dim=0) if exact else squared.mean(dim=0)
    error = squared / row_norm2 / KERNEL_SCALE**2
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


def kernel_checkpoint_key(metrics: dict[str, object]) -> tuple[float, float] | None:
    """Rank kernel agreement subject to the 99% curve acceptance constraint.

    Higher is better. Curve agreement breaks exact kernel ties; it cannot
    compensate for worse kernels once the curve constraint is satisfied.
    """
    curves = float(metrics["value"]["samples_all_within_1pct"])
    kernels = float(metrics["kernel"]["rows_within_5pct"])
    if not (math.isfinite(curves) and math.isfinite(kernels)):
        return None
    return (kernels, curves) if curves >= 0.99 else None


def _save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_with_kernels(config: KernelTrainingConfig) -> Path:
    if config.kernel_row_weight > 0 and config.kernel_train_split != "train":
        raise ValueError("kernel row mining requires kernel_train_split='train'")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    base = torch.load(config.base_checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_split_policy(base)
    normalizer = Normalizer.from_payload(base)
    device = torch.device(config.device)
    model = FourHeadForwardModel(
        width=config.width,
        blocks=config.blocks,
        profile_features=config.profile_features,
    )
    architecture = {
        "width": config.width,
        "blocks": config.blocks,
        "profile_features": config.profile_features,
    }
    base_architecture = {
        "width": 256, "blocks": 4, "profile_features": False,
        **(base.get("architecture") or {}),
    }
    warm_start = all(
        base_architecture[key] == value for key, value in architecture.items()
    )
    last_path = config.output_dir / "last.pt"
    if config.require_warm_start:
        if last_path.exists():
            raise ValueError(
                f"require_warm_start refuses automatic resume from {last_path}; "
                "use a new output_dir for fresh fine-tuning"
            )
        if not warm_start:
            raise ValueError(
                "require_warm_start: checkpoint architecture does not match the "
                f"requested architecture: base={base_architecture}, "
                f"requested={architecture}"
            )
    if warm_start:
        model.load_state_dict(base["model"])
    print(json.dumps({
        "initialization": ("resume" if last_path.exists() else
                           "base_checkpoint" if warm_start else "random"),
        "base_checkpoint": str(config.base_checkpoint),
        "resume_checkpoint": str(last_path) if last_path.exists() else None,
        "architecture": architecture,
        "torch_version": str(torch.__version__),
        "kernel_training_source": str(Path(__file__).resolve()),
        "configuration": config.to_dict(),
    }), flush=True)
    model.to(device)
    forward = normalizer.physical(model)

    train_rows = load_split_rows(config.dataset_dir, "train", config.cache_dir)
    apply_corrections(train_rows, config.corrections)
    kernel_rows = load_kernel_rows(
        config.kernel_dir, config.kernel_train_split, config.corrections
    )
    if config.kernel_train_fraction < 1:
        available = len(kernel_rows["sample_id"])
        kernel_rows = select_kernel_fraction(
            kernel_rows, config.kernel_train_fraction, config.seed
        )
        kept = np.sort(kernel_rows["sample_id"].astype(np.uint64))
        print(json.dumps({"kernel_train_subset": {
            "fraction": config.kernel_train_fraction,
            "models": len(kept), "available_models": available,
            "sample_id_checksum": hashlib.sha256(kept.tobytes()).hexdigest()[:16],
        }}), flush=True)
    if config.kernel_row_weight > 0 and not mask_for_split(
        kernel_rows["sample_id"], "train"
    ).all():
        raise ValueError("kernel row mining found non-training sample IDs")
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
    best_path = config.output_dir / "best.pt"
    best_kernel_path = config.output_dir / "best-kernel.pt"
    history_path = config.output_dir / "history.json"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_score = float("inf")
    best_kernel_key: tuple[float, float] | None = None

    def checkpoint(
        epoch: int, score: float, validation: dict[str, object]
    ) -> dict[str, object]:
        return {
            **{key: value for key, value in base.items()
               if key not in {"model", "optimizer", "scheduler", "selection"}},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": score,
            "validation": validation,
            "architecture": architecture,
            "kernel_training_config": config.to_dict(),
            "base_checkpoint": str(config.base_checkpoint),
            "kernel_training_runtime": {
                "torch_version": str(torch.__version__),
                "device": str(device),
                "layer_norm": "elementary_ops_v1",
            },
        }

    def retain_kernel_checkpoint(payload: dict[str, object]) -> None:
        nonlocal best_kernel_key
        validation = payload.get("validation")
        if validation is None:
            # Old checkpoints did not embed their validation metrics. Only
            # recover metrics for an epoch whose weights are actually present.
            validation = next((
                row["validation"] for row in history
                if row["epoch"] == payload["epoch"]
            ), None)
        if validation is None:
            return
        key = kernel_checkpoint_key(validation)
        if key is not None and (best_kernel_key is None or key > best_kernel_key):
            best_kernel_key = key
            _save(best_kernel_path, {
                **payload,
                "validation": validation,
                "selection": {
                    "metric": "kernel.rows_within_5pct",
                    "minimum_curve_pass_rate": 0.99,
                    "tie_breaker": "value.samples_all_within_1pct",
                },
            })

    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        rng = np.random.default_rng(config.seed + start_epoch)
        if history_path.exists():
            history = [
                record for record in json.loads(history_path.read_text())["epochs"]
                if record["epoch"] < start_epoch
            ]
        if best_kernel_path.exists():
            saved = torch.load(best_kernel_path, map_location="cpu", weights_only=False)
            best_kernel_key = kernel_checkpoint_key(saved["validation"])
            del saved
        # Seed the new criterion when resuming an older run. Unsaved historical
        # epochs cannot be reconstructed from history.json.
        if best_path.exists():
            retain_kernel_checkpoint(torch.load(
                best_path, map_location="cpu", weights_only=False,
            ))
        retain_kernel_checkpoint(payload)
    else:
        model.eval()
        initial = evaluate_rows(forward, validation_rows)
        best_score = _selection_score(initial)
        print(json.dumps({
            "epoch": -1,
            "score": best_score,
            "samples_all_within_1pct": initial["value"]["samples_all_within_1pct"],
            "kernel_rows_within_5pct": initial["kernel"]["rows_within_5pct"],
            "kernel_median": initial["kernel"]["median_relative_l2"],
        }), flush=True)
        if not math.isfinite(best_score) or best_score > config.max_initial_score:
            raise ValueError(
                f"initial validation score {best_score} is non-finite or exceeds "
                "max_initial_score="
                f"{config.max_initial_score}; check the checkpoint, loaded code "
                "and data before training"
            )
        history.append({"epoch": -1, "score": best_score, "validation": initial})
        # A fine-tune can regress for every epoch. Keep the actual initial best,
        # rather than falling back to a worse last.pt if no update beats it.
        # A fresh run also replaces artifacts from an interrupted initial-only
        # run, even if its new initial model misses the curve constraint.
        best_kernel_path.unlink(missing_ok=True)
        payload = checkpoint(-1, best_score, initial)
        _save(best_path, payload)
        retain_kernel_checkpoint(payload)

    # Per-sample hard-example weights from the latest relative error.
    sample_weight = torch.ones(len(train_vs), device=device)
    kernel_weight = torch.ones(len(kernel_vs), device=device)
    row_bank = None
    # Mining must not perturb either of the baseline batch RNG streams.
    mining_rng = np.random.default_rng(config.seed + 104729 + start_epoch)
    row_rng = np.random.default_rng(config.seed + 130363 + start_epoch)
    for epoch in range(start_epoch, config.epochs):
        if config.kernel_row_weight > 0 and (
            row_bank is None or epoch % config.kernel_mining_interval == 0
        ):
            mining_started = time.time()
            model.eval()
            pool = torch.as_tensor(mining_rng.choice(
                len(kernel_vs), min(config.kernel_mining_samples, len(kernel_vs)),
                replace=False,
            ), device=device)
            row_bank, mining_stats = mine_kernel_rows(
                forward, kernel_vs, kernel_labels, kernel_mask, pool,
            )
            print(json.dumps({"kernel_row_mining": {
                "epoch": epoch, "split": "train", **mining_stats,
                "seconds": time.time() - mining_started,
            }}), flush=True)
        model.train()
        started = time.time()
        order = torch.as_tensor(rng.permutation(len(train_vs)), device=device)
        if config.hard_example_power > 0:
            probability = sample_weight / sample_weight.sum()
            order = torch.multinomial(probability, len(train_vs), replacement=True)
        value_total = 0.0
        kernel_total = 0.0
        row_total = 0.0
        row_passed = 0
        row_count = 0
        grad_norm_total = 0.0
        grad_norm_max = 0.0
        clipped_steps = 0
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
            if config.kernel_row_weight > 0 and len(row_bank):
                selected = row_bank[torch.as_tensor(row_rng.integers(
                    0, len(row_bank), config.kernel_row_batch_size,
                ), device=device)]
                row_models, row_modes, row_frequencies = selected.unbind(1)
                row_loss, row_relative = exact_row_kernel_loss(
                    forward, kernel_vs[row_models],
                    kernel_labels[row_models, row_modes, row_frequencies],
                    row_modes, row_frequencies,
                    kernel_floor=KERNEL_FLOOR, percent_scale=KERNEL_SCALE,
                )
                loss = loss + config.kernel_row_weight * row_loss
                row_total += float(row_loss.detach())
                row_passed += int((row_relative < 0.05).sum())
                row_count += len(row_relative)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}, step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            ))
            grad_norm_total += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)
            clipped_steps += grad_norm > config.max_grad_norm
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
            "grad_norm_mean": grad_norm_total / steps_per_epoch,
            "grad_norm_max": grad_norm_max,
            "grad_clip_fraction": clipped_steps / steps_per_epoch,
            "seconds": time.time() - started,
            "score": score,
            "validation": metrics,
        }
        if config.kernel_row_weight > 0:
            record.update({
                "kernel_row_loss": row_total / steps_per_epoch,
                "mined_row_pass_fraction": row_passed / row_count if row_count else None,
                "mined_rows": len(row_bank),
            })
        history.append(record)
        payload = checkpoint(epoch, min(best_score, score), metrics)
        _save(last_path, payload)
        retain_kernel_checkpoint(payload)
        if score < best_score:
            best_score = score
            _save(best_path, payload)
        history_path.write_text(json.dumps({"epochs": history}, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "epoch", "value_loss", "kernel_loss", "learning_rate",
                        "grad_norm_mean", "grad_norm_max", "grad_clip_fraction",
                        "seconds", "score",
                    )
                }
                | {
                    "samples_all_within_1pct": metrics["value"]["samples_all_within_1pct"],
                    "kernel_rows_within_5pct": metrics["kernel"]["rows_within_5pct"],
                    "kernel_median": metrics["kernel"]["median_relative_l2"],
                }
                | {key: record[key] for key in (
                    "kernel_row_loss", "mined_row_pass_fraction", "mined_rows",
                ) if key in record}
            ),
            flush=True,
        )
    if best_kernel_path.exists():
        return best_kernel_path
    return best_path if best_path.exists() else last_path
