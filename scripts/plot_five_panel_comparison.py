#!/usr/bin/env python3
"""Plot one Vs profile beside true and predicted four-mode dispersion curves."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from swave.config import load_dataset_config
from swave.dataset import validate_dataset_files
from swave.geology import ModelKind
from swave.inference import ForwardPredictor
from swave.splits import mask_for_split


def plot_comparison(
    *,
    vs: NDArray[np.floating[Any]],
    true_curves: NDArray[np.floating[Any]],
    predicted_curves: NDArray[np.floating[Any]],
    valid_mask: NDArray[np.bool_],
    frequencies: NDArray[np.floating[Any]],
    sample_id: int,
    model_kind: str,
    output: Path | str,
    thickness_km: float,
    anomaly_first_layer: int,
    anomaly_last_layer: int,
) -> list[dict[str, float]]:
    """Write the five-panel comparison figure and return per-mode errors."""
    vs_values = np.asarray(vs, dtype=np.float64)
    true_values = np.asarray(true_curves, dtype=np.float64)
    predicted_values = np.asarray(predicted_curves, dtype=np.float64)
    mask_values = np.asarray(valid_mask, dtype=np.bool_)
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    if vs_values.shape != (20,):
        raise ValueError("vs must have shape (20,)")
    expected_curve_shape = (4, frequency_values.size)
    if (
        true_values.shape != expected_curve_shape
        or predicted_values.shape != expected_curve_shape
        or mask_values.shape != expected_curve_shape
    ):
        raise ValueError(
            "true curves, predictions, and mask must have shape "
            f"{expected_curve_shape}"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    velocity_m_s = vs_values * 1000.0
    true_m_s = true_values * 1000.0
    predicted_m_s = predicted_values * 1000.0
    thickness_m = thickness_km * 1000.0
    depth_edges = np.arange(vs_values.size + 1) * thickness_m

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(23, 5),
        gridspec_kw={"width_ratios": [1.1, 1, 1, 1, 1]},
    )
    profile_x = np.repeat(velocity_m_s, 2)
    profile_y = np.column_stack(
        [depth_edges[:-1], depth_edges[1:]]
    ).reshape(-1)
    axes[0].plot(profile_x, profile_y, color="black", linewidth=2)
    axes[0].axhspan(
        (anomaly_first_layer - 1) * thickness_m,
        anomaly_last_layer * thickness_m,
        color="red",
        alpha=0.10,
        label=(
            f"Candidate zone (layer {anomaly_first_layer}"
            f"–{anomaly_last_layer})"
        ),
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Vs (m/s)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title(
        f"Vs Profile\n{model_kind.replace('_', ' ').title()}, "
        f"sample {sample_id}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    metrics: list[dict[str, float]] = []
    for mode in range(4):
        axis = axes[mode + 1]
        valid = (
            mask_values[mode]
            & np.isfinite(true_m_s[mode])
            & np.isfinite(predicted_m_s[mode])
        )
        if not np.any(valid):
            raise ValueError(f"mode {mode} contains no valid comparison cells")
        difference = np.abs(predicted_m_s[mode, valid] - true_m_s[mode, valid])
        mae = round(float(np.mean(difference)), 6)
        relative = round(
            float(
                np.mean(
                    difference
                    / np.maximum(np.abs(true_m_s[mode, valid]), 1e-12)
                )
                * 100.0
            ),
            6,
        )
        metrics.append({"mae_m_s": mae, "relative_percent": relative})

        true_line = np.where(valid, true_m_s[mode], np.nan)
        axis.plot(
            frequency_values,
            true_line,
            color="black",
            linewidth=1.8,
            label="True",
        )
        predicted_line = np.where(valid, predicted_m_s[mode], np.nan)
        axis.plot(
            frequency_values,
            predicted_line,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="4-head NN",
        )
        axis.set_title(f"M{mode}\nMAE={mae:.1f} m/s, Rel={relative:.2f}%")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Phase velocity (m/s)")
        axis.grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return metrics


def _load_sample(
    dataset_dir: Path, sample_id: int | None
) -> tuple[int, str, NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    for shard_path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(shard_path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            kinds = np.asarray(handle["model_kind"], dtype=np.uint8)
            if sample_id is None:
                rows = np.flatnonzero(
                    (kinds == int(ModelKind.COUPLED))
                    & mask_for_split(sample_ids, "test")
                )
            else:
                rows = np.flatnonzero(sample_ids == sample_id)
            if not rows.size:
                continue
            row = int(rows[0])
            selected_id = int(sample_ids[row])
            kind = ModelKind(int(kinds[row])).name
            return (
                selected_id,
                kind,
                np.asarray(handle["vs"][row], dtype=np.float32),
                np.asarray(handle["phase_velocity"][row], dtype=np.float32),
                np.asarray(handle["valid_mask"][row], dtype=np.bool_),
            )
    if sample_id is None:
        raise ValueError("no coupled sample was found in the test split")
    raise ValueError(f"sample {sample_id} was not found in {dataset_dir}")


def _validate_inputs(dataset_dir: Path, checkpoint: Path) -> None:
    manifest = validate_dataset_files(dataset_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("dataset_config_hash") != manifest.config_hash:
        raise ValueError("checkpoint dataset configuration hash does not match")


def _parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 20-layer Vs profile and true/predicted M0–M3 curves"
        )
    )
    parser.add_argument("--dataset-dir", default="data/production", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="runs/production-48g/best.pt",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="results/five-panel-comparison.png",
        type=Path,
    )
    parser.add_argument("--config", default="configs/dataset.toml", type=Path)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(arguments)
    _validate_inputs(arguments.dataset_dir, arguments.checkpoint)
    config = load_dataset_config(arguments.config)
    sample_id, kind, vs, true_curves, valid_mask = _load_sample(
        arguments.dataset_dir, arguments.sample_id
    )
    predictor = ForwardPredictor.load(
        arguments.checkpoint, device=arguments.device
    )
    frequencies, predictions = predictor.predict_with_frequencies(vs)
    metrics = plot_comparison(
        vs=vs,
        true_curves=true_curves,
        predicted_curves=predictions,
        valid_mask=valid_mask,
        frequencies=frequencies,
        sample_id=sample_id,
        model_kind=kind,
        output=arguments.output,
        thickness_km=config.geology.thickness_km,
        anomaly_first_layer=config.geology.anomaly_first_layer,
        anomaly_last_layer=config.geology.anomaly_last_layer,
    )
    print(
        json.dumps(
            {
                "sample_id": sample_id,
                "model_kind": kind,
                "output": str(arguments.output),
                "modes": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
