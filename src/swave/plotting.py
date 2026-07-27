"""Noninteractive scientific plots for models, dispersion, and training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .geology import GeneratedModel
from .solver import DispersionResult


def _prepare_output(output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_model(
    model: GeneratedModel,
    output: Path | str,
    *,
    thickness_km: float = 0.1,
) -> Path:
    """Plot Vs, Vp, and density against layer-top depth."""
    path = _prepare_output(output)
    depth = np.arange(model.vs.size, dtype=np.float64) * thickness_km
    figure, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    axes[0].step(model.vs, depth, where="post", label="Vs")
    axes[0].step(model.vp, depth, where="post", label="Vp")
    axes[0].set_xlabel("Velocity (km/s)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].step(model.density, depth, where="post", color="tab:brown")
    axes[1].set_xlabel("Density (g/cm³)")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.invert_yaxis()
    figure.suptitle(
        f"Sample {model.sample_id}: {model.kind.name.replace('_', ' ').title()}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_dispersion(
    result: DispersionResult,
    output: Path | str,
    *,
    comparisons: Mapping[str, DispersionResult] | None = None,
) -> Path:
    """Plot valid modal phase velocities and optional strategy comparisons."""
    path = _prepare_output(output)
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for mode, color in enumerate(colors):
        valid = result.valid_mask[mode]
        axis.plot(
            result.frequencies[valid],
            result.phase_velocity[mode, valid],
            color=color,
            label=f"mode {mode}",
        )
    if comparisons:
        line_styles = ("--", ":", "-.")
        for style_index, (name, comparison) in enumerate(comparisons.items()):
            for mode, color in enumerate(colors):
                valid = comparison.valid_mask[mode]
                axis.plot(
                    comparison.frequencies[valid],
                    comparison.phase_velocity[mode, valid],
                    color=color,
                    linestyle=line_styles[style_index % len(line_styles)],
                    alpha=0.55,
                    label=f"{name} mode {mode}",
                )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Phase velocity (km/s)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_training_history(history_json: Path | str, output: Path | str) -> Path:
    """Plot checkpoint-adjacent training loss and validation MAE."""
    with Path(history_json).open(encoding="utf-8") as handle:
        epochs = json.load(handle)["epochs"]
    if not epochs:
        raise ValueError("training history contains no epochs")
    path = _prepare_output(output)
    indexes = [int(item["epoch"]) + 1 for item in epochs]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(indexes, [item["training_loss"] for item in epochs])
    axes[0].set_ylabel("Normalized Smooth-L1")
    axes[1].plot(indexes, [item["validation_mae_km_s"] for item in epochs])
    axes[1].set_ylabel("Validation MAE (km/s)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
