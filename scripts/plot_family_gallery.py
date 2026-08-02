#!/usr/bin/env python3
"""Plot one representative Vs profile and its dispersion curves per model family.

The figure has one column per family (NORMAL, LOW_VELOCITY, HIGH_VELOCITY,
COUPLED): the top row shows the Vs step profile with the anomaly-candidate
depth zone shaded, the bottom row shows the M0–M3 dispersion curves stored in
the dataset (masked cells are omitted).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from swave.config import load_dataset_config
from swave.dataset import validate_dataset_files
from swave.geology import ModelKind

KIND_TITLES = {
    ModelKind.NORMAL: "Normal",
    ModelKind.LOW_VELOCITY: "Low-velocity",
    ModelKind.HIGH_VELOCITY: "High-velocity",
    ModelKind.COUPLED: "Coupled high+low",
}
MODE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def pick_samples(shard_path: Path) -> dict[ModelKind, int]:
    """Return the first row index of each model family in a shard."""
    with h5py.File(shard_path, "r") as handle:
        kinds = handle["model_kind"][:]
    selected: dict[ModelKind, int] = {}
    for kind in ModelKind:
        rows = np.where(kinds == int(kind))[0]
        if rows.size:
            selected[kind] = int(rows[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/production-w64")
    parser.add_argument("--config", default="configs/dataset.toml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--output", default="results/family-gallery.png")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    validate_dataset_files(dataset_dir)
    config = load_dataset_config(args.config)
    thickness_m = config.geology.thickness_km * 1000.0
    anomaly_top = (config.geology.anomaly_first_layer - 1) * thickness_m
    anomaly_bottom = config.geology.anomaly_last_layer * thickness_m

    shard_path = dataset_dir / f"shard-{args.shard:05d}.h5"
    selected = pick_samples(shard_path)
    if len(selected) < len(ModelKind):
        raise RuntimeError(f"shard {args.shard} lacks some families: {selected}")

    with h5py.File(shard_path, "r") as handle:
        rows = {kind: row for kind, row in selected.items()}
        vs = {k: handle["vs"][r] for k, r in rows.items()}
        curves = {k: handle["phase_velocity"][r] for k, r in rows.items()}
        masks = {k: handle["valid_mask"][r] for k, r in rows.items()}
        sample_ids = {k: int(handle["sample_id"][r]) for k, r in rows.items()}
        frequencies = handle["frequencies"][:] if "frequencies" in handle else None
    if frequencies is None:
        frequencies = np.arange(0.5, 60.25, 0.5)

    kinds = list(ModelKind)
    figure, axes = plt.subplots(
        2, len(kinds), figsize=(4.6 * len(kinds), 8.5), sharey="row"
    )
    depth_edges = np.arange(21) * thickness_m

    for col, kind in enumerate(kinds):
        ax_profile, ax_disp = axes[0, col], axes[1, col]

        profile_x = np.repeat(vs[kind] * 1000.0, 2)
        profile_y = np.column_stack(
            [depth_edges[:-1], depth_edges[1:]]
        ).reshape(-1)
        ax_profile.plot(profile_x, profile_y, color="black", linewidth=2)
        ax_profile.axhspan(
            anomaly_top, anomaly_bottom, color="red", alpha=0.08,
            label="anomaly zone" if col == 0 else None,
        )
        ax_profile.set_ylim(depth_edges[-1] + 500.0, 0.0)
        ax_profile.set_xlabel("Vs (m/s)")
        ax_profile.grid(True, alpha=0.3)
        ax_profile.set_title(
            f"{KIND_TITLES[kind]}\nsample {sample_ids[kind]}", fontsize=11
        )

        for mode in range(4):
            valid = masks[kind][mode]
            ax_disp.plot(
                frequencies[valid],
                curves[kind][mode][valid] * 1000.0,
                color=MODE_COLORS[mode],
                linewidth=1.6,
                label=f"M{mode}",
            )
        ax_disp.set_xlim(frequencies[0], frequencies[-1])
        ax_disp.set_xlabel("Frequency (Hz)")
        ax_disp.grid(True, alpha=0.3)
        if col == 0:
            ax_disp.legend(loc="upper left", fontsize=9)

    axes[0, 0].set_ylabel("Depth (m)")
    axes[1, 0].set_ylabel("Phase velocity (m/s)")
    figure.suptitle("Model families: Vs profiles and M0–M3 dispersion curves")
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    print(f"saved {output_path}")
    for kind in kinds:
        print(f"  {kind.name}: sample_id={sample_ids[kind]}")


if __name__ == "__main__":
    main()
