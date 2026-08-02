#!/usr/bin/env python3
"""Compare three representative samples per model family in one figure.

Layout: one column per family (NORMAL, LOW_VELOCITY, HIGH_VELOCITY, COUPLED).
Top row overlays the three Vs step profiles; bottom row overlays the M0–M3
dispersion curves. Color encodes the representative sample (consistent across
both rows); line style encodes the mode in the bottom row. Representatives are
the 25th/50th/75th percentile samples of the family ranked by mean Vs, taken
from one shard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from swave.config import load_dataset_config
from swave.geology import ModelKind

KIND_TITLES = {
    ModelKind.NORMAL: "Normal",
    ModelKind.LOW_VELOCITY: "Low-velocity",
    ModelKind.HIGH_VELOCITY: "High-velocity",
    ModelKind.COUPLED: "Coupled high+low",
}
# 验证过的三色分类调色板（light 模式全对通过）
REP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
REP_NAMES = ["Rep A", "Rep B", "Rep C"]
MODE_STYLES = ["-", "--", "-.", ":"]
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"


def pick_representatives(kinds: np.ndarray, vs: np.ndarray, kind: ModelKind) -> list[int]:
    """Rows of the 25/50/75-percentile samples of a family ranked by mean Vs."""
    rows = np.where(kinds == int(kind))[0]
    if rows.size < 3:
        raise RuntimeError(f"family {kind.name} has only {rows.size} samples")
    order = rows[np.argsort(vs[rows].mean(axis=1))]
    picks = [order[int(q * (len(order) - 1))] for q in (0.25, 0.5, 0.75)]
    return [int(r) for r in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/production-w64")
    parser.add_argument("--config", default="configs/dataset.toml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--output", default="results/family-representatives.png")
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    thickness_m = config.geology.thickness_km * 1000.0
    anomaly_top = (config.geology.anomaly_first_layer - 1) * thickness_m
    anomaly_bottom = config.geology.anomaly_last_layer * thickness_m

    shard_path = Path(args.dataset_dir) / f"shard-{args.shard:05d}.h5"
    with h5py.File(shard_path, "r") as handle:
        kinds_all = handle["model_kind"][:]
        vs_all = handle["vs"][:]
        curves_all = handle["phase_velocity"][:]
        masks_all = handle["valid_mask"][:]
        ids_all = handle["sample_id"][:]
        frequencies = handle["frequencies"][:] if "frequencies" in handle else None
    if frequencies is None:
        frequencies = np.arange(0.5, 60.25, 0.5)

    families = list(ModelKind)
    reps = {kind: pick_representatives(kinds_all, vs_all, kind) for kind in families}

    figure, axes = plt.subplots(
        2, len(families), figsize=(4.6 * len(families), 8.5), sharey="row"
    )
    depth_edges = np.arange(21) * thickness_m

    for col, kind in enumerate(families):
        ax_profile, ax_disp = axes[0, col], axes[1, col]
        ax_profile.axhspan(anomaly_top, anomaly_bottom, color="#eda100", alpha=0.10)

        for rep_idx, row in enumerate(reps[kind]):
            color = REP_COLORS[rep_idx]
            profile_x = np.repeat(vs_all[row] * 1000.0, 2)
            profile_y = np.column_stack(
                [depth_edges[:-1], depth_edges[1:]]
            ).reshape(-1)
            ax_profile.plot(profile_x, profile_y, color=color, linewidth=2)

            for mode in range(4):
                valid = masks_all[row][mode]
                ax_disp.plot(
                    frequencies[valid],
                    curves_all[row][mode][valid] * 1000.0,
                    color=color,
                    linestyle=MODE_STYLES[mode],
                    linewidth=1.8,
                )

        ax_profile.set_ylim(depth_edges[-1] + 500.0, 0.0)
        ax_profile.set_xlim(200, 2700)
        ax_profile.set_xlabel("Vs (m/s)", color=SECONDARY)
        ax_profile.grid(True, color=GRID, linewidth=0.8)
        rep_ids = ", ".join(f"{REP_NAMES[i]}=#{ids_all[r]}" for i, r in enumerate(reps[kind]))
        ax_profile.set_title(f"{KIND_TITLES[kind]}\n{rep_ids}", fontsize=10, color=INK)

        ax_disp.set_xlim(frequencies[0], frequencies[-1])
        ax_disp.set_xlabel("Frequency (Hz)", color=SECONDARY)
        ax_disp.grid(True, color=GRID, linewidth=0.8)

    axes[0, 0].set_ylabel("Depth (m)")
    axes[1, 0].set_ylabel("Phase velocity (m/s)")

    rep_handles = [
        Line2D([0], [0], color=REP_COLORS[i], linewidth=2, label=REP_NAMES[i])
        for i in range(3)
    ]
    mode_handles = [
        Line2D([0], [0], color=SECONDARY, linestyle=MODE_STYLES[m],
               linewidth=1.8, label=f"M{m}")
        for m in range(4)
    ]
    legend1 = figure.legend(
        handles=rep_handles, loc="center", ncol=3, frameon=False,
        bbox_to_anchor=(0.30, 0.955), title="Representative sample",
        title_fontsize=10,
    )
    figure.legend(
        handles=mode_handles, loc="center", ncol=4, frameon=False,
        bbox_to_anchor=(0.72, 0.955), title="Mode (bottom row)",
        title_fontsize=10,
    )
    figure.add_artist(legend1)
    figure.suptitle(
        "Model families: three representative samples each — Vs profiles (top) "
        "and M0–M3 dispersion curves (bottom)",
        y=0.995, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="#fcfcfb")
    print(f"saved {output_path}")
    for kind in families:
        ids = [int(ids_all[r]) for r in reps[kind]]
        print(f"  {kind.name}: sample_ids={ids}")


if __name__ == "__main__":
    main()
