"""Render the same-holdout depth comparison used in report.tex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    supervised = json.loads(args.supervised.read_text(encoding="utf-8"))[
        "inversion_comparison"
    ]["per_layer"]
    inversion = json.loads(args.inversion.read_text(encoding="utf-8"))[
        "experiment_scopes"
    ]["full"]["groups"]["noise"]["clean"]["vs"]["per_layer"]

    depth = np.asarray([row["depth_km"] for row in supervised])
    supervised_mae = np.asarray([row["mae_km_s"] for row in supervised])
    supervised_bias = np.asarray([row["bias_km_s"] for row in supervised])
    inversion_mae = np.asarray(inversion["mae_km_s"])
    inversion_bias = np.asarray(inversion["bias_km_s"])

    if not (
        len(depth)
        == len(supervised_mae)
        == len(supervised_bias)
        == len(inversion_mae)
        == len(inversion_bias)
        == 20
    ):
        raise ValueError("expected exactly 20 aligned depth layers")

    blue = "#2563EB"
    orange = "#D97706"
    ink = "#252A34"
    grid = "#D8DCE3"
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)

    axes[0].plot(
        depth,
        inversion_mae,
        color=orange,
        linestyle="--",
        marker="o",
        markersize=4,
        label="L-BFGS-B (clean)",
    )
    axes[0].plot(
        depth,
        supervised_mae,
        color=blue,
        linestyle="-",
        marker="s",
        markersize=4,
        label="Supervised ensemble",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Layer-wise MAE (log scale)")
    axes[0].set_ylabel("MAE (km/s)")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].axhline(0.0, color=ink, linewidth=0.9)
    axes[1].plot(
        depth,
        inversion_bias,
        color=orange,
        linestyle="--",
        marker="o",
        markersize=4,
        label="L-BFGS-B (clean)",
    )
    axes[1].plot(
        depth,
        supervised_bias,
        color=blue,
        linestyle="-",
        marker="s",
        markersize=4,
        label="Supervised ensemble",
    )
    axes[1].set_title("Layer-wise bias")
    axes[1].set_ylabel("Bias (km/s)")

    for axis in axes:
        axis.set_xlabel("Depth (km)")
        axis.set_xlim(-0.03, 1.93)
        axis.set_xticks(np.arange(0.0, 2.0, 0.2))
        axis.grid(True, color=grid, linewidth=0.7, alpha=0.8)
        axis.tick_params(colors=ink)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Vs recovery on the same 90–99 holdout (100,000 samples)",
        color=ink,
        fontsize=12,
        fontweight="semibold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
