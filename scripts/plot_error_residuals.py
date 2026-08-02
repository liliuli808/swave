#!/usr/bin/env python3
"""Plot NN surrogate error residuals versus frequency, by family and mode.

The curve-overlay figures make the errors invisible (median |error| is ~0.08
m/s on a ~2000 m/s axis). This figure shows them directly: for each model
family (rows) and mode (columns), the distribution of |NN - truth| across
held-out test samples as a function of frequency — median line, p5–p95 band,
and worst-case envelope, on a log scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SWAVE_ROOT = Path("/home/smbu/swave")
sys.path.insert(0, str(SWAVE_ROOT / "src"))

from swave.inference import ForwardPredictor  # noqa: E402

KIND_TITLES = {0: "Normal", 1: "Low-velocity", 2: "High-velocity",
               3: "Coupled high+low"}
BAND_COLOR = "#cde2fb"
MEDIAN_COLOR = "#3987e5"
MAX_COLOR = "#0d366b"
GRID = "#e1e0d9"
SECONDARY = "#52514e"
FREQUENCIES = np.arange(0.5, 60.0 + 0.25, 0.5)


def load_test_errors(dataset_dir: Path, shards: range, checkpoint: str):
    """Return (kinds, errors_m/s, mask) for test-split rows of the shards."""
    predictor = ForwardPredictor.load(checkpoint, device="cpu")
    kinds_all, errs_all, masks_all = [], [], []
    for shard in shards:
        shard_path = dataset_dir / f"shard-{shard:05d}.h5"
        if not shard_path.exists():
            continue
        with h5py.File(shard_path, "r") as handle:
            sample_ids = handle["sample_id"][:]
            test = (sample_ids % 100) >= 95
            if not test.any():
                continue
            vs = handle["vs"][test]
            true = handle["phase_velocity"][test]
            mask = handle["valid_mask"][test]
            nn = predictor.predict(vs)
            kinds_all.append(handle["model_kind"][test])
            errs_all.append(np.abs(nn - true) * 1000.0)
            masks_all.append(mask)
    return (np.concatenate(kinds_all), np.concatenate(errs_all),
            np.concatenate(masks_all))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",
                        default=str(SWAVE_ROOT / "data/production-w64"))
    parser.add_argument("--checkpoint",
                        default=str(SWAVE_ROOT / "runs/production-48g/best.pt"))
    parser.add_argument("--shards", default="95-99",
                        help="shard range, e.g. 95-99")
    parser.add_argument("--output",
                        default=str(SWAVE_ROOT / "results/nn-error-residuals.png"))
    args = parser.parse_args()

    lo, hi = (int(x) for x in args.shards.split("-"))
    kinds, errs, masks = load_test_errors(
        Path(args.dataset_dir), range(lo, hi + 1), args.checkpoint)
    print(f"test samples: {len(kinds)}")

    figure, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True,
                                sharey=True)
    for row, (kind_id, kind_name) in enumerate(KIND_TITLES.items()):
        sel = kinds == kind_id
        for mode in range(4):
            ax = axes[row, mode]
            band_lo, band_hi, median, worst = (np.full(FREQUENCIES.size, np.nan)
                                               for _ in range(4))
            for fi in range(FREQUENCIES.size):
                valid = masks[sel, mode, fi]
                if valid.sum() < 10:
                    continue
                values = errs[sel, mode, fi][valid]
                band_lo[fi] = np.percentile(values, 5)
                band_hi[fi] = np.percentile(values, 95)
                median[fi] = np.median(values)
                worst[fi] = values.max()
            ax.fill_between(FREQUENCIES, band_lo, band_hi, color=BAND_COLOR,
                            linewidth=0)
            ax.plot(FREQUENCIES, median, color=MEDIAN_COLOR, linewidth=2)
            ax.plot(FREQUENCIES, worst, color=MAX_COLOR, linewidth=1)
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 50)
            ax.grid(True, color=GRID, linewidth=0.8, which="both")
            if row == 0:
                ax.set_title(f"M{mode}", fontsize=12)
            if row == 3:
                ax.set_xlabel("Frequency (Hz)", color=SECONDARY)
        axes[row, 0].set_ylabel(f"{kind_name}\n|error| (m/s)")

    handles = [
        Patch(facecolor=BAND_COLOR, label="p5–p95 band"),
        Line2D([0], [0], color=MEDIAN_COLOR, linewidth=2, label="Median"),
        Line2D([0], [0], color=MAX_COLOR, linewidth=1, label="Worst sample"),
    ]
    figure.legend(handles=handles, loc="center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 0.955))
    figure.suptitle(
        "NN surrogate |error| vs frequency by model family and mode "
        f"(held-out test samples, n={len(kinds)})", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="#fcfcfb")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
