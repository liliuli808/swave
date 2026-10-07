"""Compare forward surrogates with the physical solver in value and kernel.

Value agreement is measured on every row of the test split (IDs ending 85-89);
kernel agreement on the test rows that have physical kernel labels. The last
checkpoint given is plotted; earlier ones only appear in the summary and the
error-distribution figure.

    python scripts/evaluate_forward_kernels.py \
        --checkpoint base=runs/production-48g/best.pt \
        --checkpoint finetuned=runs/kernel-finetune/best.pt \
        --dataset-dir data/production --kernel-dir data/kernels \
        --cache-dir data/cache --output-dir results/forward-kernel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from swave.empirical import brocher05
from swave.kernel_training import (
    Normalizer,
    apply_corrections,
    evaluate_rows,
    kernel_metrics,
    load_kernel_rows,
    load_split_rows,
    network_kernels,
    predict,
)
from swave.network import model_from_checkpoint
from swave.secular import LayeredModel, RayleighSecular

FREQUENCIES = np.arange(0.5, 60.0 + 0.25, 0.5)
DEPTH_TOP = np.arange(20) * 0.1
MODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
KIND_NAMES = ["normal", "low velocity", "high velocity", "coupled HVL+LVL"]
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _load(path: Path) -> callable:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(payload)
    model.eval()
    return Normalizer.from_payload(payload).physical(model)


def _representatives(kernel_rows, kinds_by_id) -> list[int]:
    kinds = np.array([kinds_by_id[int(i)] for i in kernel_rows["sample_id"]])
    chosen = []
    for kind in range(4):
        candidates = np.flatnonzero(kinds == kind)
        # A fully valid model keeps all four modes on the plots.
        full = [i for i in candidates if kernel_rows["valid_mask"][i].sum() > 470]
        chosen.append(int((full or list(candidates))[0]))
    return chosen


def plot_dispersion(rows, prediction, chosen, output: Path) -> None:
    figure, axes = plt.subplots(
        3, 4, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1.1, 2, 1.1]}
    )
    for column, index in enumerate(chosen):
        vs = rows["vs"][index]
        profile = axes[0, column]
        profile.step(
            np.r_[vs, vs[-1]], np.r_[DEPTH_TOP, 2.1], where="post", color=INK, lw=1.6
        )
        profile.invert_yaxis()
        profile.set_xlabel("Vs (km/s)")
        profile.set_ylabel("Depth (km)" if column == 0 else "")
        profile.set_title(f"{KIND_NAMES[column]} (sample {rows['sample_id'][index]})")
        curves = axes[1, column]
        errors = axes[2, column]
        for mode in range(4):
            valid = rows["valid_mask"][index, mode]
            target = rows["phase_velocity"][index, mode]
            curves.plot(
                FREQUENCIES[valid], target[valid], color=MODE_COLORS[mode], lw=2,
                label=f"M{mode} physical",
            )
            curves.plot(
                FREQUENCIES[valid], prediction[index, mode][valid], color=INK,
                lw=1, ls="--", label="network" if mode == 0 else None,
            )
            relative = 100 * (prediction[index, mode] - target) / target
            errors.plot(FREQUENCIES[valid], relative[valid], color=MODE_COLORS[mode], lw=1.2)
        curves.set_xlabel("Frequency (Hz)")
        curves.set_ylabel("Phase velocity (km/s)" if column == 0 else "")
        errors.axhspan(-1, 1, color="#e4e3df", alpha=0.6, lw=0)
        errors.set_xlabel("Frequency (Hz)")
        errors.set_ylabel("Relative error (%)" if column == 0 else "")
        limit = max(1.2, float(np.nanmax(np.abs(
            np.where(rows["valid_mask"][index],
                     100 * (prediction[index] - rows["phase_velocity"][index])
                     / rows["phase_velocity"][index], 0)))) * 1.2)
        errors.set_ylim(-limit, limit)
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Network vs physical (Pan & Chen secular-function roots): dispersion curves; "
        "grey band = ±1 %"
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_kernel_maps(physical, network, mask, vs, output: Path, title: str) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    extent = [FREQUENCIES[0], FREQUENCIES[-1], 2.0, 0.0]
    for mode in range(4):
        valid = mask[mode]
        phys = np.where(valid[:, None], physical[mode], np.nan).T
        net = np.where(valid[:, None], network[mode], np.nan).T
        limit = np.nanmax(np.abs(phys))
        norm = TwoSlopeNorm(0.0, -limit, limit)
        for column, (data, name) in enumerate(
            [(phys, "physical"), (net, "network"), (net - phys, "network − physical")]
        ):
            axis = axes[mode, column]
            image = axis.imshow(
                data, aspect="auto", extent=extent, cmap="RdBu_r",
                norm=norm if column < 2 else TwoSlopeNorm(0.0, -limit / 10, limit / 10),
                interpolation="nearest",
            )
            axis.grid(False)
            axis.set_title(f"M{mode} {name}")
            if column == 0:
                axis.set_ylabel("Layer top depth (km)")
            if mode == 3:
                axis.set_xlabel("Frequency (Hz)")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02,
                            label="dc/dVs" if column < 2 else "dc/dVs (±10 % scale)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_kernel_profiles(physical, network, mask, output: Path, title: str) -> None:
    picks = [3.0, 10.0, 30.0]
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for mode in range(4):
        axis = axes[mode]
        for frequency in picks:
            index = round(frequency / 0.5) - 1
            if not mask[mode, index]:
                continue
            alpha = {3.0: 1.0, 10.0: 0.7, 30.0: 0.45}[frequency]
            axis.step(physical[mode, index], DEPTH_TOP + 0.05, where="mid",
                      color=MODE_COLORS[mode], lw=2.2, alpha=alpha,
                      label=f"{frequency:g} Hz physical")
            axis.plot(network[mode, index], DEPTH_TOP + 0.05, "o", ms=4,
                      mfc="none", mec=INK, alpha=alpha,
                      label=f"{frequency:g} Hz network")
        axis.set_title(f"Mode {mode}")
        axis.set_xlabel("dc/dVs")
        axis.legend(frameon=False, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Layer mid depth (km)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_spectrum(vs, target, mask, prediction, output: Path, title: str) -> None:
    vp, density = brocher05(vs.astype(float))
    secular = RayleighSecular(
        LayeredModel(DEPTH_TOP, density, vs.astype(float), vp)
    )
    velocities = np.linspace(0.2, float(vs.max()) * 1.02, 500)
    image = np.full((velocities.size, FREQUENCIES.size), np.nan)
    for column, frequency in enumerate(FREQUENCIES):
        values = []
        for velocity in velocities:
            try:
                values.append(secular.evaluate(frequency, velocity))
            except ArithmeticError:
                values.append(np.nan)
        values = np.abs(np.asarray(values))
        values = np.log10(values / np.nanmax(values) + 1e-300)
        image[:, column] = values
    figure, axis = plt.subplots(figsize=(8.5, 5.6))
    shown = axis.imshow(
        np.clip(image, -8, 0), origin="lower", aspect="auto", cmap="Greys_r",
        extent=[FREQUENCIES[0], FREQUENCIES[-1], velocities[0], velocities[-1]],
    )
    axis.grid(False)
    figure.colorbar(shown, ax=axis, label="log10 |F(f, c)| (column-normalized)")
    for mode in range(4):
        valid = mask[mode]
        axis.plot(FREQUENCIES[valid], target[mode][valid], "o", ms=3.2,
                  mfc="none", mec=MODE_COLORS[mode], label=f"M{mode} physical roots")
        axis.plot(FREQUENCIES[valid], prediction[mode][valid], color=MODE_COLORS[mode],
                  lw=1.2, label=f"M{mode} network")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Phase velocity (km/s)")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_distributions(results, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = ["--", "-"]
    for (name, data), style in zip(results.items(), styles * 4, strict=False):
        for mode in range(4):
            values = np.sort(data["point_rel"][mode])
            survival = 1.0 - np.arange(values.size) / values.size
            axes[0].loglog(values * 100, survival, color=MODE_COLORS[mode], ls=style,
                           lw=1.4, label=f"{name} M{mode}")
            kernel = np.sort(data["kernel_rel"][mode])
            survival = 1.0 - np.arange(kernel.size) / kernel.size
            axes[1].loglog(kernel * 100, survival, color=MODE_COLORS[mode], ls=style,
                           lw=1.4, label=f"{name} M{mode}")
    axes[0].axvline(1.0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("Point relative error of phase velocity (%)")
    axes[0].set_ylabel("Fraction of cells exceeding")
    axes[1].axvline(5.0, color=MUTED, lw=0.8)
    axes[1].set_xlabel("Kernel row relative L2 error (%)")
    axes[1].set_ylabel("Fraction of kernel rows exceeding")
    for axis in axes:
        axis.legend(frameon=False, fontsize=6.5, ncol=2)
    figure.suptitle("Error survival curves on the test split (dashed: first, solid: last)")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=path; may repeat, last one is plotted")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--corrections", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = load_split_rows(args.dataset_dir, "test", args.cache_dir)
    apply_corrections(test_rows, args.corrections)
    kernel_rows = load_kernel_rows(args.kernel_dir, "test", args.corrections)
    kinds_by_id = dict(zip(test_rows["sample_id"].tolist(),
                           test_rows["model_kind"].tolist(), strict=True))
    chosen = _representatives(kernel_rows, kinds_by_id)
    physical_kernel = kernel_rows["kernel"].astype(np.float32)
    kernel_mask = kernel_rows["kernel_mask"]

    summary: dict[str, object] = {}
    distributions: dict[str, dict[str, list]] = {}
    forward = None
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        forward = _load(Path(path))
        value = evaluate_rows(forward, {
            key: test_rows[key] for key in ("vs", "phase_velocity", "valid_mask")
        })["value"]
        prediction = predict(forward, test_rows["vs"].astype(np.float32))
        network_kernel = network_kernels(forward, kernel_rows["vs"].astype(np.float32))
        kernel = kernel_metrics(network_kernel, physical_kernel, kernel_mask)
        summary[name] = {"checkpoint": path, "test_rows": len(test_rows["vs"]),
                         "kernel_test_rows": len(kernel_rows["vs"]),
                         "value": value, "kernel": kernel}
        relative = np.abs(prediction - test_rows["phase_velocity"]) / test_rows[
            "phase_velocity"]
        row_error = np.linalg.norm(network_kernel - physical_kernel, axis=-1) / np.maximum(
            np.linalg.norm(physical_kernel, axis=-1), 1e-12)
        distributions[name] = {
            "point_rel": [relative[:, m][test_rows["valid_mask"][:, m]] for m in range(4)],
            "kernel_rel": [row_error[:, m][kernel_mask[:, m]] for m in range(4)],
        }
        print(json.dumps({name: {
            "samples_all_within_1pct": value["samples_all_within_1pct"],
            "points_within_1pct": value["points_within_1pct"],
            "max_relative_error": value["max_relative_error"],
            "kernel_rows_within_5pct": kernel["rows_within_5pct"],
            "kernel_median": kernel["median_relative_l2"]}}), flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_distributions(distributions, args.output_dir / "error-survival.png")

    vs = kernel_rows["vs"].astype(np.float32)
    prediction = predict(forward, vs)
    network_kernel = network_kernels(forward, vs[chosen])
    plot_dispersion(kernel_rows, prediction, chosen, args.output_dir / "dispersion-comparison.png")
    for slot, index in enumerate(chosen):
        tag = KIND_NAMES[slot].split()[0]
        label = f"{KIND_NAMES[slot]} model, sample {kernel_rows['sample_id'][index]}"
        plot_kernel_maps(physical_kernel[index], network_kernel[slot], kernel_mask[index],
                         vs[index], args.output_dir / f"kernel-maps-{tag}.png",
                         f"Sensitivity kernels dc/dVs (Vp, density via Brocher 2005): {label}")
        plot_kernel_profiles(physical_kernel[index], network_kernel[slot], kernel_mask[index],
                             args.output_dir / f"kernel-profiles-{tag}.png",
                             f"Kernel depth profiles: {label}")
        plot_spectrum(vs[index], kernel_rows["phase_velocity"][index],
                      kernel_rows["valid_mask"][index], prediction[index],
                      args.output_dir / f"dispersion-spectrum-{tag}.png",
                      f"Dispersion spectrum |F(f,c)| with roots and network: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
