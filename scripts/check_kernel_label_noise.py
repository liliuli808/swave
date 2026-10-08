"""Estimate the noise floor of physical kernel labels.

Physical kernels are finite differences of the secular determinant with
DEFAULT_STEP_KM_S = 1e-5. If recomputing labels at other step sizes changes
them by ~5-10% relative L2, the 99%-within-5% acceptance target is
unreachable for those rows no matter how well the network trains.

Recomputes kernels for a stratified sample of test-split models at three
finite-difference steps and reports, per model kind and mode, how much the
labels themselves move. Also prints the kind distribution of the
kernel-labelled train models (class-imbalance check).

Usage (on the GPU machine, from the repo root):
    python3 scripts/check_kernel_label_noise.py \
        --dataset-dir data/production --kernel-dir data/kernels \
        --cache-dir data/cache --corrections results/kissing-repair/corrections.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swave.kernel_training import (
    apply_corrections,
    load_kernel_rows,
    load_split_rows,
)
from swave.kernels import sensitivity_kernels

KIND_NAMES = ["normal", "low velocity", "high velocity", "coupled HVL+LVL"]
STEPS = [1e-6, 1e-5, 1e-4]
REFERENCE_STEP = 1e-5
FREQ_STEP_HZ = 0.5
PER_KIND = 75


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/production", type=Path)
    parser.add_argument("--kernel-dir", default="data/kernels", type=Path)
    parser.add_argument("--cache-dir", default="data/cache", type=Path)
    parser.add_argument("--corrections", default="results/kissing-repair/corrections.npz", type=Path)
    args = parser.parse_args()

    rows = load_split_rows(args.dataset_dir, "train", args.cache_dir)
    apply_corrections(rows, args.corrections)
    kinds_train, counts_train = np.unique(rows["model_kind"], return_counts=True)
    print("train split kinds:", dict(zip(
        [KIND_NAMES[k] for k in kinds_train.tolist()], counts_train.tolist())))

    kernel_rows = load_kernel_rows(args.kernel_dir, "test", args.corrections)
    kind_by_id = dict(zip(rows["sample_id"].tolist(), rows["model_kind"].tolist()))
    kinds_test = np.array([kind_by_id[int(s)] for s in kernel_rows["sample_id"]])

    train_kernel_ids = load_kernel_rows(args.kernel_dir, "train", args.corrections)["sample_id"]
    kinds_kernel_train = np.array([kind_by_id[int(s)] for s in train_kernel_ids])
    kinds_k, counts_k = np.unique(kinds_kernel_train, return_counts=True)
    print("kernel-labelled train kinds:", dict(zip(
        [KIND_NAMES[k] for k in kinds_k.tolist()], counts_k.tolist())))

    # stratified sample of test models
    rng = np.random.default_rng(0)
    picked = np.concatenate([
        rng.choice(np.where(kinds_test == k)[0], size=min(PER_KIND, int((kinds_test == k).sum())),
                   replace=False)
        for k in range(len(KIND_NAMES))
    ])

    vs = kernel_rows["vs"][picked].astype(np.float64)
    phase = kernel_rows["phase_velocity"][picked].astype(np.float64)
    mask = kernel_rows["valid_mask"][picked]
    freqs = np.arange(FREQ_STEP_HZ, 60.0 + FREQ_STEP_HZ / 2, FREQ_STEP_HZ)
    assert freqs.size == phase.shape[-1], (freqs.size, phase.shape)

    kernels = {}
    for step in STEPS:
        print(f"computing kernels at step {step:.0e} ...", flush=True)
        kernels[step] = sensitivity_kernels(vs, phase, mask, freqs, step=step)

    ref = kernels[REFERENCE_STEP]
    for kind in range(len(KIND_NAMES)):
        sel = np.where(kinds_test[picked] == kind)[0]
        for mode in range(phase.shape[1]):
            disagreements, low_freq = [], []
            for i in sel:
                m = mask[i, mode]
                if not m.any():
                    continue
                target = ref[i, mode, m]  # (freqs, layers)
                n = np.linalg.norm(target)
                if n < 1e-12:
                    continue
                for step in STEPS:
                    if step == REFERENCE_STEP:
                        continue
                    d = np.linalg.norm(kernels[step][i, mode, m] - target) / n
                    disagreements.append(d)
                    if (freqs[m] < 2.0).any():
                        low = freqs[m] < 2.0
                        nl = np.linalg.norm(target[low])
                        if nl >= 1e-12:
                            low_freq.append(np.linalg.norm(kernels[step][i, mode, m][low] - target[low]) / nl)
            if disagreements:
                d = np.array(disagreements)
                lf = np.array(low_freq) if low_freq else np.array([np.nan])
                print(f"{KIND_NAMES[kind]:<18} M{mode}: step-sweep label drift "
                      f"median {100*np.median(d):6.3f}%  p95 {100*np.percentile(d,95):6.3f}%  | "
                      f"below 2 Hz median {100*np.nanmedian(lf):6.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
