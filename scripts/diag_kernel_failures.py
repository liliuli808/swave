"""Localize the kernel rows whose relative L2 error exceeds 5%.

Answers three questions about the failing tail that summary.json cannot:
  1. Which modes / frequencies are the failures concentrated at?
  2. Which model kinds (normal / low velocity / high velocity / coupled) fail?
  3. Do failing rows have abnormally small physical-kernel norms (i.e. is the
     relative error inflated by a near-zero denominator near mode kissing)?

Usage (on the GPU machine, from the repo root):
    python scripts/diag_kernel_failures.py \
        --checkpoint runs/kernel-wide-768/best.pt \
        --dataset-dir data/production --kernel-dir data/kernels \
        --cache-dir data/cache --corrections results/kissing-repair/corrections.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swave.kernel_training import (
    Normalizer,
    apply_corrections,
    load_kernel_rows,
    load_split_rows,
    network_kernels,
)
from swave.network import model_from_checkpoint

KIND_NAMES = ["normal", "low velocity", "high velocity", "coupled HVL+LVL"]
THRESHOLD = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-dir", default="data/production", type=Path)
    parser.add_argument("--kernel-dir", default="data/kernels", type=Path)
    parser.add_argument("--cache-dir", default="data/cache", type=Path)
    parser.add_argument("--corrections", default="results/kissing-repair/corrections.npz", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(payload)
    model.eval()
    forward = Normalizer.from_payload(payload).physical(model)

    rows = load_split_rows(args.dataset_dir, "test", args.cache_dir)
    apply_corrections(rows, args.corrections)
    kernel_rows = load_kernel_rows(args.kernel_dir, "test", args.corrections)
    kind_by_id = dict(zip(rows["sample_id"].tolist(), rows["model_kind"].tolist()))

    physical = kernel_rows["kernel"].astype(np.float32)
    mask = kernel_rows["kernel_mask"]
    network = network_kernels(forward, kernel_rows["vs"].astype(np.float32))

    norm_phys = np.linalg.norm(physical, axis=-1)
    err = np.linalg.norm(network - physical, axis=-1) / np.maximum(norm_phys, 1e-12)
    freqs = np.arange(0.5, 60.0 + 0.25, 0.5)[: err.shape[-1]]

    bad = (err > THRESHOLD) & mask
    total, n_bad = int(mask.sum()), int(bad.sum())
    print(f"rows >5%: {n_bad}/{total} = {100 * n_bad / total:.3f}%")
    print(f"median kernel norm (all valid rows): {np.median(norm_phys[mask]):.4g}")

    for m in range(err.shape[1]):
        bm = bad[:, m]
        print(f"\n=== mode {m}: bad {int(bm.sum())}/{int(mask[:, m].sum())} "
              f"({100 * bm.sum() / max(mask[:, m].sum(), 1):.3f}%) ===")
        if not bm.any():
            continue

        # failure count per frequency
        hist = np.bincount(np.where(bm)[1], minlength=err.shape[-1])
        top = np.argsort(hist)[::-1][:12]
        print("worst frequencies (Hz: #failing rows):")
        print("  " + ", ".join(f"{freqs[i]:.1f}:{int(hist[i])}" for i in sorted(top)))
        frac = hist / np.maximum(mask[:, m].sum(axis=0), 1)
        worst = int(np.argmax(frac))
        print(f"highest failure fraction: {freqs[worst]:.1f} Hz "
              f"({100 * frac[worst]:.1f}% of rows at that frequency)")

        # failure count per model kind
        kinds = np.array([kind_by_id[int(s)] for s in kernel_rows["sample_id"]])
        for k, name in enumerate(KIND_NAMES):
            sub = bm[kinds == k]
            if sub.size:
                print(f"  {name:<18} {100 * sub.mean():6.2f}% of its rows fail")

        # denominator diagnosis: are failing rows near-zero-norm rows?
        print(f"median |kernel| of failing rows:  {np.median(norm_phys[bm]):.4g}")
        print(f"median |kernel| of passing rows:  {np.median(norm_phys[mask & ~bad]):.4g}")
        print(f"failing rows with |kernel| < 1% of median: "
              f"{100 * (norm_phys[bm] < 0.01 * np.median(norm_phys[mask])).mean():.1f}%")

        # error excluding the worst 2.4%: how close is the body to the target?
        e = np.sort(err[:, m][mask[:, m]])
        print(f"p95 error: {100 * e[int(0.95 * (e.size - 1))]:.3f}%  "
              f"p99: {100 * e[int(0.99 * (e.size - 1))]:.3f}%  "
              f"max: {100 * e[-1]:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
