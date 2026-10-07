"""Find and re-solve dataset cells where a mode-kissing pair was missed.

A missed kissing pair removes two roots at one frequency and shifts every
higher mode up, which shows as a phase velocity *increasing* with frequency or
as a large disagreement with a trained surrogate. Each suspicious
``(sample, frequency)`` cell is re-solved with the ``consensus`` strategy
(Pan & Chen truncated-model roots united with deep quadratic search). Only
cells whose roots change are written to ``corrections.npz``; the original
SHA-verified shards are never modified.

    python scripts/repair_kissing_defects.py data/production data/repairs \
        --checkpoint runs/production-48g/best.pt --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from swave.config import PhysicsConfig
from swave.kernel_training import Normalizer, load_split_rows, predict
from swave.network import model_from_checkpoint
from swave.secular import LayeredModel
from swave.solver import DispersionSolver

FREQUENCIES = np.arange(0.5, 60.0 + 0.25, 0.5)
SPLITS = ("train", "validation", "test", "inversion")


def suspicious_cells(
    phase: np.ndarray, mask: np.ndarray, prediction: np.ndarray, threshold: float
) -> np.ndarray:
    """Return a ``(rows, frequencies)`` boolean array of cells to re-solve."""
    increasing = (np.diff(phase, axis=2) > 1e-6) & mask[:, :, 1:] & mask[:, :, :-1]
    cells = np.zeros(mask.shape[::2], dtype=bool)
    cells[:, 1:] |= increasing.any(axis=1)
    relative = np.where(mask, np.abs(prediction - phase) / phase, 0.0)
    cells |= (relative > threshold).any(axis=1)
    return cells


def _resolve(task: tuple[np.ndarray, int]) -> np.ndarray:
    vs, index = task
    solution = DispersionSolver(
        LayeredModel.from_vs(vs.astype(np.float64)), PhysicsConfig()
    ).solve_frequency(float(FREQUENCIES[index]), "consensus")
    roots = np.full(4, np.nan)
    roots[: solution.roots.size] = solution.roots
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(payload)
    model.eval()
    forward = Normalizer.from_payload(payload).physical(model)

    records = {"sample_id": [], "frequency_index": [], "phase_velocity": []}
    report = {}
    for split in SPLITS:
        rows = load_split_rows(args.dataset_dir, split, args.cache_dir)
        prediction = predict(forward, rows["vs"].astype(np.float32))
        cells = np.argwhere(
            suspicious_cells(
                rows["phase_velocity"], rows["valid_mask"], prediction, args.threshold
            )
        )
        with Pool(args.workers) as pool:
            solved = pool.map(
                _resolve,
                [(rows["vs"][row], int(index)) for row, index in cells],
                chunksize=64,
            )
        changed = 0
        for (row, index), roots in zip(cells, solved, strict=True):
            old = np.where(
                rows["valid_mask"][row, :, index],
                rows["phase_velocity"][row, :, index],
                np.nan,
            )
            if np.allclose(roots, old, atol=1e-5, rtol=0.0, equal_nan=True):
                continue
            changed += 1
            records["sample_id"].append(int(rows["sample_id"][row]))
            records["frequency_index"].append(int(index))
            records["phase_velocity"].append(roots)
        report[split] = {
            "rows": len(rows["vs"]),
            "checked_cells": len(cells),
            "corrected_cells": changed,
        }
        print(json.dumps({split: report[split]}), flush=True)
    np.savez(
        args.output_dir / "corrections.npz",
        sample_id=np.asarray(records["sample_id"], dtype=np.uint64),
        frequency_index=np.asarray(records["frequency_index"], dtype=np.int64),
        phase_velocity=np.asarray(records["phase_velocity"], dtype=np.float64).reshape(
            -1, 4
        ),
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
