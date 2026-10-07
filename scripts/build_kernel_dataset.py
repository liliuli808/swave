"""Compute physical dc/dVs kernel labels for a deterministic subset of rows.

Rows are chosen by split and ``(sample_id // 100) % stride == 0`` so the subset
is stable across runs. One ``kernels-NNNNN.h5`` file is written per dataset
shard; existing complete files are skipped, so the command is resumable.

    python scripts/build_kernel_dataset.py data/production data/kernels \
        --split train --stride 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from swave.kernels import sensitivity_kernels
from swave.splits import mask_for_split

FREQUENCIES = np.arange(0.5, 60.0 + 0.25, 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    output = args.output_dir / args.split
    output.mkdir(parents=True, exist_ok=True)
    for shard in sorted(args.dataset_dir.glob("shard-*.h5")):
        target = output / shard.name.replace("shard-", "kernels-")
        if target.exists():
            continue
        started = time.time()
        with h5py.File(shard, "r") as handle:
            sample_id = np.asarray(handle["sample_id"], dtype=np.uint64)
            selected = mask_for_split(sample_id, args.split) & (
                (sample_id // 100) % args.stride == 0
            )
            rows = np.flatnonzero(selected)
            vs = np.asarray(handle["vs"], dtype=np.float64)[rows]
            phase = np.asarray(handle["phase_velocity"], dtype=np.float64)[rows]
            mask = np.asarray(handle["valid_mask"], dtype=np.bool_)[rows]
        kernels = np.empty((rows.size, 4, FREQUENCIES.size, 20), np.float32)
        for start in range(0, rows.size, args.chunk):
            stop = start + args.chunk
            kernels[start:stop] = sensitivity_kernels(
                vs[start:stop], phase[start:stop], mask[start:stop], FREQUENCIES
            )
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        with h5py.File(temporary, "w") as handle:
            handle["sample_id"] = sample_id[rows]
            handle["row"] = rows.astype(np.int64)
            handle["vs"] = vs.astype(np.float32)
            handle["phase_velocity"] = phase.astype(np.float32)
            handle["valid_mask"] = mask
            handle.create_dataset(
                "kernel", data=kernels, chunks=(64, 4, FREQUENCIES.size, 20)
            )
        temporary.replace(target)
        print(
            f"{target.name}: {rows.size} rows in {time.time() - started:.1f} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
