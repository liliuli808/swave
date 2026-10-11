"""Check whether averaging trained models fixes the kernel rows they fail.

The kernel of an averaged network is the average of the members' kernels,
so averaging the predicted Jacobians gives the ensemble's kernel exactly. The
report also counts, for every valid row, how many members fail it: failures
that the members do not share point to model variance, failures shared by all
point to rows the recipe cannot generalize to.

    PYTHONPATH=src python scripts/ensemble_kernel_check.py \
        --checkpoint runs/kernel-wide-kw3-fixed/best-kernel.pt \
        --checkpoint runs/kernel-wide-kw3-fixed-wd1e-2/best-kernel.pt \
        --checkpoint runs/kernel-wide-pf-fixed/best-kernel.pt \
        --device cuda --output results/kernel-ensemble/validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swave.kernel_diagnostics import summarize_kernel_errors
from swave.kernel_training import (
    Normalizer,
    load_kernel_rows,
    load_split_rows,
    network_kernels,
    predict,
    value_metrics,
)
from swave.network import model_from_checkpoint
from swave.splits import validate_checkpoint_split_policy

THRESHOLD = 0.05


def relative_l2(network: np.ndarray, physical: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(physical, axis=-1)
    return np.linalg.norm(network - physical, axis=-1) / np.maximum(norm, 1e-12)


def shared_failures(member_errors: list[np.ndarray], mask: np.ndarray) -> dict[str, object]:
    """Distribution of how many members fail each valid row (exact 5% fails)."""
    failing = np.stack([~(error[mask] < THRESHOLD) for error in member_errors])
    count = failing.sum(axis=0)
    rows = int(mask.sum())
    any_failed = int((count > 0).sum())
    return {
        "rows": rows,
        "rows_failed_by_n_members": {
            str(n): int((count == n).sum()) for n in range(len(member_errors) + 1)
        },
        "failed_by_any": any_failed,
        "failed_by_all": int((count == len(member_errors)).sum()),
        "fraction_of_any_failed_by_all": (
            float((count == len(member_errors)).sum() / any_failed) if any_failed else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path,
                        help="member checkpoint file; give at least two")
    parser.add_argument("--dataset-dir", default="data/production", type=Path)
    parser.add_argument("--kernel-dir", default="data/kernels", type=Path)
    parser.add_argument("--cache-dir", default="data/cache", type=Path)
    parser.add_argument("--corrections", default="results/kissing-repair/corrections.npz",
                        type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--output", type=Path, help="save the report as JSON")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        parser.error("give at least two --checkpoint members")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    torch.set_num_threads(args.threads)

    rows = load_split_rows(args.dataset_dir, args.split, args.cache_dir)
    kind_by_id = dict(zip(rows["sample_id"].tolist(), rows["model_kind"].tolist()))
    del rows
    kernel_rows = load_kernel_rows(args.kernel_dir, args.split, args.corrections)
    kinds = np.asarray([kind_by_id[int(sample)] for sample in kernel_rows["sample_id"]])
    physical = kernel_rows["kernel"].astype(np.float32)
    vs = kernel_rows["vs"].astype(np.float32)
    mask = kernel_rows["kernel_mask"]
    norm = np.linalg.norm(physical, axis=-1)
    frequencies = np.arange(physical.shape[2], dtype=np.float64) * 0.5 + 0.5
    print(f"loaded {len(vs)} kernel-labelled {args.split} models", flush=True)

    kernel_sum = np.zeros_like(physical)
    value_sum = None
    member_errors = []
    members = []
    for path in args.checkpoint:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_split_policy(payload)
        model = model_from_checkpoint(payload).to(args.device)
        model.eval()
        forward = Normalizer.from_payload(payload).physical(model)
        kernel = network_kernels(forward, vs, batch_size=args.batch_size)
        values = predict(forward, vs)
        kernel_sum += kernel
        value_sum = values if value_sum is None else value_sum + values
        error = relative_l2(kernel, physical)
        del kernel
        member_errors.append(error)
        summary = summarize_kernel_errors(
            error, norm, mask, kernel_rows["sample_id"], kinds, frequencies)
        curves = value_metrics(values, kernel_rows["phase_velocity"],
                               kernel_rows["valid_mask"])["samples_all_within_1pct"]
        members.append({"checkpoint": str(path), "epoch": payload.get("epoch"),
                        "samples_all_within_1pct": curves, "kernel": summary})
        print(f"{path}: kernel rows <5% {summary['overall']['rows_within_5pct']:.4%}, "
              f"curves {curves:.4%}", flush=True)
        del payload, model

    count = len(args.checkpoint)
    ensemble_error = relative_l2(kernel_sum / count, physical)
    ensemble = summarize_kernel_errors(
        ensemble_error, norm, mask, kernel_rows["sample_id"], kinds, frequencies)
    ensemble_curves = value_metrics(value_sum / count, kernel_rows["phase_velocity"],
                                    kernel_rows["valid_mask"])["samples_all_within_1pct"]
    overlap = shared_failures(member_errors, mask)
    report = {
        "split": args.split, "threshold_relative_l2": THRESHOLD, "members": members,
        "ensemble": {"samples_all_within_1pct": ensemble_curves, "kernel": ensemble},
        "shared_failures": overlap,
    }

    print(f"\nensemble of {count}: kernel rows <5% "
          f"{ensemble['overall']['rows_within_5pct']:.4%}, curves {ensemble_curves:.4%}")
    print(f"{'group':<18}" + "".join(f"{'member ' + str(i):>11}" for i in range(count))
          + f"{'ensemble':>11}")
    groups = [("all", lambda summary: summary["overall"])]
    groups += [(kind, lambda summary, kind=kind: summary["by_kind"][kind])
               for kind in ensemble["by_kind"]]
    groups += [(f"M{mode}", lambda summary, mode=mode: summary["by_mode"][mode])
               for mode in ensemble["by_mode"]]
    for name, pick in groups:
        cells = [pick(member["kernel"])["rows_within_5pct"] for member in members]
        cells.append(pick(ensemble)["rows_within_5pct"])
        print(f"{name:<18}" + "".join(
            f"{'n/a':>11}" if cell is None else f"{cell:>11.4%}" for cell in cells))
    print("rows failed by n members: "
          + json.dumps(overlap["rows_failed_by_n_members"]))
    print(f"failed by all members: {overlap['failed_by_all']} of "
          f"{overlap['failed_by_any']} rows failed by any")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
