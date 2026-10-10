"""Diagnose kernel failures on validation data before choosing another experiment.

Run from the repository root with PYTHONPATH=src. Test data is available only
through an explicit --split test for a final audit, not for training decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swave.kernel_diagnostics import summarize_history, summarize_kernel_errors
from swave.kernel_training import (
    Normalizer,
    kernel_checkpoint_key,
    load_kernel_rows,
    load_split_rows,
    network_kernels,
    predict,
    value_metrics,
)
from swave.network import model_from_checkpoint
from swave.splits import validate_checkpoint_split_policy


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.4f}%"


def select_checkpoint(path: Path) -> tuple[Path, list[dict]]:
    """For a run directory, rank only weights that are actually on disk."""
    if not path.is_dir():
        return path, []
    history_path = path / "history.json"
    history = json.loads(history_path.read_text())["epochs"] if history_path.exists() else []
    by_epoch = {row["epoch"]: row["validation"] for row in history}
    candidates = []
    eligible = []
    for name in ("best-kernel.pt", "best.pt", "last.pt"):
        checkpoint = path / name
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = payload.get("epoch")
        validation = payload.get("validation", by_epoch.get(epoch))
        key = kernel_checkpoint_key(validation) if validation is not None else None
        candidates.append({
            "checkpoint": str(checkpoint), "epoch": epoch,
            "samples_all_within_1pct": validation["value"]["samples_all_within_1pct"]
            if validation is not None else None,
            "kernel_rows_within_5pct": validation["kernel"]["rows_within_5pct"]
            if validation is not None else None,
            "eligible": key is not None,
        })
        if key is not None:
            eligible.append((key, checkpoint))
        del payload
    if not eligible:
        raise ValueError(
            f"no saved checkpoint in {path} has validation curves >=99%; "
            "supply a checkpoint file directly to diagnose an infeasible model"
        )
    return max(eligible, key=lambda item: item[0])[1], candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="checkpoint file, or run directory to select the best "
                        "saved kernels with validation curve pass rate >=99%%")
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
    parser.add_argument("--output", type=Path,
                        help="save the full diagnostic report as JSON")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    torch.set_num_threads(args.threads)

    checkpoint, candidates = select_checkpoint(args.checkpoint)
    if candidates:
        print(json.dumps({"available_checkpoints": candidates}), flush=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_split_policy(payload)
    model = model_from_checkpoint(payload).to(args.device)
    model.eval()
    forward = Normalizer.from_payload(payload).physical(model)
    print(f"checkpoint={checkpoint}, epoch={payload.get('epoch')}, "
          f"split={args.split}, device={args.device}", flush=True)

    history_path = checkpoint.parent / "history.json"
    history = None
    if history_path.exists():
        history = summarize_history(json.loads(history_path.read_text())["epochs"])
        print(json.dumps({"history": history}), flush=True)

    rows = load_split_rows(args.dataset_dir, args.split, args.cache_dir)
    kind_by_id = dict(zip(rows["sample_id"].tolist(), rows["model_kind"].tolist()))
    del rows
    kernel_rows = load_kernel_rows(args.kernel_dir, args.split, args.corrections)
    kinds = np.asarray([kind_by_id[int(sample)] for sample in kernel_rows["sample_id"]])
    print(f"loaded {len(kinds)} kernel-labelled models", flush=True)

    physical = kernel_rows["kernel"].astype(np.float32)
    vs = kernel_rows["vs"].astype(np.float32)
    network = network_kernels(forward, vs, batch_size=args.batch_size)
    norm = np.linalg.norm(physical, axis=-1)
    error = np.linalg.norm(network - physical, axis=-1) / np.maximum(norm, 1e-12)
    del network, physical
    report = summarize_kernel_errors(
        error, norm, kernel_rows["kernel_mask"], kernel_rows["sample_id"], kinds,
        np.arange(error.shape[-1], dtype=np.float64) * 0.5 + 0.5,
    )
    report.update({
        "checkpoint": str(checkpoint),
        "available_checkpoints": candidates,
        "checkpoint_epoch": payload.get("epoch"),
        "split": args.split,
        "device": args.device,
        "history": history,
        "value_on_kernel_models": value_metrics(
            predict(forward, vs), kernel_rows["phase_velocity"], kernel_rows["valid_mask"],
        ),
    })
    overall = report["overall"]
    print(f"kernel rows <5%: {_percent(overall['rows_within_5pct'])}; "
          f"failed {overall['failed_rows']}/{overall['rows']}")
    print(f"finite error bands: {json.dumps(overall['error_band_counts'])}; "
          f"nonfinite rows: {overall['nonfinite_rows']}")
    for mode, group in report["by_mode"].items():
        print(f"\nM{mode}: pass {_percent(group['rows_within_5pct'])}, "
              f"failed {group['failed_rows']}/{group['rows']}")
        for kind, values in group["by_kind"].items():
            print(f"  {kind:<18} pass {_percent(values['rows_within_5pct']):>10}; "
                  f"failed {values['failed_rows']}/{values['rows']}")
        worst = sorted(group["by_frequency"],
                       key=lambda item: item["failed_rows"], reverse=True)[:5]
        print("  largest failure counts (Hz, failed/valid): " + ", ".join(
            f"{item['frequency_hz']:.1f}: {item['failed_rows']}/{item['rows']}"
            for item in worst if item["failed_rows"]
        ))
        print(f"  finite error quantiles: {group['finite_error_quantiles']}")
        print(f"  physical kernel norm: {group['physical_kernel_norm']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
