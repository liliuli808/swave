"""Audit individual failing kernel rows and random reference rows, read-only.

Defaults to validation. A diagnostic JSON supplies the worst failing rows;
four background models per kind add one valid row per mode. Only the selected
models' labels are loaded. This is a targeted consistency check, not an
estimate of dataset-wide label accuracy or a proof of physical correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads

from swave.kernel_audit import (
    STEPS,
    audit_kernel_rows,
    kernel_model_metadata,
    select_audit_rows,
    summarize_audit,
)
from swave.kernel_training import load_kernel_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/production", type=Path)
    parser.add_argument("--kernel-dir", default="data/kernels", type=Path)
    parser.add_argument("--cache-dir", default="data/cache", type=Path,
                        help="Accepted for compatibility; reads shard metadata directly.")
    parser.add_argument("--corrections", default="results/kissing-repair/corrections.npz", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--diagnostics", type=Path, help="JSON from diag_kernel_failures.py")
    parser.add_argument("--reference-models-per-kind", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference_models_per_kind < 0 or args.threads < 1:
        parser.error("reference-models-per-kind must be nonnegative; threads must be positive")
    diagnostic = json.loads(args.diagnostics.read_text()) if args.diagnostics else None
    if diagnostic is not None and diagnostic.get("split") != args.split:
        parser.error("diagnostic split does not match --split")
    targets = diagnostic["worst_rows"] if diagnostic is not None else []
    set_num_threads(args.threads)
    ids, kinds = kernel_model_metadata(args.dataset_dir, args.kernel_dir, args.split)
    rng = np.random.default_rng(args.seed)
    reference_ids = np.concatenate([
        rng.choice(ids[kinds == kind], min(args.reference_models_per_kind,
                                         int((kinds == kind).sum())), replace=False)
        for kind in np.unique(kinds)
    ])
    requested = np.unique(np.concatenate([
        reference_ids, np.asarray([row["sample_id"] for row in targets], dtype=np.uint64),
    ]))
    if not len(requested):
        parser.error("no audit models selected")
    rows = load_kernel_rows(args.kernel_dir, args.split, args.corrections, sample_ids=requested)
    chosen = select_audit_rows(rows, targets, reference_ids, seed=args.seed)
    print(json.dumps({"label_audit": {
        "split": args.split, "models": len(requested), "rows": len(chosen),
        "diagnostic_rows": sum(row["selection"] == "diagnostic_failure" for row in chosen),
        "step_sizes_km_s": STEPS, "root_strategy": "consensus", "root_tolerance": 1e-11,
    }}), flush=True)

    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(json.dumps({"audit_progress": {"completed": done, "total": total}}), flush=True)

    records = audit_kernel_rows(rows, chosen, progress=progress)
    kind_by_id = dict(zip(ids.tolist(), kinds.tolist(), strict=True))
    for record in records:
        record["model_kind"] = int(kind_by_id[record["sample_id"]])
    report = {
        "split": args.split, "diagnostics": str(args.diagnostics) if args.diagnostics else None,
        "checkpoint": diagnostic.get("checkpoint") if diagnostic else None,
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "step_sizes_km_s": STEPS, "root_strategy": "consensus", "root_tolerance": 1e-11,
        "summary": summarize_audit(records), "rows": records,
        "note": "Targeted failures and stratified random references do not estimate population "
                "prevalence. All checks share the existing secular function. Agreement does not "
                "certify physical correctness; disagreements do not automatically justify "
                "changing labels or acceptance masks.",
    }
    print(json.dumps({"label_audit_summary": report["summary"]}, allow_nan=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
